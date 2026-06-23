# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch>=2.4",
#     "transformers>=4.46",
#     "accelerate>=1.0",
#     "lerobot",
#     "bitsandbytes>=0.44",
#     "safetensors",
#     "pillow",
#     "numpy",
#     "torchvision",
# ]
# ///
"""
Train an Inverse Dynamics Model (IDM) on a LeRobot dataset.

Given two image frames (t and t+H), the IDM predicts the H actions between them.
Trained on teleoperation data (real frames + real actions), then used to label
synthetically generated videos with pseudo-actions.

Architecture (matches DreamGen paper):
    SigLIP-2 large (frozen) → 4-layer SA backbone → 8-layer flow-matching DiT

Usage:
    uv run train_idm.py --dataset /path/to/dataset --output ./idm
    uv run train_idm.py --dataset your-hf/dataset --output ./idm --epochs 50
    uv run train_idm.py --dataset /path --camera observation.images.top
    uv run train_idm.py --dataset /path --list-cameras  # inspect dataset first
"""

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from safetensors.torch import save_file
from transformers import AutoModel


SIGLIP_MODEL = "google/siglip2-large-patch16-256"
SIGLIP_DIM = 1024
SIGLIP_SIZE = 256


# ── Model ─────────────────────────────────────────────────────────────────────

class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim)
        )
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half).float() / max(half - 1, 1))
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = t[:, None] * self.freqs[None]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return self.proj(emb)


class SABlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff    = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = self.norm1(x)
        x = x + self.attn(n, n, n, need_weights=False)[0]
        x = x + self.ff(self.norm2(x))
        return x


class DiTBlock(nn.Module):
    """Diffusion Transformer block: self-attn + cross-attn to visual context + AdaLN from timestep."""
    def __init__(self, dim: int, heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn1 = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn2 = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False)
        self.ff    = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        s1, b1, s2, b2, s3, b3 = [a.unsqueeze(1) for a in self.adaLN(t_emb).chunk(6, dim=-1)]

        h = self.norm1(x) * (1 + s1) + b1
        x = x + self.attn1(h, h, h, need_weights=False)[0]

        h = self.norm2(x) * (1 + s2) + b2
        x = x + self.attn2(h, ctx, ctx, need_weights=False)[0]

        h = self.norm3(x) * (1 + s3) + b3
        x = x + self.ff(h)
        return x


class IDM(nn.Module):
    def __init__(
        self,
        action_dim: int,
        action_horizon: int = 16,
        hidden: int = SIGLIP_DIM,
        backbone_layers: int = 4,
        dit_layers: int = 8,
        heads: int = 16,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon

        # SigLIP-2 vision encoder — frozen, weights not saved in checkpoint
        self.siglip = AutoModel.from_pretrained(SIGLIP_MODEL, torch_dtype=torch.bfloat16)
        self.siglip.requires_grad_(False)

        # SA backbone fuses patch tokens from both frames
        self.backbone = nn.ModuleList([
            SABlock(hidden, heads, dropout) for _ in range(backbone_layers)
        ])

        # Flow matching DiT head
        self.t_emb      = TimestepEmbedding(hidden)
        self.action_in  = nn.Linear(action_dim, hidden)
        self.action_out = nn.Linear(hidden, action_dim)
        self.dit        = nn.ModuleList([
            DiTBlock(hidden, heads, dropout) for _ in range(dit_layers)
        ])

    def _preprocess(self, frame: torch.Tensor) -> torch.Tensor:
        """[B, C, H, W] ∈ [0,1] → resized & normalized for SigLIP-2."""
        if frame.shape[-2:] != (SIGLIP_SIZE, SIGLIP_SIZE):
            frame = TF.resize(frame, [SIGLIP_SIZE, SIGLIP_SIZE], antialias=True)
        return frame * 2.0 - 1.0  # [0,1] → [-1,1]

    def encode_frames(self, frame_t: torch.Tensor, frame_tH: torch.Tensor) -> torch.Tensor:
        """Returns fused visual context [B, 2*(N+1), hidden] in float32."""
        pt  = self._preprocess(frame_t).to(dtype=self.siglip.dtype, device=next(self.siglip.parameters()).device)
        ptH = self._preprocess(frame_tH).to(dtype=self.siglip.dtype, device=next(self.siglip.parameters()).device)
        with torch.no_grad():
            ft  = self.siglip.vision_model(pixel_values=pt).last_hidden_state    # [B, N+1, 1024]
            ftH = self.siglip.vision_model(pixel_values=ptH).last_hidden_state
        ctx = torch.cat([ft, ftH], dim=1).float()  # [B, 2*(N+1), 1024]
        for block in self.backbone:
            ctx = block(ctx)
        return ctx

    def forward(self, frame_t: torch.Tensor, frame_tH: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Flow matching training loss (MSE on velocity field)."""
        B, device = actions.shape[0], actions.device
        ctx = self.encode_frames(frame_t, frame_tH)

        noise  = torch.randn_like(actions)
        t      = torch.rand(B, device=device)
        noisy  = (1 - t[:, None, None]) * actions + t[:, None, None] * noise
        target = noise - actions

        t_emb = self.t_emb(t)
        x = self.action_in(noisy)
        for block in self.dit:
            x = block(x, ctx, t_emb)
        return F.mse_loss(self.action_out(x), target)

    @torch.no_grad()
    def get_actions(self, frame_t: torch.Tensor, frame_tH: torch.Tensor, steps: int = 16) -> torch.Tensor:
        """Euler ODE: integrate from noise z_1 → actions z_0."""
        B, device = frame_t.shape[0], frame_t.device
        ctx = self.encode_frames(frame_t, frame_tH)

        z  = torch.randn(B, self.action_horizon, self.action_dim, device=device)
        dt = 1.0 / steps
        for i in range(steps, 0, -1):
            t_emb = self.t_emb(torch.full((B,), i / steps, device=device))
            x = self.action_in(z)
            for block in self.dit:
                x = block(x, ctx, t_emb)
            z = z - dt * self.action_out(x)
        return z


# ── Dataset helpers ────────────────────────────────────────────────────────────

def _lerobot_dataset(*args, **kwargs):
    try:
        from lerobot.datasets import LeRobotDataset
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset(*args, **kwargs)


def _local_kwargs(path: Path) -> dict:
    """lerobot v3+ splits local datasets into repo_id=<folder> + root=<parent>."""
    return {"repo_id": "snapshot", "root": path}


def load_dataset(dataset_arg: str):
    """Accept a local path or HF repo_id."""
    local = Path(dataset_arg)
    if local.exists():
        return _lerobot_dataset(**_local_kwargs(local))
    return _lerobot_dataset(repo_id=dataset_arg)


def load_dataset_with_windows(dataset_arg: str, camera: str, H: int, fps: float):
    delta = {
        camera:   [0.0, H / fps],
        "action": [i / fps for i in range(H)],
    }
    local = Path(dataset_arg)
    if local.exists():
        return _lerobot_dataset(**_local_kwargs(local), delta_timestamps=delta)
    return _lerobot_dataset(repo_id=dataset_arg, delta_timestamps=delta)


def discover_cameras(dataset) -> list[str]:
    return [k for k in dataset.features if k.startswith("observation.images.")]


def discover_action_dim(dataset) -> int:
    shape = dataset.features["action"]["shape"]
    return shape[-1] if len(shape) > 1 else shape[0]


def load_action_stats(dataset) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    try:
        stats = dataset.meta.stats["action"]
        mean = torch.tensor(stats["mean"], dtype=torch.float32)
        std  = torch.tensor(stats["std"],  dtype=torch.float32).clamp(min=1e-6)
        return mean, std
    except Exception:
        return None, None


# ── Main ──────────────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, scheduler, epoch: int, output_dir: Path) -> None:
    """Save inference weights (safetensors) + full resume state (pt)."""
    model_state = {k: v for k, v in model.state_dict().items() if not k.startswith("siglip.")}
    save_file(model_state, str(output_dir / f"checkpoint_epoch{epoch:04d}.safetensors"))
    torch.save({
        "epoch": epoch,
        "model": model_state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, output_dir / "resume.pt")


def main():
    parser = argparse.ArgumentParser(description="Train IDM on a LeRobot dataset")
    parser.add_argument("--dataset",       required=True, help="Local path or HuggingFace repo id")
    parser.add_argument("--camera",        default=None,  help="Camera key (auto-detects first found)")
    parser.add_argument("--output",        default="./idm")
    parser.add_argument("--epochs",        type=int,   default=50)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--batch-size",    type=int,   default=64,
                        help="Batch size (64 suits A100 80GB; use 8-16 on a 4090)")
    parser.add_argument("--num-workers",   type=int,   default=8,
                        help="DataLoader workers (8 for A100 node; 4 for desktop)")
    parser.add_argument("--action-horizon",type=int,   default=16, help="H: actions predicted per frame pair")
    parser.add_argument("--inference-steps",type=int,  default=16, help="ODE steps at inference time")
    parser.add_argument("--save-every",    type=int,   default=10)
    parser.add_argument("--resume",        action="store_true",
                        help="Resume from <output>/resume.pt if it exists")
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--list-cameras",  action="store_true", help="Print cameras and exit")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda")

    # ── Probe dataset ──
    print(f"Probing dataset: {args.dataset}")
    probe = load_dataset(args.dataset)
    cameras   = discover_cameras(probe)
    action_dim = discover_action_dim(probe)
    fps        = probe.fps
    output_dir = Path(args.output)

    print(f"  Cameras:    {cameras}")
    print(f"  Action dim: {action_dim}")
    print(f"  FPS:        {fps}")
    print(f"  Episodes:   {probe.num_episodes}")
    print(f"  Frames:     {len(probe)}")

    if args.list_cameras:
        return

    camera = args.camera or cameras[0]
    if camera not in cameras:
        raise ValueError(f"Camera '{camera}' not found. Available: {cameras}")
    print(f"  Using cam:  {camera}")

    action_mean, action_std = load_action_stats(probe)
    if action_mean is not None:
        action_mean = action_mean.to(device)
        action_std  = action_std.to(device)
        print(f"  Action norm: enabled (from dataset stats)")
    else:
        print(f"  Action norm: disabled (no stats found)")
    del probe

    # ── Dataset with temporal windows ──
    H = args.action_horizon
    print(f"\nLoading dataset with H={H} action horizon at {fps} fps...")
    dataset = load_dataset_with_windows(args.dataset, camera, H, fps)
    loader  = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    print(f"  {len(dataset)} training samples, {len(loader)} batches/epoch")

    # ── Build model ──
    print(f"\nBuilding IDM...")
    model = IDM(action_dim=action_dim, action_horizon=H).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # ── Optimizer (8-bit AdamW, SigLIP frozen so no waste on its params) ──
    import bitsandbytes as bnb
    optimizer = bnb.optim.AdamW8bit(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.95, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(loader)
    )

    # ── Resume ──
    start_epoch = 1
    resume_path = output_dir / "resume.pt"
    if args.resume and resume_path.exists():
        print(f"Resuming from {resume_path}...")
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        # Merge saved weights into current state (SigLIP keys absent in ckpt — keep current)
        state = model.state_dict()
        state.update(ckpt["model"])
        model.load_state_dict(state)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        print(f"  Resumed at epoch {start_epoch}/{args.epochs}")
    elif args.resume:
        print(f"  No resume.pt found in {output_dir} — starting fresh")

    # ── Training loop ──
    print(f"\nTraining for {args.epochs} epochs (starting at {start_epoch})...")
    interrupted = False
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            epoch_loss = 0.0

            for batch in loader:
                # batch[camera]: [B, 2, C, H, W] — two timestamps
                # batch["action"]: [B, H, action_dim]
                frame_t  = batch[camera][:, 0].to(device)
                frame_tH = batch[camera][:, 1].to(device)
                actions  = batch["action"].to(device, dtype=torch.float32)

                if action_mean is not None:
                    actions = (actions - action_mean) / action_std

                optimizer.zero_grad()
                loss = model(frame_t, frame_tH, actions)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()

            avg = epoch_loss / len(loader)
            lr  = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:4d}/{args.epochs}  loss={avg:.4f}  lr={lr:.2e}")

            # Save resume state every epoch so Ctrl+C loses at most one epoch
            torch.save({
                "epoch": epoch,
                "model": {k: v for k, v in model.state_dict().items() if not k.startswith("siglip.")},
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }, output_dir / "resume.pt")

            if epoch % args.save_every == 0:
                save_checkpoint(model, optimizer, scheduler, epoch, output_dir)
                print(f"  Checkpoint saved: checkpoint_epoch{epoch:04d}.safetensors")

    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted — saving weights...")

    # ── Save final weights + config ──
    final_weights = output_dir / "idm.safetensors"
    model_state = {k: v for k, v in model.state_dict().items() if not k.startswith("siglip.")}
    save_file(model_state, str(final_weights))

    config = dict(
        action_dim=action_dim,
        action_horizon=H,
        camera=camera,
        fps=fps,
        inference_steps=args.inference_steps,
        action_mean=action_mean.tolist() if action_mean is not None else None,
        action_std=action_std.tolist()  if action_std  is not None else None,
    )
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))

    if interrupted:
        print(f"Weights saved to: {final_weights}")
        print(f"Resume with: uv run train_idm.py ... --resume")
    else:
        print(f"\nIDM saved to: {output_dir}")
        print(f"  Weights: {final_weights}")
        print(f"  Config:  {output_dir / 'config.json'}")
        print(f"\nNext: uv run extract_actions.py --idm {output_dir} --video generated.mp4")


if __name__ == "__main__":
    main()
