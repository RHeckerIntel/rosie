"""
Train an Inverse Dynamics Model (IDM) on a LeRobot dataset.

Given two image frames (t and t+H), the IDM predicts the H actions between them.
Trained on teleoperation data (real frames + real actions), then used to label
synthetically generated videos with pseudo-actions.

Usage:
    uv run train_idm.py --dataset /path/to/dataset --output ./idm
    uv run train_idm.py --dataset your-hf/dataset --output ./idm --epochs 50
    uv run train_idm.py --dataset /path --camera observation.images.top
    uv run train_idm.py --dataset /path --list-cameras  # inspect dataset first
"""

import argparse
import json
import random
from pathlib import Path

import torch
from safetensors.torch import save_file

from rosie.idm import IDM


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
