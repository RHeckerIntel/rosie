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

import numpy as np
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


def load_dataset_with_windows(dataset_arg: str, cameras: list[str], H: int, fps: float):
    delta = {"action": [i / fps for i in range(H)]}
    for cam in cameras:
        delta[cam] = [0.0, H / fps]
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


# ── Feature caching ───────────────────────────────────────────────────────────

def precompute_features(
    model, loader, cameras: list[str], device, cache_dir: str
) -> tuple[torch.Tensor, "np.ndarray"]:
    """Run SigLIP once per dataset item and cache to disk as a memory-mapped file.

    Uses compact sequential slots so allocation = N_unique_items × token_size,
    not max_global_index × token_size (which can be huge on large datasets).

    Returns (index_map, mmap) where:
      index_map[abs_idx]  → slot  (int32 tensor, lives in CPU RAM, tiny)
      mmap[slot]          → [n_tokens, hidden] float16 features (on disk, demand-paged)
    """
    cache_dir  = Path(cache_dir)
    map_path   = cache_dir / "index_map.pt"
    feat_path  = cache_dir / "features.npy"
    shape_path = cache_dir / "shape.pt"

    if map_path.exists() and feat_path.exists():
        print(f"Loading feature cache from {cache_dir} ...")
        index_map  = torch.load(map_path, map_location="cpu", weights_only=True)
        shape      = torch.load(shape_path, weights_only=True)
        mmap       = np.memmap(feat_path, dtype="float16", mode="r", shape=tuple(shape))
        print(f"  {shape[0]:,} slots  {np.prod(shape) * 2 / 1e9:.2f} GB on disk")
        return index_map, mmap

    print("Pre-computing SigLIP features (runs once per dataset) ...")
    model.eval()
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Probe token dimensions ──
    probe = next(iter(loader))
    with torch.no_grad():
        probe_out = model.siglip_encode(
            [probe[cam][:1, 0].to(device) for cam in cameras],
            [probe[cam][:1, 1].to(device) for cam in cameras],
        )
    _, n_tokens, hidden = probe_out.shape

    # ── Collect all unique absolute indices ──
    all_abs = set()
    for batch in loader:
        all_abs.update(batch["index"].tolist())
    all_abs = sorted(all_abs)
    N = len(all_abs)

    # ── Build compact index_map: abs_index → slot (0..N-1) ──
    max_abs   = all_abs[-1]
    index_map = torch.full((max_abs + 1,), -1, dtype=torch.int32)
    for slot, abs_idx in enumerate(all_abs):
        index_map[abs_idx] = slot

    # ── Allocate memmap on disk ──
    shape = (N, n_tokens, hidden)
    mmap  = np.memmap(feat_path, dtype="float16", mode="w+", shape=shape)
    print(f"  {N:,} items  |  {np.prod(shape) * 2 / 1e9:.2f} GB → {feat_path}")

    # ── Fill ──
    for batch in loader:
        abs_indices = batch["index"].tolist()
        frames_t  = [batch[cam][:, 0].to(device) for cam in cameras]
        frames_tH = [batch[cam][:, 1].to(device) for cam in cameras]
        with torch.no_grad():
            tokens = model.siglip_encode(frames_t, frames_tH).cpu().to(torch.float16).numpy()
        for b, abs_idx in enumerate(abs_indices):
            mmap[int(index_map[abs_idx])] = tokens[b]

    mmap.flush()
    torch.save(index_map, map_path)
    torch.save(list(shape), shape_path)
    print(f"  Cache written.")
    model.train()
    return index_map, mmap


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
    parser.add_argument("--cameras",       default=None, nargs="+",
                        help="Camera keys to train on (auto-detects all found if omitted)")
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
    parser.add_argument("--cache-features", default=None, metavar="PATH",
                        help="Pre-compute SigLIP features to PATH (.pt). Huge speedup: "
                             "SigLIP runs once instead of every step.")
    parser.add_argument("--compile",       action="store_true",
                        help="torch.compile backbone + DiT (~20-40%% faster on H100/A100)")
    parser.add_argument("--resume",        action="store_true",
                        help="Resume from <output>/resume.pt if it exists")
    parser.add_argument("--export",        action="store_true",
                        help="Export idm.safetensors from resume.pt and exit (no training)")
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--list-cameras",  action="store_true", help="Print cameras and exit")
    args = parser.parse_args()

    # ── Export-only mode ──
    if args.export:
        output_dir  = Path(args.output)
        resume_path = output_dir / "resume.pt"
        if not resume_path.exists():
            raise FileNotFoundError(f"No resume.pt found in {output_dir}")
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        save_file(ckpt["model"], str(output_dir / "idm.safetensors"))
        print(f"Exported epoch {ckpt['epoch']} → {output_dir / 'idm.safetensors'}")
        return

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

    cameras = args.cameras or cameras
    missing = [c for c in cameras if c not in discover_cameras(probe)]
    if missing:
        raise ValueError(f"Cameras not found: {missing}. Available: {discover_cameras(probe)}")
    print(f"  Using cams: {cameras}")

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
    dataset = load_dataset_with_windows(args.dataset, cameras, H, fps)
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

    # ── Write config early so extract_actions.py can run against resume.pt ──
    config = dict(
        action_dim=action_dim,
        action_horizon=H,
        cameras=cameras,
        fps=fps,
        inference_steps=args.inference_steps,
        action_mean=action_mean.tolist() if action_mean is not None else None,
        action_std=action_std.tolist()   if action_std  is not None else None,
    )
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))

    # ── Build model ──
    print(f"\nBuilding IDM (num_cameras={len(cameras)})...")
    model = IDM(action_dim=action_dim, action_horizon=H, num_cameras=len(cameras)).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    if args.compile:
        print("  torch.compile: backbone + DiT (first epoch will be slow) ...")
        model.backbone = torch.nn.ModuleList([
            torch.compile(b, mode="reduce-overhead") for b in model.backbone
        ])
        model.dit = torch.nn.ModuleList([
            torch.compile(d, mode="reduce-overhead") for d in model.dit
        ])

    # ── Optimizer (8-bit AdamW, SigLIP frozen so no waste on its params) ──
    import bitsandbytes as bnb
    optimizer = bnb.optim.AdamW8bit(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.95, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(loader)
    )

    # ── Feature cache (build once, skip SigLIP every step) ──
    # index_map: CPU int32 tensor  [max_abs_idx+1] → slot
    # feat_mmap: numpy memmap on disk, demand-paged by OS
    feat_index_map = None
    feat_mmap      = None
    if args.cache_features:
        feat_index_map, feat_mmap = precompute_features(
            model, loader, cameras, device, args.cache_features
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
                # batch[cam]: [B, 2, C, H, W] — two timestamps per camera
                # batch["action"]: [B, H, action_dim]
                actions = batch["action"].to(device, dtype=torch.float32)

                if action_mean is not None:
                    actions = (actions - action_mean) / action_std

                if feat_mmap is not None:
                    slots  = feat_index_map[batch["index"]].numpy()
                    cached = torch.from_numpy(feat_mmap[slots].copy()).to(device, dtype=torch.float32)
                    kwargs = dict(cached_tokens=cached)
                    frames_t = frames_tH = []
                else:
                    frames_t  = [batch[cam][:, 0].to(device) for cam in cameras]
                    frames_tH = [batch[cam][:, 1].to(device) for cam in cameras]
                    kwargs = {}

                optimizer.zero_grad()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(frames_t, frames_tH, actions, **kwargs)
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
        cameras=cameras,
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
