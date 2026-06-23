# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch>=2.4",
#     "transformers>=4.46",
#     "accelerate>=1.0",
#     "lerobot",
#     "safetensors",
#     "imageio>=2.34",
#     "imageio-ffmpeg",
#     "pillow",
#     "numpy",
#     "torchvision",
#     "pyarrow",
# ]
# ///
"""
Run a trained IDM over a video to produce pseudo-actions, then write a LeRobot dataset.

The IDM slides a window over every consecutive frame pair (t, t+H) in the video,
predicts H actions between them, and takes the first action as the label for frame t.
The last H frames have no label and are dropped.

Usage:
    uv run extract_actions.py --idm ./idm --video generated.mp4 --output ./synthetic_dataset
    uv run extract_actions.py --idm ./idm --video generated.mp4 --task "pick up the block"
    uv run extract_actions.py --idm ./idm --video generated.mp4 --batch-size 64
"""

import argparse
import json
import math
import shutil
from pathlib import Path

import imageio
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoModel


SIGLIP_MODEL = "google/siglip2-large-patch16-256"
SIGLIP_DIM   = 1024
SIGLIP_SIZE  = 256


# ── Model (mirrors train_idm.py) ──────────────────────────────────────────────

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
        return self.proj(torch.cat([emb.sin(), emb.cos()], dim=-1))


class SABlock(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff    = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = self.norm1(x)
        x = x + self.attn(n, n, n, need_weights=False)[0]
        return x + self.ff(self.norm2(x))


class DiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn1 = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn2 = nn.MultiheadAttention(dim, heads, batch_first=True)
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
        return x + self.ff(h)


class IDM(nn.Module):
    def __init__(self, action_dim: int, action_horizon: int = 16,
                 hidden: int = SIGLIP_DIM, backbone_layers: int = 4,
                 dit_layers: int = 8, heads: int = 16):
        super().__init__()
        self.action_dim     = action_dim
        self.action_horizon = action_horizon

        self.siglip   = AutoModel.from_pretrained(SIGLIP_MODEL, torch_dtype=torch.bfloat16)
        self.siglip.requires_grad_(False)
        self.backbone = nn.ModuleList([SABlock(hidden, heads) for _ in range(backbone_layers)])
        self.t_emb    = TimestepEmbedding(hidden)
        self.action_in  = nn.Linear(action_dim, hidden)
        self.action_out = nn.Linear(hidden, action_dim)
        self.dit = nn.ModuleList([DiTBlock(hidden, heads) for _ in range(dit_layers)])

    def _preprocess(self, frame: torch.Tensor) -> torch.Tensor:
        if frame.shape[-2:] != (SIGLIP_SIZE, SIGLIP_SIZE):
            frame = TF.resize(frame, [SIGLIP_SIZE, SIGLIP_SIZE], antialias=True)
        return (frame * 2.0 - 1.0).to(dtype=self.siglip.dtype)

    @torch.no_grad()
    def encode_frames(self, frame_t: torch.Tensor, frame_tH: torch.Tensor) -> torch.Tensor:
        dev = next(self.siglip.parameters()).device
        ft  = self.siglip.vision_model(pixel_values=self._preprocess(frame_t).to(dev)).last_hidden_state
        ftH = self.siglip.vision_model(pixel_values=self._preprocess(frame_tH).to(dev)).last_hidden_state
        ctx = torch.cat([ft, ftH], dim=1).float()
        for block in self.backbone:
            ctx = block(ctx)
        return ctx

    @torch.no_grad()
    def get_actions(self, frame_t: torch.Tensor, frame_tH: torch.Tensor, steps: int = 16) -> torch.Tensor:
        B, device = frame_t.shape[0], frame_t.device
        ctx = self.encode_frames(frame_t, frame_tH)
        z   = torch.randn(B, self.action_horizon, self.action_dim, device=device)
        dt  = 1.0 / steps
        for i in range(steps, 0, -1):
            t_emb = self.t_emb(torch.full((B,), i / steps, device=device))
            x = self.action_in(z)
            for block in self.dit:
                x = block(x, ctx, t_emb)
            z = z - dt * self.action_out(x)
        return z  # [B, H, action_dim]


# ── Video loading ─────────────────────────────────────────────────────────────

def load_video_frames(path: Path) -> list[np.ndarray]:
    """Returns list of HWC uint8 numpy arrays."""
    reader = imageio.get_reader(str(path), "ffmpeg")
    frames = [f for f in reader]
    reader.close()
    return frames


def frames_to_tensor(frames: list[np.ndarray], device: torch.device) -> torch.Tensor:
    """HWC uint8 numpy → [N, C, H, W] float32 in [0, 1]."""
    arr = np.stack(frames).astype(np.float32) / 255.0       # [N, H, W, C]
    return torch.from_numpy(arr).permute(0, 3, 1, 2).to(device)  # [N, C, H, W]


# ── IDM inference ─────────────────────────────────────────────────────────────

def run_inference(
    model: IDM,
    frame_tensors: torch.Tensor,   # [N, C, H, W]
    H: int,
    batch_size: int,
    steps: int,
) -> np.ndarray:
    """
    Slide IDM over (frame_t, frame_{t+H}) pairs.
    Returns actions [N-H, action_dim] — one action per labeled frame.
    """
    N = frame_tensors.shape[0]
    n_labeled = N - H
    action_dim = model.action_dim
    all_actions = np.zeros((n_labeled, action_dim), dtype=np.float32)

    for start in range(0, n_labeled, batch_size):
        end = min(start + batch_size, n_labeled)
        idx = list(range(start, end))

        ft  = frame_tensors[idx]          # [B, C, H, W]
        ftH = frame_tensors[[i + H for i in idx]]

        # get_actions returns [B, H, action_dim] — take step 0 as the label for frame t
        preds = model.get_actions(ft, ftH, steps=steps)  # [B, H, action_dim]
        all_actions[start:end] = preds[:, 0, :].cpu().float().numpy()

        print(f"  Frames {end}/{n_labeled}", end="\r")

    print()
    return all_actions


# ── LeRobot dataset writer ────────────────────────────────────────────────────

def write_lerobot_dataset(
    output_dir: Path,
    repo_id: str,
    camera: str,
    frames: list[np.ndarray],   # HWC uint8, only the labeled frames (N-H)
    actions: np.ndarray,         # [N-H, action_dim]
    fps: float,
    task: str,
) -> None:
    """
    Writes a minimal LeRobot v2-compatible dataset:
        meta/info.json
        meta/episodes.jsonl
        meta/tasks.jsonl
        meta/stats.json
        data/chunk-000/episode_000000.parquet
        videos/chunk-000/{camera}/episode_000000.mp4
    """
    n_frames, action_dim = actions.shape
    h, w = frames[0].shape[:2]

    # ── Directory structure ──
    data_dir  = output_dir / "data"  / "chunk-000"
    video_dir = output_dir / "videos" / "chunk-000" / camera
    meta_dir  = output_dir / "meta"
    for d in [data_dir, video_dir, meta_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Video ──
    print("  Writing video...")
    video_path = video_dir / "episode_000000.mp4"
    writer = imageio.get_writer(str(video_path), fps=fps, codec="libx264", quality=8)
    for f in frames:
        writer.append_data(f)
    writer.close()

    # ── Parquet ──
    print("  Writing parquet...")
    action_list = pa.array(
        [row.tolist() for row in actions],
        type=pa.list_(pa.float32(), action_dim),
    )
    table = pa.table({
        "frame_index":              pa.array(range(n_frames), type=pa.int64()),
        "episode_index":            pa.array([0] * n_frames,  type=pa.int64()),
        "timestamp":                pa.array([i / fps for i in range(n_frames)], type=pa.float32()),
        "action":                   action_list,
        "index":                    pa.array(range(n_frames), type=pa.int64()),
        "episode_data_index_from":  pa.array([0] * n_frames,  type=pa.int64()),
        "episode_data_index_to":    pa.array([n_frames] * n_frames, type=pa.int64()),
        "next.done":                pa.array([False] * (n_frames - 1) + [True], type=pa.bool_()),
    })
    pq.write_table(table, str(data_dir / "episode_000000.parquet"))

    # ── Meta files ──
    info = {
        "codebase_version": "v2.0",
        "robot_type": "unknown",
        "total_episodes": 1,
        "total_frames": n_frames,
        "total_tasks": 1,
        "total_videos": 1,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            camera: {
                "dtype": "video",
                "shape": [3, h, w],
                "names": ["channel", "height", "width"],
                "info": {
                    "video.fps": fps,
                    "video.codec": "libx264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "action": {
                "dtype": "float32",
                "shape": [action_dim],
                "names": None,
            },
            "timestamp":           {"dtype": "float32", "shape": [1], "names": None},
            "frame_index":         {"dtype": "int64",   "shape": [1], "names": None},
            "episode_index":       {"dtype": "int64",   "shape": [1], "names": None},
            "index":               {"dtype": "int64",   "shape": [1], "names": None},
            "next.done":           {"dtype": "bool",    "shape": [1], "names": None},
        },
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2))

    (meta_dir / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "tasks": [task], "length": n_frames}) + "\n"
    )
    (meta_dir / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": task}) + "\n"
    )

    # Action stats
    mean = actions.mean(axis=0).tolist()
    std  = actions.std(axis=0).clip(1e-6).tolist()
    mn   = actions.min(axis=0).tolist()
    mx   = actions.max(axis=0).tolist()
    stats = {
        "action": {"mean": mean, "std": std, "min": mn, "max": mx},
    }
    (meta_dir / "stats.json").write_text(json.dumps(stats, indent=2))

    print(f"  Dataset written: {output_dir}")
    print(f"    {n_frames} frames  |  action_dim={action_dim}  |  fps={fps}")
    print(f"    Load with: LeRobotDataset(repo_id='{repo_id}', root='{output_dir.parent}')")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract pseudo-actions from a video using a trained IDM")
    parser.add_argument("--idm",        required=True, help="Path to IDM output dir (contains idm.safetensors + config.json)")
    parser.add_argument("--video",      required=True, help="Input video (.mp4)")
    parser.add_argument("--output",     default=None,  help="Output LeRobot dataset dir (default: <video_stem>_dataset)")
    parser.add_argument("--task",       default="robot manipulation task",
                                        help="Task description stored in the dataset")
    parser.add_argument("--repo-id",    default=None,  help="LeRobot repo_id for the output dataset")
    parser.add_argument("--batch-size", type=int, default=32,  help="IDM inference batch size")
    parser.add_argument("--cpu",        action="store_true", help="Force CPU (default: CUDA)")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu else "cuda")
    video_path = Path(args.video)
    idm_dir    = Path(args.idm)

    output_dir = Path(args.output) if args.output else video_path.parent / f"{video_path.stem}_dataset"
    output_dir.mkdir(parents=True, exist_ok=True)

    repo_id = args.repo_id or output_dir.name

    # ── Load IDM config ──
    config_path = idm_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {idm_dir}")
    config = json.loads(config_path.read_text())

    action_dim  = config["action_dim"]
    H           = config["action_horizon"]
    fps         = config["fps"]
    camera      = config["camera"]
    steps       = config["inference_steps"]
    action_mean = np.array(config["action_mean"], dtype=np.float32) if config.get("action_mean") else None
    action_std  = np.array(config["action_std"],  dtype=np.float32) if config.get("action_std")  else None

    print(f"IDM config: action_dim={action_dim}, H={H}, fps={fps}, camera={camera}")

    # ── Build and load IDM ──
    print("Loading IDM weights...")
    model = IDM(action_dim=action_dim, action_horizon=H).to(device)

    weights_path = idm_dir / "idm.safetensors"
    if not weights_path.exists():
        # Try latest checkpoint
        ckpts = sorted(idm_dir.glob("checkpoint_epoch*.safetensors"))
        if not ckpts:
            raise FileNotFoundError(f"No IDM weights found in {idm_dir}")
        weights_path = ckpts[-1]
        print(f"  Using checkpoint: {weights_path.name}")

    saved = load_file(str(weights_path))
    state = model.state_dict()
    state.update(saved)
    model.load_state_dict(state)
    model.eval()
    print(f"  Loaded: {weights_path.name}")

    # ── Load video ──
    print(f"Loading video: {video_path}")
    raw_frames = load_video_frames(video_path)
    print(f"  {len(raw_frames)} frames at {fps} fps")

    if len(raw_frames) <= H:
        raise ValueError(f"Video has only {len(raw_frames)} frames but H={H} — need at least {H+1} frames")

    # ── Run inference ──
    print(f"Running IDM inference (H={H}, batch={args.batch_size}, steps={steps})...")
    frame_tensors = frames_to_tensor(raw_frames, device)
    actions_norm = run_inference(model, frame_tensors, H, args.batch_size, steps)

    # Unnormalize
    if action_mean is not None and action_std is not None:
        actions = actions_norm * action_std + action_mean
    else:
        actions = actions_norm

    print(f"  Predicted {len(actions)} actions  |  shape={actions.shape}")
    print(f"  Action range: [{actions.min():.3f}, {actions.max():.3f}]")

    # Save raw numpy alongside the dataset for easy inspection
    np.save(str(output_dir / "actions.npy"), actions)
    print(f"  Raw actions saved: {output_dir / 'actions.npy'}")

    # ── Write LeRobot dataset ──
    print("Writing LeRobot dataset...")
    labeled_frames = raw_frames[:len(actions)]  # drop last H frames (no future frame to condition on)
    write_lerobot_dataset(output_dir, repo_id, camera, labeled_frames, actions, fps, args.task)


if __name__ == "__main__":
    main()
