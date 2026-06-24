#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""
Print the actions stored in an actions.npy file.

Usage:
    uv run show_actions.py <path/to/actions.npy>
    uv run show_actions.py <path/to/dataset_dir>   # auto-finds actions.npy inside
"""

import sys
from pathlib import Path

import numpy as np


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run show_actions.py <actions.npy or dataset_dir>")
        sys.exit(1)

    p = Path(sys.argv[1])
    if p.is_dir():
        p = p / "actions.npy"

    if not p.exists():
        print(f"File not found: {p}")
        sys.exit(1)

    actions = np.load(str(p))
    n_frames, action_dim = actions.shape

    print(f"File:       {p}")
    print(f"Shape:      {actions.shape}  ({n_frames} frames × {action_dim} dims)")
    print(f"Range:      [{actions.min():.4f}, {actions.max():.4f}]")
    print()

    # Per-dimension stats
    print(f"{'dim':<5} {'mean':>10} {'std':>10} {'min':>10} {'max':>10}")
    print("-" * 50)
    for d in range(action_dim):
        col = actions[:, d]
        print(f"{d:<5} {col.mean():>10.4f} {col.std():>10.4f} {col.min():>10.4f} {col.max():>10.4f}")

    print()

    # Per-frame values (clamp to first/last 10 if long)
    limit = 10
    indices = list(range(min(limit, n_frames)))
    if n_frames > 2 * limit:
        indices += ["..."]
        indices += list(range(n_frames - limit, n_frames))
    elif n_frames > limit:
        indices = list(range(n_frames))

    header = f"{'frame':<7}" + "".join(f"  dim{d:02d}" for d in range(action_dim))
    print(header)
    print("-" * len(header))
    for i in indices:
        if i == "...":
            print("  ...")
            continue
        vals = "".join(f"  {actions[i, d]:7.4f}" for d in range(action_dim))
        print(f"{i:<7}{vals}")


if __name__ == "__main__":
    main()
