# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch>=2.4",
#     "diffusers>=0.32",
#     "transformers>=4.46",
#     "accelerate>=1.0",
#     "peft>=0.13",
#     "bitsandbytes>=0.44",
#     "imageio>=2.34",
#     "imageio-ffmpeg",
#     "pillow",
#     "numpy",
#     "ftfy",
#     "safetensors",
#     "torchvision",
# ]
# ///
"""
LoRA fine-tune WAN2.1 I2V on a single MP4 video.

Usage:
    uv run fine_tune.py --video robot.mp4 --prompt "robot arm picks up a cup"
    uv run fine_tune.py --video robot.mp4 --prompt "..." --output ./lora --epochs 75 --rank 4
    uv run fine_tune.py --video robot.mp4 --prompt "..." --quantize  # QLoRA for 4090
"""

import argparse
import random
from pathlib import Path

import imageio
import torch
import torch.nn.functional as F
from PIL import Image
from peft import LoraConfig
from torchvision.transforms.functional import to_tensor

from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.quantizers import PipelineQuantizationConfig
from transformers import CLIPVisionModel


MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"

# Matches DreamGen paper: rank=4, alpha=4 on attention projections
LORA_TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0"]


# ── Data ──────────────────────────────────────────────────────────────────────

def extract_clips(video_path: Path, num_frames: int, stride: int, size: tuple[int, int]) -> list[list[Image.Image]]:
    reader = imageio.get_reader(str(video_path), "ffmpeg")
    all_frames = [Image.fromarray(f).resize(size, Image.LANCZOS) for f in reader]
    reader.close()

    clips = [all_frames[s:s + num_frames] for s in range(0, len(all_frames) - num_frames + 1, stride)]
    if not clips:
        raise ValueError(
            f"Video has {len(all_frames)} frames but needs at least {num_frames}. "
            "Use a longer video or reduce --clip-frames."
        )
    return clips


def frames_to_tensor(frames: list[Image.Image], device: torch.device) -> torch.Tensor:
    """Convert PIL frames to [1, 3, T, H, W] in [-1, 1]."""
    tensors = [to_tensor(f) * 2 - 1 for f in frames]
    return torch.stack(tensors, dim=1).unsqueeze(0).to(device)


@torch.no_grad()
def encode_video(vae: AutoencoderKLWan, frames: list[Image.Image], device: torch.device) -> torch.Tensor:
    video = frames_to_tensor(frames, device).to(vae.dtype)
    latents = vae.encode(video).latent_dist.sample()
    mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(latents)
    std  = 1.0 / torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(latents)
    return (latents - mean) * std


# ── Model setup ───────────────────────────────────────────────────────────────

def load_pipeline(quantize: bool) -> WanImageToVideoPipeline:
    image_encoder = CLIPVisionModel.from_pretrained(
        MODEL_ID, subfolder="image_encoder", torch_dtype=torch.float32
    )
    vae = AutoencoderKLWan.from_pretrained(
        MODEL_ID, subfolder="vae", torch_dtype=torch.float32
    )

    kwargs: dict = dict(vae=vae, image_encoder=image_encoder, torch_dtype=torch.bfloat16)
    if quantize:
        kwargs["quantization_config"] = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={"bnb_4bit_compute_dtype": torch.bfloat16, "bnb_4bit_quant_type": "nf4"},
            components_to_quantize=["transformer"],
        )

    pipe = WanImageToVideoPipeline.from_pretrained(MODEL_ID, **kwargs)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    return pipe


def apply_lora(pipe: WanImageToVideoPipeline, rank: int, alpha: int) -> None:
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.0,
        bias="none",
    )
    pipe.transformer.add_adapter(lora_config)
    pipe.transformer.enable_gradient_checkpointing()

    trainable = sum(p.numel() for p in pipe.transformer.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in pipe.transformer.parameters())
    print(f"LoRA trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")


# ── Pre-computation ───────────────────────────────────────────────────────────

@torch.no_grad()
def precompute(
    pipe: WanImageToVideoPipeline,
    prompt: str,
    clips: list[list[Image.Image]],
    device: torch.device,
    cache_dir: Path,
) -> tuple[list[Path], torch.Tensor]:
    """Encode all clips and write each to disk. Re-uses existing files on restart."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Text embedding is shared — encode once and keep tiny in RAM
    prompt_file = cache_dir / "prompt_embeds.pt"
    if not prompt_file.exists():
        # Encode on CPU — text encoder stays there; avoids device conflicts with bnb-quantized transformer
        prompt_embeds, _ = pipe.encode_prompt(prompt, torch.device("cpu"), 1, False)
        torch.save(prompt_embeds.cpu(), prompt_file)
    prompt_embeds = torch.load(prompt_file, weights_only=True)

    pipe.image_encoder.to(device)
    pipe.vae.to(device)

    h, w = clips[0][0].height, clips[0][0].width
    num_frames = len(clips[0])
    clip_files: list[Path] = []

    for i, clip in enumerate(clips):
        clip_file = cache_dir / f"clip_{i:06d}.pt"
        clip_files.append(clip_file)

        if clip_file.exists():
            print(f"  Clip {i + 1}/{len(clips)} (cached)", end="\r")
            continue

        first_pil = clip[0]
        first_pixel = pipe.video_processor.preprocess(first_pil, height=h, width=w).to(device, torch.float32)

        _, condition = pipe.prepare_latents(
            image=first_pixel,
            batch_size=1,
            num_channels_latents=pipe.vae.config.z_dim,
            height=h, width=w,
            num_frames=num_frames,
            dtype=torch.float32, device=device,
        )
        image_embeds = pipe.encode_image(first_pil, device)
        clean_latents = encode_video(pipe.vae, clip, device)

        torch.save({
            "image_embeds": image_embeds.cpu(),
            "clean_latents": clean_latents.cpu().float(),
            "condition": condition.cpu().float(),
        }, clip_file)
        print(f"  Clip {i + 1}/{len(clips)}", end="\r")

    print()
    pipe.image_encoder.cpu()
    pipe.vae.cpu()
    torch.cuda.empty_cache()

    return clip_files, prompt_embeds


# ── Training ──────────────────────────────────────────────────────────────────

def training_step(
    transformer,
    batch: dict,
    prompt_embeds: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    dtype = next(p for p in transformer.parameters()).dtype

    clean   = batch["clean_latents"].to(device, dtype)
    cond    = batch["condition"].to(device, dtype)
    t_emb   = prompt_embeds.to(device, dtype)
    i_emb   = batch["image_embeds"].to(device, dtype)

    noise  = torch.randn_like(clean)
    t      = torch.rand(1, device=device)                   # continuous [0, 1]
    t_int  = (t * 1000).round().long()                      # [0, 1000] for transformer

    # Flow matching: linear interpolation between data and noise
    noisy  = (1 - t) * clean + t * noise
    target = noise - clean                                   # velocity field

    model_input = torch.cat([noisy, cond], dim=1)

    pred = transformer(
        hidden_states=model_input,
        timestep=t_int,
        encoder_hidden_states=t_emb,
        encoder_hidden_states_image=i_emb,
        return_dict=False,
    )[0]

    return F.mse_loss(pred.float(), target.float())


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune WAN2.1 I2V on an MP4")
    parser.add_argument("--video", required=True, help="Input MP4 path")
    parser.add_argument("--prompt", required=True, help="Text description of the motion in the video")
    parser.add_argument("--output", default="./lora", help="Directory to save LoRA weights")
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=4, help="LoRA rank (DreamGen default: 4)")
    parser.add_argument("--alpha", type=int, default=4, help="LoRA alpha (DreamGen default: 4)")
    parser.add_argument("--clip-frames", type=int, default=81, help="Frames per training clip (must be 4k+1)")
    parser.add_argument("--clip-stride", type=int, default=16, help="Frame stride between clips")
    parser.add_argument("--cache-dir", default=None, help="Where to store precomputed latents (default: <output>/cache)")
    parser.add_argument("--quantize", action="store_true", help="4-bit base model (QLoRA) for 4090")
    parser.add_argument("--save-every", type=int, default=25, help="Save checkpoint every N epochs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ──
    print(f"Loading model (quantize={args.quantize})...")
    pipe = load_pipeline(args.quantize)
    apply_lora(pipe, rank=args.rank, alpha=args.alpha)
    # Transformer stays on CPU until after precomputation to free GPU for encoders

    # ── Extract clips ──
    print(f"Extracting clips from {args.video}...")
    size = (720, 480)  # 480p — (width, height) for PIL resize
    clips = extract_clips(Path(args.video), args.clip_frames, args.clip_stride, size)
    print(f"  {len(clips)} clips of {args.clip_frames} frames each")

    # ── Precompute embeddings (disk-cached) ──
    cache_dir = Path(args.cache_dir) if args.cache_dir else output_dir / "cache"
    print(f"Precomputing embeddings → {cache_dir}  (skips existing files on restart)")
    clip_files, prompt_embeds = precompute(pipe, args.prompt, clips, device, cache_dir)

    # Encoders are back on CPU; move transformer to GPU for training
    pipe.transformer.to(device)
    torch.cuda.empty_cache()

    # ── Optimizer — only LoRA params ──
    import bitsandbytes as bnb
    optimizer = bnb.optim.AdamW8bit(
        [p for p in pipe.transformer.parameters() if p.requires_grad],
        lr=args.lr,
    )

    # ── Training loop ──
    print(f"\nTraining for {args.epochs} epochs over {len(clip_files)} clips...")
    for epoch in range(1, args.epochs + 1):
        random.shuffle(clip_files)
        epoch_loss = 0.0

        for clip_file in clip_files:
            batch = torch.load(clip_file, map_location="cpu", weights_only=True)
            optimizer.zero_grad()
            loss = training_step(pipe.transformer, batch, prompt_embeds, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in pipe.transformer.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            epoch_loss += loss.item()

        avg = epoch_loss / len(clip_files)
        print(f"Epoch {epoch:4d}/{args.epochs}  loss={avg:.4f}")

        if epoch % args.save_every == 0:
            ckpt = output_dir / f"checkpoint_epoch{epoch:04d}"
            pipe.save_lora_weights(str(ckpt))
            print(f"  Saved checkpoint: {ckpt}")

    # ── Save final LoRA ──
    pipe.save_lora_weights(str(output_dir))
    print(f"\nLoRA weights saved to: {output_dir}")
    print(f"Use with: uv run generate_video.py ... --lora {output_dir}")


if __name__ == "__main__":
    main()
