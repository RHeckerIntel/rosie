"""
LoRA fine-tune WAN2.1 I2V on a single MP4 video.

Usage:
    uv run train_wan.py --video robot.mp4 --prompt "robot arm picks up a cup"
    uv run train_wan.py --video robot.mp4 --prompt "..." --output ./lora --epochs 75
    uv run train_wan.py --video robot.mp4 --prompt "..." --resume
    uv run train_wan.py --video robot.mp4 --prompt "..." --compile      # torch.compile, ~30% faster after warmup
    uv run train_wan.py --video robot.mp4 --prompt "..." --quantize     # QLoRA for 4090 (not needed on H100)
    uv run train_wan.py --video robot.mp4 --prompt "..." --offload      # CPU offload for <24GB VRAM
"""

import argparse
import random
from pathlib import Path

import imageio
import torch
import torch.nn.functional as F
from PIL import Image
from peft import LoraConfig
from safetensors.torch import save_file, load_file
from torchvision.transforms.functional import to_tensor

from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.quantizers import PipelineQuantizationConfig
from transformers import CLIPVisionModel


MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
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

def load_pipeline(quantize: bool, offload: bool, flash_attn: bool) -> WanImageToVideoPipeline:
    image_encoder = CLIPVisionModel.from_pretrained(
        MODEL_ID, subfolder="image_encoder", torch_dtype=torch.float32
    )
    vae = AutoencoderKLWan.from_pretrained(
        MODEL_ID, subfolder="vae", torch_dtype=torch.float32
    )

    kwargs: dict = dict(vae=vae, image_encoder=image_encoder, torch_dtype=torch.bfloat16)

    if flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"

    if quantize:
        kwargs["quantization_config"] = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={"bnb_4bit_compute_dtype": torch.bfloat16, "bnb_4bit_quant_type": "nf4"},
            components_to_quantize=["transformer"],
        )

    pipe = WanImageToVideoPipeline.from_pretrained(MODEL_ID, **kwargs)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    if offload:
        pipe.enable_model_cpu_offload() if quantize else pipe.enable_sequential_cpu_offload()

    return pipe


def apply_lora(pipe: WanImageToVideoPipeline, rank: int, alpha: int) -> None:
    pipe.transformer.add_adapter(LoraConfig(
        r=rank, lora_alpha=alpha,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.0, bias="none",
    ))
    pipe.transformer.enable_gradient_checkpointing()
    trainable = sum(p.numel() for p in pipe.transformer.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in pipe.transformer.parameters())
    print(f"LoRA params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")


# ── Pre-computation ───────────────────────────────────────────────────────────

@torch.no_grad()
def precompute(
    pipe: WanImageToVideoPipeline,
    prompt: str,
    clips: list[list[Image.Image]],
    device: torch.device,
    cache_dir: Path,
) -> tuple[list[Path], torch.Tensor]:
    cache_dir.mkdir(parents=True, exist_ok=True)

    prompt_file = cache_dir / "prompt_embeds.pt"
    if not prompt_file.exists():
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

        first_pil   = clip[0]
        # image_processor (VaeImageProcessor) → [1, 3, H, W] in [-1, 1]
        # video_processor returns 5D with T=0 for a single PIL, which breaks prepare_latents
        first_pixel = pipe.image_processor.preprocess(first_pil, height=h, width=w).to(device, torch.float32)
        _, condition = pipe.prepare_latents(
            image=first_pixel, batch_size=1,
            num_channels_latents=pipe.vae.config.z_dim,
            height=h, width=w, num_frames=num_frames,
            dtype=torch.float32, device=device,
        )
        image_embeds  = pipe.encode_image(first_pil, device)
        clean_latents = encode_video(pipe.vae, clip, device)

        torch.save({
            "image_embeds":  image_embeds.cpu(),
            "clean_latents": clean_latents.cpu().float(),
            "condition":     condition.cpu().float(),
        }, clip_file)
        print(f"  Clip {i + 1}/{len(clips)}", end="\r")

    print()
    pipe.image_encoder.cpu()
    pipe.vae.cpu()
    torch.cuda.empty_cache()
    return clip_files, prompt_embeds


# ── Training step ─────────────────────────────────────────────────────────────

def training_step(transformer, batch: dict, prompt_embeds: torch.Tensor, device: torch.device) -> torch.Tensor:
    dtype = next(p for p in transformer.parameters()).dtype
    clean  = batch["clean_latents"].to(device, dtype)
    cond   = batch["condition"].to(device, dtype)
    t_emb  = prompt_embeds.to(device, dtype)
    i_emb  = batch["image_embeds"].to(device, dtype)

    noise = torch.randn_like(clean)
    t     = torch.rand(1, device=device)
    t_int = (t * 1000).round().long()

    noisy  = (1 - t) * clean + t * noise
    target = noise - clean

    pred = transformer(
        hidden_states=torch.cat([noisy, cond], dim=1),
        timestep=t_int,
        encoder_hidden_states=t_emb,
        encoder_hidden_states_image=i_emb,
        return_dict=False,
    )[0]
    return F.mse_loss(pred.float(), target.float())


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def lora_state(transformer) -> dict:
    return {k: v for k, v in transformer.state_dict().items() if "lora" in k}


def save_resume(transformer, optimizer, scheduler, epoch: int, output_dir: Path) -> None:
    torch.save({
        "epoch":     epoch,
        "lora":      lora_state(transformer),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, output_dir / "resume.pt")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune WAN2.1 I2V")
    parser.add_argument("--video",       required=True)
    parser.add_argument("--prompt",      required=True)
    parser.add_argument("--output",      default="./lora")
    parser.add_argument("--epochs",      type=int,   default=75)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--rank",        type=int,   default=4)
    parser.add_argument("--alpha",       type=int,   default=4)
    parser.add_argument("--clip-frames", type=int,   default=81,  help="Must be 4k+1 (17,33,49,65,81...)")
    parser.add_argument("--clip-stride", type=int,   default=16)
    parser.add_argument("--grad-accum",  type=int,   default=4,   help="Gradient accumulation steps")
    parser.add_argument("--cache-dir",   default=None)
    parser.add_argument("--save-every",  type=int,   default=25)
    parser.add_argument("--resume",      action="store_true", help="Resume from <output>/resume.pt")
    parser.add_argument("--compile",     action="store_true", help="torch.compile the transformer (~30%% faster after warmup)")
    parser.add_argument("--flash-attn",  action="store_true", help="Flash Attention 2 (requires: pip install flash-attn)")
    parser.add_argument("--quantize",    action="store_true", help="4-bit QLoRA (for GPUs with <24GB VRAM)")
    parser.add_argument("--offload",     action="store_true", help="CPU offload (for GPUs with <24GB VRAM)")
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device     = torch.device("cuda")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir  = Path(args.cache_dir) if args.cache_dir else output_dir / "cache"

    # ── Load model ──
    print(f"Loading WAN2.1 I2V (quantize={args.quantize}, offload={args.offload}, flash_attn={args.flash_attn})...")
    pipe = load_pipeline(args.quantize, args.offload, args.flash_attn)
    apply_lora(pipe, args.rank, args.alpha)

    # ── Extract + precompute clips ──
    print(f"Extracting clips from {args.video}...")
    clips = extract_clips(Path(args.video), args.clip_frames, args.clip_stride, size=(720, 480))
    print(f"  {len(clips)} clips × {args.clip_frames} frames")

    print(f"Precomputing embeddings → {cache_dir}")
    clip_files, prompt_embeds = precompute(pipe, args.prompt, clips, device, cache_dir)

    # Move transformer to GPU after precompute (encoders return to CPU above)
    if not args.offload:
        pipe.transformer.to(device)
    torch.cuda.empty_cache()

    if args.compile:
        print("Compiling transformer (first epoch will be slow)...")
        pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead")

    # ── Optimizer + scheduler ──
    import bitsandbytes as bnb
    trainable_params = [p for p in pipe.transformer.parameters() if p.requires_grad]
    optimizer = bnb.optim.AdamW8bit(trainable_params, lr=args.lr)
    total_steps = args.epochs * len(clip_files) // args.grad_accum
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # ── Resume ──
    start_epoch = 1
    resume_path = output_dir / "resume.pt"
    if args.resume and resume_path.exists():
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        state = pipe.transformer.state_dict()
        state.update(ckpt["lora"])
        pipe.transformer.load_state_dict(state)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {ckpt['epoch']}")
    elif args.resume:
        print("No resume.pt found — starting fresh")

    # ── Training loop ──
    print(f"\nTraining {args.epochs} epochs, {len(clip_files)} clips, grad_accum={args.grad_accum}...")
    interrupted = False
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            random.shuffle(clip_files)
            epoch_loss = 0.0
            optimizer.zero_grad()

            for step, clip_file in enumerate(clip_files):
                batch = torch.load(clip_file, map_location="cpu", weights_only=True)
                loss  = training_step(pipe.transformer, batch, prompt_embeds, device)
                (loss / args.grad_accum).backward()
                epoch_loss += loss.item()

                if (step + 1) % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            avg = epoch_loss / len(clip_files)
            lr  = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:4d}/{args.epochs}  loss={avg:.4f}  lr={lr:.2e}")

            save_resume(pipe.transformer, optimizer, scheduler, epoch, output_dir)

            if epoch % args.save_every == 0:
                ckpt_dir = output_dir / f"checkpoint_epoch{epoch:04d}"
                pipe.save_lora_weights(str(ckpt_dir))
                print(f"  Checkpoint: {ckpt_dir}")

    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted — saving weights...")

    pipe.save_lora_weights(str(output_dir))
    if interrupted:
        print(f"LoRA saved. Resume with: uv run train_wan.py ... --resume")
    else:
        print(f"\nLoRA saved to: {output_dir}")
        print(f"Use with: uv run generate_video.py ... --lora {output_dir}")


if __name__ == "__main__":
    main()
