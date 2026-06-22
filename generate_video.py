# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch>=2.4",
#     "diffusers>=0.32",
#     "transformers>=4.46",
#     "accelerate>=1.0",
#     "bitsandbytes>=0.44",
#     "imageio>=2.34",
#     "imageio-ffmpeg",
#     "pillow",
#     "numpy",
#     "ftfy",
# ]
# ///
"""
Generate a video from a single image frame using WAN2.1 I2V.

Usage:
    uv run generate_video.py --image frame.png --prompt "robot arm picks up a red cube"
    uv run generate_video.py --image frame.png --prompt "..." --output out.mp4 --resolution 720p --frames 81
    uv run generate_video.py --image frame.png --prompt "..." --lora ./my_lora.safetensors --seed 42

Memory modes (pick one based on your VRAM):
    default                  sequential CPU offload, ~16GB peak — safe for 4090
    --quantize               4-bit transformer via bitsandbytes, ~12GB peak, slightly lower quality
    --no-offload             everything on GPU, fastest but needs ~22GB free
"""

import argparse
import time
import numpy as np
import torch
from pathlib import Path
from PIL import Image

from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.quantizers import PipelineQuantizationConfig
from diffusers.utils import export_to_video
from transformers import CLIPVisionModel


MODELS = {
    "480p": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
    "720p": "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
}

MAX_AREA = {
    "480p": 480 * 832,
    "720p": 720 * 1280,
}


def resize_to_wan(image: Image.Image, pipe: WanImageToVideoPipeline, resolution: str) -> tuple[Image.Image, int, int]:
    """Resize image preserving aspect ratio, snapped to WAN's patch grid."""
    mod = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    max_area = MAX_AREA[resolution]
    ratio = image.height / image.width
    height = round(np.sqrt(max_area * ratio)) // mod * mod
    width = round(np.sqrt(max_area / ratio)) // mod * mod
    return image.resize((width, height), Image.LANCZOS), height, width


def load_pipeline(model_id: str, offload: bool, quantize: bool) -> WanImageToVideoPipeline:
    image_encoder = CLIPVisionModel.from_pretrained(
        model_id, subfolder="image_encoder", torch_dtype=torch.float32
    )
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float32
    )

    kwargs: dict = dict(vae=vae, image_encoder=image_encoder, torch_dtype=torch.bfloat16)

    if quantize:
        # 4-bit NF4 quantization on the transformer only — cuts it from ~28GB to ~7GB
        kwargs["quantization_config"] = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={
                "bnb_4bit_compute_dtype": torch.bfloat16,
                "bnb_4bit_quant_type": "nf4",
            },
            components_to_quantize=["transformer"],
        )

    pipe = WanImageToVideoPipeline.from_pretrained(model_id, **kwargs)

    if offload:
        if quantize:
            # sequential offload moves tensors to meta device which bnb 4-bit can't handle;
            # model_cpu_offload (CPU, not meta) is compatible and still saves significant VRAM
            pipe.enable_model_cpu_offload()
        else:
            pipe.enable_sequential_cpu_offload()
    else:
        pipe.to("cuda")

    # Always safe to enable; cuts VAE memory with no quality loss
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    return pipe


def main():
    parser = argparse.ArgumentParser(description="WAN2.1 image-to-video generation")
    parser.add_argument("--image", required=True, help="Input PNG frame")
    parser.add_argument("--prompt", required=True, help="Text description of the motion")
    parser.add_argument("--negative-prompt", default=(
        "static, no motion, blurry, low quality, worst quality, "
        "deformed, disfigured, JPEG artifacts"
    ))
    parser.add_argument("--output", default="output.mp4")
    parser.add_argument("--resolution", choices=["480p", "720p"], default="480p",
                        help="480p fits comfortably on 24GB; 720p needs ~20GB with offload")
    parser.add_argument("--frames", type=int, default=81,
                        help="Number of frames (must be 4k+1: 17, 33, 49, 65, 81...)")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-offload", action="store_true",
                        help="Disable CPU offload (faster if you have enough VRAM)")
    parser.add_argument("--quantize", action="store_true",
                        help="4-bit quantize the transformer (bitsandbytes NF4, ~12GB peak)")
    parser.add_argument("--lora", default=None,
                        help="Path to LoRA weights (.safetensors) to load before generation")
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--throttle", type=float, default=0.0, metavar="SECS",
                        help="Seconds to sleep between denoising steps (e.g. 0.5 keeps GPU ~50-60%%)")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model_id = MODELS[args.resolution]
    print(f"Loading {args.resolution} model: {model_id}")
    pipe = load_pipeline(model_id, offload=not args.no_offload, quantize=args.quantize)

    if args.lora:
        print(f"Loading LoRA: {args.lora}")
        pipe.load_lora_weights(args.lora)
        pipe.set_adapters(["default"], adapter_weights=[args.lora_scale])

    image = Image.open(image_path).convert("RGB")
    image, height, width = resize_to_wan(image, pipe, args.resolution)
    print(f"Resized input to {width}x{height}")

    generator = torch.Generator("cuda").manual_seed(args.seed) if args.seed is not None else None

    callback = None
    if args.throttle > 0:
        def callback(pipe, step, timestep, kwargs):
            time.sleep(args.throttle)
            return kwargs

    print(f"Generating {args.frames} frames...")
    output = pipe(
        image=image,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=height,
        width=width,
        num_frames=args.frames,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        callback_on_step_end=callback,
    ).frames[0]

    export_to_video(output, args.output, fps=args.fps)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
