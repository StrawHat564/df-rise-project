"""Shared Stable Diffusion loading and DDIM inference loop.

The paper uses deterministic DDIM (eta=0) with 30 steps on a latent diffusion
model (Stable Diffusion v1.5, CLIP ViT-L/14 text encoder, 512x512).

This module centralizes:
- pipeline loading (with float16 + safety-cleaner off)
- the core denoise loop with hooks so DF-RISE / DF-CAM / feature variants can
  instrument each step without forking the loop.
"""
from __future__ import annotations

import torch
import yaml
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
import numpy as np


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_stable_diffusion_components(cfg: dict, device: str = "cuda"):
    """Return (tokenizer, text_encoder, vae, unet, scheduler) for SD1.x."""
    repo = cfg["model"]["name"]

    tokenizer = CLIPTokenizer.from_pretrained(repo, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(repo, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(repo, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(repo, subfolder="unet").to(device)

    scheduler = DDIMScheduler.from_pretrained(repo, subfolder="scheduler")
    scheduler.set_timesteps(cfg["model"]["num_inference_steps"])

    if cfg["model"].get("fp16", True):
        unet, vae, text_encoder = unet.half(), vae.half(), text_encoder.half()

    return dict(
        tokenizer=tokenizer, text_encoder=text_encoder, vae=vae,
        unet=unet, scheduler=scheduler,
    )


def encode_prompt(components, prompt: str, device: str = "cuda") -> torch.Tensor:
    """Return text embeddings for a prompt (plus unconditional embeddings)."""
    tokenizer = components["tokenizer"]
    text_encoder = components["text_encoder"]

    text_input = tokenizer(
        prompt, padding="max_length", max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    ).to(device)
    text_emb = text_encoder(text_input.input_ids)[0]

    uncond_input = tokenizer(
        [""], padding="max_length", max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).to(device)
    uncond_emb = text_encoder(uncond_input.input_ids)[0]

    return torch.cat([uncond_emb, text_emb], dim=0)


@torch.no_grad()
def denoise_with_hooks(
    components: dict,
    prompt: str,
    seed: int = 0,
    guidance_scale: float = 7.5,
    num_steps: int | None = None,
    timestep_override: list[int] | None = None,
    step_hook=None,
    device: str = "cuda",
):
    """Run the DDIM reverse process.

    Args:
        components: dict from load_stable_diffusion_components
        step_hook: Callable(latent, noisy_latent, t, output_latent, t_idx)
            invoked after each UNet forward so external tools can compute
            saliency / CAM / feature maps per step. Returns None; must not
            modify the denoise trajectory.
        timestep_override: arbitrary list of timesteps (e.g. from
            exponential_scheduler) to use instead of the scheduler default.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    scheduler, unet, vae = components["scheduler"], components["unet"], components["vae"]

    if num_steps:
        scheduler.set_timesteps(num_steps)
    timesteps = timestep_override if timestep_override is not None else scheduler.timesteps

    # 1) sample initial latent x_T ~ N(0, I) in latent space (4, 64, 64)
    latents = torch.randn((1, 4, 64, 64), generator=gen, device=device, dtype=unet.dtype)

    emb = encode_prompt(components, prompt, device)

    for t_idx, t in enumerate(timesteps):
        # store the pre-step latent so tools can perturb it if they want
        current = latents.detach().clone()

        # CFG: two forward passes (uncond + cond)
        latent_input = torch.cat([latents] * 2)
        latent_input = latent_input / scheduler.init_noise_sigma if t_idx == 0 else latent_input

        noise_pred = unet(
            latent_input, torch.as_tensor([t] * 2, device=device), emb
        ).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        # DDIM deterministic update (eta=0)
        prev = scheduler.step(noise_pred, t, latents, eta=0.0).prev_sample
        prev = prev.to(dtype=latents.dtype)   # scheduler can promote to fp32
        if step_hook is not None:
            step_hook(noise_pred=noise_pred, current=current, t=t, t_idx=t_idx, prev=prev)
        latents = prev

    # VAE decode final latent -> image
    image = vae.decode(latents / 0.18215).sample[0]
    image = (image.clamp(-1, 1) + 1) / 2
    image = image.permute(1, 2, 0).float().cpu().numpy()
    return image, {"latents": latents, "timesteps": timesteps, "seed": seed}