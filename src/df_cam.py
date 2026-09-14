"""DF-CAM: Diffusion gradient-weighted Class Activation Mapping (Park et al. 2024).

Internal attribution: backprop from the U-Net's noise-prediction output to a
chosen feature map, then Grad-CAM-style weighted activation map.

Paper Eq 10-11:
    α_k = (1/λ) Σ_i Σ_j ∂R_t / ∂A^k_ij        (global-avg-pool of gradients)
    L_{DF-CAM} = ReLU( Σ_k α_k · A^k )        (linear combo + ReLU)

where the 'target score' is the SUM of all pixels of the output representation
R_t (the noise prediction), i.e. score = R_t.sum(), and A^k are the activation
maps of the hooked layer.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _register_forward_backward_hooks(module: nn.Module):
    """Capture activations (forward) and gradients (backward) of a module."""
    activations, gradients = {}, {}

    def fwd_hook(m, _, out):
        activations["out"] = out[0] if isinstance(out, tuple) else out

    def bwd_hook(m, grad_input, grad_output):
        g = grad_output[0] if isinstance(grad_output, tuple) else grad_output
        gradients["out"] = g

    fh = module.register_forward_hook(fwd_hook)
    bh = module.register_full_backward_hook(bwd_hook)
    return fh, bh, activations, gradients


@torch.enable_grad()
def df_cam_step(
    unet,
    text_emb,          # (2, 77, 768)
    latent,            # (1, 4, 64, 64)
    t: int,
    target_module: nn.Module,
    guidance_scale: float = 7.5,
    device: str = "cuda",
) -> torch.Tensor:
    """Compute a DF-CAM heatmap for step t against a hooked U-Net module.

    Args:
        target_module: an nn.Module inside `unet` (e.g. a mid-block or
            decoder block) whose activations become A^k and whose gradients
            derive α_k.
    Returns:
        heatmap: (h, w) in [0,1] upsampled to the latent resolution.
    """
    fh, bh, acts, grads = _register_forward_backward_hooks(target_module)

    # ---- forward: noise prediction to use as target ----
    latent.requires_grad_(True)
    z = torch.cat([latent] * 2)
    pred = unet(z, torch.as_tensor([t] * 2, device=device), text_emb).sample
    pred_uncond, pred_text = pred.chunk(2)
    target_score = pred_text.sum()          # sum of pixels of output rep as target

    # ---- backward: get grads wrt the hooked module's output ----
    target_score.backward(retain_graph=True)

    A = acts["out"]                          # (B, C, h, w)
    G = grads["out"]                         # (B, C, h, w)

    # α_k = global-avg-pool over gradient map
    alpha = G.mean(dim=(2, 3), keepdim=True)           # (B, C, 1, 1)
    cam = torch.relu((alpha * A).sum(dim=1, keepdim=True))  # (B, 1, h, w)

    fh.remove(); bh.remove()
    latent.requires_grad_(False)

    # multiple batch entries averaged, normalized
    cam = cam[0, 0] if cam.shape[0] == 1 else cam.mean(0)
    cam = torch.nn.functional.interpolate(
        cam.unsqueeze(0).unsqueeze(0), size=latent.shape[-2:], mode="bilinear"
    )[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.detach()