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
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Compute a DF-CAM heatmap for step t against a hooked U-Net module.

    device defaults to the latent's own device (auto-follows the pipeline).
    Args:
        target_module: an nn.Module inside `unet` (e.g. a mid-block or
            decoder block) whose activations become A^k and whose gradients
            derive α_k.
    Returns:
        heatmap: (h, w) in [0,1] upsampled to the latent resolution.
    """
    device = device if device is not None else latent.device
    # Match the UNet's canonical dtype (fp16 on GPU, fp32 on CPU) so that
    # fp32/fp16-dtype drift can't crash the forward. Same pattern as df_rise.
    tdtype = next(unet.parameters()).dtype
    text_emb = text_emb.to(device=device, dtype=tdtype)
    latent = latent.to(device=device, dtype=tdtype).detach().requires_grad_(True)

    # Hooks MUST be removed on every exit path (success or exception), else
    # they stay attached to the module and fire on every future forward, keep
    # stale refs alive and corrupt later activations -> try/finally.
    fh = bh = None
    try:
        fh, bh, acts, grads = _register_forward_backward_hooks(target_module)

        # ---- forward: noise prediction to use as target ----
        # Batch shape mirrors denoise_with_hooks (uncond + cond copies). Only
        # the cond branch's output score is backpropagated, so the uncond
        # result is discarded (`_`).
        z = torch.cat([latent] * 2)
        pred = unet(z, torch.as_tensor([t] * 2, device=device), text_emb).sample
        _, pred_text = pred.chunk(2)
        # sum of pixels of the output rep as target; float() avoids fp16
        # overflow when summing 4*64*64 output values.
        target_score = pred_text.float().sum()

        # ---- backward: get grads wrt the hooked module's output ----
        # retain_graph is unneeded: only one backward and no reuse of the graph.
        target_score.backward()

        A = acts["out"]                          # (B, C, h, w)
        G = grads["out"]                         # (B, C, h, w)

        # α_k = global-avg-pool over gradient map
        alpha = G.mean(dim=(2, 3), keepdim=True)           # (B, C, 1, 1)
        cam = torch.relu((alpha * A).sum(dim=1, keepdim=True))  # (B, 1, h, w)
    finally:
        if fh is not None:
            fh.remove()
        if bh is not None:
            bh.remove()
    latent.requires_grad_(False)

    # multiple batch entries averaged, normalized
    cam = cam[0, 0] if cam.shape[0] == 1 else cam.mean(0)
    cam = torch.nn.functional.interpolate(
        cam.unsqueeze(0).unsqueeze(0), size=latent.shape[-2:],
        mode="bilinear", align_corners=False
    )[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.detach()