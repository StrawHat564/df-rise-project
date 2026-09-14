# dfrise-project

Reproduction of **DF-RISE** (Park, Ju, Lee, ESWA 2024 — [Explaining Generative Diffusion Models via Visual Analysis](https://arxiv.org/abs/2402.10404)) plus a **feature-level extension** (colour / sharpness / texture attribution).

The Kaggle implementation runs through the paper's three tools, then extends DF-RISE with feature-absence masks to get *concept-resolved* + *temporally-resolved* attribution.

## What is being implemented implementing (in order)

| # | Tool | Paper section | Status |
|---|------|---------------|--------|
| 1 | **DF-RISE** — spatial saliency per step via masking + structure-similarity | §3.1 | skeleton |
| 2 | **DF-CAM** — internal concept heatmap via Grad-CAM on U-Net | §3.2 | skeleton |
| 3 | **Exponential time-step sampling** — concentrate steps in a temporal stage | §3.3 | skeleton |
| 4 | **Evaluation** — deletion/insertion games + AUC, LIME baseline | §4.1 | skeleton |
| 5 | **Feature-importance extension** — colour/sharpness/texture masks + colour-sensitive similarity | *novel* | skeleton |

## Repository structure

```
dfrise-project/
├── src/
│   ├── df_rise.py          # DF-RISE: masks + structure similarity + accumulation
│   ├── df_cam.py           # DF-CAM: forward/backward hooks → heatmap
│   ├── exponential_scheduler.py
│   ├── feature_importance.py  # novel colour/sharpness/texture variant
│   ├── eval_utils.py       # deletion/insertion games, AUC
│   ├── visualization.py
│   └── pipeline.py         # shared Stable Diffusion loading + DDIM loop
├── configs/default.yaml
├── notebooks/              # Kaggle notebooks (import from src/)
└── outputs/                # gitignored
```

## Setup — local (optional, for debugging small steps)

```bash
cd Project/dfrise-project
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision diffusers transformers accelerate peft open-clip-torch numpy matplotlib pyyaml scipy scikit-image piqa
```

## Model choice

The paper tests on **Stable Diffusion** (they cite [40] as the Laion-5B pretrained LDM — `CompVis/stable-diffusion-v1-4`, the same architecture lineage as `runwayml/stable-diffusion-v1-5`). Both give DDIM + CLIP ViT-L/14 + 512×512 latents of 64×64 with 4 channels. **Use v1-5** (more commonly available, same UNet architecture) and record that in the config. ControlNet variants can be added later for the cross-conditioning novelty (roadmap step).

## Notes on the paper's key formulas (so the code matches)

- **DF-RISE** (Eq 1–9, Alg 1): pop N random Gaussian-threshold binary masks `M_n` over the latent `R_t` (input to U-Net); forward `f(R_t)` and `f(R_t⊙M_n)`; score = SSIM **structure term** `s(a,b) = (σ_ab + C3)/(σ_a σ_b + C3)`; accumulate `Σ M_n ⊙ s(...)`; min–max normalize.
- **DF-CAM** (Eq 10–11): target score = sum of all pixels of the U-Net *output* representation; `α_k = (1/Σλ) Σ_i Σ_j ∂R_t/∂A^k_ij`; heatmap = `ReLU(Σ_k α_k A^k)`.
- **Exponential sampling** (Eq 12–14): `δ^(l+γ)=T`, `δ = e^(ln T/(l+γ))`, `p_t = T − δ^(t+γ)`. γ<30 → dense 1000–800 (early); γ=60 in paper; mirror for latter stage.

## Implementation priority

1. Get `pipeline.py` + `df_rise.py` working on 1–2 prompts → verify saliency maps look sensible.
2. Add `df_cam.py` — sanity-check against DF-RISE on the same steps (they should correlate).
3. Add `eval_utils.py` — deletion/insertion + AUC (this is the "does it work" gate).
4. Add exponential sampling and compare early/later-stage concept emphasis (Fig 1/6 in paper).
5. **Novelty**: `feature_importance.py` with colour/sharpness/texture features (design in `p4_research_notes.md` §F.17).

