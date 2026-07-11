# P0: Adaptive Penumbra Width Validation

Date: 2026-07-11

## Question

Does the learned width head recover a physically meaningful penumbra width,
and does width-conditioned feedback improve binary shadow detection beyond the
adaptive soft target itself?

## Independent physical protocol

The ISTD test set provides 540 aligned triplets: a shadow image, binary mask,
and shadow-free image. The binary mask is used only to locate boundary normals.
The width target is measured independently from the paired images:

1. Resize each triplet to 512 x 512.
2. Compute robust log attenuation as the median channel value of
   `log(clean) - log(shadow)`.
3. Sample attenuation along the outward boundary normal.
4. Normalize each profile with local shadow-side and lit-side endpoint levels.
5. Measure the distance between the 80% and 20% crossings.
6. Reject profiles with weak contrast or poor outward monotonicity.

The standard filter retains 63,205 profiles from all 540 images. A stricter
filter retains 56,866 profiles, so the result is not driven by marginal cases.

## Physical width results

| Model | Image Pearson | Image Spearman | Profile Pearson | Width MAE (px) |
|---|---:|---:|---:|---:|
| Untrained width control | -0.119 | -0.106 | -0.021 | 6.242 |
| Width-only training | **0.578** | **0.533** | **0.290** | **3.688** |
| Joint training | 0.551 | 0.521 | 0.281 | 4.133 |

The width-only predictor increases monotonically across physical groups:

| Physical group | Measured width (px) | Predicted width (px) |
|---|---:|---:|
| Narrow | 2.668 | 5.398 |
| Medium | 3.111 | 6.476 |
| Wide | 4.407 | 8.119 |

The model learns the ordering of penumbra widths but systematically
overestimates their magnitude. The continuous penumbra confidence also agrees
strongly with the paired-image attenuation profile (Pearson 0.849; 0.860 under
the stricter profile filter).

## Causal matched ablation

Both runs use seed 20260711, the same initializer, adaptive RGB/GT soft target,
optimizer, 500-step schedule, and trainable penumbra/binary heads. The only
difference is whether the width predictor, width loss, and width-conditioned
residual are enabled.

### SBU at 250 steps

| Variant | BER | FPR | FNR | F1 |
|---|---:|---:|---:|---:|
| Adaptive target only | **2.7346** | **2.8542** | 2.6151 | **93.1193** |
| Adaptive target + width joint | 2.7357 | 2.8571 | **2.6143** | 93.1144 |

For joint minus target-only, paired bootstrap over 638 test images gives BER
delta +0.00107 with 95% CI [+0.00042, +0.00169]. The small FNR reduction has a
CI crossing zero, while FPR and F1 worsen. At 500 steps, the same ordering
remains: 2.7400 versus 2.7417 BER.

### Direct SBU-to-ISTD transfer

| Variant | BER | FPR | FNR | F1 |
|---|---:|---:|---:|---:|
| Adaptive target only | 3.0731 | **2.2646** | 3.8815 | 92.3295 |
| Adaptive target + width joint | **3.0682** | 2.2649 | **3.8714** | **92.3340** |

The ISTD BER delta is -0.00490 with paired-bootstrap 95% CI
[-0.00587, -0.00401]. This is a stable but very small cross-domain recall gain,
not a main benchmark improvement.

## Decision

1. Keep adaptive soft-target construction and continuous penumbra confidence
   as the central second contribution.
2. Keep the width head as a physically validated auxiliary representation:
   it demonstrably distinguishes narrow and wide penumbrae without extra
   manual width labels.
3. Do not claim that the current scalar `gamma_width` feedback improves SBU.
   The matched experiment rejects that claim.
4. Disable or omit width feedback in the main model unless it is explicitly
   presented as a small cross-domain analysis. The width head itself can remain
   auxiliary for diagnostics and physical validation.
5. Stop tuning the third decimal of the 2.73 SBU checkpoint. Further work is
   justified only for a redesigned coupling mechanism with separate
   shadow-side support and lit-side suppression, followed by the same matched
   controls.

## Paper-ready evidence

- Main mechanism table: physical width correlation and MAE against ISTD pairs.
- Group table: measured and predicted width for narrow/medium/wide penumbrae.
- Causal ablation table: adaptive target only versus width joint.
- Figure: paired shadow/shadow-free attenuation profiles with measured and
  predicted 20-80% widths, plus a width scatter plot.
- Limitation: predicted widths are calibrated too high in absolute pixels.
