# SBU Soft-Shadow Split Pilot

Date: 2026-07-11

## Purpose

Test whether the current IG-PaSCL checkpoints support the claim that the
method is especially strong on soft shadows, rather than only on the standard
hard-shadow-dominated SBU/ISTD benchmarks.

## Automatic split

- Source: 638 standard SBU test RGB images and binary GT masks.
- No model prediction is used for sample selection.
- Every image is resized to 512 x 512.
- RGB luminance is sampled along outward GT-boundary normals.
- Each reliable profile is normalized with local shadow-side and lit-side
  endpoints, then measured by its unclipped 20--80% transition width.
- Per-image softness score: reliability-weighted 75th percentile width.
- Provisional soft set: top 15% (96 images).
- Provisional hard set: bottom 15% (96 images).

The soft score range is 12.97--29.50 px; the hard range is 1.47--4.39 px.
Visual inspection of the top 24 cases shows clear overall separation, but some
texture/material-edge cases remain and require manual exclusion before the
split is used in the paper.

## Existing-checkpoint screen

Metrics below use the same frozen provisional split.

| Checkpoint | Soft BER | Soft FPR | Soft FNR | Soft F1 | Hard BER |
|---|---:|---:|---:|---:|---:|
| Host | **3.4256** | 3.1015 | **3.7496** | 91.5939 | 2.2929 |
| Inner-margin | 3.4491 | 2.8096 | 4.0886 | 91.9931 | 2.2511 |
| Band-width 10 | 3.4934 | 2.7180 | 4.2689 | 92.0813 | 2.2541 |
| Refine | 3.5037 | 2.5888 | 4.4186 | 92.2610 | 2.2469 |
| Global-best | 3.5140 | 2.5764 | 4.4516 | 92.2686 | 2.2474 |
| TV0.20 | 3.5234 | 2.5679 | 4.4788 | 92.2713 | 2.2455 |
| IG-PaSCL main | 3.5259 | **2.5554** | 4.4965 | **92.2873** | **2.2488** |

## Interpretation

The current IG-PaSCL family is precision-oriented. On the provisional soft
set, it consistently lowers FPR and raises F1, but misses more weak shadow
pixels, increasing FNR enough that BER is worse than the Host. The main model
therefore does not currently support a claim of superior soft-shadow BER.

The inner-shadow margin is the most promising existing mechanism: it reduces
soft FNR from 4.4965 to 4.0886 and approaches Host BER, but still does not beat
it. A new soft-shadow experiment should start from this observation and train
an explicit shadow-side support loss on the frozen soft split, while preserving
the outer-side precision gain.

## Decision gate

1. Manually review the 96 soft and 96 hard candidates and freeze the final
   split before new model selection.
2. Add boundary-band BER/F1 on the reviewed split.
3. Train a matched soft-recall variant only if the reviewed split retains the
   same failure pattern.
4. Do not deploy external competing methods until Ours first beats its Host on
   the reviewed soft split; otherwise the comparison cannot support the paper
   story.
