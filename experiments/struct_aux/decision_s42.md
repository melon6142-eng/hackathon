# struct_aux seed42 decision (2026-10-15 17:43 CST)

Rule: expand to 7/123/2026 only if eq4 beats eq3 by >=0.03 OR aux alone beats ViT-B orig (12.948) / aux0 control.

## Verdict: STOP — do not expand

- eq3 RMSE: 12.539
- eq4 (eq3+aux0): 12.554 (delta -0.015, worse)
- eq4 (eq3+aux0.3): 12.596 (worse)
- eq3 orig→aux0: 12.514 (beat by 0.025 < 0.03)
- aux0 alone: 12.974 (vs ViT 12.948 — worse)
- aux0.3 alone: 13.176 (vs ViT 12.948 / aux0 12.974 — worse)

Note: aux0.3 improves AUPRC 0.375 and P@18 0.333 and hi_bias -23.79, but expand gate is RMSE-based and failed.
Did not kill other jobs (aux*_s1 dirs present); only withheld 7/123/2026 expansion.
