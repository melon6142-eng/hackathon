# struct_aux 중간 보고 (2026-10-15 17:00 KST)
## Task A
- 백본: timm vit_base_patch14_dinov2.lvd142m (HF timm, license tag apache-2.0). BL1 DINO 특징 = dino/vitb14_252 (같은 ViT-B/14로 추정, 사전 추출 npy).
- 재계산 0.44·BL1(sub_bl1.csv)+0.283·orig3+0.277·trans3 vs cand_BL1_Bavgorig-trans_cv12.61.csv: corr 0.99999999, max|Δ| 0.052, RMS 0.042 (가중치 반올림 수준). vs ref_current_nnls_weights_test.csv: max|Δ| 0.048.
- 실제 업로드한 11.33 파일 원본은 Mac/box에서 못 찾음 → cand CSV가 제출본이라는 가정.
## Task B (seed 42, fold rs42 진행 중)
| fold | aux0 (대조) | aux0.3 |
|---|---|---|
| 0 | 14.134 | 14.553 |
| 1 | 11.182 | 11.298 |
| 2 | 12.765 | 12.889 |
