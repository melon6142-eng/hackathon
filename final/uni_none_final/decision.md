# decision — uni_none vs final_clean

규칙(사전 고정): 3seed eq3에서 uni_none이 fold seed 4개 중 3개 이상에서 final_clean보다 낮고, 평균 Δ ≤ −0.03이면 uni_none 메인, 아니면 final_clean 메인.

- 결과: uni 승 4/4, 평균 Δeq3 -0.052, 평균 ΔnCV -0.056
- 판정: **uni_none 메인**
- 메인 CSV 후보: submission_uninone_eq3_ckpt.csv (ckpt 재추론, 없으면 submission_uninone_eq3.csv)

## ckpt 재학습 확인 (fold seed 42)
- seed 42: OOF bit-identical=True, OOF 12.661 (기존 12.661), test corr vs 기존 1.000000, max|Δ| 0.0000; ckpt 재추론 vs 학습중 예측 max|Δ| 0.00e+00
- seed 1: OOF bit-identical=True, OOF 12.675 (기존 12.675), test corr vs 기존 1.000000, max|Δ| 0.0000; ckpt 재추론 vs 학습중 예측 max|Δ| 0.00e+00
- seed 2: OOF bit-identical=True, OOF 12.652 (기존 12.652), test corr vs 기존 1.000000, max|Δ| 0.0000; ckpt 재추론 vs 학습중 예측 max|Δ| 0.00e+00

제출하지 않음. 상세 표는 eval.md.