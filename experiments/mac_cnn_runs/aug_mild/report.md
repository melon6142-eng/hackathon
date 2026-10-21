# Mild aug eval (rot±12° + crop≥0.85, no blur)

mild tags: ['dinoB_orig_mild_s42', 'dinoB_orig_mild_s1', 'dinoB_orig_mild_s2']

- dinoB_orig_mild_s42: OOF 13.902 hi -27.4 lo 16.5
- dinoB_orig_mild_s1: OOF 13.757 hi -28.2 lo 16.7
- dinoB_orig_mild_s2: OOF 13.779 hi -28.2 lo 15.8
- mild avg: OOF 13.615 hi -27.9

## nCV 10×5 vs base
| 구성 | nCV | sd | Δ | hi | lo |
|---|---|---|---|---|---|
| base | 12.601 | 0.023 | +0.000 | -27.2 | 17.4 |
| Bo←mild_avg | 12.650 | 0.031 | +0.050 | -27.4 | 17.4 |
| base+mild_col | 12.622 | 0.023 | +0.022 | -27.3 | 17.5 |
| Bo soft←dinoB_orig_mild_s42 | 12.626 | 0.029 | +0.025 | -27.3 | 17.4 |
| base+dinoB_orig_mild_s42 col | 12.609 | 0.026 | +0.009 | -27.2 | 17.4 |
| Bo soft←dinoB_orig_mild_s1 | 12.610 | 0.028 | +0.009 | -27.3 | 17.5 |
| base+dinoB_orig_mild_s1 col | 12.630 | 0.022 | +0.029 | -27.3 | 17.5 |
| Bo soft←dinoB_orig_mild_s2 | 12.607 | 0.026 | +0.006 | -27.2 | 17.4 |
| base+dinoB_orig_mild_s2 col | 12.628 | 0.027 | +0.027 | -27.2 | 17.5 |

## 채택

- Bo←mild_avg가 base보다 나은 seed: 0/10
- **채택: 아니오**
- 기준 hi -27.2

blur 증강은 이미 실패. 이번은 회전+크롭만.