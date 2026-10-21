### Solo OOF
| tag | OOF | hi | lo | corr Bo / ml42 |
|---|---|---|---|---|
| d12_ml_orig_s1 | 13.317 | -24.2 | 17.2 | 0.979 / 0.965 |
| d12_ml_orig_s2 | 13.173 | -26.5 | 17.2 | 0.976 / 0.963 |
| d12_ml_trans_s42 | 13.295 | -26.0 | 17.1 | 0.931 / 0.928 |
| d12_ml_asym | 13.296 | -24.4 | 17.2 | 0.967 / 0.970 |
| d12_ml_orig_s42_frs7 | 13.036 | -24.5 | 16.1 | 0.932 / 0.923 |
| mlavg(3) | 13.001 | -24.7 | 17.1 | 0.985 / - |

### nCV 10×5 (NNLS+icpt), Δ vs 12.601 base
| 구성 | nCV | sd | Δ | hi | lo |
|---|---|---|---|---|---|
| base | 12.601 | 0.023 | +0.000 | -27.2 | 17.4 |
| night best: Bo←ml42+dldl+canon | 12.563 | 0.025 | -0.037 | -26.9 | 17.5 |
| night best +d12_ml_orig_s1(→Borig) | 12.576 | 0.025 | -0.025 | -26.8 | 17.5 |
| night best +d12_ml_orig_s1(new col) | 12.563 | 0.025 | -0.037 | -26.9 | 17.5 |
| night best +d12_ml_orig_s2(→Borig) | 12.571 | 0.026 | -0.029 | -26.9 | 17.5 |
| night best +d12_ml_orig_s2(new col) | 12.565 | 0.024 | -0.035 | -26.9 | 17.5 |
| night best +d12_ml_trans_s42(→Btrans) | 12.567 | 0.026 | -0.034 | -26.8 | 17.5 |
| night best +d12_ml_trans_s42(new col) | 12.567 | 0.024 | -0.033 | -26.9 | 17.5 |
| night best +d12_ml_asym(→Borig) | 12.566 | 0.025 | -0.034 | -26.9 | 17.4 |
| night best +d12_ml_asym(new col) | 12.566 | 0.027 | -0.034 | -26.8 | 17.5 |
| base +mlavg(col) | 12.627 | 0.027 | +0.026 | -27.1 | 17.5 |
| Bo←mlseeds | 12.598 | 0.027 | -0.003 | -27.0 | 17.4 |
| Bo←mlseeds+dldl+canon | 12.581 | 0.026 | -0.019 | -26.9 | 17.5 |
| Bo←mlseeds+dldl+canon, Bt←mltrans | 12.586 | 0.026 | -0.015 | -26.9 | 17.5 |
| Bo←mlseeds+dldl+canon +asym(col) | 12.589 | 0.026 | -0.012 | -26.9 | 17.5 |
| ALL: Bo←mlseeds+dldl+canon, Bt←mltrans, +asym(col) | 12.593 | 0.027 | -0.007 | -26.9 | 17.5 |
| -- 전 시드(s0~s3) 평균, 사후 drop 없음 -- |||||| 
| [all-seed] base | 12.598 | 0.022 | -0.003 | -27.1 | 17.4 |
| [all-seed] Bo←ml42+dldl+canon | 12.566 | 0.023 | -0.035 | -26.8 | 17.5 |
| [all-seed] Bo←mlseeds+dldl+canon | 12.580 | 0.025 | -0.021 | -26.9 | 17.5 |
| [all-seed] Bo←mlseeds+dldl+canon, Bt←mltrans | 12.585 | 0.025 | -0.015 | -26.9 | 17.5 |
| -- fold rs 비교: ml s42 solo OOF rs42=12.994 / rs7=13.036 (hi -23.5/-24.5), corr 0.923 -- ||||||
| night best, ml42→ml(rs7) | 12.560 | 0.025 | -0.041 | -26.8 | 17.5 |
| base +ml(rs7) col | 12.588 | 0.026 | -0.013 | -26.8 | 17.2 |
| -- 전 시드(s0~s3) 평균, 사후 drop 없음 -- |||||| 
| [all-seed] base | 12.598 | 0.022 | -0.003 | -27.1 | 17.4 |
| [all-seed] Bo←ml42+dldl+canon | 12.566 | 0.023 | -0.035 | -26.8 | 17.5 |
| [all-seed] Bo←mlseeds+dldl+canon | 12.580 | 0.025 | -0.021 | -26.9 | 17.5 |
| [all-seed] Bo←mlseeds+dldl+canon, Bt←mltrans | 12.585 | 0.025 | -0.015 | -26.9 | 17.5 |
| -- fold rs 비교: ml s42 solo OOF rs42=12.994 / rs7=13.036 (hi -23.5/-24.5), corr 0.923 -- ||||||
| night best, ml42→ml(rs7) | 12.560 | 0.025 | -0.041 | -26.8 | 17.5 |
| base +ml(rs7) col | 12.588 | 0.026 | -0.013 | -26.8 | 17.2 |

### 재현성: 12.563 구성의 ml42 자리를 단일 모델로 대체 (vs base)
| 대체 | nCV | Δ | repeat win(10) | 샘플 SSE win율 | hi |
|---|---|---|---|---|---|
| ml42 | 12.563 | -0.037 | 10/10 | 0.544 | -26.9 |
| d12_ml_orig_s1 | 12.587 | -0.013 | 10/10 | 0.528 | -27.0 |
| d12_ml_orig_s2 | 12.582 | -0.019 | 10/10 | 0.558 | -27.1 |
| ml7 | 12.560 | -0.041 | 10/10 | 0.528 | -26.8 |
| trans_s42→Bt | 12.581 | -0.020 | 9/10 | 0.540 | -27.0 |
| (ml 없음: dldl+canon만) | 12.576 | -0.025 | 10/10 | 0.556 | -27.0 |

BEST: night best, ml42→ml(rs7) nCV=12.560
no candidate (>12.55)
