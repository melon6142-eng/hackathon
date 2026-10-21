BL1 OOF 12.949 hi -28.3

| 모델 | OOF | hi편향 | BL1+모델 nCV |
|---|---|---|---|
| B_orig_avg | 12.948 | -26.0 | 12.626±0.022 |
| B_trans_avg | 12.947 | -26.4 | 12.622±0.024 |
| cnx_full_w | 13.890 | -26.7 | 12.890±0.028 |
| dinoB_orig | 13.084 | -25.6 | 12.653±0.023 |
| dinoB_orig_s1 | 12.972 | -26.1 | 12.612±0.023 |
| dinoB_orig_s2 | 13.161 | -26.2 | 12.701±0.023 |
| dinoB_trans | 13.045 | -26.0 | 12.638±0.025 |
| dinoB_trans_s1 | 13.108 | -26.2 | 12.647±0.020 |
| dinoB_trans_s2 | 13.115 | -26.9 | 12.684±0.024 |
| dino_orig | 14.210 | -28.0 | 12.965±0.032 |
| dino_trans | 13.774 | -26.4 | 12.875±0.033 |
| dino_trans_w | 13.948 | -25.0 | 12.935±0.030 |

| 블렌드 | nCV(20회) | 가중치 |
|---|---|---|
| BL1+B1seed(orig,trans) | 12.624±0.031 | {'BL1': np.float64(0.469), 'dinoB_orig': np.float64(0.255), 'dinoB_trans': np.float64(0.276), 'c': np.float64(0.0)} |
| BL1+Bavg(orig,trans) | 12.606±0.031 | {'BL1': np.float64(0.44), 'B_orig_avg': np.float64(0.283), 'B_trans_avg': np.float64(0.277), 'c': np.float64(0.0)} |
| BL1+all | 12.696±0.042 | {'BL1': np.float64(0.404), 'cnx_full_w': np.float64(0.066), 'dinoB_orig': np.float64(0.08), 'dinoB_trans': np.float64(0.13), 'dino_orig': np.float64(0.0), 'dino_trans': np.float64(0.001), 'dino_trans_w': np.float64(0.0), 'dinoB_trans_s1': np.float64(0.107), 'dinoB_orig_s2': np.float64(0.0), 'dinoB_trans_s2': np.float64(0.0), 'dinoB_orig_s1': np.float64(0.213), 'c': np.float64(0.0)} |

BEST BL1+Bavg(orig,trans) nCV 12.606  test mean 193.86 (BL1 193.90)
CSV ./runs/dino_ft2/cand_BL1_Bavgorig-trans_cv12.61.csv