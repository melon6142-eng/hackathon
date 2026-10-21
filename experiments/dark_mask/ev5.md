| 모델 | 단독 OOF | hi편향 |
|---|---|---|
| BL1 | 12.949 | -28.3 |
| Borig | 12.948 | -26.0 |
| Btrans | 12.947 | -26.4 |
| dinoB_mask_s42 | 13.857 | -28.1 |
| d4_pad266 | 13.717 | -27.3 |
| d4_binreg | 13.560 | -26.3 |
| dinoB_mask_s1 | 13.943 | -27.0 |
| dinoB_mask_s2 | 13.939 | -28.6 |
| d4_lds | 13.365 | -23.7 |
| d4_s448 | 13.524 | -24.4 |
| d4_rank | 13.031 | -26.0 |
| d4_crop140 | 14.233 | -30.2 |
| Bmask_avg | 13.772 | -27.9 |

기준 BL1+Borig+Btrans nCV 12.606±0.031 hi -27.2 lo 17.4

| 조합 | nCV | hi | lo |
|---|---|---|---|
| BL1+Borig+Btrans+d4_rank | 12.603±0.036 | -27.2 | 17.2 |
| BL1+Borig+Btrans+d4_rank+d4_crop140 | 12.603±0.036 | -27.2 | 17.2 |
| BL1+Borig+Btrans+d4_binreg+d4_rank | 12.604±0.035 | -27.2 | 17.2 |
| BL1+Borig+Btrans+d4_crop140 | 12.606±0.031 | -27.2 | 17.4 |
| BL1+Borig+Btrans+d4_binreg | 12.608±0.029 | -27.2 | 17.4 |
| BL1+Borig+Btrans+d4_binreg+d4_crop140 | 12.608±0.029 | -27.2 | 17.4 |
| BL1+Borig+Btrans+d4_lds+d4_rank | 12.613±0.038 | -27.1 | 17.2 |
| BL1+Borig+Btrans+d4_pad266+d4_rank | 12.613±0.038 | -27.2 | 17.2 |
| BL1+Borig+Btrans+d4_pad266 | 12.620±0.033 | -27.3 | 17.5 |
| BL1+Borig+Btrans+d4_pad266+d4_crop140 | 12.620±0.033 | -27.3 | 17.5 |
| BL1+Borig+Btrans+d4_rank+Bmask_avg | 12.621±0.040 | -27.2 | 17.3 |
| BL1+Borig+Btrans+d4_pad266+d4_binreg | 12.622±0.031 | -27.3 | 17.5 |

0.03 이상 개선 없음 → CSV 미생성