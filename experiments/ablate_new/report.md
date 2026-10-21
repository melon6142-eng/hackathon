# 새 경계·밝기 지표 ablation (2026-10-12)
특징: gen/{gen,acf,bd,morph} 중 노이즈·블러 4개(sig_raw,noise_lag1,noise_wav,noise_res) 제외 + 대비변환(CLAHE/감마2/1-99% stretch) 신규 51개(ctr_feats.csv). hand에서 sharp 제외, tex 전부 제외. CV 5fold×3 seed nested. 공정 비교 지표는 repRMSE(seed별 RMSE 평균; 'RMSE'는 seed 평균 OOF라 약간 낙관적).
```
A0 gen지표만(노이즈제외) Ridge: RMSE 13.871 | hi(>230,n=18) bias -31.4 | lo(<170,n=38) bias +19.6 | repRMSE 13.932
A0 gen지표만(노이즈제외) LGB: RMSE 14.153 | hi(>230,n=18) bias -29.1 | lo(<170,n=38) bias +20.0 | repRMSE 14.361
A 새지표(gen+대비변환) Ridge: RMSE 13.834 | hi(>230,n=18) bias -31.3 | lo(<170,n=38) bias +19.2 | repRMSE 13.920
A 새지표(gen+대비변환) LGB: RMSE 13.718 | hi(>230,n=18) bias -29.0 | lo(<170,n=38) bias +19.2 | repRMSE 13.930
B A+hand(형상)+grain Ridge: RMSE 13.610 | hi(>230,n=18) bias -30.9 | lo(<170,n=38) bias +19.0 | repRMSE 13.684
B A+hand(형상)+grain LGB: RMSE 13.600 | hi(>230,n=18) bias -28.2 | lo(<170,n=38) bias +18.6 | repRMSE 13.805
C B+DINO Ridge블렌드: RMSE 12.879 | hi(>230,n=18) bias -28.9 | lo(<170,n=38) bias +17.2 | repRMSE 12.971
C B+DINO + B-LGB: RMSE 12.946 | hi(>230,n=18) bias -28.1 | lo(<170,n=38) bias +17.0 | repRMSE 12.953
D BL1(oof_blend1): RMSE 12.949 | hi(>230,n=18) bias -28.3 | lo(<170,n=38) bias +17.7 | repRMSE 12.949
E BL1구성+새지표블록 Ridge: RMSE 12.919 | hi(>230,n=18) bias -28.5 | lo(<170,n=38) bias +17.3 | repRMSE 13.018
E BL1 + 새지표 LGB: RMSE 12.946 | hi(>230,n=18) bias -27.5 | lo(<170,n=38) bias +17.4 | repRMSE 12.950
E BL1 + B(LGB,Ridge): RMSE 12.989 | hi(>230,n=18) bias -27.6 | lo(<170,n=38) bias +17.4 | repRMSE 12.997
```
## 추가 분석 (an.py, analysis_feats.csv)
```
n_feat=271, hi=18, mid(예측200~230,실제<=230)=142

R² median: hand+grain 0.682, DINO 0.689, 전체 0.698; R²_all>0.8 비율 0.38, <0.5 비율 0.30

잔차 CV R²: Ridge -0.010, LGB -0.075

max|corr(잔차)| 0.131 (bwidth_med), 순열 null 95% 0.159

hi vs mid: max AUC 0.762/0.291, |d|>0.8 개수 4

hi vs mid 상위(|d|):
              d_hi_vs_mid  auc_hi_vs_mid  corr_resid  r2_all
Lmaj_ac10           0.938          0.731       0.005   0.869
acf_major           0.896          0.759       0.043   0.759
Lmaj                0.846          0.758       0.048   0.713
z-0.5_euler        -0.814          0.291       0.033   0.661
hf_16_32           -0.799          0.315       0.005   0.962
st_g90             -0.767          0.304      -0.002   0.951
z0_perim           -0.707          0.304       0.016   0.908
z-0.5_wmean         0.706          0.727      -0.017   0.612
z-1.5_perim        -0.703          0.316      -0.013   0.764
rf_bgrad_med       -0.703          0.330      -0.003   0.901
bwidth_q99          0.695          0.688       0.019   0.787
z-2_euler          -0.686          0.310       0.001   0.584

잔차상관 상위:
              corr_resid  corr_y  r2_all
bwidth_med        -0.131  -0.056   0.391
abs128_wmean       0.123   0.349   0.178
abs128_max         0.111   0.359   0.235
ws_blen_std       -0.105  -0.160   0.615
rw_bstep_q99       0.102   0.177   0.348
abs128_area        0.099   0.389   0.411
ws_blen_q99       -0.095  -0.140   0.668
ws_blen_mean      -0.094  -0.120   0.670
rw_bdark_std       0.089   0.202   0.473
rw_lowc_frac       0.087   0.081   0.419

중복 낮은(R²_all 하위) 지표:
             r2_all  corr_y  corr_resid
acf_ang      -0.025   0.023       0.039
st_ang       -0.017   0.010       0.057
acf_angle    -0.016  -0.014      -0.005
z-1_perc     -0.013  -0.021       0.001
abs128_perc  -0.005   0.084       0.010
abs144_perc   0.008   0.057       0.024
abs96_top     0.034   0.003      -0.039
g2_q95        0.041   0.017      -0.048
abs144_top    0.047   0.084       0.032
I_q75         0.057  -0.070      -0.050```
## 결론
- 가설 기각: 질감·노이즈를 빼도 새 지표 단독 13.68~13.93으로 BL1(12.95)보다 나쁨. B+DINO 12.95~12.97, BL1+새 지표 12.95~13.02로 개선 없음.
- 새 지표 중앙값 R² 0.70이 hand+grain+DINO로 설명됨(중복). 남는 정보도 잔차 예측력 0(CV R² −0.01/−0.08). 잔차 최대 상관 0.131은 순열 null 0.159 미만.
- 고경도 vs 예측 200~230 시편에서 큰 결정립·신장(acf_major, Lmaj, AUC 0.73~0.76)이 갈림. 하지만 기존 특징과 중복(R² 0.71~0.87)이고 고경도 편향은 −27.5~−31로 그대로임.
