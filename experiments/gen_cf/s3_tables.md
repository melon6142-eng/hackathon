## 재현 세트: 실제 이미지 i의 적합 파라미터로 새로 생성한 이미지에 대한 예측 vs 실제 라벨 y_i

| 모델 | Pearson r | Spearman | 예측 SD | 실제 OOF 대비 |
|---|---|---|---|---|
| FrozenDINO_Ridge | 0.392 | 0.402 | 10.8 | - |
| Handcrafted_Ridge | 0.207 | 0.229 | 9.4 | - |
| ViT_mild_dinoB_orig_mild_s42 | 0.363 | 0.353 | 12.7 | - |
| ViT_mild_dinoB_orig_mild_s1 | 0.361 | 0.357 | 13.0 | - |
| ViT_mild_dinoB_orig_mild_s2 | 0.368 | 0.360 | 12.5 | - |
| ViT_orig_vit_s1 | 0.357 | 0.356 | 12.2 | - |
| ViT_orig_vit_s2 | 0.353 | 0.356 | 12.1 | - |
| ViT_orig_vit_s42 | 0.365 | 0.363 | 12.0 | - |

## 기울기
var,feat,model,real_slope,real_rho,model_slope_med,q25,q75,sign_agree_bases,sign_agree_basefold,slope_in_dist,frac_in
n_grains,g_Amean,FrozenDINO_Ridge,5.8,0.34,2.84,0.57,5.52,0.8,0.76,2.79,0.9
n_grains,g_Amean,Handcrafted_Ridge,5.8,0.34,4.38,2.74,5.76,0.87,0.89,4.39,0.9
n_grains,g_Amean,ViT_mild_dinoB_orig_mild_s42,5.8,0.34,3.58,1.06,6.6,0.83,0.79,3.89,0.9
n_grains,g_Amean,ViT_mild_dinoB_orig_mild_s1,5.8,0.34,3.61,0.83,7.13,0.83,0.81,3.47,0.9
n_grains,g_Amean,ViT_mild_dinoB_orig_mild_s2,5.8,0.34,2.35,0.53,6.79,0.83,0.75,2.39,0.9
n_grains,g_Amean,ViT_orig_vit_s1,5.8,0.34,2.85,1.13,5.49,0.83,0.82,2.79,0.9
n_grains,g_Amean,ViT_orig_vit_s2,5.8,0.34,2.54,1.19,5.24,0.83,0.8,2.08,0.9
n_grains,g_Amean,ViT_orig_vit_s42,5.8,0.34,2.15,1.14,5.03,0.87,0.79,2.09,0.9
aspect,e_ar,FrozenDINO_Ridge,5.89,0.28,3.89,2.52,6.56,0.8,0.8,3.31,0.89
aspect,e_ar,Handcrafted_Ridge,5.89,0.28,6.54,3.8,8.65,0.97,0.98,6.15,0.89
aspect,e_ar,ViT_mild_dinoB_orig_mild_s42,5.89,0.28,3.87,2.26,5.15,0.93,0.87,3.68,0.89
aspect,e_ar,ViT_mild_dinoB_orig_mild_s1,5.89,0.28,3.74,2.3,5.38,0.93,0.91,3.68,0.89
aspect,e_ar,ViT_mild_dinoB_orig_mild_s2,5.89,0.28,3.05,1.61,4.87,0.93,0.87,3.05,0.89
aspect,e_ar,ViT_orig_vit_s1,5.89,0.28,4.13,1.42,5.71,0.93,0.89,4.35,0.89
aspect,e_ar,ViT_orig_vit_s2,5.89,0.28,3.82,1.3,5.54,0.9,0.86,4.02,0.89
aspect,e_ar,ViT_orig_vit_s42,5.89,0.28,4.32,2.1,5.76,0.93,0.88,4.21,0.89
ph_frac,ph_frac,FrozenDINO_Ridge,5.8,0.29,5.03,0.28,9.81,0.77,0.76,4.33,0.93
ph_frac,ph_frac,Handcrafted_Ridge,5.8,0.29,6.09,1.5,10.93,0.8,0.79,6.25,0.93
ph_frac,ph_frac,ViT_mild_dinoB_orig_mild_s42,5.8,0.29,7.14,0.93,14.11,0.77,0.75,5.63,0.93
ph_frac,ph_frac,ViT_mild_dinoB_orig_mild_s1,5.8,0.29,6.6,-0.5,10.93,0.73,0.73,6.01,0.93
ph_frac,ph_frac,ViT_mild_dinoB_orig_mild_s2,5.8,0.29,6.63,0.13,12.19,0.73,0.77,5.82,0.93
ph_frac,ph_frac,ViT_orig_vit_s1,5.8,0.29,6.33,-0.33,13.59,0.73,0.72,5.79,0.93
ph_frac,ph_frac,ViT_orig_vit_s2,5.8,0.29,6.45,-0.26,14.59,0.73,0.73,5.46,0.93
ph_frac,ph_frac,ViT_orig_vit_s42,5.8,0.29,6.01,-0.14,12.46,0.73,0.73,5.84,0.93
pore_frac,po_frac,FrozenDINO_Ridge,-5.78,-0.36,1.74,0.51,2.52,0.83,0.8,2.04,0.9
pore_frac,po_frac,Handcrafted_Ridge,-5.78,-0.36,-1.34,-2.52,-0.9,0.93,0.91,-1.34,0.9
pore_frac,po_frac,ViT_mild_dinoB_orig_mild_s42,-5.78,-0.36,0.7,-0.22,2.01,0.69,0.67,0.96,0.9
pore_frac,po_frac,ViT_mild_dinoB_orig_mild_s1,-5.78,-0.36,1.26,-0.01,2.76,0.72,0.74,1.42,0.9
pore_frac,po_frac,ViT_mild_dinoB_orig_mild_s2,-5.78,-0.36,1.43,-0.02,2.65,0.69,0.7,1.57,0.9
pore_frac,po_frac,ViT_orig_vit_s1,-5.78,-0.36,0.63,-0.35,1.48,0.62,0.66,0.66,0.9
pore_frac,po_frac,ViT_orig_vit_s2,-5.78,-0.36,0.35,-0.58,1.76,0.62,0.59,0.81,0.9
pore_frac,po_frac,ViT_orig_vit_s42,-5.78,-0.36,1.1,0.12,2.01,0.76,0.67,1.19,0.9
bnd_c,b_contrast,FrozenDINO_Ridge,5.55,0.34,1.86,-0.51,5.64,0.7,0.66,3.43,0.86
bnd_c,b_contrast,Handcrafted_Ridge,5.55,0.34,6.7,2.62,9.22,0.97,0.93,5.9,0.86
bnd_c,b_contrast,ViT_mild_dinoB_orig_mild_s42,5.55,0.34,0.51,-0.74,6.57,0.6,0.59,0.45,0.86
bnd_c,b_contrast,ViT_mild_dinoB_orig_mild_s1,5.55,0.34,1.62,-0.84,7.13,0.7,0.64,0.79,0.86
bnd_c,b_contrast,ViT_mild_dinoB_orig_mild_s2,5.55,0.34,1.16,-0.21,7.27,0.63,0.6,0.71,0.86
bnd_c,b_contrast,ViT_orig_vit_s1,5.55,0.34,1.52,-0.46,4.62,0.7,0.67,1.54,0.86
bnd_c,b_contrast,ViT_orig_vit_s2,5.55,0.34,0.77,-0.74,4.7,0.67,0.62,0.79,0.86
bnd_c,b_contrast,ViT_orig_vit_s42,5.55,0.34,1.25,-0.37,6.12,0.67,0.69,1.3,0.86
noise,x_sig,FrozenDINO_Ridge,-0.61,-0.04,1.37,-0.09,2.99,0.73,0.71,1.0,0.85
noise,x_sig,Handcrafted_Ridge,-0.61,-0.04,-0.5,-1.54,-0.08,0.77,0.74,-0.47,0.85
noise,x_sig,ViT_mild_dinoB_orig_mild_s42,-0.61,-0.04,0.98,-1.24,1.89,0.63,0.62,1.11,0.85
noise,x_sig,ViT_mild_dinoB_orig_mild_s1,-0.61,-0.04,0.68,-0.99,1.26,0.67,0.59,0.79,0.85
noise,x_sig,ViT_mild_dinoB_orig_mild_s2,-0.61,-0.04,0.15,-1.18,0.89,0.53,0.54,0.3,0.85
noise,x_sig,ViT_orig_vit_s1,-0.61,-0.04,0.81,-0.49,1.88,0.67,0.65,0.77,0.85
noise,x_sig,ViT_orig_vit_s2,-0.61,-0.04,1.08,-0.82,1.62,0.6,0.62,0.94,0.85
noise,x_sig,ViT_orig_vit_s42,-0.61,-0.04,0.7,-0.57,1.38,0.63,0.6,0.37,0.85


## 측정 특징 (level 0/3/6 평균) vs 실제 5–95%
| 변수 | 특징 | L0/L3/L6 | 실제 |
|---|---|---|---|
| n_grains | g_Amean | 895 / 627 / 661 | 388–874 |
| aspect | e_ar | 1.64 / 1.76 / 2.08 | 1.58–2.41 |
| ph_frac | ph_frac | 0.054 / 0.0791 / 0.0913 | 0.0125–0.176 |
| pore_frac | po_frac | 0.0104 / 0.0294 / 0.0457 | 0–0.0383 |
| bnd_c | b_contrast | 3.43 / 2.96 / 2.87 | 2.59–3.67 |
| noise | x_sig | 0.0124 / 0.0329 / 0.0707 | 0.0186–0.0722 |