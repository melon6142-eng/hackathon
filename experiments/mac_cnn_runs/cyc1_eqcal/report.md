# cyc1 (2026-10-15, CPU only — GPU는 ssl_train/p3_ablate가 사용 중)
1. equalmean 독립 fold 검증 (BL1+orig avg+trans avg 균등평균, 10x5 nCV)
   k=3: rs7 12.795 / rs123 12.538 / rs2026 12.796 → 12.710±0.149 (NNLS 12.796, 3/3 fold 개선 -0.08~-0.10) → 재현됨
   k=6: 12.704±0.203 (k=3 대비 이득 없음)
2. 후보정(선형 stretch, isotonic 50%): eq보다 3/3 fold 악화 → 기각. hi bias -27 그대로.
3. GBM(HistGB, hand feats 213d) 4번째 멤버: solo 14.4~14.8, 잔차상관 0.87, eq4 3/3 악화(+0.11~0.14) → 기각.
