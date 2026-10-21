## Box light: seed leave-one-out & per-seed columns (02:10)
base (avg3) nCV 12.601 hi -27.2
- drop seed-idx 0 (42): nCV 12.613 Δ+0.012
- drop seed-idx 1 (1): nCV 12.626 Δ+0.025
- drop seed-idx 2 (2): nCV 12.586 Δ-0.015
- BL1+6seeds as cols: nCV 12.649 Δ+0.049
- full NNLS w: BL1=0.437, dinoB_orig=0.081, dinoB_orig_s1=0.224, dinoB_orig_s2=0.000, dinoB_trans=0.137, dinoB_trans_s1=0.120, dinoB_trans_s2=0.000, c=0.000

### OOF correlations (max off-diag among ViT-B)
- highest ViT-B pair corr: dinoB_orig_s1–dinoB_orig_s2 = 0.9725
- lowest ViT-B pair corr: dinoB_orig–dinoB_trans_s1 = 0.9216
- corr(BL1,BL1)=1.0000
- corr(BL1,dinoB_orig)=0.8658
- corr(BL1,dinoB_orig_s1)=0.8686
- corr(BL1,dinoB_orig_s2)=0.8705
- corr(BL1,dinoB_trans)=0.8650
- corr(BL1,dinoB_trans_s1)=0.8637
- corr(BL1,dinoB_trans_s2)=0.8660

- fraction of hi samples underpredicted by ALL 7 heads: 1.00
- mean hi residual BL1=-28.3, avg-orig=-26.0, avg-trans=-26.4
