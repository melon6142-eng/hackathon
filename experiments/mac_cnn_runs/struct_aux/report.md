# struct_aux 결과 (fold rs42, ViT-B orig 설정, aux 0.3 vs 대조 aux 0)

```
                    RMSE  hi_bias  lo_bias  hi_RMSE  mid_RMSE  AUPRC    P18
BL1               12.949  -28.290   17.701   29.658    10.876  0.321  0.389
ViT orig 3s (기존)  12.948  -25.968   17.177   27.813    10.951  0.298  0.222
ViT orig s        13.084  -25.565   17.172   27.350    11.101  0.322  0.333
ViT orig s1       12.972  -26.114   17.203   28.258    10.961  0.296  0.278
ViT orig s2       13.161  -26.225   17.157   28.086    11.220  0.270  0.222
eq3               12.539  -26.885   17.224   28.388    10.475  0.335  0.278
aux0 s42          12.974  -24.438   17.169   26.160    11.122  0.312  0.278
aux0 _s1          13.031  -25.494   17.003   27.351    11.096  0.321  0.333
aux0 _s2          13.159  -25.193   17.660   27.042    11.310  0.340  0.333
aux0 avg3s        12.926  -25.042   17.277   26.783    11.027  0.317  0.278
eq4 (eq3+aux0)    12.568  -26.424   17.237   27.962    10.532  0.341  0.278
eq3 orig→aux0     12.527  -26.577   17.257   28.060    10.494  0.343  0.278
aux0.3 s42        13.176  -23.792   17.378   25.822    11.431  0.375  0.333
aux0.3 _s1        13.112  -25.049   17.019   27.045    11.329  0.326  0.333
aux0.3 _s2        13.199  -25.783   17.158   27.430    11.355  0.304  0.222
aux0.3 avg3s      13.061  -24.875   17.185   26.699    11.254  0.328  0.278
eq4 (eq3+aux0.3)  12.589  -26.383   17.214   27.943    10.574  0.335  0.278
eq3 orig→aux0.3   12.562  -26.521   17.226   28.034    10.558  0.340  0.278
eq4 aux0 per-image SSE delta: total 365, hi -432, lo 265, mid 532; improved 0.49
eq4 aux0.3 per-image SSE delta: total 631, hi -451, lo 152, mid 931; improved 0.48
```

aux 타깃: grain_size_logDT grain_size_spread gb_density orient_coherence dark_frac dark_largest_conn dark_logncomp pore_logcount pore_logsize fftE_lt4 fftE_4_16 fftE_16_64 fftE_gt64 uneven_band16_64 uneven_edge
표준화: fold별 train idx 통계만. test·합성·Phase2/3 가중치 미사용.
