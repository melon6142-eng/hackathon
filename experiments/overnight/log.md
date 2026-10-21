# DAKER overnight log (UTC+8)

## 2026-10-13 02:07 — checkpoint 1 (01:47 fire, run ~02:07)

### Status
- Mac Studio screen: none (dino6 finished 00:59)
- GPU: idle (no dino_ft process)
- Best blend unchanged: BL1 + ViT-B orig/trans 3-seed avg, nCV **12.601**, hi −27.2, lo +17.4

### Finished since last eval (already on box as runs/dino6/)
| tag | solo OOF | +base nCV | Δ | hi | note |
|---|---|---|---|---|---|
| d6_maskcoh | 13.880 | 12.627 | +0.026 | −27.3 | mask+coherence channels |
| d6_maskcoh_rot | 13.964 | 12.621 | +0.021 | −27.4 | +rot TTA |
| d6_maskaux | 14.076 | 12.623 | +0.023 | −27.3 | mask + aux heads 0.3 |
| d6_maskaux_rot | 13.958 | 12.626 | +0.025 | −27.4 | +rot TTA |
| d4_rank (prior) | 13.031 | 12.602 | +0.001 | −27.2 | pairwise rank — null |

**Failure reason:** mask/coherence/aux and rank all stay highly correlated with the existing ViT-B heads; NNLS puts near-zero weight on them. High-hardness underprediction (~−27) unchanged. No candidate CSV (need nCV ≤ 12.55 / ≥0.05 gain).

### Quick diversity recheck (box)
- +u6 / +u8 / expanded trans avg: Δ +0.002~+0.04 (worse). Corrs with base trans ≈0.90–0.94.

### Next (launched this checkpoint)
1. **Seed avg expand:** `dinoB_orig_s3`, `dinoB_trans_s3` (same recipe as s1/s2)
2. **DINO pool representation:** `d7_cls_trans` (CLS-only), `d7_mean_trans` (patch-mean-only) on trans channels — diversity for blend
- screen `dino7`, log `run_dino7.log`
- No submit; did not kill any process

### Brainstorm
- Sent results to 조애기; will use reply at next checkpoint if present.

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

### Note from seed LOO
Dropping seed2 from both orig/trans averages gives nCV **12.586** (Δ−0.015 vs 12.601). Still above 12.55 → no candidate CSV. Separate 6-column NNLS overfits (12.649). All 7 heads underpredict every hi sample — need dissimilar errors, not more of the same.

### ETA
~4 runs × ~37–45 min ≈ finish ~04:30–05:00 UTC+8 (before 05:47 checkpoint; 03:47 may see orig_s3±trans_s3).

## 02:15 — 조애기 3차 (패치 풀링 / 다른 백본 / DLDL) 착수
- 중복 확인: screen dino7(시드 s3 + d7 cls/mean 풀 미세조정)만 실행 중 → 겹치지 않음, 건드리지 않음
- frozen 특징은 맥스튜디오 **CPU**로 추출(GPU는 dino7이 사용 중): screen `featx`, `cnn/featx.py`, 순서 dinoB → clipL → dinoL → maeL → supL
  - 특징: CLS·mean·GeM(p=3, 부호보존) × (중간층, 마지막층) + 토큰 PCA64(train+test 비지도) 상관행렬 상삼각 → 박스에서 PCA256(train+test 비지도)
  - 평가: `overnight/evf.py` (Ridge, 외부 10×5 nested 안에서 Ridge 재학습 + 내부 OOF로 NNLS)
- DLDL: `cnn/dino_ft8.py` (60 bin, σ=15, KL, 기대값 디코드), screen `dino8`이 dino7 종료 대기 후 fold0만 실행

### 1) DINOv2 ViT-B frozen 패치 풀링 (02:27) — 효과 없음
| 특징 | 단독 OOF | hi | lo | +base nCV | Δ | 블렌드 hi | corr(BL1/Bo/Bt) | NNLS w (BL1/Bo/Bt/new) |
|---|---|---|---|---|---|---|---|---|
| dinoB_gem | 13.630 | -30.2 | 17.6 | 12.608 | +0.008 | -27.3 | 0.94/0.85/0.85 | 0.44/0.28/0.28/0.00 |
| dinoB_cov | 15.051 | -34.4 | 22.2 | 12.619 | +0.019 | -27.4 | 0.70/0.71/0.69 | 0.44/0.28/0.28/0.00 |
| dinoB_clsmean | 14.058 | -30.3 | 19.1 | 12.605 | +0.005 | -27.2 | 0.91/0.81/0.80 | 0.44/0.28/0.28/0.00 |
| dinoB_gem+cov | 13.516 | -30.1 | 17.8 | 12.614 | +0.013 | -27.3 | 0.93/0.86/0.86 | 0.44/0.28/0.28/0.00 |
| dinoB_all | 13.600 | -29.8 | 18.0 | 12.609 | +0.008 | -27.3 | 0.94/0.86/0.85 | 0.44/0.28/0.28/0.00 |
- 공분산 풀은 상관 0.70으로 낮지만 단독 15.05로 약해 NNLS 가중 0. GeM(13.63)은 BL1과 0.94 상관 → 중복.

### 2) 다른 계열 백본 frozen + Ridge
| 특징 | 단독 OOF | hi | lo | +base nCV | Δ | 블렌드 hi | corr(BL1/Bo/Bt) | NNLS w (BL1/Bo/Bt/new) |
|---|---|---|---|---|---|---|---|---|
| clipL_clsmean | 14.265 | -31.9 | 21.4 | 12.604 | +0.003 | -27.2 | 0.87/0.80/0.80 | 0.44/0.28/0.28/0.00 |
| clipL_gem | 13.731 | -30.6 | 19.7 | 12.608 | +0.008 | -27.3 | 0.91/0.85/0.85 | 0.44/0.28/0.28/0.00 |
| clipL_cov | 14.967 | -32.6 | 25.8 | 12.612 | +0.012 | -27.4 | 0.69/0.71/0.70 | 0.44/0.28/0.28/0.00 |
| clipL_all | 13.733 | -29.8 | 20.5 | 12.613 | +0.012 | -27.3 | 0.90/0.85/0.85 | 0.44/0.28/0.28/0.00 |
| dinoL_clsmean | 14.412 | -31.3 | 20.9 | 12.601 | +0.000 | -27.2 | 0.85/0.76/0.77 | 0.44/0.28/0.28/0.00 |
| dinoL_gem | 13.669 | -30.3 | 19.0 | 12.601 | +0.001 | -27.2 | 0.92/0.84/0.85 | 0.44/0.28/0.28/0.00 |
| dinoL_cov | 14.792 | -33.2 | 23.4 | 12.601 | +0.001 | -27.6 | 0.68/0.67/0.69 | 0.39/0.28/0.25/0.07 |
| dinoL_all | 13.681 | -28.6 | 19.1 | 12.603 | +0.003 | -27.2 | 0.90/0.84/0.84 | 0.44/0.28/0.28/0.00 |
| maeL_clsmean | 13.897 | -32.6 | 19.9 | 12.601 | +0.000 | -27.2 | 0.92/0.83/0.84 | 0.44/0.28/0.28/0.00 |
| maeL_gem | 13.960 | -33.2 | 20.2 | 12.601 | +0.000 | -27.2 | 0.92/0.83/0.85 | 0.44/0.28/0.28/0.00 |
| maeL_cov | 16.052 | -39.1 | 25.1 | 12.603 | +0.002 | -27.2 | 0.65/0.61/0.63 | 0.44/0.28/0.28/0.00 |
| maeL_all | 14.000 | -33.3 | 20.5 | 12.601 | +0.000 | -27.2 | 0.92/0.83/0.84 | 0.44/0.28/0.28/0.00 |
| supL_clsmean | 13.732 | -31.1 | 20.9 | 12.605 | +0.005 | -27.2 | 0.90/0.85/0.85 | 0.44/0.28/0.28/0.00 |
| supL_gem | 13.682 | -31.4 | 19.8 | 12.603 | +0.002 | -27.2 | 0.92/0.86/0.86 | 0.44/0.28/0.28/0.00 |
| supL_cov | 15.077 | -35.0 | 24.9 | 12.606 | +0.006 | -27.3 | 0.69/0.74/0.74 | 0.44/0.28/0.28/0.00 |
| supL_all | 13.596 | -30.4 | 20.2 | 12.603 | +0.003 | -27.2 | 0.91/0.87/0.87 | 0.44/0.28/0.28/0.00 |
| multi5_gem | 13.285 | -28.7 | 18.1 | 12.605 | +0.005 | -27.2 | 0.94/0.88/0.88 | 0.44/0.28/0.28/0.00 |
| multi5_gem+cov | 13.311 | -29.0 | 18.5 | 12.604 | +0.004 | -27.2 | 0.94/0.88/0.88 | 0.44/0.28/0.28/0.00 |
- 결론: 5개 백본(dinoB/clipL/dinoL/maeL/supL) × 풀링 4종 + 5백본 결합까지 모두 NNLS 가중 ≈0, Δ ≥ 0. 공분산 풀만 상관 0.65~0.74로 낮지만 단독 14.8~16.1로 약함. 후보 없음.

## 2026-10-13 ~04:05 — checkpoint (dino7 done, GPU idle → launch)

### Pulled from Mac Studio → overnight/
- `dinoB_orig_s3`, `dinoB_trans_s3`, `d7_cls_trans`, `d7_mean_trans` (oof.npy+test.npy each)
- Base seeds already on box (dino_ft / dino_ft2 s1/s2)

### Solo OOF (confirm)
| tag | OOF | hi | lo |
|---|---|---|---|
| dinoB_orig_s3 | 13.112 | −24.9 | 16.8 |
| dinoB_trans_s3 | 13.029 | −25.7 | 17.3 |
| d7_cls_trans | 13.158 | −25.6 | 16.9 |
| d7_mean_trans | 13.069 | −27.1 | 16.7 |

### Nested CV (10×5, NNLS+intercept) vs base **12.601** (hi −27.2)
| 구성 | 단독 | nCV | Δ | hi | note |
|---|---|---|---|---|---|
| +dinoB_orig_s3 | 13.112 | 12.613 | +0.013 | −27.2 | w_new≈0, corr Bo=0.979 |
| +dinoB_trans_s3 | 13.029 | 12.610 | +0.010 | −27.2 | w≈0.19 on s3 but Δ+ |
| +d7_cls_trans | 13.158 | 12.610 | +0.009 | −27.2 | w_new=0, corr Bt=0.978 |
| +d7_mean_trans | 13.069 | 12.620 | +0.019 | −27.3 | w≈0.04 |
| expand orig4+trans3 | - | 12.603 | +0.003 | −27.2 | |
| expand orig3+trans4 | - | 12.594 | −0.006 | −27.2 | best expand |
| expand orig4+trans4 | - | 12.598 | −0.003 | −27.1 | |
| replace Bt←cls | 13.158 | 12.633 | +0.032 | −27.1 | worse |
| replace Bt←mean | 13.069 | 12.614 | +0.013 | −27.3 | worse |
| orig4+trans4+cls+mean | - | 12.619 | +0.018 | −27.2 | |
| drop_s2 + s3 avg | - | **12.586** | −0.014 | −27.0 | still >12.55 |
| BL1+8seeds+cls+mean cols | - | 12.660 | +0.059 | −27.4 | overfit |
| LOO4 drop idx2 (seed2) | - | 12.586 | −0.014 | −27.0 | same as prior LOO |

**Failure:** all new heads corr ≥0.97 with existing ViT-B; shared hi underprediction (~−25~−27) unbroken. NNLS≈0 or tiny. **No candidate CSV** (need ≤12.55).

### Frozen-backbone (prior, same night)
dinoB/clipL/dinoL/maeL/supL × pools + multi5: all NNLS w≈0, Δ≥0. Logged above.

### Brainstorm → 조애기
- SendToAgent **unavailable to executor subagent** (cursor MCP not in this context). Parent must relay Korean checkpoint + ask 2–3 next ideas (draft in executor report). Avoid: frozen multi-backbone Ridge, CLS/mean pool FT, seed avg, mask, brightness bins, residual/rank, DLDL(running).

### Launched (screen `dino8`, GPU was idle)
1. **Must:** finish DLDL `d8_dldl_orig` folds 1–4 (fold0 already; script skips existing). `run_dino8b.sh` → `DINO8DONE`
2. **Chained:** `d9_cls_orig` = dino_ft7 `--chan orig --pool cls` (cross-channel diversity vs d7_cls_trans). Log `run_dino9.log` → `DINO9DONE`
- ETA: ~4×4.2min DLDL ≈17min + ~21min d9 ≈ **~04:45–05:00 CST** finish
- No kill, no submit, no GitHub data push

### Best nCV still **12.601** (drop_s2 trick 12.586 not enough for candidate)

## 04:08 — dino10 대기열 추가 (맥스튜디오, screen dino10)
- 코드: cnn/dino_ft10.py (dino_ft5 기반, --mode hiexp|canon), run_dino10.sh, 로그 run_dino10.log
- dino8 screen + dino_ft*.py + run_dino8* 프로세스가 모두 사라지면 자동 시작 (d8_dldl_orig → d9_cls_orig 이후)
- 공통: ViT-B dinov2 252, unfreeze 4, 5fold rs42, seed 42, orig 채널, bs16 lr5e-5
- d10_hiexp: 폴드 train 내 상위 40% 경도(≥train 60분위)만 학습, val 전체/test 전체 예측, 8-way TTA
- d10_canon: 구조텐서 주방향을 가로로 회전(BORDER_REFLECT_101), train/test 동일 정렬, aug·TTA는 수평/수직 뒤집기 4-way
- 검증: py_compile OK, 더미 줄무늬(0/30/60/90/135°) 회전 후 잔여각 0~5.5° 확인
- 출력: cnn/runs/d10_{hiexp,canon}/{oof,test}.npy

### 다음 점검에서 d10_hiexp 평가법
1. 전 구간: 기존 최고 블렌드(BL1+B_orig_avg+B_trans_avg, 12.601)에 d10_hiexp oof를 NNLS 후보로 추가, nested 반복 CV RMSE·hi편향 비교.
2. 상위 구간 게이트 혼합: base 블렌드 예측 p의 상위 q(예 10/20/30/40%) 구간에서만 p'=(1-w)p+w·hiexp, w∈{0..1}; (q,w)는 inner CV에서 선택, outer로 평가(nested). 개선 ≥0.05일 때만 후보 CSV.
- 주의: hiexp는 고경도만 학습해 저구간 예측이 높게 치우침 → 전 구간 단순 추가는 NNLS 가중≈0 예상, 게이트 혼합이 본 평가.
- d10_canon: 단독 OOF/hi편향, base와 corr, 블렌드 추가 nCV.

## 2026-10-13 ~05:55 — checkpoint (d8/d9/d10 done)

### Pulled from Mac Studio → overnight/
- `d8_dldl_orig` (oof/test **assembled from fold0..4.npz** on Studio: oof[va]=pv, test=mean(pt); convention verified exact on d9: max|diff|=0), `d9_cls_orig`, `d10_hiexp`, `d10_canon`
- 평가 스크립트: `ev_d8d10.py` → `ev_d8d10.md`, `ev_d8d10.log`, `ev_d8d10_extra.log`

### Solo OOF
| tag | OOF | hi(>230) | lo(<170) | corr BL1/Bo/Bt |
|---|---|---|---|---|
| d8_dldl_orig | 13.063 | −24.5 | 17.9 | 0.857/0.967/0.933 |
| d9_cls_orig | 13.276 | −25.2 | 17.4 | 0.858/0.976/0.940 |
| d10_hiexp | 20.884 | −22.2 | 40.0 | 0.642/0.655/0.648 |
| d10_canon | 13.463 | −26.5 | 17.6 | 0.843/0.890/0.872 |

### Nested CV (10×5, NNLS+intercept) vs base **12.601** (hi −27.2)
| 구성 | nCV | Δ | hi | NNLS w (BL1,Bo,Bt,new..) |
|---|---|---|---|---|
| +d8_dldl_orig | 12.605 | +0.005 | −26.9 | .434/.100/.244/**.223** |
| +d9_cls_orig | 12.613 | +0.012 | −27.2 | new 0 |
| +d10_hiexp (simple add) | 12.601 | +0.000 | −27.2 | new 0 (예상대로) |
| +d10_canon | 12.627 | +0.026 | −27.2 | .402/.221/.256/.121 |
| +d8+d9 | 12.610 | +0.010 | −26.9 | d9 0 |
| +d8+canon / +d9+canon / +d8+d9+canon / +all4 | 12.628 / 12.637 / 12.631 / 12.631 | +0.03 | −26.9~−27.3 | |
| Borig avg ← +d8 (4 members) | 12.590 | −0.011 | −27.1 | |
| Borig avg ← +d9 (4) | 12.606 | +0.005 | −27.2 | |
| Borig avg ← +canon (4) | 12.587 | −0.014 | −27.2 | |
| Borig avg ← +d8+d9 (5) | 12.597 | −0.004 | −27.1 | |
| drop_s2+s3 (ref) | 12.586 | −0.014 | −27.0 | |
| drop_s2+s3 +d8 / +d9 col | 12.593 / 12.593 | −0.008 | | |
| drop_s2+s3, Borig ← +canon (4) | 12.572 | −0.029 | −27.0 | |
| **drop_s2+s3, Borig ← +canon+d8 (5)** | **12.563** | −0.038 | −26.8 | .396/.364/.240 |
| base, Borig ← +canon+d8 (5) | 12.576 | −0.025 | −27.0 | |
| **d10_hiexp GATED** (top-q of p, p'=(1−w)p+w·hiexp, (q,w) inner CV, outer nested) | 12.621 | +0.021 | −26.8 | picks 거의 q=0.1, w 0~0.4 불안정 |
- gated 비중첩(낙관) 진단: q=0.1,w=0.2에서 in-sample 12.523→12.512 (−0.011)뿐, q≥0.2는 w=0. → nested에선 악화.

### 실패 원인
- d8(DLDL)/d9(CLS orig): ViT-B 기존 헤드와 corr 0.97~0.98, hi 편향 동일(−24~−25). 모델 출력만 바뀌고 표현은 그대로.
- **d10_hiexp 핵심 관찰**: 상위 40%(≥~205)만 학습했는데도 y>230에서 여전히 **−22.2** 과소예측, 200–230 구간 편향 0. → 고경도 꼬리는 '학습 분포의 평균회귀' 문제가 아니라, >230 이미지가 DINO 표현상 ~210대 이미지와 구분되지 않음(정보 부족 or 라벨 노이즈). 게이트 혼합이 안 되는 이유.
- d10_canon: corr 0.89로 가장 다양하지만 단독 13.46으로 약해 열 추가는 악화; 평균 멤버로만 소폭 이득.
- 최고 12.563은 drop_s2(사후 선택) + 평균멤버 확장 조합이라 선택편향 있음, 12.55 미달 → **후보 CSV 미작성**.

### Launched (screen `dino11`, 05:59 CST, GPU idle였음)
- 코드 `cnn/dino_ft11.py` (dino_ft10/ft7 복제, py_compile + CPU dry-run 양 모드 OK), `run_dino11.sh`, 로그 `run_dino11.log` → `DINO11DONE`
1. `d11_multilayer`: ViT-B dinov2 252 orig, unfreeze4, 헤드 입력 = [CLS_L12, LN된 patch 토큰 mean/std @ block 6,9,12] (표현 변경: 중간층 텍스처 통계)
2. `d11_cropaug`: 스케일 보존 랜덤 크롭(원본 171/199/228px → 168/196/224, dynamic_img_size, p=0.75), test는 전체 252 8-way TTA (입자 크기 스케일 유지한 국소 뷰 다양성)
- ETA: 각 ~21–25분 → multilayer ~06:22, cropaug ~06:47 CST 전후
- 다음 점검: 두 run pull → 같은 ev 패턴(열 추가 + Borig 평균 멤버) 평가. No kill, no submit, no GitHub data push.

### Best nCV this round: 12.563 (drop_s2+s3, Borig←+canon+d8) — 후보 기준(≤12.55) 미달

## 2026-10-13 06:05 geo_tail (박스)
단조 기하측정 상위밴드 재정렬 12.770(악화), 결정립 면적 Ridge 블렌드 12.601(무변화). y>230 상관은 dark_frac +0.63 등 있으나 예측 상위20% 내 |ρ|≤0.13. 후보 없음. 상세 runs/geo_tail/report.md

## 2026-10-13 07:55 — 아침 보고 체크포인트 (dino11 평가)
- 맥스튜디오: dino11 06:38 종료(DINO11DONE), screen 없음, GPU 유휴. 프로세스 건드리지 않음.
- Pulled → overnight/d11_multilayer, overnight/d11_cropaug (oof/test). 평가 `ev_d11.py` → `ev_d11.md`, `ev_d11.log`

| tag | 단독 OOF | hi | lo | corr BL1/Bo/Bt |
|---|---|---|---|---|
| d11_multilayer | 12.994 | −23.5 | 16.9 | 0.871/0.968/0.939 |
| d11_cropaug | 13.589 | −22.5 | 18.1 | 0.855/0.956/0.925 |

| 구성 | nCV | Δ | hi |
|---|---|---|---|
| +d11_multilayer (열) | 12.587 | −0.013 | −26.6 |
| +d11_cropaug (열) | 12.607 | +0.006 | −27.2 |
| Borig avg ← +multilayer | 12.581 | −0.020 | −27.0 |
| Borig avg ← +multilayer+d8+canon (사후선택 없음 최고) | **12.563** | −0.037 | −26.9 |
| [선택편향] drop_s2+s3, Borig ← +canon+d8+multilayer | **12.552** | −0.049 | −26.7 |

- d11_multilayer: ViT-B 단독 첫 13 이하(12.994), 중간층 텍스처 통계가 소폭 다양성 제공. cropaug는 단독 약하고 가중 0.
- 최고 12.552도 Δ0.049 < 0.05 + drop_s2 사후선택 → **후보 CSV 미작성**. 제출 안 함.
