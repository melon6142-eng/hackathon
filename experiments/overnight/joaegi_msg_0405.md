[DAKER overnight checkpoint ~04:05 Asia/Shanghai 2026-10-13]

dino7 완료 결과 공유합니다. 공개 코드/외부 힌트 없이, 아이디어만 부탁드립니다.

## 새 결과 (ViT-B FT)
| tag | 단독 OOF | hi bias | +base nCV | Δ vs 12.601 |
|---|---|---|---|---|
| dinoB_orig_s3 | 13.112 | −24.9 | 12.613 | +0.013 |
| dinoB_trans_s3 | 13.029 | −25.7 | 12.610 | +0.010 |
| d7_cls_trans (CLS-only) | 13.158 | −25.6 | 12.610 | +0.009 |
| d7_mean_trans (mean-only) | 13.069 | −27.1 | 12.620 | +0.019 |
| seed expand orig3+trans4 | - | −27.2 | 12.594 | −0.006 |
| drop seed2 + s3 | - | −27.0 | 12.586 | −0.014 |

모든 신규 헤드가 기존 ViT-B와 corr≥0.97, NNLS 가중≈0. 고경도(>230) 과소예측 ≈−25~−27 공유. 후보 CSV 없음(목표 nCV≤12.55).

## Frozen 백본 Ridge (같은 밤, 이미 실패)
dinoB/clipL/dinoL/maeL/supL × GeM/cov/clsmean/all + multi5 결합 → 전부 NNLS w≈0, Δ≥0.

## 현재 best
BL1 + ViT-B orig/trans 3-seed avg → **nCV 12.601**, hi≈−27. Public 최고 참고만: 우리 쪽은 여전히 같은 꼬리 오차.

## 지금 돌리는 것
- DLDL (d8_dldl_orig) folds 1–4 이어서
- d9_cls_orig (orig 채널 + CLS-only) 체인

## 부탁
위와 **겹치지 않는** 다음 아이디어 2–3개만 주세요. 이미 했거나 돌리는 것(피해주세요):
- frozen multi-backbone Ridge
- CLS/mean pool FT
- seed avg expand
- mask channels
- brightness bins
- residual/rank correction
- DLDL (이미 런칭)

우선순위: **DINO representation 변화**, **diversity ensemble**, **공유 고경도 과소예측을 깨는** 방향. 공개 대회 코드/외부 힌트 금지 — 아이디어만.
