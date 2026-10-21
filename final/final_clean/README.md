# DAKER final_clean — 규정준수 최종 파이프라인 (2026-10-16)
위치: Mac Studio `~/AI/daker/final_clean/` (box 사본 `./final_clean/`). 기존 파일은 읽기만 함, 새 폴더에만 씀. 제출하지 않음.

## 실행
`cd ~/AI/daker/final_clean && tmux new -d -s final_clean "./run_all.sh all"`  (main = fold seed 42 + 최종 CSV, extra = fold seed 7/123/2026 OOF)
진행: `progress.txt`, 로그 `logs/`. 환경: `~/AI/daker/venv` (torch 2.14.1, timm 1.0.30, sklearn 1.9.1, numpy 2.5.4, cv2 5.0.0) + `pylib/`(scikit-image 0.26.0, PyWavelets; --target 설치, 기존 venv 미변경). HF_HUB_OFFLINE=1.

| 단계 | 스크립트 | 출력 |
|---|---|---|
| 0 가중치 해시 | run_all.sh | logs/env.txt, weights_dinov2_sha256.json |
| 1 DINO 특징 | src/feat_dino.py | feat/dino_{train,test}.npy |
| 2 손 특징(hand/tex/grain/raw) | src/feat_hand.py | feat/hand_{train,test}.csv |
| 3 BL1 | src/bl1.py | bl1/oof_fs*.npy, bl1/test.npy |
| 4 ViT 6개 (orig/trans × seed 42/1/2) | src/vit.py | vit/<tag>/fold*.pt(+sha256 in log.txt), fold*_norm.json, oof.npy, test.npy |
| 5 블렌드+CSV | src/blend.py | submission_final_clean.csv, results.md/json |

## 가중치 (sha256)
- Meta torch.hub `dinov2_vitb14_pretrain.pth` 0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73 (Apache-2.0, 2023) — BL1
- timm `vit_base_patch14_dinov2.lvd142m` rev 4685c99d `model.safetensors` 55cbb5d887b336d430e649c277b85a1429e724871f9d02ac16203235886d8c7b (HF 원본 LFS 해시와 일치, 2023-05-09) — ViT
- 합성 이미지 / Phase 2·3 가중치 사용 안 함. test 이미지는 추론에만 사용.

## 누수 감사 (무엇을 바꿨나)
| # | 위치 | 옛 동작 | 판정 | 새 동작 |
|---|---|---|---|---|
| 1 | ViT `build()` 채널 정규화 | train/test 각각 자기 집합의 mean/std로 정규화 → **test 통계 사용** | 위반 소지 | fold-train 이미지로 mean/std 적합 → 같은 값을 val·test에 적용 (fold*_norm.json 저장) |
| 2 | ViT trans 채널 p1/p99 창 | train 500장 전체(val fold 포함) 픽셀 | val 누수(CV) | fold-train 픽셀로만 계산 |
| 3 | ViT 라벨 표준화 mu/sd | train 500장 전체 라벨 | val 라벨 누수(미미) | fold-train 라벨 |
| 4 | ViT seed | seed 42 실행이 원본과 비트 불일치(시드 미고정 의심) | 재현성 | `torch/np.manual_seed(seed)` + fold별 seed+1000·f 명시, seed 42/1/2 |
| 5 | BL1 poly2 top-14 선택 (`runs/feats.py HX`) | 전체 train 라벨 상관으로 열 선택·표준화 후 CV → OOF 낙관 편향 | val 라벨 누수 | `PolyTop`을 각 학습 집합(outer fold-train, inner fold-train, 최종 full train) 안에서 적합. 단 Ridge alpha 선택용 3단 내부 CV에서는 상위 집합 선택을 재사용 |
| 6 | BL1 StandardScaler/Ridge | pipeline 안에서 학습 집합에만 적합 | 문제 없음 | 유지 |
| 7 | BL1 DINO 특징 | 이미지별 /255 + 고정 ImageNet mean/std | 문제 없음 | 동일 재추출 (옛 train 특징과 max diff 1.6e-4) |
| 8 | grain / raw 특징 | 이미지별 (median/std/percentile 모두 이미지 내부) | 문제 없음 | 동일 코드 |
| 9 | hand / tex 특징 (`open/combo_v1/feat`) | **생성 코드가 box·Mac 어디에도 없음** → 감사 불가 | 출처 불명 | 같은 열 의미로 이미지별 재구현(dark/pore/연결성분/coherence/상관길이/sharp/z-분위수; GLCM 5×5, LBP 8/16/24, Gabor 4, FFT 16) — 값은 옛 것과 근사일 뿐 동일하지 않음 |
| 10 | 블렌드 | OOF에 NNLS(+절편) | 문제 없음 | 동일 방법, 새 OOF로 재적합 |
| 11 | CSV 순서 | sample_submission ID 순서 | – | assert로 ID·열 순서 확인 |
