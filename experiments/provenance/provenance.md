# DAKER 최종 제출 출처 감사 (2026-10-15 22:40 KST)

대상 제출 파일: `runs/dino_ft2/cand_BL1_Bavgorig-trans_cv12.61.csv` (sha256 816abe2f…d594), Public 11.33 후보

## 구성요소 표
| 구성 | 사전학습 가중치 출처 | 체크포인트 sha256 | 라이선스 | 공개일 | 재현 결과 |
|---|---|---|---|---|---|
| BL1 DINO 특징 `runs/dino/vitb14_252_{train,test}.npy` | **torch.hub `facebookresearch/dinov2` → `dinov2_vitb14`** (timm 아님). 다운로드 URL https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth | 0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73 (오늘 Mac에 새로 내려받은 파일; 원래 추출 때 쓴 box 캐시는 사라져 원본 해시는 확인 불가) | Apache-2.0 (repo README "DINOv2 code and model weights are released under the Apache License 2.0"; LICENSE 2023-08-31 커밋 81b2b64로 변경) | repo 최초 커밋 2023-04-17 | 20 train + 20 test 재추출: max abs diff 4.8e-5 / 3.6e-5, 코사인 ≥0.9999998 → **재현됨** |
| ViT-B fine-tune 6개 (dinoB_orig/trans, seed 0/1/2) | timm `vit_base_patch14_dinov2.lvd142m`, HF 리비전 4685c99dabffe5affac90bd99dbffd25801ae58d | model.safetensors 55cbb5d887b336d430e649c277b85a1429e724871f9d02ac16203235886d8c7b = HF API LFS sha256 **일치** | HF 카드 태그 apache-2.0, config.json에는 `cc-by-nc-4.0` | HF 최초 커밋 2023-05-09 (마지막 2025-01-21) | 가중치 미저장(fold*.npz에는 예측만) → 테스트 예측 재생성 불가, 재학습 필요 |
| BL1 예측 `sub_bl1.csv` | 위 DINO 특징 + 손 특징 → Ridge 계열 | 08c792d2…a9912 (Mac `cnn/sub_bl1.csv`, box 사본 2개 모두 동일) | – | – | 저장된 파일만 확인, 재학습 재현은 안 함 |
| 최종 블렌드 | sub_bl1 + ViT orig 3seed평균 + trans 3seed평균 | – | – | – | 정확 가중치 0.4398/0.2831/0.2773로 잔차 3e-13 → **완전 재현**. 반올림 가중치 0.44/0.283/0.277이면 최대 0.052 차이 |

## 전처리 (BL1 특징, `runs/dino/feat_dino.py`)
흑백 PNG → cv2 INTER_AREA 252×252 → /255 → 3채널 복제 → ImageNet mean/std → get_intermediate_layers(n=4, cls 포함) → 층마다 [CLS, patch 평균, 표준편차, GeM(p=3)] → D4 8방향 TTA 평균. 768×4×4 = 12288차원. train 500×12288, test 1000×12288.

## 남은 공백
1. BL1 블렌드 기반 파일 `out/submission_blend1_cv12.95.csv`는 box·Mac 어디에도 없음. eval3.py가 이걸 BL1로 썼지만, sub_bl1.csv로 대체해도 블렌드가 잔차 0으로 재현되므로 내용은 같다고 봄.
2. 원래 추출 때 쓴 dinov2 체크포인트의 해시와 repo 커밋은 남아 있지 않음. 같은 URL의 현재 파일로 재현됐으니 같은 가중치로 판단.
3. timm 쪽 라이선스 표기가 갈림(카드 apache-2.0, config cc-by-nc-4.0). 대회 규정은 둘 다 허용하지만 증빙에 둘 다 적어 둘 것.
4. 파일 mtime(box 10/15 17:05)은 box 복원 시점이라 생성 시각 증빙이 안 됨. Mac 사본 lcurve/.../vitb14_252_train.npy는 10/13 14:14이고, 해시가 box와 같음(7a22492c…ac13). Mac sub_bl1.csv는 10/12 11:39, ViT 실행은 10/12 13:59부터.
5. ViT fine-tune 체크포인트가 없어서 테스트 예측을 다시 만들려면 6개 모델을 재학습해야 함(시드 비결정성 때문에 비트 단위 일치는 보장 안 됨).
