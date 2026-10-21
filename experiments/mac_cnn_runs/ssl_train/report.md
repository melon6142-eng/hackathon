# DAKER 규정준수 트랙: train 500장 VICReg 자기지도 사전학습 (합성·외부·test 학습 없음)

- 방법: 외부 fold마다 학습 fold 이미지만으로 DINOv2 ViT-B(원본 lvd142m) 마지막 블록 + projector를 VICReg로 학습. 두 뷰 = 같은 train 이미지의 D4 회전/반전 + 감마·대비·밝기·약한 blur·노이즈. 검증 이미지는 사전학습에 들어가지 않음.
- 평가: 8-TTA frozen 특징(CLS+patch mean) + RidgeCV (Phase 3와 동일). eq3 = BL1+ViT orig3+ViT trans3 균등, eq4 = eq3 + SSL ridge 균등.
- 변형: v1 = 8ep lr3e-5 블록4개 / v2 = 30ep lr5e-5 블록6개 / v3 = 30ep lr1e-4 블록4개

## v1
| seed | DINO ridge | SSL ridge | eq3 | eq4(+SSL) | Δ | SSL 고편향 | SSL 저편향 | eq4 AUPRC | eq4 P@18 | eq4 고편향 |
|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 13.803 | 13.424 | 12.539 | 12.610 | +0.071 | -28.0 | +17.7 | 0.337 | 0.278 | -27.2 |

## v2
| seed | DINO ridge | SSL ridge | eq3 | eq4(+SSL) | Δ | SSL 고편향 | SSL 저편향 | eq4 AUPRC | eq4 P@18 | eq4 고편향 |
|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 13.803 | 13.295 | 12.539 | 12.533 | -0.005 | -28.5 | +17.0 | 0.342 | 0.333 | -27.3 |

## v3
| seed | DINO ridge | SSL ridge | eq3 | eq4(+SSL) | Δ | SSL 고편향 | SSL 저편향 | eq4 AUPRC | eq4 P@18 | eq4 고편향 |
|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 13.803 | 13.369 | 12.539 | 12.535 | -0.003 | -29.8 | +17.3 | 0.346 | 0.389 | -27.6 |

## 판정
seed 42에서 세 변형 모두 기준 미달(eq4 개선 <0.03, SSL 단독 >13.0). 4 seed 확장 없이 중단. 채택 안 함, 테스트 CSV 미생성.

