# hackathon

## daker-hardness — DAKER 합성 미세조직 경도 예측

256×256 흑백 합성 미세조직 이미지로 경도(hardness)를 예측하는 회귀 대회입니다. 평가 지표는 RMSE이고, Public은 test의 30%(300장), Private는 나머지 70%로 채점합니다.

> **데이터는 포함하지 않았습니다.** 대회 규칙상 원본 데이터(train/test 이미지, train.csv, zip)는 재배포하지 않습니다. 재현하려면 대회 페이지에서 받아 `open/`에 두세요.

### 최고 모델: BL1
- 5-fold 반복 CV **12.95**, Public **11.77** (이전 제출 12.52)
- 구성: 측정 특징 Ridge(밝기, 어두운 영역, GLCM, LBP, FFT, Gabor, 결정립, 2차 조합항 등 약 170개) + DINOv2-B(252px, 층 9~12 patch mean+std, D4 8방향 평균) Ridge 블렌드
- 제출 파일: `daker-hardness/out/submission_blend1_cv12.95.csv`

### 실험 결과 (CV RMSE)
| 실험 | 결과 | 비고 |
|---|---|---|
| BL1 (기준) | 12.95~13.03 | 최고 |
| + 밝기 단계별 배치·형상 | 13.00 | 효과 없음 (단독 13.95) |
| + 패턴 거리 비교 | 13.05 | 효과 없음 (단독 14.13) |
| 양 끝 보정(예측 늘리기) | 13.09 | 분포 강제 맞춤은 14.5로 악화 |
| 양상별 역산(군집별 역회귀) | 단독 13.6~14.2 | 순수 역산은 26~28 |
| ConvNeXt 전체 이미지 미세조정 (Mac Studio) | 14.03 | 블렌드 12.97~13.00 |
| ConvNeXt 패치 학습 (Mac Studio) | 13.99 | 고경도 −28~−32 편향 동일 |
| 생성 규칙 역추적 + 경계 지표 | 12.95~13.01 | 경계 단독 14.61 |

생성 규칙 역추적 결과: 한 방향으로 늘린 Voronoi 구조에 2상 셀, 타원 기공, 노이즈·블러를 얹은 형태. 결정립이 크고 많이 늘어날수록, 어두운 영역이 넓게 이어질수록 경도가 높음. 2상 비율 15~45%에서 경도 최대. 노이즈·블러는 경도와 무관.

### CV–Public 갭 분석
CV 13.03 대비 Public 11.77인 주된 이유는 train/test 분포 차이가 아니라 **Public 30% 표본 분산**입니다. 오차가 경도 양 끝에 몰려 있어서, 300장 표본에 극단값이 적게 들어가면 점수가 크게 좋아집니다. 시뮬레이션에서 11.77 이하가 나올 확률은 약 2.5%였습니다. Private는 12.8~13.2 정도로 예상되므로, 모델 선택은 반복 CV를 기준으로 합니다.

### 구조
- `daker-hardness/runs/` 실험 스크립트, 로그, 그림(`viz/`, `gen/`)
- `daker-hardness/studio_cnn/` Mac Studio CNN 학습 코드와 로그
- `daker-hardness/journal.md`, `out/BEST.md` 실험 일지와 최고 기록
- `daker-hardness/gallery.html` 학습 이미지 경도 갤러리 (이미지는 `open/train/`이 있어야 표시됨)

### 재현 방법
1. 대회 데이터를 받아 `open/`(train/, test/, train.csv)에 풉니다.
2. `python -m venv venv && pip install numpy scipy scikit-image scikit-learn opencv-python torch timm pillow lightgbm`
3. `runs/` 아래 특징 추출 스크립트(측정 특징, `runs/dino/feat_dino*.py`)를 실행해 특징을 만든 뒤, 블렌드 스크립트로 CV와 제출 파일을 만듭니다. 세부 순서는 `journal.md`를 참고하세요.
