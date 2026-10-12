# 현재 최고 후보
- **out/submission_blend1_cv12.95.csv** — 엄밀 중첩 CV 12.95 (기존 D0 13.41 / Public 12.52). 예상 Public ≈ 12.0~12.1 (CV-Public 차 ~0.9 경향).
  - 구성: [Ridge on 측정특징 HX(hand+tex+grain+상위14 2차항)] 0.39 + [Ridge on DINOv2-B 252 D4 층9~12 패치토큰 평균+std] 0.72 (전체 train 재적합)
  - 기존 12.52 제출과 예측 상관 0.952
- 재현: runs/make_sub1.py (venv: ./venv)
