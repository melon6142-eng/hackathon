# 1) 진짜 라벨 추가 — 조사 결과

## 찾은 것

- `./open/mlccfx_v1/submission_d0_blend.csv`
- `./open/sample_submission.csv`
- `./open/train`
- `./open/train.csv`

## 판정

- `open/train.csv`: ID,hardness — **500행**, 이미지 `open/train/*.png` **500장**과 1:1 일치.
- `open/test/`: **1000장**, 라벨 없음 (sample_submission만).
- 미사용 train 후보·추가 시편·외부 라벨 CSV: **없음**.
- 기존 label_ssl에서도 고유 경도 500/500, GMM 1성분 → 숨은 레시피 라벨 없음.

## 막힘

**새 측정 라벨이 필요합니다.** 가짜 라벨은 만들지 않습니다.

## 필요 형식 (추가 시)

```csv
ID,hardness
TRAIN_<id>,214.32
EXTRA_batch3_01,241.05
```
- 이미지: `open/train/<ID>.png` (또는 별도 폴더 + 경로 컬럼 `path,hardness`).
- 특히 고경도(>230)·저경도(<170) 꼬리 우선.
- 동일 시편의 다른 FOV도 가능하나, 그룹 ID를 같이 주시면 4번 그룹 검증에 바로 씁니다.