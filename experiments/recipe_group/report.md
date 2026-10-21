# 4) 레시피/공정 메타·그룹 구조

## 조사

- train ID 접두어: [('TRAIN', 500)]
- 고유 hardness: 500/500 (전부 다른 연속값)
- 파일명: `TRAIN_<id>.png` 형태만. 배치·레시피·시편 코드 없음.
- CSV 컬럼: `ID,hardness`만. 메타 컬럼 없음.
- REVIEW_BRIEF / label_ssl: 레시피·이산 묶음 가설 **이미 기각** (GMM k=1, 클러스터 R²≤0.21).

## 판정

**그룹 정보 없음.** GroupKFold·그룹 가중 재평가는 실행하지 않음 (가짜 구조 금지: 파일명 클러스터링 안 함).

## 대안 (데이터 확보 시)

1. 측정 시 `batch_id` / `specimen_id` / `recipe_id` 컬럼 추가.
2. 같은 시편 다중 FOV면 specimen 단위 GroupKFold로 누수 차단.
3. 그룹이 생기면 그때 GroupKFold nCV와 표본 유효 N 재추정.

## 채택

채택 불가 (정보 부재). Δ 없음. 고경도 편향 변화 없음.