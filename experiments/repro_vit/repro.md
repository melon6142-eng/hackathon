# ViT 6개 재현 재학습 (진행 중, 시작 2026-10-15 22:42 KST)
- 원본 스크립트: cnn/dino_ft.py (sha256 66da852a…9c16, mtime 10/12 15:06; dinoB_orig/trans 첫 실행(13:59/14:10)은 seed 인자 추가 전 → 기본값 42로 간주)
- 변경점: 출력경로 runs/repro_vit/<tag>, fold별 state_dict 저장+sha256 로그. 그 외 동일 (diff는 README 참조)
- 환경: torch 2.14.1 timm 1.0.30 sklearn 1.9.1 numpy 2.5.4 cv2 5.0.0 macOS-26.5.2-arm64-arm-64bit mps True; HF_HUB_OFFLINE=1, timm 캐시 rev 4685c99dabffe5affac90bd99dbffd25801ae58d
- 결정성 주의: MPS는 결정적 연산 보장 없음, 원본 당시 torch/timm 버전 기록 없음. 원본 dinoB_orig는 동시 2회 실행 로그가 동일(fold0 ep9 va 15.283)했으나 재학습은 14.126 → 비트 일치 아님, 통계적 재현만 기대.
- 참고: build()가 train/test를 각각 자체 평균·표준편차로 정규화(원본 그대로 유지; test 라벨/학습 사용은 없음).
- 다른 작업(cyc2_hibias) 동시 실행 중이라 fold당 ~11분, 예상 완료 ~10/16 01:40 KST.
## 비교 결과
(compare.py 실행 후 compare.md에 기록)
