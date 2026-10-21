# dino5(마스크 3채널 ViT-B) 진행 기록
- 2026-10-12 20:27 CST: Mac Studio에 cnn/dino_ft5.py(chan=mask) 생성
  - 3채널: 원본 / 이미지별 1·2단계 마스크(1.0/0.5/0) / 부호 있는 거리변환(어두운 상 안쪽 +, 바깥 −, /10, ±1로 자름)
- run_dino5.sh를 screen dino5로 시작. dino4 screen과 dino_ft4.py가 끝날 때까지 대기한 뒤 dinoB_mask_s42, s1, s2를 순서대로 실행
  - skip-if-done, pgrep으로 중복 방지, 로그는 cnn/run_dino5.log
  - 설정: ViT-B 252, unfreeze 4, bs16, lr5e-5, KFold(5, rs42), D4 8방향 TTA
- 이어서 할 일:
  1. Mac의 cnn/runs/{dinoB_mask_s*, d4_*}/{oof,test}.npy를 박스 runs/dino_ft3/<tag>/로 복사
  2. `cd runs/dark_mask && ../../venv/bin/python ev5.py` 실행 → ev5.md
  3. 0.03 이상 개선되면 cand_*.csv를 MacBook ~/Downloads/open/에 복사
- 상태: dino4가 d4_binreg 학습 중이고, rank, crop140, s448이 남아 있음(예상 2~3시간). dino5는 그 뒤 약 75~90분
- 20:41 CST 확인: d4_binreg 진행 중(로그가 버퍼링돼 지연 표시), dino5 대기 중. dino4 지금까지 결과: d4_lds 13.365(hi −23.7), d4_pad266 13.717. 박스 평가는 아직 안 함.

- 2026-10-13 00:49 CST: dino4·dino5 모두 완료 확인 후 평가
  - 단독 OOF: mask seeds 13.86~13.94, d4_rank 13.031, d4_lds 13.365, d4_s448 13.524, d4_binreg 13.560, d4_pad266 13.717, d4_crop140 14.233
  - 기준 BL1+Borig+Btrans nCV 12.606; 최고 블렌드 +d4_rank → 12.603 (Δ0.003)
  - 0.03 미만 → cand CSV 미생성. 제출 없음. dino6(d6_maskaux)은 실행 중이라 건드리지 않음.
