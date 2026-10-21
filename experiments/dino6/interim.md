# dino6 interim (2026-10-12 22:18 UTC+8 대기 시작)
- 맥스튜디오 screen dino6 / run_dino6.sh / run_dino6.log. dino4·dino5 screen과 dino_ft4/5 프로세스가 모두 사라지면 시작.
- 변형1(쌍별 랭킹) 생략: d4_rank가 같은 pairwise softplus 랭킹 손실. 단독 13.031, 블렌드 추가 12.602(+0.001), hi -27.2 → 효과 없음.
- d6_maskcoh: 채널 = 원본, 어두운 마스크(1단계 1.0/2단계 0.5), 구조텐서 coherence 맵(σ=4).
- d6_maskaux: mask 채널 + 보조 헤드 3개(log 덩어리 면적 가중평균, 장축비, 어두운 면적비; z-score, Huber, 가중 0.3).
- 공통: ViT-B 252, unfreeze4, 30ep, bs16, lr5e-5, 5폴드 rs42, seed42. 체크포인트 저장 안 함.
- 회전 TTA(C): 기존 체크포인트 없음(fold npz는 예측만). dino6에 내장 → oof_rot/test_rot.npy (D4 8-way × {0,22.5,45,67.5}°, reflection 패딩).
- 예상: dino4 ~22:45, dino5 ~23:55, dino6 ~01:10 (UTC+8).
## 평가 절차
1. 맥 cnn/runs/<tag>/{oof,test,oof_rot,test_rot}.npy → 박스 runs/dino6/<tag>/ 로 CopyToBox (d6_maskcoh, d6_maskaux, 원하면 dinoB_mask_s42/1/2도).
2. ./venv/bin/python runs/dino6/ev6.py → ev6.md. nCV 0.05 이상 개선이면 runs/round2/ 에 후보 CSV 자동 생성.
