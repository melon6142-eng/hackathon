# DAKER 최종 블렌드 재현 절차 (end-to-end)
환경: Mac Studio, `~/AI/daker/venv/bin/python` (env.txt 참조), 장치 MPS. HF_HUB_OFFLINE=1, timm `vit_base_patch14_dinov2.lvd142m` rev 4685c99d (HF 캐시).
1. 이미지 텐서: `~/AI/daker/cnn/imgs.pt` (train 500, test 1000 원본 PNG에서 생성; 학습엔 train만 사용).
2. BL1: `runs/dino/feat_dino.py`로 torch.hub dinov2_vitb14 특징 추출 → Ridge 계열 → `cnn/sub_bl1.csv` (+ OOF `ref/oof_blend1.npy`).
3. ViT 6개 재학습(+fold별 ckpt, sha256 로그): `tmux new -d -s repro_orig "./chain.sh orig"; tmux new -d -s repro_trans "./chain.sh trans"`
   - 스크립트 `dino_ft_repro.py` = 원본 `cnn/dino_ft.py`(sha256 66da852a…9c16) + 출력경로 변경 + ckpt 저장만 추가.
   - 설정: size 252, unfreeze 4, epochs 30, bs 16, lr 5e-5, weight 0, seeds 42/1/2, KFold(5, shuffle, rs=42).
   - 진행: `progress.txt`, 로그 `logs_<tag>.out`, `<tag>/log.txt`.
4. ckpt만으로 추론 확인: chain.sh가 자동 실행 `infer_from_ckpt.py <tag> <chan>` → `<tag>/test_ckpt.npy`, `ckpt_infer.json`.
5. 블렌드/비교: `python compare.py` → `compare.md`, `sub_repro_blend.csv`
   블렌드 = 0.4398·BL1 + 0.2831·mean(orig 3seed) + 0.2773·mean(trans 3seed) + c(0.0031).
