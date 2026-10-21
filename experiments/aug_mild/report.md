# Mild physical aug — GPU 대기열 (3번)

## 설정
- 기준: ViT-B orig (`dino_ft.py`와 동일: 252, bs16, lr5e-5, unfreeze4, fold_rs=42)
- **추가만**: 연속 회전 ±12° + random crop scale [0.85, 1] (reflection). **blur 제외**(이미 실패).
- 기존 8-way rot/flip + brightness±0.1은 유지.
- seeds: 42, 1, 2. fold ckpt 저장.
- 출력: `cnn/runs/dinoB_orig_mild_s{42,1,2}/`, 평가·리포트 `runs/aug_mild/`.
- GPU: screen `augMild`가 Exp D `INDEPDONE` 후 실행. **indepD 미kill**.

## 상태
대기 중 (D: rs2026 진행 중). 예상 D 완료 ~23:00 CST, mild+평가 ~00:15 CST.

## 참고
- 공격적 cropaug(d11) 단독 OOF 13.589로 이미 실패 → 이번은 **약한** crop만.
