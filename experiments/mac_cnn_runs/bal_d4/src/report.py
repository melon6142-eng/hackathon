import json,glob,os,numpy as np
from common import *
Y=y();FC='~/AI/daker/final_clean';A=open(f'{O}/analysis.md').read();R=json.load(open(f'{O}/results.json'))
L=['# bal_d4 리포트: 경도분위 균형 샘플링 × D4 (규정준수, 연구용 단일 실험)\n',
'설정: final_clean ViT-B orig(timm dinov2), fold-train 정규화, KFold(5,rs=42), 학습 seed 42/1/2, 30ep, TTA D4×8 공통. test는 추론만, 합성 없음, 제출 없음.\n',
'arm: base=uniform+D4+밝기±0.1(final_clean 재사용) / uni_none=증강 없음 / uni_d4=D4만(무손실) / bal_none=균형샘플링 / bal_d4b=균형+D4+밝기. 균형=fold-train 라벨 등간격 10bin, 확률∝1/√bin빈도, 복원추출(분위수 그룹은 정의상 동일빈도라 1/√freq가 균등이 되므로 등간격 bin 사용).\n',
'## Step 1 감사: 기존 실험 재사용\n','| 계획 arm | 기존 기록 | 판정 |','|---|---|---|',
'| uniform+D4 | final_clean ViT orig = 이미 랜덤 D4+밝기±0.1+D4 TTA로 학습 | 기준선으로 재사용 |',
'| TTA | 기준선에 D4×8 내장; cyc2 TTA24는 이득 없음(+0.008) | 재실행 안 함 |',
'| mild 회전±12°+crop | aug_mild: 단독 13.6~13.9, nCV +0.05 악화 | 재실행 안 함 |',
'| 고경도 가중 loss | cyc2 hiw1: hi편향 −24.9→−23.2 대신 단독 +0.27 악화; dino_trans_w 13.95 | 재실행 안 함(샘플링은 미시도) |',
'| VICReg(train-only) | ssl_train: seed 42 세 변형 기준 미달 | 재실행 안 함 |',
'| 증강 없음 / 균형 샘플링 | 기록 없음 | 신규 실행 |\n','기존 기준선 노출량: permutation 샘플러, loss 가중 1 → 10분위 모두 fold당 약 1200 visit(균등). 즉 기존 학습에서 양 끝 분위가 덜 노출된 적은 없음.\n',A]
ex=sorted(glob.glob(f'{O}/arms_fs/*/oof.npy'))
if ex:
    L+=['\n## 독립 fold seed 확장 (학습 seed 42)\n','| run | RMSE | 기준(final_clean orig_s42 동일 fs) |','|---|---|---|']
    for p in ex:
        t=p.split('/')[-2];fs=t.split('_fs')[1];b=f'{FC}/vit/orig_s42_fs{fs}/oof.npy'
        L.append(f"| {t} | {rm(np.load(p),Y):.3f} | {rm(np.load(b),Y):.3f} |" if os.path.exists(b) else f"| {t} | {rm(np.load(p),Y):.3f} | 미완 |")
else: L.append('\n## 독립 fold seed 확장\n기준(3seed 평균 RMSE가 기준보다 ≥0.05 개선)을 넘은 arm 없음 → 확장 안 함.')
L.append('\n## A/B 해석 기준\n- 균형/D4 arm이 RMSE를 반복 개선하고 held-out 임베딩 kNN/probe R²가 오르면 A(학습방식 부족) 쪽.\n- 노출을 바꿔도 RMSE·임베딩 분리도가 그대로면 B(원본 다양성 부족) 쪽(학습곡선 80→400 단조감소와 함께). 결론은 위 표 숫자로만 판단.')
open(f'{O}/report.md','w').write('\n'.join(L));print('ok')
