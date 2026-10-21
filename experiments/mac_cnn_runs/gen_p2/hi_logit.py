# Phase3 점수 검증: fold-train에서만 학습한 >230 로지스틱 점수 (frozen DINO vs frozen PRE vs 회귀 OOF). 4 fold seed
import numpy as np,csv,os
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score as ap
y=np.array([float(r['hardness']) for r in csv.DictReader(open('~/AI/daker/open/train.csv'))]);hi=y>230;H=np.where(hi)[0]
Fd=np.load('feat_dino_tta.npy');L=['| rs | 점수 | AUPRC | P@18 |','|---|---|---|---|'];RK={}
for rs in (42,7,123,2026):
    if not all(os.path.exists(f'feat_pre_rs{rs}_f{f}.npy') for f in range(5)): continue
    folds=list(KFold(5,shuffle=True,random_state=rs).split(y));sd=np.zeros(500);sn=np.zeros(500)
    for f,(tr,va) in enumerate(folds):
        Fs=np.load(f'feat_pre_rs{rs}_f{f}.npy')
        for F_,o in ((Fd,sd),(Fs,sn)):
            m,s=F_[tr].mean(0),F_[tr].std(0)+1e-6;c=LogisticRegressionCV(Cs=np.logspace(-4,0,9),cv=4,scoring='average_precision',max_iter=3000,class_weight='balanced').fit((F_[tr]-m)/s,hi[tr]);o[va]=c.decision_function((F_[va]-m)/s)
    for nm,o in (('logit frozen DINO',sd),('logit frozen PRE',sn),('회귀 frozen PRE',np.load(f'oof_frozen_pre_rs{rs}.npy'))):
        L.append(f'| {rs} | {nm} | {ap(hi,o):.3f} | {hi[np.argsort(-o)[:18]].mean():.3f} |');RK[(rs,nm)]=(-o).argsort().argsort()[H]+1
    L.append(f'| {rs} | DINO vs PRE 로짓 점수 상관 | {np.corrcoef(sd,sn)[0,1]:.2f} | |')
print('\n'.join(L));open('hi_logit.md','w').write('\n'.join(L)+'\n');np.save('hi_logit_ranks.npy',RK,allow_pickle=True)
