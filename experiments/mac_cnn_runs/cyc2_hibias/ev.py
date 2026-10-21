# eval vs equalmean(BL1+orig avg3+trans avg3) per indep RS. Variant replaces orig_s42 in orig avg.
import glob,os,numpy as np,pandas as pd
R='~/AI/daker/cnn/runs';I=R+'/indepfold';B=os.path.dirname(os.path.abspath(__file__))
y=pd.read_csv(R+'/lcurve/w/open/train.csv').hardness.values;hi=y>230
r_=lambda p:np.sqrt(((p-y)**2).mean());hb=lambda p:(p-y)[hi].mean()
L=['# cyc2 고경도 편향/TTA 평가 (기준 equalmean)','','| rs | variant | TTA | solo | solo hi | eq3 | Δ vs eq | eq hi |','|---|---|---|---|---|---|---|---|'];D={}
for rs in (7,123,2026):
    b=np.load(f'{I}/bl1_rs{rs}.npy');o={s:np.load(f'{I}/rs{rs}_orig_s{s}/oof.npy') for s in (1,2,42)}
    t=np.mean([np.load(f) for f in glob.glob(f'{I}/rs{rs}_trans_s*/oof.npy')],0)
    e0=np.mean([b,np.mean(list(o.values()),0),t],0)
    L.append(f'| {rs} | equalmean 기준 | 8 | {r_(o[42]):.3f} | {hb(o[42]):.1f} | {r_(e0):.3f} | 0 | {hb(e0):.1f} |')
    for v in ('base','log','hiw1','log_hiw1'):
        d=f'{B}/rs{rs}_{v}'
        for tt,fn in (('8','oof.npy'),('24','oof_x.npy')):
            if not os.path.exists(f'{d}/{fn}'): continue
            p=np.load(f'{d}/{fn}');e=np.mean([b,np.mean([o[1],o[2],p],0),t],0)
            L.append(f'| {rs} | {v} | {tt} | {r_(p):.3f} | {hb(p):.1f} | {r_(e):.3f} | {r_(e)-r_(e0):+.3f} | {hb(e):.1f} |');D.setdefault((v,tt),[]).append(r_(e)-r_(e0))
L+=['','## 요약 (Δ<0이 개선, 3/3 fold 개선이어야 재현)']
for (v,tt),d in sorted(D.items()): L.append(f'- {v} TTA{tt}: n={len(d)} Δ mean {np.mean(d):+.3f}, 개선 fold {sum(x<0 for x in d)}/{len(d)}')
open(f'{B}/report.md','w').write('\n'.join(L)+'\n');print('\n'.join(L))
