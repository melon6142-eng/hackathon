import os,numpy as np,pandas as pd
from scipy.optimize import nnls
from sklearn.model_selection import KFold
R='~/AI/daker'; C=f'{R}/cnn/runs'; O=f'{R}/runs/aug_mild'
y=pd.read_csv(f'{R}/open/train.csv').hardness.values
hi=y>230; lo=y<170
r=lambda p: np.sqrt(((p-y)**2).mean())
# base blend members on mac flat runs/
def ld(tag,k='oof'):
    for p in [f'{C}/{tag}/{k}.npy', f'{C}/../{tag}/{k}.npy']:
        if os.path.exists(p): return np.load(p)
    raise FileNotFoundError(tag)
# mac layout: dinoB_orig etc under cnn/runs/
av=lambda tags: np.mean([ld(t) for t in tags],0)
Bo=np.column_stack([np.load(f'{R}/cnn/oof_blend1.npy'),
    av(['dinoB_orig','dinoB_orig_s1','dinoB_orig_s2']),
    av(['dinoB_trans','dinoB_trans_s1','dinoB_trans_s2'])])
mild_tags=[t for t in ['dinoB_orig_mild_s42','dinoB_orig_mild_s1','dinoB_orig_mild_s2'] if os.path.exists(f'{C}/{t}/oof.npy')]
L=['# Mild aug eval (rot±12° + crop≥0.85, no blur)','',f'mild tags: {mild_tags}','']
solos=[]
for t in mild_tags:
    o=ld(t); solos.append(o)
    L.append(f'- {t}: OOF {r(o):.3f} hi {(o-y)[hi].mean():.1f} lo {(o-y)[lo].mean():.1f}')
if not mild_tags:
    open(f'{O}/report.md','w').write('\n'.join(L+['NO RESULTS'])); raise SystemExit
Mavg=np.mean(solos,0)
L.append(f'- mild avg: OOF {r(Mavg):.3f} hi {(Mavg-y)[hi].mean():.1f}')

def ncv(X,Rps=10):
    out=[]
    for s in range(Rps):
        pp=np.zeros(len(y))
        for tr,va in KFold(5,shuffle=True,random_state=100+s).split(X):
            w,_=nnls(np.column_stack([X[tr],np.ones(len(tr))]),y[tr])
            pp[va]=np.column_stack([X[va],np.ones(len(va))])@w
        out.append((r(pp),(pp-y)[hi].mean(),(pp-y)[lo].mean()))
    o=np.array(out); return o[:,0].mean(),o[:,0].std(),o[:,1].mean(),o[:,2].mean()

base=ncv(Bo)
# replace Borig avg with mild avg
Br=Bo.copy(); Br[:,1]=Mavg
# add as new col
Ba=np.column_stack([Bo,Mavg])
# replace each seed into avg partially
L+=['','## nCV 10×5 vs base','| 구성 | nCV | sd | Δ | hi | lo |','|---|---|---|---|---|---|']
def rep(lab,X):
    m=ncv(X); L.append(f'| {lab} | {m[0]:.3f} | {m[1]:.3f} | {m[0]-base[0]:+.3f} | {m[2]:.1f} | {m[3]:.1f} |'); return m
rep('base',Bo)
rep('Bo←mild_avg',Br)
rep('base+mild_col',Ba)
for t,o in zip(mild_tags,solos):
    Bx=Bo.copy(); Bx[:,1]=(2*Bo[:,1]+o)/3  # soft mix one seed
    rep(f'Bo soft←{t}',Bx)
    rep(f'base+{t} col',np.column_stack([Bo,o]))
# seed consistency of replace
improve=0
for s in range(10):
    # compare base vs Bo←mild for this seed only
    def one(X,s):
        pp=np.zeros(len(y))
        for tr,va in KFold(5,shuffle=True,random_state=100+s).split(X):
            w,_=nnls(np.column_stack([X[tr],np.ones(len(tr))]),y[tr])
            pp[va]=np.column_stack([X[va],np.ones(len(va))])@w
        return r(pp)
    if one(Br,s)<one(Bo,s): improve+=1
adopt = improve>=7 and (ncv(Br)[0] < base[0]-0.01)
L+=['',f'## 채택','',
f'- Bo←mild_avg가 base보다 나은 seed: {improve}/10',
f'- **채택: {"예" if adopt else "아니오"}**',
f'- 기준 hi {base[2]:.1f}',
'','blur 증강은 이미 실패. 이번은 회전+크롭만.',
]
open(f'{O}/report.md','w').write('\n'.join(L))
open(f'{C}/aug_mild/report.md','w').write('\n'.join(L))
print('\n'.join(L))
