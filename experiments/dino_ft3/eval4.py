# python eval4.py out.md ; reads dino_ft3/d4_*/oof.npy
import numpy as np,pandas as pd,glob,os,sys
from scipy.optimize import nnls
from sklearn.model_selection import KFold
D='./runs';tr=pd.read_csv('./open/train.csv');y=tr['hardness'].values
b=np.load(f'{D}/oof_blend1.npy');bt=pd.read_csv('./out/submission_blend1_cv12.95.csv').iloc[:,1].values.copy()
ld=lambda t:(np.mean([np.load(f'{D}/{p}/oof.npy') for p in t],0),np.mean([np.load(f'{D}/{p}/test.npy') for p in t],0))
Bo=ld(['dino_ft/dinoB_orig','dino_ft2/dinoB_orig_s1','dino_ft2/dinoB_orig_s2']);Bt=ld(['dino_ft/dinoB_trans','dino_ft2/dinoB_trans_s1','dino_ft2/dinoB_trans_s2'])
base=[(b,bt),Bo,Bt];r=lambda p:np.sqrt(((p-y)**2).mean());hi=y>230
def ncv(cols,reps=20):
    X=np.column_stack(cols+[np.ones(500)]);o=[];P=np.zeros(500)
    for s in range(reps):
        p=np.zeros(500)
        for a,v in KFold(5,shuffle=True,random_state=100+s).split(X): w,_=nnls(X[a],y[a]);p[v]=X[v]@w
        o.append(r(p));P+=p/reps
    return np.mean(o),P
m0,P0=ncv([c[0] for c in base]);ref=Bt[0]
L=[f'기준 블렌드 BL1+B원본평균+B변환평균 nCV {m0:.3f}, 고경도편향 {(P0-y)[hi].mean():.1f}','',
   '| 실험 | OOF | 고경도편향 | 18장 |오차| 평균 (B변환평균 대비) | 18장 중 개선 수 | 기준블렌드+실험 nCV |','|---|---|---|---|---|---|']
best=(m0,None)
for p in sorted(glob.glob(f'{D}/dino_ft3/d4_*/oof.npy')):
    t=p.split('/')[-2];o=np.load(p);te=np.load(p.replace('oof','test'))
    e1=np.abs(o-y)[hi];e0=np.abs(ref-y)[hi];m,P=ncv([c[0] for c in base]+[o])
    L.append(f'| {t} | {r(o):.3f} | {(o-y)[hi].mean():.1f} | {e1.mean():.1f} ({e0.mean():.1f}) | {(e1<e0).sum()}/18 | {m:.3f} |')
    if m<best[0]-0.005: best=(m,t,o,te)
if best[1]:
    m,t,o,te=best;X=np.column_stack([c[0] for c in base]+[o,np.ones(500)]);w=nnls(X,y)[0]
    pt=np.column_stack([c[1] for c in base]+[te,np.ones(len(bt))])@w
    s=pd.read_csv('./out/submission_blend1_cv12.95.csv');s.iloc[:,1]=pt;fn=f'{D}/dino_ft3/cand_base_{t}_cv{m:.2f}.csv';s.to_csv(fn,index=False)
    L+=['',f'BEST +{t} nCV {m:.3f} w={w.round(3)} -> {fn}']
else: L+=['','기준 블렌드 대비 0.005 이상 개선된 실험 없음']
open(sys.argv[1],'w').write('\n'.join(L));print('\n'.join(L))
