# usage: python ev5.py  (dino_ft3/<tag>/{oof,test}.npy 사용: dinoB_mask_s* 와 d4_*)
import numpy as np,pandas as pd,glob,os,itertools
from scipy.optimize import nnls
from sklearn.model_selection import KFold
H='./runs/hi_rule';D='./runs/dino_ft3'
y=pd.read_csv('./open/train.csv').hardness.values
C=np.load(f'{H}/comp_oof.npy');Ct=np.load(f'{H}/comp_test.npy')
M={'BL1':(C[:,0],Ct[:,0]),'Borig':(C[:,1],Ct[:,1]),'Btrans':(C[:,2],Ct[:,2])}
for p in glob.glob(f'{D}/*/oof.npy'):
    t=os.path.basename(os.path.dirname(p));M[t]=(np.load(p),np.load(p.replace('oof','test')))
mk=[t for t in M if t.startswith('dinoB_mask')]
if len(mk)>1:M['Bmask_avg']=(np.mean([M[t][0] for t in mk],0),np.mean([M[t][1] for t in mk],0))
r=lambda p:np.sqrt(((p-y)**2).mean());hb=lambda p:(p-y)[y>230].mean();lb=lambda p:(p-y)[y<170].mean()
def ncv(ts,reps=20):
    X=np.column_stack([M[t][0] for t in ts]+[np.ones(500)]);o=[];P=np.zeros(500)
    for s in range(reps):
        p=np.zeros(500)
        for tr,va in KFold(5,shuffle=True,random_state=100+s).split(X):p[va]=X[va]@nnls(X[tr],y[tr])[0]
        o.append(r(p));P+=p/reps
    return np.mean(o),np.std(o),hb(P),lb(P)
L=['| 모델 | 단독 OOF | hi편향 |','|---|---|---|']+[f'| {t} | {r(M[t][0]):.3f} | {hb(M[t][0]):.1f} |' for t in M]
base=['BL1','Borig','Btrans'];b0=ncv(base)
L+=['',f'기준 BL1+Borig+Btrans nCV {b0[0]:.3f}±{b0[1]:.3f} hi {b0[2]:.1f} lo {b0[3]:.1f}','','| 조합 | nCV | hi | lo |','|---|---|---|---|']
extra=[t for t in M if t not in base and not (t.startswith('dinoB_mask_s') and 'Bmask_avg' in M)]
res=[]
for k in [1,2]:
    for c in itertools.combinations(extra,k):
        m=ncv(base+list(c));res.append((m,base+list(c)))
res.sort(key=lambda t:t[0][0])
for m,ts in res[:12]:L.append(f'| {"+".join(ts)} | {m[0]:.3f}±{m[1]:.3f} | {m[2]:.1f} | {m[3]:.1f} |')
m,ts=res[0] if res else (b0,base)
if m[0]<=b0[0]-0.03:
    X=np.column_stack([M[t][0] for t in ts]+[np.ones(500)]);w=nnls(X,y)[0]
    pt=np.column_stack([M[t][1] for t in ts]+[np.ones(len(Ct))])@w
    sub=pd.read_csv('./out/submission_blend1_cv12.95.csv');sub.iloc[:,1]=pt
    fn=f'./runs/dark_mask/cand_{"_".join(ts)}_cv{m[0]:.2f}.csv';sub.to_csv(fn,index=False)
    L+=['',f'후보 {fn} 가중 {dict(zip(ts+["c"],w.round(3)))} test평균 {pt.mean():.2f}']
else:L+=['','0.03 이상 개선 없음 → CSV 미생성']
open('./runs/dark_mask/ev5.md','w').write('\n'.join(L));print('\n'.join(L))
