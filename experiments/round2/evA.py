import numpy as np,pandas as pd,sys
from scipy.optimize import nnls
from sklearn.model_selection import KFold
D='./runs';y=pd.read_csv('./open/train.csv')['hardness'].values
ld=lambda t,k:np.load(f'{D}/{t}/{k}.npy');av=lambda ts,k:np.mean([ld(t,k) for t in ts],0)
O=['dino_ft/dinoB_orig','dino_ft2/dinoB_orig_s1','dino_ft2/dinoB_orig_s2'];T=['dino_ft/dinoB_trans','dino_ft2/dinoB_trans_s1','dino_ft2/dinoB_trans_s2']
B=np.column_stack([np.load(f'{D}/oof_blend1.npy'),av(O,'oof'),av(T,'oof')])
r=lambda p,t:np.sqrt(((p-t)**2).mean())
def blend(tr,va):
    A=np.column_stack([B,np.ones(500)]);w=nnls(A[tr],y[tr])[0];return A[tr]@w,A[va]@w
C=np.arange(-5,31,1.0)
def adj(pt,p,c,q,soft):
    th=np.quantile(pt,q)
    if soft==0: g=(p>=th).astype(float)
    else: g=1/(1+np.exp(-(p-th)/soft))
    return p+c*g
def m(tr,va,q,soft):
    s=np.zeros(len(C))
    for i,j in KFold(5,shuffle=True,random_state=7).split(tr):
        a,b=tr[i],tr[j];pt,pv=blend(a,b);s+=[((adj(pt,pv,c,q,soft)-y[b])**2).sum() for c in C]
    c=C[s.argmin()];pt,pv=blend(tr,va);return adj(pt,pv,c,q,soft),c
V={'base':None,'hard_q90':(.9,0),'soft_q90_s2':(.9,2),'soft_q90_s5':(.9,5),'soft_q85_s5':(.85,5)}
R=10;hi=y>230;lo=y<170;res={k:[] for k in V};cs={k:[] for k in V}
for s in range(R):
    pp={k:np.zeros(500) for k in V}
    for tr,va in KFold(5,shuffle=True,random_state=100+s).split(B):
        for k,v in V.items():
            if v is None: pp[k][va]=blend(tr,va)[1]
            else: p,c=m(tr,va,*v);pp[k][va]=p;cs[k].append(c)
    for k in V: res[k].append((r(pp[k],y),(pp[k]-y)[hi].mean(),(pp[k]-y)[lo].mean()))
L=['| 변형 | nCV(10회) | std | hi편향 | lo편향 | 고른 상수(빈도 상위) |','|---|---|---|---|---|---|']
for k in V:
    a=np.array(res[k]);vc=pd.Series(cs[k]).value_counts().head(3).to_dict() if cs[k] else '-'
    L.append(f'| {k} | {a[:,0].mean():.3f} | {a[:,0].std():.3f} | {a[:,1].mean():.1f} | {a[:,2].mean():.1f} | {vc} |')
open('evA.md','w').write('\n'.join(L));print('\n'.join(L))
