import numpy as np,pandas as pd,sys,os
sys.path.insert(0,'./runs');os.chdir('..')
from feats import HX
from scipy.optimize import nnls
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix,diags,identity
from scipy.sparse.linalg import spsolve
y=pd.read_csv('./open/train.csv').hardness.values
H=np.vstack([HX('train'),HX('test')]);H=np.nan_to_num(StandardScaler().fit_transform(np.nan_to_num(H)))
Z=np.load('label_ssl/Z64.npy');Z=Z/Z[:500].std(0)
F=np.hstack([H,Z]);n=500
r=lambda p:np.sqrt(((p-y)**2).mean());hb=lambda p:(p-y)[y>230].mean();lb=lambda p:(p-y)[y<170].mean()
def fit(X,t,w=None,a=300):return Ridge(a).fit(X,t,sample_weight=w)
# graph
nb=NearestNeighbors(n_neighbors=11).fit(F);d,ix=nb.kneighbors(F);s=np.median(d[:,1:])
N=len(F);W=csr_matrix((np.exp(-(d[:,1:]/s)**2).ravel(),(np.repeat(np.arange(N),10),ix[:,1:].ravel())),shape=(N,N));W=(W+W.T)/2
Lg=diags(np.asarray(W.sum(1)).ravel())-W
def prop(lab_idx,t,base,lam):
    # min sum_lab (f-t)^2 + lam f'Lf  + small pull to base (ridge pred) on all
    m=np.zeros(N);m[lab_idx]=1;b=np.zeros(N);b[lab_idx]=t
    A=diags(m+0.05)+lam*Lg;return spsolve(A.tocsc(),b+0.05*base)
res={}
for rep in range(3):
  P={k:np.zeros(n) for k in ['ridge','st_ridge','st_ridge_all','lap1','lap5']}
  for tr,va in KFold(5,shuffle=True,random_state=rep).split(np.arange(n)):
    te=np.r_[va,np.arange(n,N)]  # unlabeled = val + test (transductive, labels unseen)
    m1=fit(F[tr],y[tr]);p=m1.predict(F);P['ridge'][va]=p[va]
    mh=fit(H[tr],y[tr]).predict(H);md=fit(Z[tr],y[tr],a=30).predict(Z);dis=np.abs(mh-md)
    u=np.arange(n,N);c=u[dis[u]<np.quantile(dis[u],.5)]
    for k,uu in [('st_ridge',c),('st_ridge_all',u)]:
      X=np.vstack([F[tr],F[uu]]);t=np.r_[y[tr],p[uu]];w=np.r_[np.ones(len(tr)),.5*np.ones(len(uu))]
      P[k][va]=fit(X,t,w).predict(F[va])
    for lam,k in [(1,'lap1'),(5,'lap5')]:
      P[k][va]=prop(tr,y[tr],p,lam)[va]
  for k,v in P.items():res.setdefault(k,[]).append(v)
oo={k:np.mean(v,0) for k,v in res.items()}
b=np.load('oof_blend1.npy');M=lambda t:np.mean([np.load(f'dino_ft2/{x}/oof.npy') if os.path.exists(f'dino_ft2/{x}/oof.npy') else np.load(f'dino_ft/{x}/oof.npy') for x in t],0)
bo=M(['dinoB_orig','dinoB_orig_s1','dinoB_orig_s2']);bt=M(['dinoB_trans','dinoB_trans_s1','dinoB_trans_s2'])
def ncv(cols,reps=20):
    X=np.column_stack(cols+[np.ones(n)]);o=[]
    for s_ in range(reps):
        p=np.zeros(n)
        for a_,v_ in KFold(5,shuffle=True,random_state=100+s_).split(X): w,_=nnls(X[a_],y[a_]);p[v_]=X[v_]@w
        o.append(r(p))
    return np.mean(o),np.std(o)
L=['','## (3) 준지도 (누수 없음: 폴드 내 train만 라벨, val+test는 비라벨)','| 방법 | 단독 OOF(3회평균) | hi편향 | lo편향 | 최고블렌드+방법 nCV |','|---|---|---|---|---|']
m,s_=ncv([b,bo,bt]);L.append(f'| (기준) BL1+Bavg | – | – | – | {m:.3f}±{s_:.3f} |')
for k,v in oo.items():
    m,s_=ncv([b,bo,bt,v]);L.append(f'| {k} | {np.mean([r(x) for x in res[k]]):.3f} | {hb(v):.1f} | {lb(v):.1f} | {m:.3f}±{s_:.3f} |')
np.savez('label_ssl/oof_ssl.npz',**oo)
open('label_ssl/interim.md','a').write('\n'.join(L)+'\n');print('\n'.join(L))
