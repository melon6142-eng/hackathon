# Exp D eval: per fold seed RS, BL1 (HX in-fold PolyTop + frozen DINO L9-12 ms, inner NNLS) on KFold(5,RS) + FT members from runs/indepfold/rs{RS}_*
import os,sys,glob;os.environ['OMP_NUM_THREADS']='1'
B=os.path.dirname(os.path.abspath(__file__));I=f'{B}/../indepfold';W=f'{B}/../lcurve/w';os.chdir(f'{W}/runs');sys.path[:0]=['lib','.']
import numpy as np,pandas as pd,feats,hxfold;feats.R=hxfold.R=f'{W}/open'
from feats import DINO;from hxfold import base,PolyTop;from blend import fit_ridge,inner_oof
from scipy.optimize import nnls;from sklearn.model_selection import KFold;from joblib import Parallel,delayed
y=pd.read_csv(f'{W}/open/train.csv').hardness.values;hi=y>230;lo=y<170
Xr,nb,cand=base('train');Xb=DINO('train',sel=[(L,k) for L in range(4) for k in (1,2)])
r_=lambda p:np.sqrt(((p-y)**2).mean())
def bl1_fold(tr,va):
    pt=PolyTop(nb,cand).fit(Xr[tr],y[tr]);XA=pt.transform(Xr)
    O=np.column_stack([inner_oof(X[tr],y[tr]) for X in (XA,Xb)]);mu=y[tr].mean();w,_=nnls(O-mu,y[tr]-mu)
    P=np.column_stack([fit_ridge(X[tr],y[tr]).predict(X[va]) for X in (XA,Xb)]);return va,mu+(P-mu)@w
def bl1(rs):
    f=f'{I}/bl1_rs{rs}.npy'
    if os.path.exists(f): return np.load(f)
    o=np.zeros(len(y))
    for va,p in Parallel(n_jobs=5)(delayed(bl1_fold)(tr,va) for tr,va in KFold(5,shuffle=True,random_state=rs).split(y)): o[va]=p
    np.save(f,o);return o
A1=lambda X:np.column_stack([X,np.ones(len(X))])
def ncv(X,R=10):
    o=[]
    for s in range(R):
        pp=np.zeros(len(y))
        for tr,va in KFold(5,shuffle=True,random_state=100+s).split(X): pp[va]=A1(X[va])@nnls(A1(X[tr]),y[tr])[0]
        o.append((r_(pp),(pp-y)[hi].mean(),(pp-y)[lo].mean()))
    o=np.array(o);return o.mean(0),o[:,0].std()
def same_fold(X,rs):  # stacker weights fit on the other 4 folds of the SAME RS split
    pp=np.zeros(len(y))
    for tr,va in KFold(5,shuffle=True,random_state=rs).split(X): pp[va]=A1(X[va])@nnls(A1(X[tr]),y[tr])[0]
    return r_(pp),(pp-y)[hi].mean()
L=['# Seed bagging (indep fold)','','| rs | k seeds/chan | orig avg | trans avg | blend nCV | sd | Δ vs 12.601 |','|---|---|---|---|---|---|---|']
R={}
for rs in (7,123,2026):
    b=bl1(rs)
    def ld(c): return [np.load(f) for f in sorted(glob.glob(f'{I}/rs{rs}_{c}_s*/oof.npy'))+sorted(glob.glob(f'{B}/rs{rs}_{c}_s*/oof.npy'))]
    oo,tt=ld('orig'),ld('trans');n=min(len(oo),len(tt))
    for k in sorted(set([1,3,n])):
        if k>n: continue
        X=np.column_stack([b,np.mean(oo[:k],0),np.mean(tt[:k],0)]);(m,h,l),sd=ncv(X);R.setdefault(k,[]).append(m)
        L.append(f'| {rs} | {k} | {r_(X[:,1]):.3f} | {r_(X[:,2]):.3f} | {m:.3f} | {sd:.3f} | {m-12.601:+.3f} |')
L.append('')
for k,v in sorted(R.items()): L.append(f'k={k}: {len(v)} rs, blend nCV mean {np.mean(v):.3f} ± {np.std(v,ddof=1) if len(v)>1 else 0:.3f}')
open(f'{B}/report.md','w').write('\n'.join(L)+'\n');print('\n'.join(L))
