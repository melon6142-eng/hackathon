# Exp D eval: per fold seed RS, BL1 (HX in-fold PolyTop + frozen DINO L9-12 ms, inner NNLS) on KFold(5,RS) + FT members from runs/indepfold/rs{RS}_*
import os,sys,glob;os.environ['OMP_NUM_THREADS']='1'
B=os.path.dirname(os.path.abspath(__file__));W=f'{B}/../lcurve/w';os.chdir(f'{W}/runs');sys.path[:0]=['lib','.']
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
    f=f'{B}/bl1_rs{rs}.npy'
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
L=['# Exp D 독립 fold 검증 결과 (auto)','','| fold rs | 멤버 | BL1 | orig avg | trans avg | 블렌드 nCV(10×5) | sd | Δ vs 12.601 | hi bias | 같은-fold stack | 가중치 |','|---|---|---|---|---|---|---|---|---|---|---|']
S=[]
for rs in (7,123,2026):
    oo=[np.load(f) for f in sorted(glob.glob(f'{B}/rs{rs}_orig_s*/oof.npy'))];tt=[np.load(f) for f in sorted(glob.glob(f'{B}/rs{rs}_trans_s*/oof.npy'))]
    if not oo or not tt: L.append(f'| {rs} | 미완료 (orig {len(oo)}, trans {len(tt)}) |||||||||| ');continue
    b=bl1(rs);X=np.column_stack([b,np.mean(oo,0),np.mean(tt,0)]);(m,h,l),sd=ncv(X);sf=same_fold(X,rs);w=nnls(A1(X),y)[0]
    solo=' / '.join(f'{r_(o):.2f}' for o in oo)+' | '+' / '.join(f'{r_(o):.2f}' for o in tt)
    L.append(f'| {rs} | o{len(oo)} t{len(tt)} | {r_(b):.3f} | {r_(X[:,1]):.3f} | {r_(X[:,2]):.3f} | {m:.3f} | {sd:.3f} | {m-12.601:+.3f} | {h:.1f} | {sf[0]:.3f} | {np.round(w[:3],2).tolist()} |')
    L.append(f'|  | 시드별 단독 orig / trans | {solo} |||||||||');S.append(m)
    np.save(f'{B}/blendX_rs{rs}.npy',X)
if S: L+=['',f'완료 fold seed {len(S)}개: 블렌드 nCV 평균 {np.mean(S):.3f} ± {np.std(S,ddof=1) if len(S)>1 else 0:.3f} (Δ vs 12.601 {np.mean(S)-12.601:+.3f})']
open(f'{B}/eval_auto.md','w').write('\n'.join(L)+'\n');print('\n'.join(L))
