# BL1 = NNLS blend of two Ridge models: [handcrafted + fold-internal poly2(top-14 |corr|)] and [DINO L9-12 patch mean+std].
# Same estimator recipe as runs/lib/blend.py + runs/feats.py, but the top-14 selection / its standardization is now
# fitted inside each training set (fold-train for OOF, full train for test). StandardScaler lives inside the Ridge pipeline.
import os; os.environ['OMP_NUM_THREADS']='4'
import numpy as np,pandas as pd,sys,json
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.optimize import nnls
from common import *
AL=np.logspace(-1,6,15)
def load(sp):
    h=pd.read_csv(f'{O}/feat/hand_{sp}.csv').drop(columns='ID')
    cols=list(h.columns); ht=[c for c in cols if c.startswith(('h_','t_','g_'))]; r_=[c for c in cols if c.startswith('r_')]
    X=h[ht+r_].values.astype(float); nb=len(ht)
    cand=[i for i,c in enumerate(ht+r_) if not c.startswith('t_')]   # hand+grain+raw, as original
    F=np.load(f'{O}/feat/dino_{sp}.npy'); D=F.shape[1]//16
    Xd=np.hstack([F[:,(L*4+k)*D:(L*4+k+1)*D] for L in range(4) for k in (1,2)])
    return X,nb,cand,Xd
class PolyTop:
    def __init__(s,nb,cand,k=14): s.nb,s.cand,s.k=nb,cand,k
    def fit(s,X,y):
        C=X[:,s.cand]; c=np.abs([np.corrcoef(C[:,j],y)[0,1] if C[:,j].std()>0 else 0 for j in range(C.shape[1])])
        s.sel=[s.cand[j] for j in np.argsort(np.nan_to_num(c))[-s.k:]]; s.mu=X[:,s.sel].mean(0); s.sd=X[:,s.sel].std(0)+1e-9; return s
    def transform(s,X): return np.hstack([X[:,:s.nb],PolynomialFeatures(2,include_bias=False).fit_transform((X[:,s.sel]-s.mu)/s.sd)])
def fit_ridge(X,y):
    best=None
    for a in AL:
        e=0
        for t,v in KFold(5,shuffle=True,random_state=0).split(X):
            m=make_pipeline(StandardScaler(),Ridge(a)).fit(X[t],y[t]); e+=((m.predict(X[v])-y[v])**2).sum()
        if best is None or e<best[0]: best=(e,a)
    return make_pipeline(StandardScaler(),Ridge(best[1])).fit(X,y),best[1]
def member_fit_predict(kind,Xtr,ytr,Xte,meta):
    if kind=='hx': pt=PolyTop(*meta).fit(Xtr,ytr); Xtr,Xte=pt.transform(Xtr),pt.transform(Xte)
    m,a=fit_ridge(Xtr,ytr); return m.predict(Xte)
def bl1_fit_predict(Xh,Xd,yv,Xht,Xdt,meta):
    O_=np.zeros((len(yv),2))
    for t,v in KFold(5,shuffle=True,random_state=1).split(Xh):
        O_[v,0]=member_fit_predict('hx',Xh[t],yv[t],Xh[v],meta); O_[v,1]=member_fit_predict('d',Xd[t],yv[t],Xd[v],meta)
    mu=yv.mean(); w,_=nnls(O_-mu,yv-mu)
    P=np.column_stack([member_fit_predict('hx',Xh,yv,Xht,meta),member_fit_predict('d',Xd,yv,Xdt,meta)])
    return mu+(P-mu)@w,w,P
if __name__=='__main__':
    Y=y(); Xh,nb,cand,Xd=load('train'); Xht,_,_,Xdt=load('test'); meta=(nb,cand)
    os.makedirs(f'{O}/bl1',exist_ok=True); res={}
    for fs in [int(s) for s in sys.argv[1].split(',')]:
        fn=f'{O}/bl1/oof_fs{fs}.npy'
        if os.path.exists(fn): continue
        oof=np.zeros(500); W=[]
        for tr,va in KFold(5,shuffle=True,random_state=fs).split(Xh):
            p,w,_=bl1_fit_predict(Xh[tr],Xd[tr],Y[tr],Xh[va],Xd[va],meta); oof[va]=p; W.append(w)
        np.save(fn,oof); res[fs]={'rmse':rm(oof,Y),'hi':hb(oof,Y),'lo':lb(oof,Y),'w':np.mean(W,0).tolist()}
        print('BL1 fs',fs,res[fs],flush=True)
    if not os.path.exists(f'{O}/bl1/test.npy'):
        p,w,P=bl1_fit_predict(Xh,Xd,Y,Xht,Xdt,meta); np.save(f'{O}/bl1/test.npy',p); print('BL1 test w',w,'mean',p.mean(),flush=True)
    json.dump(res,open(f'{O}/bl1/res_{sys.argv[1].replace(",","_")}.json','w'),indent=1)
