"""Strict nested CV for tail calibration on BL1. Fast path. OMP=2."""
import os; os.environ['OMP_NUM_THREADS']='2'; os.environ['OPENBLAS_NUM_THREADS']='2'
import sys; sys.path.insert(0,'lib'); sys.path.insert(0,'.')
import numpy as np, json, time
from sklearn.model_selection import KFold
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge, QuantileRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from scipy.optimize import nnls
from blend import blend_fit_predict, fit_ridge
from feats import HX, DINO
from cv import y as get_y

AL=np.logspace(0,5,8)  # fewer alphas for speed

def rmse(a,b): return float(np.sqrt(((a-b)**2).mean()))
def bias_report(p,yv):
    hi=yv>230; lo=yv<170
    return {'rmse':rmse(p,yv),
            'hi_n':int(hi.sum()),'hi_bias':float(np.mean(p[hi]-yv[hi])) if hi.any() else None,
            'lo_n':int(lo.sum()),'lo_bias':float(np.mean(p[lo]-yv[lo])) if lo.any() else None}

def fit_ridge_fast(X,y,alphas=AL):
    best=None
    for a in alphas:
        e=0
        for t,v in KFold(5,shuffle=True,random_state=0).split(X):
            m=make_pipeline(StandardScaler(),Ridge(a)).fit(X[t],y[t])
            e+=((m.predict(X[v])-y[v])**2).sum()
        if best is None or e<best[0]: best=(e,a)
    return make_pipeline(StandardScaler(),Ridge(best[1])).fit(X,y)

def blend_fp(Xs,y,Xte):
    def inner_oof(X,yy):
        o=np.zeros(len(yy))
        for t,v in KFold(5,shuffle=True,random_state=1).split(X):
            o[v]=fit_ridge_fast(X[t],yy[t]).predict(X[v])
        return o
    O=np.column_stack([inner_oof(X,y) for X in Xs]); mu=y.mean()
    w,_=nnls(np.column_stack([O-mu]),y-mu)
    P=np.column_stack([fit_ridge_fast(X,y).predict(Xt) for X,Xt in zip(Xs,Xte)])
    return mu+(P-mu)@w, w

def blend_inner_oof(Xs,y):
    o=np.zeros(len(y))
    for t,v in KFold(5,shuffle=True,random_state=7).split(Xs[0]):
        p,_=blend_fp([X[t] for X in Xs],y[t],[X[v] for X in Xs]); o[v]=p
    return o

def stretch_k(p_oof,y,ks=np.linspace(0.9,2.2,27)):
    mu=y.mean(); best=(1e99,1.0)
    for k in ks:
        e=rmse(mu+k*(p_oof-mu),y)
        if e<best[0]: best=(e,float(k))
    return best[1]

def apply_stretch(p,y_tr,k):
    return y_tr.mean()+k*(p-y_tr.mean())

def fit_isotonic(p_oof,y):
    return IsotonicRegression(out_of_bounds='clip').fit(p_oof,y)

def quantile_map(p,p_ref,y_ref):
    qs=np.linspace(0,1,101)
    pref_q=np.quantile(p_ref,qs); y_q=np.quantile(y_ref,qs)
    r=np.interp(p,pref_q,qs,left=0,right=1)
    return np.interp(r,qs,y_q)

def weighted_ridge_fp(X,y,Xt,power=2.0):
    mu=y.mean(); w=np.abs(y-mu)**power; w=w/(w.mean()+1e-12)
    best=None
    for a in AL:
        e=0
        for t,v in KFold(5,shuffle=True,random_state=0).split(X):
            sc=StandardScaler().fit(X[t])
            m=Ridge(a).fit(sc.transform(X[t]),y[t],sample_weight=w[t])
            e+=((m.predict(sc.transform(X[v]))-y[v])**2).sum()
        if best is None or e<best[0]: best=(e,a)
    sc=StandardScaler().fit(X)
    m=Ridge(best[1]).fit(sc.transform(X),y,sample_weight=w)
    return m.predict(sc.transform(Xt))

def nested_cal(Xs,y,reps=3):
    methods=['base','stretch','isotonic','qmap','ext_w','qreg']
    oofs={m:[] for m in methods}
    ks_chosen=[]
    for r in range(reps):
        oo={m:np.zeros(len(y)) for m in methods}
        for fold,(tr,va) in enumerate(KFold(5,shuffle=True,random_state=42+r).split(Xs[0])):
            t1=time.time()
            Xtr=[X[tr] for X in Xs]; Xva=[X[va] for X in Xs]; ytr=y[tr]
            p_va,_=blend_fp(Xtr,ytr,Xva)
            oo['base'][va]=p_va
            p_tr_oof=blend_inner_oof(Xtr,ytr)
            k=stretch_k(p_tr_oof,ytr); ks_chosen.append(k)
            oo['stretch'][va]=apply_stretch(p_va,ytr,k)
            oo['isotonic'][va]=fit_isotonic(p_tr_oof,ytr).predict(p_va)
            oo['qmap'][va]=quantile_map(p_va,p_tr_oof,ytr)
            # extreme-weighted blend of two streams
            def w_inner(X,yy):
                o=np.zeros(len(yy))
                for t,v in KFold(5,shuffle=True,random_state=1).split(X):
                    o[v]=weighted_ridge_fp(X[t],yy[t],X[v],2.0)
                return o
            O=np.column_stack([w_inner(X,ytr) for X in Xtr]); mu=ytr.mean()
            ww,_=nnls(np.column_stack([O-mu]),ytr-mu)
            P=np.column_stack([weighted_ridge_fp(X,ytr,Xt,2.0) for X,Xt in zip(Xtr,Xva)])
            oo['ext_w'][va]=mu+(P-mu)@ww
            # QuantileRegressor on PCA(HX) — median; also blend 0.1/0.9 for interval mid if needed
            pca=PCA(24,random_state=0).fit(Xtr[0]); Z=pca.transform(Xtr[0]); Zva=pca.transform(Xva[0])
            try:
                qr=QuantileRegressor(quantile=0.5,alpha=1.0,solver='highs').fit(Z,ytr)
                oo['qreg'][va]=qr.predict(Zva)
            except Exception:
                oo['qreg'][va]=p_va
            print(f'  rep{r} fold{fold} {time.time()-t1:.0f}s base_fold={rmse(p_va,y[va]):.2f} k={k:.2f}',flush=True)
        for m in methods: oofs[m].append(oo[m])
        print(f'rep{r}',{m:round(rmse(oo[m],y),3) for m in methods},flush=True)
    out={'ks_mean':float(np.mean(ks_chosen))}
    for m in methods:
        arr=np.array(oofs[m]); mean_oof=arr.mean(0)
        rms=[rmse(o,y) for o in arr]
        out[m]={'rmse_mean':float(np.mean(rms)),'rmse_reps':rms,'oof_bias':bias_report(mean_oof,y)}
        np.save(f'cal/oof_{m}.npy',mean_oof)
    return out

# export helpers for cal3
# (module-level functions already defined)

if __name__=='__main__':
    t0=time.time(); Y=get_y()
    SEL=[(L,k) for L in range(4) for k in (1,2)]
    print('loading feats',flush=True)
    Xs=[HX('train'),DINO('train',sel=SEL)]
    print('shapes',[X.shape for X in Xs],flush=True)
    res=nested_cal(Xs,Y,reps=3)
    res['elapsed_s']=time.time()-t0
    json.dump(res,open('cal/cal1_res.json','w'),indent=1)
    print(json.dumps({k:(v if not isinstance(v,dict) else {kk:vv for kk,vv in v.items() if kk!='rmse_reps'}) for k,v in res.items()},indent=1),flush=True)
