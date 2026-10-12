"""Phenotype clustering + inverse + pairwise. Nested CV. OMP=2."""
import os; os.environ['OMP_NUM_THREADS']='2'; os.environ['OPENBLAS_NUM_THREADS']='2'
import sys; sys.path.insert(0,'lib'); sys.path.insert(0,'.')
import numpy as np, json, time, pandas as pd
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import NearestNeighbors
from blend import fit_ridge
from feats import HX, DINO, hx
from cv import y as get_y
from scipy.optimize import nnls

AL=np.logspace(0,5,8)

def rmse(a,b): return float(np.sqrt(((a-b)**2).mean()))
def bias_report(p,yv):
    hi=yv>230; lo=yv<170
    return {'rmse':rmse(p,yv),
            'hi_bias':float(np.mean(p[hi]-yv[hi])) if hi.any() else None,
            'lo_bias':float(np.mean(p[lo]-yv[lo])) if lo.any() else None}

def phenotype_feats(split='train'):
    h,_,_,_=hx(split)
    g=pd.read_csv(f'grain/grain_{split}.csv').drop(columns=['ID','bnd_density'],errors='ignore').fillna(0)
    cols=[]
    for c in ['dark0.5','dark1.0','dark1.5','dark2.0','pore','coh','x_along','x_across','x_ratio','sharp','ncc','ccmean']:
        if c in h.columns: cols.append(h[c].values)
    for c in ['inv_sqrt_d','A_mean','A_med','ori_align','ng','otsu_frac','pore_frac','ecc_mean']:
        if c in g.columns: cols.append(g[c].values)
    return np.column_stack(cols)

def cluster_fp(Ztr,Zva,k,method='kmeans'):
    sc=StandardScaler().fit(Ztr); A=sc.transform(Ztr); B=sc.transform(Zva)
    if method=='kmeans':
        km=KMeans(k,random_state=0,n_init=10).fit(A)
        return km.labels_, km.predict(B)
    gm=GaussianMixture(k,random_state=0,covariance_type='diag',max_iter=200).fit(A)
    return gm.predict(A), gm.predict(B)

def fit_ridge_fast(X,y):
    best=None
    for a in AL:
        e=0
        for t,v in KFold(min(5,max(2,len(X)//8)),shuffle=True,random_state=0).split(X):
            m=make_pipeline(StandardScaler(),Ridge(a)).fit(X[t],y[t])
            e+=((m.predict(X[v])-y[v])**2).sum()
        if best is None or e<best[0]: best=(e,a)
    return make_pipeline(StandardScaler(),Ridge(best[1])).fit(X,y)

def moe_predict(Xtr,ytr,Xva,lab_tr,lab_va,k):
    global_m=fit_ridge_fast(Xtr,ytr)
    p=np.zeros(len(Xva))
    for c in range(k):
        mtr=lab_tr==c; mva=lab_va==c
        if not mva.any(): continue
        if mtr.sum()<20:
            p[mva]=global_m.predict(Xva[mva]); continue
        p[mva]=fit_ridge_fast(Xtr[mtr],ytr[mtr]).predict(Xva[mva])
    return p

def inverse_hardness(Ftr,ytr,Fva,lab_tr=None,lab_va=None,k=1,n_feat=12):
    corr=np.array([abs(np.corrcoef(Ftr[:,j],ytr)[0,1]) if Ftr[:,j].std()>0 else 0 for j in range(Ftr.shape[1])])
    top=np.argsort(-np.nan_to_num(corr))[:n_feat]
    Ftr=Ftr[:,top]; Fva=Fva[:,top]
    def fit_params(F,y):
        a=np.zeros(F.shape[1]); b=np.zeros(F.shape[1]); sig=np.zeros(F.shape[1])
        for j in range(F.shape[1]):
            lr=LinearRegression().fit(y.reshape(-1,1),F[:,j])
            a[j]=lr.intercept_; b[j]=lr.coef_[0]
            r=F[:,j]-(a[j]+b[j]*y); sig[j]=max(r.std(),1e-6)
        return a,b,sig
    def mle_h(f,a,b,sig,h_lo,h_hi):
        den=((b/sig)**2).sum()
        if den<1e-12: return 0.5*(h_lo+h_hi)
        h=((b/sig**2)*(f-a)).sum()/den
        return float(np.clip(h,h_lo,h_hi))
    h_lo,h_hi=float(ytr.min()-5),float(ytr.max()+5)
    p=np.zeros(len(Fva))
    if lab_tr is None:
        a,b,sig=fit_params(Ftr,ytr)
        for i in range(len(Fva)): p[i]=mle_h(Fva[i],a,b,sig,h_lo,h_hi)
    else:
        ag,bg,sg=fit_params(Ftr,ytr)
        for c in range(k):
            mtr=lab_tr==c; mva=lab_va==c
            if not mva.any(): continue
            a,b,sig = fit_params(Ftr[mtr],ytr[mtr]) if mtr.sum()>=20 else (ag,bg,sg)
            for i in np.where(mva)[0]: p[i]=mle_h(Fva[i],a,b,sig,h_lo,h_hi)
    return p

def hierarchical(Ftr,ytr,Fva,lab_tr,lab_va,k):
    pca=PCA(min(20,Ftr.shape[1]),random_state=0)
    sc=StandardScaler().fit(Ftr)
    Ztr=pca.fit_transform(sc.transform(Ftr)); Zva=pca.transform(sc.transform(Fva))
    def add_c(F,lab):
        D=np.zeros((len(F),max(k-1,1)))
        for c in range(k-1): D[lab==c,c]=1
        return np.hstack([F,D])
    return fit_ridge_fast(add_c(Ztr,lab_tr),ytr).predict(add_c(Zva,lab_va))

def pairwise_knn(Ftr,ytr,Fva,n_neighbors=15):
    sc=StandardScaler().fit(Ftr); A=sc.transform(Ftr); B=sc.transform(Fva)
    rng=np.random.default_rng(0); n=len(ytr)
    n_pairs=min(6000,n*(n-1)//2)
    i1=rng.integers(0,n,n_pairs); i2=rng.integers(0,n,n_pairs); m=i1!=i2; i1,i2=i1[m],i2[m]
    dF=A[i1]-A[i2]; dY=ytr[i1]-ytr[i2]
    model=fit_ridge_fast(dF,dY)
    nn=NearestNeighbors(n_neighbors=n_neighbors).fit(A)
    _,idx=nn.kneighbors(B)
    preds=np.array([np.mean(ytr[refs]+model.predict(B[i]-A[refs])) for i,refs in enumerate(idx)])
    return preds

def nested(reps=3):
    Y=get_y()
    SEL=[(L,k) for L in range(4) for k in (1,2)]
    H=HX('train'); D=DINO('train',sel=SEL)
    Z_hand=phenotype_feats('train')
    scD=StandardScaler().fit(D); Z_dino=PCA(32,random_state=0).fit_transform(scD.transform(D))
    run_cfgs=[
        ('hand_km',3,'kmeans',Z_hand),('hand_km',4,'kmeans',Z_hand),('hand_km',5,'kmeans',Z_hand),
        ('hand_km',6,'kmeans',Z_hand),('hand_km',8,'kmeans',Z_hand),
        ('hand_gm',4,'gmm',Z_hand),('hand_gm',5,'gmm',Z_hand),
        ('dino_km',4,'kmeans',Z_dino),('dino_km',5,'kmeans',Z_dino),('dino_km',6,'kmeans',Z_dino),
    ]
    results={}
    methods=['moe_H','moe_D','moe_HD','inv','hier','pair']
    for name,k,meth,Z in run_cfgs:
        print(f'=== {name} k={k} ===',flush=True)
        oofs={m:[] for m in methods}
        for r in range(reps):
            oo={m:np.zeros(len(Y)) for m in methods}
            for tr,va in KFold(5,shuffle=True,random_state=42+r).split(Y):
                lab_tr,lab_va=cluster_fp(Z[tr],Z[va],k,meth)
                pH=moe_predict(H[tr],Y[tr],H[va],lab_tr,lab_va,k)
                pD=moe_predict(D[tr],Y[tr],D[va],lab_tr,lab_va,k)
                oo['moe_H'][va]=pH; oo['moe_D'][va]=pD
                # fold-internal NNLS weights from tiny split of tr
                # use simple equal-ish: fit weights via ridge on train preds from cluster models
                # approximate: weight by inverse MSE on tr via leave-one-cluster... use fixed from quick OOF 2-fold
                oH=np.zeros(len(tr)); oD=np.zeros(len(tr))
                for t2,v2 in KFold(3,shuffle=True,random_state=1).split(tr):
                    lt,lv=cluster_fp(Z[tr][t2],Z[tr][v2],k,meth)
                    oH[v2]=moe_predict(H[tr][t2],Y[tr][t2],H[tr][v2],lt,lv,k)
                    oD[v2]=moe_predict(D[tr][t2],Y[tr][t2],D[tr][v2],lt,lv,k)
                mu=Y[tr].mean(); w,_=nnls(np.column_stack([oH-mu,oD-mu]),Y[tr]-mu)
                if w.sum()<1e-8: w=np.array([0.4,0.6])
                oo['moe_HD'][va]=mu+np.column_stack([pH-mu,pD-mu])@w
                oo['inv'][va]=inverse_hardness(Z[tr],Y[tr],Z[va],lab_tr,lab_va,k)
                oo['hier'][va]=hierarchical(H[tr],Y[tr],H[va],lab_tr,lab_va,k)
                oo['pair'][va]=pairwise_knn(Z[tr],Y[tr],Z[va],15)
            for m in methods: oofs[m].append(oo[m])
            print(f'  rep{r}',{m:round(rmse(oo[m],Y),3) for m in methods},flush=True)
        results[f'{name}_k{k}']={}
        for m in methods:
            arr=np.array(oofs[m]); mean_oof=arr.mean(0)
            rms=[rmse(o,Y) for o in arr]
            # per-cluster RMSE diagnostic (clusters on full Z)
            lab,_=cluster_fp(Z,Z,k,meth)
            crmse={int(c):rmse(mean_oof[lab==c],Y[lab==c]) for c in range(k) if (lab==c).any()}
            results[f'{name}_k{k}'][m]={
                'rmse_mean':float(np.mean(rms)),'rmse_reps':[float(x) for x in rms],
                'bias':bias_report(mean_oof,Y),'cluster_rmse':crmse}
            np.save(f'cal/oof_{name}_k{k}_{m}.npy',mean_oof)
    print('=== global inv/pair ===',flush=True)
    for tag,Z in [('hand',Z_hand),('dino',Z_dino)]:
        o_inv=[]; o_pair=[]
        for r in range(reps):
            oi=np.zeros(len(Y)); op=np.zeros(len(Y))
            for tr,va in KFold(5,shuffle=True,random_state=42+r).split(Y):
                oi[va]=inverse_hardness(Z[tr],Y[tr],Z[va])
                op[va]=pairwise_knn(Z[tr],Y[tr],Z[va])
            o_inv.append(oi); o_pair.append(op)
            print(f'  {tag} rep{r}',round(rmse(oi,Y),3),round(rmse(op,Y),3),flush=True)
        results[f'global_{tag}']={
            'inv':{'rmse_mean':float(np.mean([rmse(o,Y) for o in o_inv])),'bias':bias_report(np.mean(o_inv,0),Y)},
            'pair':{'rmse_mean':float(np.mean([rmse(o,Y) for o in o_pair])),'bias':bias_report(np.mean(o_pair,0),Y)}}
        np.save(f'cal/oof_global_{tag}_inv.npy',np.mean(o_inv,0))
        np.save(f'cal/oof_global_{tag}_pair.npy',np.mean(o_pair,0))
    return results

if __name__=='__main__':
    t0=time.time()
    res=nested(reps=3)
    res['_elapsed_s']=time.time()-t0
    json.dump(res,open('cal/cal2_res.json','w'),indent=1,default=float)
    print('DONE',time.time()-t0,flush=True)
