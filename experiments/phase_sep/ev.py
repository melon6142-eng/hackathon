import os; os.environ['OMP_NUM_THREADS']='1'
import numpy as np,pandas as pd,json,warnings; warnings.filterwarnings('ignore')
from scipy.optimize import nnls; from scipy.stats import spearmanr
from sklearn.model_selection import KFold; from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler; from sklearn.pipeline import make_pipeline
from joblib import Parallel,delayed
import lightgbm as lgb
from interpret.glassbox import ExplainableBoostingRegressor
R='~/AI/daker'; C=f'{R}/cnn/runs'
y=pd.read_csv(f'{R}/open/train.csv').hardness.values; hi=y>230; lo=y<170; n=len(y)
F=pd.read_csv('feat_train.csv'); assert (F.ID==pd.read_csv(f'{R}/open/train.csv').ID).all()
cols=[c for c in F.columns if c not in ('ID','x_sig')]; X=F[cols].values.astype(float)
r_=lambda p,t=y: float(np.sqrt(((p-t)**2).mean()))
def mk(kind,seed=0):
    if kind=='ridge': return make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-2,4,25)))
    if kind=='lgbm': return lgb.LGBMRegressor(n_estimators=400,learning_rate=0.03,num_leaves=15,min_child_samples=10,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,reg_lambda=1.0,random_state=seed,verbose=-1,n_jobs=1)
    if kind=='ebm': return ExplainableBoostingRegressor(interactions=10,outer_bags=8,random_state=seed,n_jobs=1)
def fit1(kind,Xm,t,tr,va,seed):
    m=mk(kind,seed).fit(Xm[tr],t[tr]); out=(va,m.predict(Xm[va]))
    if kind=='ebm': out=out+({nm:float(v) for nm,v in zip(m.term_names_,m.term_importances())},)
    return out
def oof(kind,rs,Xm=X,t=y):
    folds=list(KFold(5,shuffle=True,random_state=rs).split(Xm))
    res=Parallel(n_jobs=5)(delayed(fit1)(kind,Xm,t,tr,va,rs) for tr,va in folds)
    o=np.zeros(len(t)); imps=[]
    for r in res: o[r[0]]=r[1]; imps+= [r[2]] if len(r)>2 else []
    return o,imps
A1=lambda Z:np.column_stack([Z,np.ones(len(Z))])
def ncv(Z,Rp=10):
    o=[]
    for s in range(Rp):
        pp=np.zeros(n)
        for tr,va in KFold(5,shuffle=True,random_state=100+s).split(Z): pp[va]=A1(Z[va])@nnls(A1(Z[tr]),y[tr])[0]
        o.append((r_(pp),(pp-y)[hi].mean()))
    return np.array(o)
def same_fold(Z,rs):
    pp=np.zeros(n)
    for tr,va in KFold(5,shuffle=True,random_state=rs).split(Z): pp[va]=A1(Z[va])@nnls(A1(Z[tr]),y[tr])[0]
    return pp
ld=lambda t: np.load(f'{C}/{t}/oof.npy')
av=lambda ts: np.mean([ld(t) for t in ts],0)
BASE={42:np.column_stack([np.load(f'{R}/cnn/oof_blend1.npy'),av(['dinoB_orig','dinoB_orig_s1','dinoB_orig_s2']),av(['dinoB_trans','dinoB_trans_s1','dinoB_trans_s2'])])}
for rs in (7,123,2026): BASE[rs]=np.load(f'{C}/indepfold/blendX_rs{rs}.npy')
OUT={}; L=[]; IMPS={}
for rs in (42,7,123,2026):
    B=BASE[rs]; nb=ncv(B); bp=same_fold(B,rs); res=bp-y
    row={'rs':rs,'base_ncv':nb[:,0].mean(),'base_sd':nb[:,0].std(),'base_hi':nb[:,1].mean(),'base_same':r_(bp)}
    for kind in ('ridge','lgbm','ebm'):
        o,imps=oof(kind,rs); np.save(f'oof_{kind}_rs{rs}.npy',o)
        if imps: IMPS[rs]=imps
        nn=ncv(np.column_stack([B,o])); d=nn[:,0]-nb[:,0]
        sp=same_fold(np.column_stack([B,o]),rs)
        row[kind]=dict(solo=r_(o),solo_hi=float((o-y)[hi].mean()),solo_lo=float((o-y)[lo].mean()),corr_blend=float(np.corrcoef(o,bp)[0,1]),
                       ncv=nn[:,0].mean(),d_ncv=d.mean(),n_better=int((d<0).sum()),hi=nn[:,1].mean(),same=r_(sp),d_same=r_(sp)-r_(bp),
                       w=float(nnls(A1(np.column_stack([B,o])),y)[0][3]))
        print(rs,kind,row[kind],flush=True)
    # 잔차 설명: fold 내에서 특징 -> 블렌드 잔차 (Ridge, EBM)
    for kind in ('ridge','ebm'):
        ro,_=oof(kind,rs,X,res)
        row['res_'+kind]=dict(R2=float(1-((ro-res)**2).sum()/((res-res.mean())**2).sum()),corr=float(np.corrcoef(ro,res)[0,1]),
                              corr_hi=float(np.corrcoef(ro[hi],res[hi])[0,1]))
        print(rs,'res',kind,row['res_'+kind],flush=True)
    OUT[rs]=row
json.dump(OUT,open('ev.json','w'),indent=1,default=float)
json.dump({str(k):v for k,v in IMPS.items()},open('ebm_imps.json','w'))
# 단독 nCV 10x5 (rs 100..109)
S={}
for kind in ('ridge','lgbm','ebm'):
    S[kind]=[r_(oof(kind,100+s)[0]) for s in range(10)]; print('ncv10',kind,np.mean(S[kind]),np.std(S[kind]),flush=True)
json.dump(S,open('solo_ncv.json','w'))
