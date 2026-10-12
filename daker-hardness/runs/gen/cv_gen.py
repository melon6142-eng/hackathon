import os,sys; os.environ['OMP_NUM_THREADS']='2'
import warnings; warnings.filterwarnings("ignore")
import numpy as np,pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.pipeline import make_pipeline
import lightgbm as lgb
t=pd.read_csv('./open/train.csv'); y=t.hardness.values
bl1=np.load('../oof_blend1.npy')
def load(files):
    d=None
    for f in files:
        x=pd.read_csv(f); d=x if d is None else d.merge(x,on='ID')
    d=d.replace([np.inf,-np.inf],np.nan)
    d=d.fillna(d.median(numeric_only=True))
    tr=t[['ID']].merge(d,on='ID'); return tr.drop(columns='ID'), d
def oof(X,kind,reps=3):
    X=np.asarray(X,float); O=[]
    for r in range(reps):
        o=np.zeros(len(y))
        for a,b in KFold(5,shuffle=True,random_state=42+r).split(X):
            if kind=='ridge': m=make_pipeline(StandardScaler(),RidgeCV(np.logspace(-2,4,20)))
            elif kind=='spline': m=make_pipeline(StandardScaler(),SplineTransformer(5,3),RidgeCV(np.logspace(-1,4,20)))
            else: m=lgb.LGBMRegressor(n_estimators=600,learning_rate=0.02,num_leaves=15,min_child_samples=10,subsample=0.8,subsample_freq=1,colsample_bytree=0.6,n_jobs=2,verbose=-1)
            m.fit(X[a],y[a]); o[b]=m.predict(X[b])
        O.append(o)
    return np.array(O)
def rm(o): return np.sqrt(((o-y)**2).mean())
def blend(O):
    # weight chosen by inner CV over w grid on each outer fold (nested), using rep-avg BL1
    res=[]
    for r,o in enumerate(O):
        p=np.zeros(len(y))
        for a,b in KFold(5,shuffle=True,random_state=7+r).split(o):
            ws=np.linspace(0,1,21); e=[rm_(w*o[a]+(1-w)*bl1[a],y[a]) for w in ws]; w=ws[np.argmin(e)]
            p[b]=w*o[b]+(1-w)*bl1[b]
        res.append(rm(p))
    return np.mean(res)
def rm_(p,yy): return np.sqrt(((p-yy)**2).mean())
if __name__=='__main__':
    files=sys.argv[1:]; X,_=load(files)
    print('BL1',round(rm(bl1),3),'nfeat',X.shape[1])
    for k in ['ridge','spline','lgb']:
        O=oof(X,k); print(k,'alone %.3f'%np.mean([rm(o) for o in O]),'blendBL1 %.3f'%blend(O),flush=True)
        np.save(f'oof_{k}_{len(files)}.npy',O.mean(0))
