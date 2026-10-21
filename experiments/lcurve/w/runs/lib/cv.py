import numpy as np, pandas as pd
from sklearn.model_selection import RepeatedKFold, KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
ROOT='./open'
def y(): return pd.read_csv(f'{ROOT}/train.csv').hardness.values
def nested_ridge(X, yv, alphas=np.logspace(-2,5,15), seeds=(42,), reps=3):
    """outer RepeatedKFold 5xreps; inner 5fold alpha select. returns oof mean over reps, rmse"""
    n=len(yv); oofs=[]
    for r in range(reps):
        oof=np.zeros(n)
        for tr,va in KFold(5,shuffle=True,random_state=42+r).split(X):
            best=None
            for a in alphas:
                e=0
                for t2,v2 in KFold(5,shuffle=True,random_state=0).split(tr):
                    m=make_pipeline(StandardScaler(),Ridge(a)).fit(X[tr][t2],yv[tr][t2])
                    e+=((m.predict(X[tr][v2])-yv[tr][v2])**2).sum()
                if best is None or e<best[0]: best=(e,a)
            m=make_pipeline(StandardScaler(),Ridge(best[1])).fit(X[tr],yv[tr]); oof[va]=m.predict(X[va])
        oofs.append(oof)
    oofs=np.array(oofs); rm=np.mean([np.sqrt(((o-yv)**2).mean()) for o in oofs])
    return oofs.mean(0), rm
def report(name,oof,yv):
    r=np.sqrt(((oof-yv)**2).mean()); hi=yv>230; lo=yv<170
    return f'{name}: RMSE {r:.3f} | hi(>230,n={hi.sum()}) bias {np.mean(oof[hi]-yv[hi]):+.1f} | lo(<170,n={lo.sum()}) bias {np.mean(oof[lo]-yv[lo]):+.1f}'
