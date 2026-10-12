import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.optimize import nnls
AL=np.logspace(-1,6,15)
def fit_ridge(X,y):
    best=None
    for a in AL:
        e=0
        for t,v in KFold(5,shuffle=True,random_state=0).split(X):
            m=make_pipeline(StandardScaler(),Ridge(a)).fit(X[t],y[t]); e+=((m.predict(X[v])-y[v])**2).sum()
        if best is None or e<best[0]: best=(e,a)
    return make_pipeline(StandardScaler(),Ridge(best[1])).fit(X,y)
def inner_oof(X,y):
    o=np.zeros(len(y))
    for t,v in KFold(5,shuffle=True,random_state=1).split(X): o[v]=fit_ridge(X[t],y[t]).predict(X[v])
    return o
def blend_fit_predict(Xs,y,Xte):
    """Xs list of train feature mats; returns test preds and weights (NNLS + intercept via centering)"""
    O=np.column_stack([inner_oof(X,y) for X in Xs]); mu=y.mean()
    w,_=nnls(np.column_stack([O-mu]),y-mu)
    P=np.column_stack([fit_ridge(X,y).predict(Xt) for X,Xt in zip(Xs,Xte)])
    return mu+(P-mu)@w, w
def nested_blend(Xs,y,reps=3):
    oofs=[];W=[]
    for r in range(reps):
        oof=np.zeros(len(y))
        for tr,va in KFold(5,shuffle=True,random_state=42+r).split(Xs[0]):
            p,w=blend_fit_predict([X[tr] for X in Xs],y[tr],[X[va] for X in Xs]); oof[va]=p; W.append(w)
        oofs.append(oof)
    rm=[np.sqrt(((o-y)**2).mean()) for o in oofs]
    return np.array(oofs).mean(0), np.mean(rm), np.array(rm), np.mean(W,0)
