import numpy as np, copy
from sklearn.model_selection import KFold
from blend import fit_ridge
from scipy.optimize import nnls
def _tx(spec,Xtr,ytr,Xo):
    if spec is None: return Xtr,Xo
    s=copy.deepcopy(spec).fit(Xtr,ytr); return s.transform(Xtr),s.transform(Xo)
def inner_oof(X,spec,y):
    o=np.zeros(len(y))
    for t,v in KFold(5,shuffle=True,random_state=1).split(X):
        a,b=_tx(spec,X[t],y[t],X[v]); o[v]=fit_ridge(a,y[t]).predict(b)
    return o
def fit_predict(Xs,specs,y,Xte):
    O=np.column_stack([inner_oof(X,s,y) for X,s in zip(Xs,specs)]); mu=y.mean()
    w,_=nnls(O-mu,y-mu) if len(Xs)>1 else (np.ones(1),0)
    P=[]
    for X,s,Xt in zip(Xs,specs,Xte):
        a,b=_tx(s,X,y,Xt); P.append(fit_ridge(a,y).predict(b))
    return mu+(np.column_stack(P)-mu)@w, w
def nested(Xs,specs,y,reps=3):
    oofs=[];W=[]
    for r in range(reps):
        oof=np.zeros(len(y))
        for tr,va in KFold(5,shuffle=True,random_state=42+r).split(Xs[0]):
            p,w=fit_predict([X[tr] for X in Xs],specs,y[tr],[X[va] for X in Xs]); oof[va]=p; W.append(w)
        oofs.append(oof)
    return np.array(oofs).mean(0), np.array([np.sqrt(((o-y)**2).mean()) for o in oofs]), np.mean(W,0)
