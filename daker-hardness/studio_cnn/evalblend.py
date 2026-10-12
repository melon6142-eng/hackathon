import numpy as np,csv,os,sys
from scipy.optimize import nnls
from sklearn.model_selection import KFold
R='~/AI/daker'
y=np.array([float(r['hardness']) for r in csv.DictReader(open(f'{R}/open/train.csv'))])
bl=np.load(f'{R}/cnn/oof_blend1.npy')
def rm(p):return np.sqrt(((p-y)**2).mean())
def bias(p):return (p-y)[y>230].mean(),(p-y)[y<170].mean()
print('BL1',round(rm(bl),3),np.round(bias(bl),1))
tags=[t for t in sorted(os.listdir(f'{R}/cnn/runs')) if os.path.exists(f'{R}/cnn/runs/{t}/oof.npy')]
M={t:np.load(f'{R}/cnn/runs/{t}/oof.npy') for t in tags}
for t,o in M.items(): print(t,round(rm(o),3),np.round(bias(o),1),'corr_res',round(np.corrcoef(o-y,bl-y)[0,1],2))
def cvblend(cols,reps=10):
    X=np.column_stack([bl]+[M[c] for c in cols]+[np.ones(len(y))]);sc=[]
    for r in range(reps):
        o=np.zeros(len(y))
        for tr,va in KFold(5,shuffle=True,random_state=100+r).split(X):
            w,_=nnls(X[tr],y[tr]);o[va]=X[va]@w
        sc.append(rm(o))
    w,_=nnls(X,y);return np.mean(sc),np.std(sc),w
for c in [[t] for t in tags]+([tags] if len(tags)>1 else []):
    m,s,w=cvblend(c);print('BL1+',c,f'{m:.3f}±{s:.3f}',np.round(w,3))
