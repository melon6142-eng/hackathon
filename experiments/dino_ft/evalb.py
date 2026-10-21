import numpy as np,pandas as pd,sys,os
from scipy.optimize import nnls
from sklearn.model_selection import KFold
y=pd.read_csv('./open/train.csv')['hardness'].values if os.path.exists('./open/train.csv') else None
b=np.load('./runs/oof_blend1.npy')
r=lambda p:np.sqrt(((p-y)**2).mean())
def ncv(cols,reps=10):
    X=np.column_stack(cols+[np.ones(len(y))]);out=[]
    for s in range(reps):
        p=np.zeros(len(y))
        for tr,va in KFold(5,shuffle=True,random_state=s).split(X):
            w,_=nnls(X[tr],y[tr]);p[va]=X[va]@w
        out.append(r(p))
    return np.mean(out)
print('BL1',round(r(b),3))
hi=y>230
for d in sys.argv[1:]:
    o=np.load(f'{d}/oof.npy');print(os.path.basename(d),'oof',round(r(o),3),'hi_bias',round((o-y)[hi].mean(),1),'blend_nestedCV',round(ncv([b,o]),3),'w',np.round(nnls(np.column_stack([b,o,np.ones(len(y))]),y)[0],3))
