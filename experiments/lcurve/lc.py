# Exp E learning curve: fixed outer val folds (5-fold, rs=100+r, 10 reps); train subsampled to frac of the 400 train-fold
import os,sys;os.environ['OMP_NUM_THREADS']='1'
B=os.path.dirname(os.path.abspath(__file__));os.chdir(f'{B}/w/runs');sys.path[:0]=['lib','.']
import numpy as np,pandas as pd
import feats;feats.R=f'{B}/w/open'
from feats import DINO
import hxfold;hxfold.R=feats.R
from hxfold import base,PolyTop
from blend import fit_ridge,inner_oof
from scipy.optimize import nnls
from sklearn.model_selection import KFold
from joblib import Parallel,delayed
y=pd.read_csv(f'{B}/w/open/train.csv').hardness.values
Xr,nb,cand=base('train');Xa=None;Xb=DINO('train',sel=[(L,k) for L in range(4) for k in (1,2)])
FR=[0.2,0.4,0.6,0.8,1.0];R=10
def job(r,f,tr,va,fr):
    rng=np.random.default_rng(1000*r+10*f+int(fr*10));sub=tr if fr==1 else np.sort(rng.choice(tr,int(round(fr*len(tr))),replace=False))
    pt=PolyTop(nb,cand).fit(Xr[sub],y[sub]);Xa=np.zeros((len(y),0));XA=pt.transform(Xr)
    out={}
    for nm,X in (('HX',XA),('DINO',Xb)): out[nm]=fit_ridge(X[sub],y[sub]).predict(X[va])
    O=np.column_stack([inner_oof(X[sub],y[sub]) for X in (XA,Xb)]);mu=y[sub].mean();w,_=nnls(O-mu,y[sub]-mu)
    out['BL1']=mu+(np.column_stack([out['HX'],out['DINO']])-mu)@w
    return r,f,fr,len(sub),va,out
tasks=[(r,f,tr,va,fr) for r in range(R) for f,(tr,va) in enumerate(KFold(5,shuffle=True,random_state=100+r).split(y)) for fr in FR]
res=Parallel(n_jobs=int(os.environ.get('NJ','6')),verbose=5)(delayed(job)(*t) for t in tasks)
P={(m,fr,r):np.zeros(len(y)) for m in ('HX','DINO','BL1') for fr in FR for r in range(R)};N={}
for r,f,fr,n,va,out in res:
    N[fr]=n
    for m,p in out.items(): P[(m,fr,r)][va]=p
rows=[]
for (m,fr,r),p in P.items(): rows.append(dict(model=m,frac=fr,n=N[fr],rep=r,rmse=np.sqrt(((p-y)**2).mean()),hi=(p-y)[y>230].mean(),lo=(p-y)[y<170].mean(),mid=np.sqrt(((p-y)[(y>=170)&(y<=230)]**2).mean())))
D=pd.DataFrame(rows);D.to_csv(f'{B}/lc_raw.csv',index=False)
np.save(f'{B}/lc_preds.npy',{str(k):v for k,v in P.items()},allow_pickle=True)
print(D.groupby(['model','n'])[['rmse','hi','lo','mid']].agg(['mean','std']).round(3).to_string())
