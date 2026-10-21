import os,sys,glob;os.environ['OMP_NUM_THREADS']='4'
R='~/AI/daker/cnn/runs';I=R+'/indepfold';G=R+'/bag';W=R+'/lcurve/w';B=os.getcwd();os.chdir(f'{W}/runs');sys.path[:0]=['lib','.']
import numpy as np,pandas as pd,feats,hxfold;feats.R=hxfold.R=f'{W}/open'
from hxfold import base;from sklearn.model_selection import KFold
from sklearn.ensemble import HistGradientBoostingRegressor as H
y=pd.read_csv(f'{W}/open/train.csv').hardness.values;hi=y>230
Xr,nb,cand=base('train');print('Xr',Xr.shape)
r_=lambda p:np.sqrt(((p-y)**2).mean())
def nest(X,R=10):
    o=[]
    for s in range(R):
        pp=X.mean(1);o.append(r_(pp))
    return np.mean(o)
rows=[]
for rs in (7,123,2026):
    g=np.zeros(len(y))
    for tr,va in KFold(5,shuffle=True,random_state=rs).split(y):
        g[va]=np.mean([H(max_iter=400,learning_rate=0.03,max_leaf_nodes=15,min_samples_leaf=10,l2_regularization=1.0,random_state=s).fit(Xr[tr],y[tr]).predict(Xr[va]) for s in range(3)],0)
    np.save(f'{B}/gbm_rs{rs}.npy',g)
    b=np.load(f'{I}/bl1_rs{rs}.npy')
    ld=lambda c:[np.load(f) for f in sorted(glob.glob(f'{I}/rs{rs}_{c}_s*/oof.npy'))]
    X3=np.column_stack([b,np.mean(ld('orig'),0),np.mean(ld('trans'),0)]);X4=np.column_stack([X3,g])
    rows.append((rs,r_(g),np.corrcoef(g-y,X3.mean(1)-y)[0,1],r_(X3.mean(1)),r_(X4.mean(1)),r_(0.85*X3.mean(1)+0.15*g)))
d=pd.DataFrame(rows,columns=['rs','gbm_solo','res_corr','eq3','eq4','eq3+.15gbm']);print(d.round(3).to_string());d.to_csv(f'{B}/gbm_res.csv',index=False)
