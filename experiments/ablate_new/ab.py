import os;os.environ['OMP_NUM_THREADS']='2'
import warnings;warnings.filterwarnings('ignore')
import sys;sys.path.insert(0,'./runs/lib');sys.path.insert(0,'./runs');os.chdir('./runs')
import numpy as np,pandas as pd,lightgbm as lgb
from sklearn.model_selection import KFold
from scipy.optimize import nnls
from cv import y,nested_ridge,report
from blend import nested_blend
from feats import HX,DINO
Y=y();ids=pd.read_csv('./open/train.csv').ID
NOISE=['sig_raw','noise_lag1','noise_wav','noise_res']
def L(f): d=pd.read_csv(f).set_index('ID'); return d.drop(columns=[c for c in NOISE if c in d])
G=pd.concat([L('gen/'+f) for f in ['gen_feats.csv','acf_feats.csv','bd_feats.csv','morph_feats.csv']],axis=1)
C=pd.read_csv('ablate_new/ctr_feats.csv').set_index('ID')
hand=pd.read_csv('./open/combo_v1/feat/hand_train.csv').set_index('ID').drop(columns=['sharp'])
grain=pd.read_csv('grain/grain_train.csv').set_index('ID').drop(columns=['bnd_density'])
def M(*ds): return pd.concat([d.loc[ids].reset_index(drop=True) for d in ds],axis=1).fillna(0).values
A=M(G,C); A0=M(G); B=M(G,C,hand,grain)
SEL=[(l,k) for l in range(4) for k in (1,2)]; D=DINO('train',sel=SEL); H=HX('train')
def lgbo(X,reps=3):
    O=[]
    for r in range(reps):
        o=np.zeros(len(Y))
        for t,v in KFold(5,shuffle=True,random_state=42+r).split(X):
            o[v]=lgb.LGBMRegressor(n_estimators=600,learning_rate=0.02,num_leaves=15,min_child_samples=10,subsample=0.8,subsample_freq=1,colsample_bytree=0.6,n_jobs=2,verbose=-1).fit(X[t],Y[t]).predict(X[v])
        O.append(o)
    return np.mean(O,0),np.mean([np.sqrt(((o-Y)**2).mean()) for o in O])
def nb(oofs,reps=3):  # nested NNLS blend of given OOF preds
    R=[];mu=Y.mean();P=np.column_stack(oofs);out=[]
    for r in range(reps):
        o=np.zeros(len(Y))
        for t,v in KFold(5,shuffle=True,random_state=100+r).split(P):
            w,_=nnls(P[t]-mu,Y[t]-mu);o[v]=mu+(P[v]-mu)@w
        out.append(o)
    return np.mean(out,0),np.mean([np.sqrt(((o-Y)**2).mean()) for o in out])
bl1=np.load('oof_blend1.npy')
res=[]
def add(n,o,r): s=report(n,o,Y)+f' | repRMSE {r:.3f}';print(s,flush=True);res.append(s)
for n,X in [('A0 gen지표만(노이즈제외)',A0),('A 새지표(gen+대비변환)',A),('B A+hand(형상)+grain',B)]:
    o,r=nested_ridge(X,Y);add(n+' Ridge',o,r); ol,rl=lgbo(X);add(n+' LGB',ol,rl)
    if n[0]=='B': oBr,oBl=o,ol
    if n[0]=='A' and n[1]==' ': oAr,oAl=o,ol
o,r,_,w=nested_blend([B,D],Y);add('C B+DINO Ridge블렌드',o,r)
oC,rC=nb([o,oBl]);add('C B+DINO + B-LGB',oC,rC)
add('D BL1(oof_blend1)',bl1,np.sqrt(((bl1-Y)**2).mean()))
o,r,_,w=nested_blend([H,D,A],Y);add('E BL1구성+새지표블록 Ridge',o,r)
o,r=nb([bl1,oAl]);add('E BL1 + 새지표 LGB',o,r)
o,r=nb([bl1,oBl,oBr]);add('E BL1 + B(LGB,Ridge)',o,r)
open('ablate_new/res.txt','w').write('\n'.join(res))
