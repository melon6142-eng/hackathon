import os,sys,json;os.environ['OMP_NUM_THREADS']='1'
H=os.path.dirname(os.path.abspath(__file__));C='~/AI/daker/cnn/runs';W=f'{C}/lcurve/w';os.chdir(f'{W}/runs');sys.path[:0]=['lib','.']
import numpy as np,pandas as pd,feats,hxfold;feats.R=hxfold.R=f'{W}/open'
from feats import DINO;from hxfold import base,PolyTop;from blend import fit_ridge,inner_oof
from scipy.optimize import nnls;from sklearn.model_selection import KFold;from joblib import Parallel,delayed
from sklearn.linear_model import RidgeCV;from sklearn.preprocessing import StandardScaler;from sklearn.pipeline import make_pipeline
y=pd.read_csv(f'{W}/open/train.csv').hardness.values;hi=y>230;n=len(y)
F=pd.read_csv(f'{H}/feat_new.csv');assert (F.ID.values==pd.read_csv(f'{W}/open/train.csv').ID.values).all()
GR={'P_basic':['P_frac','P_n','P_logn','P_Amean','P_Amed','P_Aq90','P_Acv','P_Amax','P_eqd'],
    'P_clust':['P_CE','P_nn_cv','P_vmr32','P_vmr64','P_fsd32','P_fsd64'],
    'G_uneven':[c for c in F.columns if c.startswith('G_')]}
GR['ALL']=sum(GR.values(),[])
Xr,nb,cand=base('train');Xb=DINO('train',sel=[(L,k) for L in range(4) for k in (1,2)])
r_=lambda p:float(np.sqrt(((p-y)**2).mean()))
def bl1_fold(XR,NB,CD,tr,va):
    pt=PolyTop(NB,CD).fit(XR[tr],y[tr]);XA=pt.transform(XR)
    O=np.column_stack([inner_oof(X[tr],y[tr]) for X in (XA,Xb)]);mu=y[tr].mean();w,_=nnls(O-mu,y[tr]-mu)
    P=np.column_stack([fit_ridge(X[tr],y[tr]).predict(X[va]) for X in (XA,Xb)]);return va,mu+(P-mu)@w
def bl1(rs,g):
    f=f'{H}/bl1_rs{rs}_{g}.npy'
    if os.path.exists(f): return np.load(f)
    if g=='none': XR,NB,CD=Xr,nb,cand
    else:
        Z=F[GR[g]].values.astype(float);k=Z.shape[1];XR=np.hstack([Xr[:,:nb],Z,Xr[:,nb:]]);NB=nb+k
        CD=[c if c<nb else c+k for c in cand]+list(range(nb,nb+k))
    o=np.zeros(n)
    for va,p in Parallel(n_jobs=5)(delayed(bl1_fold)(XR,NB,CD,tr,va) for tr,va in KFold(5,shuffle=True,random_state=rs).split(y)): o[va]=p
    np.save(f,o);return o
ld=lambda t: np.load(f'{C}/{t}/oof.npy');av=lambda ts: np.mean([ld(t) for t in ts],0)
BASE={42:np.column_stack([np.load('~/AI/daker/cnn/oof_blend1.npy'),av(['dinoB_orig','dinoB_orig_s1','dinoB_orig_s2']),av(['dinoB_trans','dinoB_trans_s1','dinoB_trans_s2'])])}
for rs in (7,123,2026): BASE[rs]=np.load(f'{C}/indepfold/blendX_rs{rs}.npy')
A1=lambda Z:np.column_stack([Z,np.ones(len(Z))])
def ncv(B,Z=None):  # Z: fold 내 잔차 Ridge 특징
    out=[]
    for s in range(10):
        pp=np.zeros(n)
        for tr,va in KFold(5,shuffle=True,random_state=100+s).split(B):
            w=nnls(A1(B[tr]),y[tr])[0];pp[va]=A1(B[va])@w
            if Z is not None:
                res=y[tr]-A1(B[tr])@w;m=make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-1,5,25))).fit(Z[tr],res);pp[va]+=m.predict(Z[va])
        out.append((r_(pp),(pp-y)[hi].mean()))
    return np.array(out)
OUT={}
for rs in (42,7,123,2026):
    B=BASE[rs];b0=ncv(B);row={'base':[b0[:,0].mean(),b0[:,1].mean()]}
    bl0=bl1(rs,'none');B0=B.copy();B0[:,0]=bl0;c0=ncv(B0);row['bl1_none']=[r_(bl0),c0[:,0].mean(),c0[:,1].mean()]
    for g in GR:
        Z=F[GR[g]].values.astype(float);r=ncv(B,Z);d=r[:,0]-b0[:,0]
        row['res_'+g]=[r[:,0].mean(),d.mean(),int((d<0).sum()),r[:,1].mean()]
        bg=bl1(rs,g);Bg=B.copy();Bg[:,0]=bg;c=ncv(Bg);d=c[:,0]-c0[:,0]
        row['bl1_'+g]=[r_(bg),r_(bg)-r_(bl0),c[:,0].mean(),d.mean(),int((d<0).sum()),c[:,1].mean()]
        print(rs,g,row['res_'+g],row['bl1_'+g],flush=True)
    OUT[rs]=row;json.dump(OUT,open(f'{H}/ev.json','w'),indent=1)
print('DONE')
