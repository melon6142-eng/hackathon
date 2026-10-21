import os;os.environ['OMP_NUM_THREADS']='1'
import numpy as np,pandas as pd,sys,warnings;warnings.filterwarnings('ignore')
sys.path.insert(0,'./runs');os.chdir('./runs')
from scipy.optimize import nnls;from scipy.stats import spearmanr
from sklearn.model_selection import KFold;from sklearn.linear_model import RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from sklearn.preprocessing import StandardScaler;from sklearn.pipeline import make_pipeline
from joblib import Parallel,delayed
from feats import hx
y=pd.read_csv('./open/train.csv').hardness.values;n=500
h,t,g,r=hx('train');BASE=pd.concat([h.drop(columns='sharp'),g],axis=1).values
G=pd.read_csv('grain_agree/ga_feats.csv').iloc[:500];GA=G.drop(columns='ID');names=list(GA.columns)
def clean(X):
    X=np.asarray(X,float);med=np.nanmedian(X,0);X=np.where(np.isfinite(X),X,med);return X[:,X.std(0)>0]
BASE,GAv=clean(BASE),clean(GA.values);BG=np.column_stack([BASE,GAv])
av=lambda ts:np.mean([np.load(f'dino_ft{x}/oof.npy') for x in ts],0)
B=np.column_stack([np.load('oof_blend1.npy'),av(['/dinoB_orig','2/dinoB_orig_s1','2/dinoB_orig_s2']),av(['/dinoB_trans','2/dinoB_trans_s1','2/dinoB_trans_s2'])])
ridge=lambda:make_pipeline(StandardScaler(),RidgeCV(np.logspace(-1,6,15)))
gbm=lambda:HGB(max_iter=300,learning_rate=.03,max_depth=3,min_samples_leaf=15)
def fit(mk,X,tr,va):
    io=np.zeros(len(tr))
    for a,b in KFold(5,shuffle=True,random_state=1).split(tr): io[b]=mk().fit(X[tr[a]],y[tr[a]]).predict(X[tr[b]])
    return io,mk().fit(X[tr],y[tr]).predict(X[va])
def nn(Atr,Ava,tr):
    w=nnls(np.column_stack([Atr,np.ones(len(Atr))]),y[tr])[0];return np.column_stack([Ava,np.ones(len(Ava))])@w
def one(s):
    P={k:np.zeros(n) for k in ['base','base+GA','GA Ridge','GA HGB','blend','blend+GA Ridge','blend+GA HGB']}
    for tr,va in KFold(5,shuffle=True,random_state=100+s).split(B):
        P['base'][va]=fit(ridge,BASE,tr,va)[1];P['base+GA'][va]=fit(ridge,BG,tr,va)[1]
        ri,rv=fit(ridge,GAv,tr,va);li,lv=fit(gbm,GAv,tr,va);P['GA Ridge'][va]=rv;P['GA HGB'][va]=lv
        P['blend'][va]=nn(B[tr],B[va],tr)
        P['blend+GA Ridge'][va]=nn(np.column_stack([B[tr],ri]),np.column_stack([B[va],rv]),tr)
        P['blend+GA HGB'][va]=nn(np.column_stack([B[tr],li]),np.column_stack([B[va],lv]),tr)
    return P
R=Parallel(8)(delayed(one)(s) for s in range(10))
rm=lambda p:np.sqrt(((p-y)**2).mean());hi=y>230;lo=y<170
L=['## 평가 (5-fold×10회 nCV)','','| 모델 | nCV | std | Δ(기준 대비) | hi>230 편향 | lo<170 편향 |','|---|---|---|---|---|---|']
ref={'base':'base','base+GA':'base','GA Ridge':'base','GA HGB':'base','blend':'blend','blend+GA Ridge':'blend','blend+GA HGB':'blend'}
for k in R[0]:
    a=np.array([rm(P[k]) for P in R]);d=a-np.array([rm(P[ref[k]]) for P in R])
    hb=np.mean([(P[k]-y)[hi].mean() for P in R]);lb=np.mean([(P[k]-y)[lo].mean() for P in R])
    L.append(f'| {k} | {a.mean():.3f} | {a.std():.3f} | {d.mean():+.3f} (±{d.std():.3f}) | {hb:.1f} | {lb:.1f} |')
res=y-np.mean([P['blend'] for P in R],0)
L+=['','## 특징별 신호 (d: y>230 vs 나머지, ρ: y>230 내 Spearman, r_res: 블렌드 잔차와 Spearman)','','| 특징 | d | ρ(y>230) | r_res |','|---|---|---|---|']
X=GA.fillna(GA.median());rows=[]
for c in names:
    x=X[c].values;d=(x[hi].mean()-x[~hi].mean())/np.sqrt((x[hi].var()+x[~hi].var())/2+1e-12)
    rows.append((c,d,spearmanr(x[hi],y[hi])[0],spearmanr(x,res)[0]))
for c,d,p,q in sorted(rows,key=lambda z:-abs(z[1])): L.append(f'| {c} | {d:+.2f} | {p:+.2f} | {q:+.2f} |')
L.append(f'\nn(y>230)={hi.sum()}; |r_res|>0.09 ≈ p<0.05')
open('grain_agree/ev.md','w').write('\n'.join(L));print('\n'.join(L))
