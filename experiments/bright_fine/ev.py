import os;os.environ['OMP_NUM_THREADS']='1'
import numpy as np,pandas as pd,sys,warnings;warnings.filterwarnings('ignore')
sys.path.insert(0,'./runs');os.chdir('./runs')
from scipy.optimize import nnls
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from joblib import Parallel,delayed
from feats import hx
O='./open';y=pd.read_csv(f'{O}/train.csv').hardness.values;n=500
h,t,g,r=hx('train');base=pd.concat([h.drop(columns='sharp'),g],axis=1).values
st=pd.read_csv('stage_ablate/ratio_feats.csv').iloc[:500]
st4=st[[c for c in st.columns if c[:2] in('ia','in') and c[2:].isdigit()]].values
bf=pd.read_csv('bright_fine/bf_feats.csv');tr_=bf.iloc[:500]
R=tr_[[c for c in bf.columns if c.startswith('R_')]].values;A=tr_[[c for c in bf.columns if c.startswith('A_')]].values
TF={m:pd.read_csv(f'bright_fine/bf_feats_{m}.csv').iloc[:500].drop(columns='ID').values for m in ['clahe','stretch','gamma']}
def cl(X):
    X=np.asarray(X,float);X=np.where(np.isfinite(X),X,np.nan);X=np.where(np.isnan(X),np.nan_to_num(np.nanmedian(X,0)),X);return X[:,X.std(0)>0]
F={'base':cl(base),'+4단계':cl(np.hstack([base,st4])),'+상대':cl(np.hstack([base,R])),'+절대':cl(np.hstack([base,A])),'+상대+절대':cl(np.hstack([base,R,A])),'+4단계+상대+절대':cl(np.hstack([base,st4,R,A])),**{f'+{m}':cl(np.hstack([base,v])) for m,v in TF.items()},'+원본+3변환':cl(np.hstack([base,R,A]+list(TF.values())))}
av=lambda ts:np.mean([np.load(f'dino_ft{x}/oof.npy') for x in ts],0)
C=np.column_stack([np.load('oof_blend1.npy'),av(['/dinoB_orig','2/dinoB_orig_s1','2/dinoB_orig_s2']),av(['/dinoB_trans','2/dinoB_trans_s1','2/dinoB_trans_s2'])])
def mk(k):return make_pipeline(StandardScaler(),RidgeCV(np.logspace(-1,6,15))) if k=='ridge' else HGB(max_iter=300,learning_rate=.03,max_depth=3,min_samples_leaf=15)
def nn(Atr,Ava,tr):
    w=nnls(np.column_stack([Atr,np.ones(len(Atr))]),y[tr])[0];return np.column_stack([Ava,np.ones(len(Ava))])@w
def one(s):
    P={}
    def put(k,va,p):P.setdefault(k,np.zeros(n))[va]=p
    for tr,va in KFold(5,shuffle=True,random_state=100+s).split(np.zeros(n)):
        put('B|최고블렌드',va,nn(C[tr],C[va],tr))
        for fk,X in F.items():
            for m in ['ridge','lgb']:
                io=np.zeros(len(tr))
                for a,b in KFold(5,shuffle=True,random_state=1).split(tr):io[b]=mk(m).fit(X[tr[a]],y[tr[a]]).predict(X[tr[b]])
                p=mk(m).fit(X[tr],y[tr]).predict(X[va]);put(f'A|{m}|{fk}',va,p)
                put(f'B|+{m}{fk}',va,nn(np.column_stack([C[tr],io]),np.column_stack([C[va],p]),tr))
    return P
Ps=Parallel(8)(delayed(one)(s) for s in range(10))
rm=lambda p:np.sqrt(((p-y)**2).mean());hi=y>230;lo=y<170;rows=[]
for k in Ps[0]:
    e=[rm(P[k]) for P in Ps];rows.append((k,np.mean(e),np.std(e),np.mean([(P[k]-y)[hi].mean() for P in Ps]),np.mean([(P[k]-y)[lo].mean() for P in Ps])))
D=pd.DataFrame(rows,columns=['cfg','nCV','sd','hi_bias','lo_bias']);D.to_csv('bright_fine/ev.csv',index=False);print(D.round(3).to_string())
# 고경도 vs 대조 효과크기
X=tr_.drop(columns='ID');ctl=(y>=170)&(y<=230)
es=((X[hi].mean()-X[ctl].mean())/np.sqrt((X[hi].var()+X[ctl].var())/2)).dropna()
top=es.abs().sort_values(ascending=False).head(10)
T=pd.DataFrame({'feat':top.index,'type':['상대' if c.startswith('R_') else '절대' for c in top.index],'d':es[top.index].values,'hi_mean':X[hi][top.index].mean().values,'ctl_mean':X[ctl][top.index].mean().values})
T.to_csv('bright_fine/top10.csv',index=False);print(T.round(4).to_string())

L=[]
for m in ['orig','clahe','stretch','gamma']:
    X=pd.read_csv('bright_fine/bf_feats.csv' if m=='orig' else f'bright_fine/bf_feats_{m}.csv').iloc[:500].drop(columns='ID')
    es=((X[hi].mean()-X[ctl].mean())/np.sqrt((X[hi].var()+X[ctl].var())/2)).dropna()
    for c in es.abs().sort_values(ascending=False).head(5).index:L.append((m,c,'상대' if c.startswith('R_') else '절대',es[c]))
pd.DataFrame(L,columns=['tf','feat','type','d']).to_csv('bright_fine/top_by_tf.csv',index=False);print(pd.DataFrame(L).round(3).to_string())
