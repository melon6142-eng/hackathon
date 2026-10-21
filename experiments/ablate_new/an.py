import os;os.environ['OMP_NUM_THREADS']='2'
import warnings;warnings.filterwarnings('ignore')
import sys;sys.path.insert(0,'./runs/lib');sys.path.insert(0,'./runs');os.chdir('./runs')
import numpy as np,pandas as pd
from sklearn.model_selection import KFold,cross_val_predict
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from cv import y
from feats import DINO
Y=y();ids=pd.read_csv('./open/train.csv').ID
NOISE=['sig_raw','noise_lag1','noise_wav','noise_res']
def L(f): d=pd.read_csv(f).set_index('ID'); return d.drop(columns=[c for c in NOISE if c in d])
N=pd.concat([L('gen/'+f) for f in ['gen_feats.csv','acf_feats.csv','bd_feats.csv','morph_feats.csv']]+[pd.read_csv('ablate_new/ctr_feats.csv').set_index('ID')],axis=1).loc[ids].reset_index(drop=True).fillna(0)
N=N.loc[:,N.std()>0]
hand=pd.read_csv('./open/combo_v1/feat/hand_train.csv').set_index('ID').drop(columns=['sharp'])
grain=pd.read_csv('grain/grain_train.csv').set_index('ID').drop(columns=['bnd_density'])
HG=pd.concat([hand,grain],axis=1).loc[ids].fillna(0).values
D=DINO('train',sel=[(l,k) for l in range(4) for k in (1,2)])
Z=StandardScaler().fit_transform(N.values); kf=KFold(5,shuffle=True,random_state=0)
def r2(X,T):
    m=make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-2,5,15),alpha_per_target=True))
    P=cross_val_predict(m,X,T,cv=kf); return 1-((P-T)**2).sum(0)/((T-T.mean(0))**2).sum(0)
out=pd.DataFrame(index=N.columns)
out['r2_handgrain']=r2(HG,Z); out['r2_dino']=r2(D,Z); out['r2_all']=r2(np.hstack([HG,D]),Z)
bl1=np.load('oof_blend1.npy'); res=Y-bl1
out['corr_y']=N.corrwith(pd.Series(Y)); out['corr_resid']=N.corrwith(pd.Series(res))
hi=Y>230; mid=(~hi)&(bl1>=200)&(bl1<=230)
a,b=N[hi],N[mid]; out['d_hi_vs_mid']=(a.mean()-b.mean())/np.sqrt((a.var()+b.var())/2+1e-12)
lab=np.r_[np.ones(hi.sum()),np.zeros(mid.sum())]
out['auc_hi_vs_mid']=[roc_auc_score(lab,np.r_[a[c],b[c]]) for c in N.columns]
out.to_csv('ablate_new/analysis_feats.csv')
# residual CV R2
m=make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-1,6,15)))
rr=[]
for s in range(3):
    P=cross_val_predict(m,N.values,res,cv=KFold(5,shuffle=True,random_state=s)); rr.append(1-((P-res)**2).sum()/((res-res.mean())**2).sum())
import lightgbm as lgb
rl=[]
for s in range(3):
    P=cross_val_predict(lgb.LGBMRegressor(n_estimators=300,learning_rate=0.02,num_leaves=7,min_child_samples=15,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,n_jobs=2,verbose=-1),N.values,res,cv=KFold(5,shuffle=True,random_state=s)); rl.append(1-((P-res)**2).sum()/((res-res.mean())**2).sum())
# permutation null for max |corr_resid|
rng=np.random.default_rng(0); nul=[np.abs(N.corrwith(pd.Series(rng.permutation(res)))).max() for _ in range(200)]
s=[f'n_feat={N.shape[1]}, hi={hi.sum()}, mid(예측200~230,실제<=230)={mid.sum()}',
 f'R² median: hand+grain {out.r2_handgrain.median():.3f}, DINO {out.r2_dino.median():.3f}, 전체 {out.r2_all.median():.3f}; R²_all>0.8 비율 {(out.r2_all>0.8).mean():.2f}, <0.5 비율 {(out.r2_all<0.5).mean():.2f}',
 f'잔차 CV R²: Ridge {np.mean(rr):+.3f}, LGB {np.mean(rl):+.3f}',
 f'max|corr(잔차)| {out.corr_resid.abs().max():.3f} ({out.corr_resid.abs().idxmax()}), 순열 null 95% {np.percentile(nul,95):.3f}',
 f'hi vs mid: max AUC {out.auc_hi_vs_mid.max():.3f}/{out.auc_hi_vs_mid.min():.3f}, |d|>0.8 개수 {(out.d_hi_vs_mid.abs()>0.8).sum()}',
 'hi vs mid 상위(|d|):\n'+out.reindex(out.d_hi_vs_mid.abs().sort_values(ascending=False).index).head(12)[['d_hi_vs_mid','auc_hi_vs_mid','corr_resid','r2_all']].round(3).to_string(),
 '잔차상관 상위:\n'+out.reindex(out.corr_resid.abs().sort_values(ascending=False).index).head(10)[['corr_resid','corr_y','r2_all']].round(3).to_string(),
 '중복 낮은(R²_all 하위) 지표:\n'+out.sort_values('r2_all').head(10)[['r2_all','corr_y','corr_resid']].round(3).to_string()]
open('ablate_new/analysis.txt','w').write('\n\n'.join(s)); print('\n\n'.join(s))
