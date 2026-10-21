import os;os.environ['OMP_NUM_THREADS']='2'
import warnings;warnings.filterwarnings('ignore')
import numpy as np,pandas as pd,sys,shap,lightgbm as lgb
sys.path.insert(0,'../lib');from cv import report
from cv_gen import load,oof,blend,y,bl1,rm
for files in [['bd_feats.csv'],['gen_feats.csv','acf_feats.csv','bd_feats.csv','morph_feats.csv']]:
    X,_=load(files)
    for k in ['spline','lgb']:
        O=oof(X,k); print(files,k,report('alone',O.mean(0),y),'| repRMSE %.3f'%np.mean([rm(o) for o in O]),'| blendBL1 %.3f'%blend(O),flush=True)
m=lgb.LGBMRegressor(n_estimators=600,learning_rate=0.02,num_leaves=15,min_child_samples=10,subsample=0.8,subsample_freq=1,colsample_bytree=0.6,n_jobs=2,verbose=-1).fit(X,y)
sv=shap.TreeExplainer(m).shap_values(X); imp=pd.Series(np.abs(sv).mean(0),X.columns).sort_values(ascending=False)
print(imp.head(25).round(3).to_string()); imp.to_csv('shap_importance.csv')
hi=y>230; print('hi-sample mean SHAP top:'); print(pd.Series(sv[hi].mean(0),X.columns).sort_values().head(8).round(2).to_string()); print(pd.Series(sv[hi].mean(0),X.columns).sort_values().tail(8).round(2).to_string())
