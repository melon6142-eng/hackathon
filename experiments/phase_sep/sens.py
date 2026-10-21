import numpy as np,pandas as pd,glob,os,json
from scipy.stats import spearmanr
from sklearn.model_selection import KFold,cross_val_predict
from sklearn.linear_model import RidgeCV; from sklearn.preprocessing import StandardScaler; from sklearn.pipeline import make_pipeline
y=pd.read_csv('~/AI/daker/open/train.csv').hardness.values
B=pd.read_csv('feat_train.csv'); cols=[c for c in B.columns if c!='ID']
rows={}; rm={}
def rr(D):
    p=cross_val_predict(make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-2,4,25))),D[[c for c in cols if c!='x_sig']].values,y,cv=KFold(5,shuffle=True,random_state=7)); return float(np.sqrt(((p-y)**2).mean()))
rm['base']=rr(B)
for f in sorted(glob.glob('sens/*.csv')):
    nm=os.path.basename(f)[:-4]; V=pd.read_csv(f)
    rows[nm]={c:spearmanr(B[c],V[c])[0] for c in cols}; rm[nm]=rr(V)
T=pd.DataFrame(rows); T['min']=T.min(1); T['median']=T.median(1); T.round(3).to_csv('sens_spearman.csv')
json.dump(rm,open('sens_ridge_rmse.json','w'),indent=1); print(T[['min','median']].sort_values('min').round(3).to_string()); print(rm)
