import os;os.environ['OMP_NUM_THREADS']='2'
import numpy as np,pandas as pd,warnings;warnings.filterwarnings('ignore')
from scipy.optimize import nnls
from sklearn.metrics import roc_auc_score as auc
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.linear_model import LogisticRegression,RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
H='./runs/hi_rule'
y=pd.read_csv('./open/train.csv').hardness.values
F=pd.read_csv('dm_feats.csv').set_index('ID');F=F.replace([np.inf,-np.inf],np.nan);F=F.fillna(F.median())
Ftr=F.loc[F.index.str.startswith('TRAIN')].sort_index();Fte=F.loc[F.index.str.startswith('TEST')].sort_index()
Xold=pd.read_pickle(f'{H}/X_train.pkl');Xold.index=Ftr.index
C=np.load(f'{H}/comp_oof.npy');one=lambda M:np.column_stack([M,np.ones(len(M))])
base=one(C)@nnls(one(C),y)[0];res=y-base;r=lambda p:np.sqrt(((p-y)**2).mean())
zone=(base>=195)&(base<=235);lab=(y>230).astype(int)
sets={'DM전체':Ftr,'DM이미지별':Ftr.filter(regex='^i'),'DM전역':Ftr.filter(regex='^g'),'기존특징':Xold,'기존+DM':pd.concat([Xold,Ftr],axis=1)}
for s in ['S1_','S2_','S12_','rel_']:sets['DM '+s[:-1]+'만']=Ftr.filter(regex=s)
L=[]
u=pd.Series({c:auc(lab[zone],Ftr[c].values[zone]) for c in Ftr.columns});uu=(u-.5).abs().sort_values(ascending=False)
L.append('[구간195-235 단변량 AUC top15 (in-sample)]');L+=[f'{c}: {u[c]:.3f}' for c in uu.index[:15]]
uo=pd.Series({c:auc(lab[zone],Xold[c].values[zone]) for c in Xold.columns});L.append(f'(기존특징 최대 단변량 |AUC-0.5|+0.5: {((uo-.5).abs().max()+.5):.3f})')
cr=pd.Series({c:np.corrcoef(Ftr[c],res)[0,1] for c in Ftr.columns});L.append('\n[잔차 상관 top10 (in-sample, |r|; 500장 우연 기준 ~0.09)]');L+=[f'{c}: {cr[c]:+.3f}' for c in cr.abs().sort_values(ascending=False).index[:10]]
L.append('\n| 특징셋 | 구간 hi AUC(nested, 5×5) | 잔차 R²(nested ridge) | 잔차보정 nCV | 기준 |');L.append('|---|---|---|---|---|')
idx=np.where(zone)[0]
for nm,X in sets.items():
    X=X.values;A=[]
    for s in range(5):
        p=np.zeros(len(idx))
        for tr,va in StratifiedKFold(5,shuffle=True,random_state=s).split(idx,lab[idx]):
            m=make_pipeline(StandardScaler(),LogisticRegression(penalty='l1',C=.1,solver='liblinear',class_weight='balanced')).fit(X[idx[tr]],lab[idx[tr]]);p[va]=m.predict_proba(X[idx[va]])[:,1]
        A.append(auc(lab[idx],p))
    R2=[];RM=[];B=[]
    for s in range(5):
        p=np.zeros(500);pb=np.zeros(500)
        for tr,va in KFold(5,shuffle=True,random_state=100+s).split(X):
            w=nnls(one(C[tr]),y[tr])[0];bt=one(C[tr])@w;bv=one(C[va])@w
            m=make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(0,5,20))).fit(X[tr],y[tr]-bt)
            p[va]=bv+m.predict(X[va]);pb[va]=bv
        RM.append(r(p));B.append(r(pb));R2.append(1-((y-p)**2).sum()/((y-pb)**2).sum())
    L.append(f'| {nm} | {np.mean(A):.3f}±{np.std(A):.3f} | {np.mean(R2):+.3f} | {np.mean(RM):.3f}±{np.std(RM):.3f} | {np.mean(B):.3f} |');print(L[-1],flush=True)
open('ev.md','w').write('\n'.join(L));print('\n'.join(L))
