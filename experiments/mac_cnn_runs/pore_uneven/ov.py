import numpy as np,pandas as pd
from scipy.stats import spearmanr
W='~/AI/daker/cnn/runs/lcurve/w'
D=[pd.read_csv(f'{W}/open/combo_v1/feat/hand_train.csv'),pd.read_csv(f'{W}/open/combo_v1/feat/tex_train.csv'),pd.read_csv(f'{W}/runs/grain/grain_train.csv').drop(columns=['bnd_density']),pd.read_csv(f'{W}/runs/raw/raw_train.csv')]
E0=pd.concat([d.drop(columns='ID') for d in D],axis=1).fillna(0);E0=E0.loc[:,E0.std()>0]
F=pd.read_csv('feat_new.csv').drop(columns='ID');y=pd.read_csv(f'{W}/open/train.csv').hardness.values
from sklearn.linear_model import RidgeCV;from sklearn.model_selection import cross_val_predict;from sklearn.preprocessing import StandardScaler;from sklearn.pipeline import make_pipeline
Er=E0.rank().values;L=['| 새 특징 | ρ(y) | 기존 최대 |ρ| 특징 | ρ | 기존 전체로 선형 설명 R²(CV) |','|---|---|---|---|---|']
for c in F.columns:
    f=F[c].rank().values;r=np.array([np.corrcoef(f,Er[:,j])[0,1] for j in range(Er.shape[1])]);j=np.nanargmax(np.abs(r))
    p=cross_val_predict(make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-2,4,20))),E0.values,F[c].values,cv=5)
    R2=1-((p-F[c])**2).sum()/((F[c]-F[c].mean())**2).sum()
    L.append(f'| {c} | {spearmanr(F[c],y)[0]:+.2f} | {E0.columns[j]} | {r[j]:+.2f} | {R2:.2f} |')
print(E0.shape);open('overlap.md','w').write('\n'.join(L)+'\n');print('\n'.join(L))
