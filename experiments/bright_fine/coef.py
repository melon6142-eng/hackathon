import os,sys;sys.path.insert(0,'./runs');os.chdir('./runs')
import numpy as np,pandas as pd,matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from feats import hx
y=pd.read_csv('./open/train.csv').hardness.values
h,t,g,r=hx('train');base=pd.concat([h.drop(columns='sharp'),g],axis=1)
base=base.fillna(base.median());base=base.loc[:,base.std()>0]
out=[];fig,ax=plt.subplots(4,3,figsize=(18,14))
for i,m in enumerate(['orig','clahe','stretch','gamma']):
    X=pd.read_csv('bright_fine/bf_feats.csv' if m=='orig' else f'bright_fine/bf_feats_{m}.csv').iloc[:500].drop(columns='ID')
    for j,(nm,pre) in enumerate([('상대 z 0.25','R_z'),('절대 16구간','A_h16_'),('절대 32구간','A_h32_')]):
        cols=[c for c in X.columns if c.startswith(pre) and X[c].std()>0]
        Z=pd.concat([base.reset_index(drop=True),X[cols]],axis=1);Z=(Z-Z.mean())/Z.std()
        md=RidgeCV(np.logspace(-1,6,30)).fit(Z.values,y);co=pd.Series(md.coef_,Z.columns)[cols]
        for c,v in co.items():out.append((m,nm,c,v,md.alpha_))
        a=ax[i,j];a.bar(range(len(co)),co.values,color=['tab:red' if v>0 else 'tab:blue' for v in co.values])
        a.set_xticks(range(len(co)));a.set_xticklabels([c.replace(pre,'') for c in cols],rotation=90,fontsize=6);a.axhline(0,c='k',lw=.5)
        a.set_title(f'{m} / {pre} (alpha={md.alpha_:.0f})',fontsize=9)
plt.tight_layout();plt.savefig('bright_fine/coef_bins.png',dpi=110)
pd.DataFrame(out,columns=['tf','set','bin','coef_std','alpha']).to_csv('bright_fine/coef_bins.csv',index=False)
