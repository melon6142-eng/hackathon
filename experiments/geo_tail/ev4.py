import numpy as np,pandas as pd,warnings;warnings.filterwarnings('ignore')
exec(open('ev.py').read().split('# --- 2)')[0])
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
P=G[[c for c in G.columns if c.startswith('ws_q')]+['ws_top5','ws_top10frac','ws_skew','ws_Acv','ws_n','ws_Amean']].copy()
P=np.log1p(P.clip(lower=0)).values
mk=lambda:make_pipeline(StandardScaler(),RidgeCV(np.logspace(-2,4,25)))
hi=y>230;lo=y<170;o={'base':[],'+ws_ridge':[]};so=[];cs=[]
for s in range(10):
    pb=np.zeros(500);pe=np.zeros(500);sl=np.zeros(500)
    for tr,va in KFold(5,shuffle=True,random_state=100+s).split(B):
        io=np.zeros(len(tr))
        for i,j in KFold(5,shuffle=True,random_state=7).split(tr): io[j]=mk().fit(P[tr[i]],y[tr[i]]).predict(P[tr[j]])
        f=mk().fit(P[tr],y[tr]).predict(P[va]);sl[va]=f
        pb[va]=blend(tr,va)[1]
        A=np.column_stack([B[tr],io,np.ones(len(tr))]);w=nnls(A,y[tr])[0];pe[va]=np.column_stack([B[va],f,np.ones(len(va))])@w
    for k,p in [('base',pb),('+ws_ridge',pe)]: o[k].append((r(p,y),(p-y)[hi].mean(),(p-y)[lo].mean()))
    so.append(r(sl,y));cs.append([np.corrcoef(sl,B[:,i])[0,1] for i in range(3)])
L=['## 4) 결정립 면적 분위수10+꼬리통계 Ridge','',f'단독 nCV RMSE {np.mean(so):.3f}; 블렌드 멤버(BL1, ViT-B orig, ViT-B trans)와 상관 {np.round(np.mean(cs,0),3).tolist()}','','| 방법 | nCV(10회) | std | hi | lo |','|---|---|---|---|---|']
for k,v in o.items(): a=np.array(v);L.append(f'| {k} | {a[:,0].mean():.3f} | {a[:,0].std():.3f} | {a[:,1].mean():.1f} | {a[:,2].mean():.1f} |')
open('ev4.md','w').write('\n'.join(L));print('\n'.join(L))
