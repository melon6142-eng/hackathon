import numpy as np,pandas as pd,sys,warnings;warnings.filterwarnings('ignore')
from scipy.optimize import nnls
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV,LogisticRegression,Ridge
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
D='./runs';y=pd.read_csv('./open/train.csv')['hardness'].values
ld=lambda t,k:np.load(f'{D}/{t}/{k}.npy')
av=lambda ts,k:np.mean([ld(t,k) for t in ts],0)
O=['dino_ft/dinoB_orig','dino_ft2/dinoB_orig_s1','dino_ft2/dinoB_orig_s2'];T=['dino_ft/dinoB_trans','dino_ft2/dinoB_trans_s1','dino_ft2/dinoB_trans_s2']
sub=pd.read_csv('./out/submission_blend1_cv12.95.csv')
B=np.column_stack([np.load(f'{D}/oof_blend1.npy'),av(O,'oof'),av(T,'oof')])
BT=np.column_stack([sub.iloc[:,1].values,av(O,'test'),av(T,'test')])
X=pd.concat([pd.read_pickle(f'{D}/hi_rule/X_train.pkl'),pd.read_pickle(f'{D}/hi_rule/X_test.pkl')]).reset_index(drop=True)
dm=pd.read_csv(f'{D}/dark_mask/dm_feats.csv');f3=pd.read_csv('feat3.csv')
c2=['grain_A_mean','grain_A_med','grain_inv_sqrt_d','grain_ecc_mean','grain_ori_align','grain_ng','hand_dark0.5','hand_dark1.0','hand_dark1.5','hand_dark2.0','grain_pore_frac','gen_st_ar','acf_acf_ratio','grain_otsu_frac','hand_coh']
P2=X[c2].copy()
for c in ['iS1_frac','iS12_frac','iS12_ar','iS12_align','iS12_A_q90']: P2[c]=dm[c].values
for c in ['hand_dark1.0','iS12_frac','grain_otsu_frac']: P2[c+'^2']=P2[c]**2
P3=pd.concat([P2,f3.drop(columns='ID')],axis=1)
P2=P2.fillna(P2.median()).values;P3=P3.fillna(P3.median()).values
r=lambda p,t:np.sqrt(((p-t)**2).mean())
def blend(tr,va,extra=None,extraT=None):
    A=np.column_stack([B]+([extra] if extra is not None else [])+[np.ones(len(B))])
    w=nnls(A[tr],y[tr])[0];return A[tr]@w,A[va]@w if extraT is None else np.column_stack([extraT,np.ones(len(extraT))])@w
def mk(kind): return make_pipeline(StandardScaler(),RidgeCV(np.logspace(-2,4,25))) if kind=='ridge' else HGB(max_iter=300,learning_rate=.03,max_depth=3,min_samples_leaf=15)
def rankmatch(ptr,ytr,p,a):
    q=np.searchsorted(np.sort(ptr),p)/len(ptr);q=np.clip(q,.5/len(ptr),1-.5/len(ptr));return (1-a)*p+a*np.quantile(ytr,q)
AL=np.linspace(0,1,11)
def m_rank(tr,va):
    s=np.zeros(len(AL))
    for itr,iva in KFold(5,shuffle=True,random_state=7).split(tr):
        a,b=tr[itr],tr[iva];pt,pv=blend(a,b)
        s+=[((rankmatch(pt,y[a],pv,al)-y[b])**2).sum() for al in AL]
    al=AL[s.argmin()];pt,pv=blend(tr,va);return rankmatch(pt,y[tr],pv,al),al
def m_feat(tr,va,P,kind):
    io=np.zeros(len(tr))
    for itr,iva in KFold(5,shuffle=True,random_state=7).split(tr):
        io[iva]=mk(kind).fit(P[tr[itr]],y[tr[itr]]).predict(P[tr[iva]])
    full=mk(kind).fit(P[tr],y[tr]).predict(P[va])
    A=np.column_stack([B[tr],io,np.ones(len(tr))]);w=nnls(A,y[tr])[0]
    return np.column_stack([B[va],full,np.ones(len(va))])@w,(io,full)
TH=[.3,.5,.7];SH=[0,.5,1]
def gate_fit(a,P):
    pt,_=blend(a,a[:1]);res=y[a]-pt;hy=(y[a]>230).astype(int)
    pr=np.zeros(len(a))
    for i,j in KFold(5,shuffle=True,random_state=3).split(a):
        pr[j]=make_pipeline(StandardScaler(),LogisticRegression(C=.1,class_weight='balanced')).fit(P[a[i]],hy[i]).predict_proba(P[a[j]])[:,1]
    clf=make_pipeline(StandardScaler(),LogisticRegression(C=.1,class_weight='balanced')).fit(P[a],hy)
    return clf,pr,res
def gate_apply(a,b,P,t,sh):
    clf,pr,res=gate_fit(a,P);g=pr>t;_,pv=blend(a,b)
    if g.sum()<10 or sh==0: return pv
    rg=make_pipeline(StandardScaler(),Ridge(30)).fit(P[a][g],res[g]);pb=clf.predict_proba(P[b])[:,1]
    return pv+sh*(pb>t)*rg.predict(P[b])
def m_gate(tr,va,P):
    best=None
    for t in TH:
        for sh in SH:
            e=0
            for itr,iva in KFold(5,shuffle=True,random_state=7).split(tr): e+=((gate_apply(tr[itr],tr[iva],P,t,sh)-y[tr[iva]])**2).sum()
            if best is None or e<best[0]: best=(e,t,sh)
    return gate_apply(tr,va,P,best[1],best[2]),list(best[1:])
meth={'base':lambda tr,va:(blend(tr,va)[1],None),'1_rank':m_rank,'1_rank_a1':lambda tr,va:(rankmatch(*blend(tr,va)[:1],y[tr],blend(tr,va)[1],1.0),None),'1_rank_a05':lambda tr,va:(rankmatch(*blend(tr,va)[:1],y[tr],blend(tr,va)[1],.5),None),
 '2_ridgeP2':lambda tr,va:m_feat(tr,va,P2,'ridge'),'2_gbmP2':lambda tr,va:m_feat(tr,va,P2,'gbm'),
 '3_ridgeP3':lambda tr,va:m_feat(tr,va,P3,'ridge'),'3_gbmP3':lambda tr,va:m_feat(tr,va,P3,'gbm'),
 '4_gateP3':lambda tr,va:m_gate(tr,va,P3)}
R=int(sys.argv[1]);out={k:[] for k in meth};hi=y>230;lo=y<170;aux={k:[] for k in meth};solo={}
for s in range(R):
    pp={k:np.zeros(500) for k in meth};so={k:np.zeros(500) for k in meth}
    for tr,va in KFold(5,shuffle=True,random_state=100+s).split(B):
        for k,f in meth.items():
            p,x=f(tr,va);pp[k][va]=p
            if isinstance(x,tuple): so[k][va]=x[1]
            elif x is not None: aux[k].append(x)
    for k in meth: out[k].append((r(pp[k],y),(pp[k]-y)[hi].mean(),(pp[k]-y)[lo].mean(),r(so[k],y) if so[k].any() else np.nan))
    print(s,{k:round(out[k][-1][0],3) for k in meth},flush=True)
L=['| 방법 | nCV(%d회) | std | hi(>230) 편향 | lo(<170) 편향 | 단독 모델 RMSE | 선택된 하이퍼 |'%R,'|---|---|---|---|---|---|---|']
for k in meth:
    a=np.array(out[k]);hp=pd.Series([str(x) for x in aux[k]]).value_counts().head(3).to_dict() if aux[k] else ''
    L.append(f'| {k} | {a[:,0].mean():.3f} | {a[:,0].std():.3f} | {a[:,1].mean():.1f} | {a[:,2].mean():.1f} | {np.nanmean(a[:,3]) if not np.isnan(a[:,3]).all() else float("nan"):.3f} | {hp} |')
open('ev.md','w').write('\n'.join(L));print('\n'.join(L))
