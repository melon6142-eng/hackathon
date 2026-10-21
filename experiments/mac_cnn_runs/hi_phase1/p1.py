import numpy as np,pandas as pd,glob,json
from scipy.optimize import nnls;from scipy.stats import spearmanr
from sklearn.model_selection import KFold;from sklearn.metrics import roc_auc_score,average_precision_score,precision_recall_curve
from sklearn.linear_model import LogisticRegression
HAVE_MPL=False
C='~/AI/daker/cnn';y=pd.read_csv(f'{C}/../open/train.csv').hardness.values;n=len(y)
hi=y>230;lo=y<170;mid=~hi&~lo;r_=lambda p:np.sqrt(((p-y)**2).mean())
def mem(rs):
    if rs==42:
        b=np.load(f'{C}/oof_blend1.npy');oo=[np.load(f'{C}/runs/dinoB_orig{s}/oof.npy') for s in ('','_s1','_s2')];tt=[np.load(f'{C}/runs/dinoB_trans{s}/oof.npy') for s in ('','_s1','_s2')]
        return np.column_stack([b,np.mean(oo,0),np.mean(tt,0)])
    return np.load(f'{C}/runs/indepfold/blendX_rs{rs}.npy')
A1=lambda X:np.column_stack([X,np.ones(len(X))])
def nnls_oof(X):  # mean of 10 nCV reps (rs 100+s)
    P=[]
    for s in range(10):
        pp=np.zeros(n)
        for tr,va in KFold(5,shuffle=True,random_state=100+s).split(X): pp[va]=A1(X[va])@nnls(A1(X[tr]),y[tr])[0]
        P.append(pp)
    return np.mean(P,0)
def basic(p): return dict(rmse=r_(p),hi_bias=(p-y)[hi].mean(),lo_bias=(p-y)[lo].mean(),hi_rmse=np.sqrt(((p-y)[hi]**2).mean()),lo_rmse=np.sqrt(((p-y)[lo]**2).mean()),mid_rmse=np.sqrt(((p-y)[mid]**2).mean()))
def ident(p):
    d={};oh=np.argsort(-p);ol=np.argsort(p)
    for K in (10,18,30,50):
        d[f'hiR@{K}']=hi[oh[:K]].sum()/hi.sum();d[f'hiP@{K}']=hi[oh[:K]].mean()
        d[f'loR@{K}']=lo[ol[:K]].sum()/lo.sum();d[f'loP@{K}']=lo[ol[:K]].mean()
    d.update(hiAUROC=roc_auc_score(hi,p),hiAUPRC=average_precision_score(hi,p),loAUROC=roc_auc_score(lo,-p),loAUPRC=average_precision_score(lo,-p),
      hi_rho=spearmanr(p[hi],y[hi])[0],lo_rho=spearmanr(p[lo],y[lo])[0],mid_rho=spearmanr(p[mid],y[mid])[0],all_rho=spearmanr(p,y)[0])
    return d
def oracle(p):
    d={};q=p.copy();q[hi]+= (y-p)[hi].mean();d['or_shift_hi']=r_(q);q[lo]+=(y-p)[lo].mean();d['or_shift_hilo']=r_(q);q[mid]+=(y-p)[mid].mean();d['or_shift_3bin']=r_(q)
    q=p.copy()
    for m in (hi,lo,mid): a,b=np.polyfit(p[m],y[m],1);q[m]=a*p[m]+b
    d['or_lin_3bin']=r_(q)
    q2=p.copy()
    for m in (hi,lo): a,b=np.polyfit(p[m],y[m],1);q2[m]=a*p[m]+b
    d['or_lin_hilo']=r_(q2);return d
def honest(X,p,rs):
    # nested: 5-fold (rs) outer; logistic on member OOFs (+spread) -> P(hi),P(lo); shifts = mean residual of true bin in train fold
    F=np.column_stack([X,X.std(1)]);F=(F-F.mean(0))/F.std(0)
    ph=np.zeros(n);pl=np.zeros(n);soft=p.copy();hard=p.copy();best=p.copy()
    for tr,va in KFold(5,shuffle=True,random_state=rs).split(F):
        ch=LogisticRegression(C=1.0,class_weight='balanced',max_iter=2000).fit(F[tr],hi[tr]);cl=LogisticRegression(C=1.0,class_weight='balanced',max_iter=2000).fit(F[tr],lo[tr])
        ph[va]=ch.predict_proba(F[va])[:,1];pl[va]=cl.predict_proba(F[va])[:,1]
        sh=(y-p)[tr][hi[tr]].mean();sl=(y-p)[tr][lo[tr]].mean()
        # calibrated (unbalanced) probs for soft correction
        chu=LogisticRegression(C=1.0,max_iter=2000).fit(F[tr],hi[tr]);clu=LogisticRegression(C=1.0,max_iter=2000).fit(F[tr],lo[tr])
        soft[va]=p[va]+chu.predict_proba(F[va])[:,1]*sh+clu.predict_proba(F[va])[:,1]*sl
        hard[va]=p[va]+(ph[va]>.5)*sh+(pl[va]>.5)*sl
    d=dict(hon_soft=r_(soft),hon_hard=r_(hard),hon_hiAUROC=roc_auc_score(hi,ph),hon_hiAUPRC=average_precision_score(hi,ph),hon_loAUROC=roc_auc_score(lo,pl),hon_loAUPRC=average_precision_score(lo,pl),
           hard_hi_flag=int((ph>.5).sum()),hard_hi_tp=int(((ph>.5)&hi).sum()),hard_lo_flag=int((pl>.5).sum()),hard_lo_tp=int(((pl>.5)&lo).sum()))
    return d,ph,pl,soft
def topk_split(p):
    oh=np.argsort(-p)[:18];inn=np.zeros(n,bool);inn[oh]=True;r=p-y
    return dict(hi_in_top18=int((hi&inn).sum()),res_hi_in=r[hi&inn].mean() if (hi&inn).any() else np.nan,res_hi_out=r[hi&~inn].mean(),
      fp_top18=int((~hi&inn).sum()),res_fp=r[~hi&inn].mean())
rows=[];store={}
for rs in (42,7,123,2026):
    X=mem(rs);P={'BL1':X[:,0],'ViT_orig':X[:,1],'ViT_trans':X[:,2],'blend_NNLS':nnls_oof(X),'blend_mean':X.mean(1)}
    for k,p in P.items():
        r=dict(rs=rs,model=k,**basic(p),**ident(p),**oracle(p),**topk_split(p))
        h,ph,pl,soft=honest(X,p,rs);r.update(h);rows.append(r);store[(rs,k)]=(p,ph,soft)
D=pd.DataFrame(rows);D.to_csv('metrics_all.csv',index=False)
# decomposition: hi-bin SSE components (bias^2 vs within-bin var), gain fractions
def dec(g):
    p=store[(g.rs,g.model)][0];e=(p-y)
    tot=(e**2).sum();shi=(e[hi]**2).sum();slo=(e[lo]**2).sum()
    return pd.Series(dict(SSE_hi_share=shi/tot,SSE_lo_share=slo/tot,hi_bias2_share=hi.sum()*e[hi].mean()**2/shi,lo_bias2_share=lo.sum()*e[lo].mean()**2/slo,
       gain_oracle_hilo=g.rmse-g.or_shift_hilo,gain_honest_soft=g.rmse-g.hon_soft,gain_honest_hard=g.rmse-g.hon_hard))
DD=pd.concat([D[['rs','model']],D.apply(dec,axis=1)],axis=1);DD['frac_recovered']=DD[['gain_honest_soft','gain_honest_hard']].max(1)/DD.gain_oracle_hilo;DD.to_csv('decomp.csv',index=False)
# per-image SSE (blend_mean, avg over rs)
pm=np.mean([store[(rs,'blend_mean')][0] for rs in (42,7,123,2026)],0);e2=(pm-y)**2;o=np.argsort(-e2)
pd.DataFrame(dict(idx=o,y=y[o],pred=pm[o],res=(pm-y)[o],sse=e2[o],cum_share=np.cumsum(e2[o])/e2.sum())).head(40).to_csv('per_image_sse_top40.csv',index=False)
json.dump(dict(top20_share=float(e2[o[:20]].sum()/e2.sum()),top50_share=float(e2[o[:50]].sum()/e2.sum()),n_hi_in_top20=int(hi[o[:20]].sum()),n_lo_in_top20=int(lo[o[:20]].sum())),open('per_image_summary.json','w'))
# figure
np.savez("fig_data.npz",y=y,**{f"{rs}_{k}":store[(rs,k)][0] for rs in (42,7,123,2026) for k in ("BL1","ViT_orig","ViT_trans","blend_NNLS","blend_mean")})
if HAVE_MPL:
    fig,ax=plt.subplots(1,3,figsize=(17,5.3))
    p=store[(42,'blend_mean')][0];c=np.where(hi,'r',np.where(lo,'b','0.6'))
    ax[0].scatter(y,p,c=c,s=12);ax[0].plot([120,280],[120,280],'k--');ax[0].axvline(230,c='r',lw=.5);ax[0].axvline(170,c='b',lw=.5);ax[0].set_xlabel('true');ax[0].set_ylabel('OOF pred');ax[0].set_title('blend_mean rs42 (red >230, blue <170)')
    for k in ('BL1','ViT_orig','ViT_trans','blend_NNLS','blend_mean'):
        pp=store[(42,k)][0];pr,rc,_=precision_recall_curve(hi,pp);ax[1].plot(rc,pr,label=f'{k} AP={average_precision_score(hi,pp):.2f}')
        pr,rc,_=precision_recall_curve(lo,-pp);ax[2].plot(rc,pr,label=f'{k} AP={average_precision_score(lo,-pp):.2f}')
    for a,t,b in ((ax[1],'PR: >230 (rs42)',hi.mean()),(ax[2],'PR: <170 (rs42)',lo.mean())): a.axhline(b,c='k',ls=':');a.set_title(t);a.set_xlabel('recall');a.set_ylabel('precision');a.legend(fontsize=8)
    plt.tight_layout();plt.savefig('fig_phase1.png',dpi=120)
pd.set_option('display.width',250);pd.set_option('display.max_columns',60)
cols=['rmse','hi_bias','lo_bias','hi_rmse','lo_rmse','hiR@18','hiP@18','hiAUROC','hiAUPRC','loAUROC','loAUPRC','hi_rho','lo_rho','or_shift_hi','or_shift_hilo','or_lin_hilo','or_lin_3bin','hon_soft','hon_hard','hi_in_top18','res_hi_in','res_hi_out']
print(D.groupby('model')[cols].mean().round(3).T);print(DD.groupby('model').mean().round(3).T);print(D[D.model=='blend_mean'][['rs']+cols].round(2).T)
print(open('per_image_summary.json').read())
