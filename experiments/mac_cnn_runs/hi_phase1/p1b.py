import numpy as np,pandas as pd;from sklearn.model_selection import KFold
d=np.load('fig_data.npz');y=d['y'];hi=y>230;lo=y<170;n=len(y);r_=lambda p:np.sqrt(((p-y)**2).mean())
out=[]
for rs in (42,7,123,2026):
    p=d[f'{rs}_blend_mean'];o=np.argsort(-p)[:18];fp=o[~hi[o]]
    row=dict(rs=rs,fp_n=len(fp),fp_y_mean=y[fp].mean(),fp_res=(p-y)[fp].mean(),thr18=p[o[-1]],hi_pred_mean=p[hi].mean(),hi_pred_max=p[hi].max(),hi_pred_min=p[hi].min(),mid_pred_max=p[(~hi)&(~lo)].max())
    # honest top-q shift: in each outer train fold, find predicted-threshold t (quantile) and shift s = mean residual of train items above t; apply to val above t
    for q in (0.99,0.98,0.964,0.94,0.9):
        pp=p.copy()
        for tr,va in KFold(5,shuffle=True,random_state=rs).split(p):
            t=np.quantile(p[tr],q);s=(y-p)[tr][p[tr]>=t].mean();pp[va]+=(p[va]>=t)*s
        row[f'topshift_{q}']=r_(pp)
    # oracle shift applied only to TPs in top18 vs all hi
    pp=p.copy();tp=o[hi[o]];pp[tp]+=(y-p)[tp].mean();row['oracle_TPtop18_only']=r_(pp)
    row['base']=r_(p);out.append(row)
D=pd.DataFrame(out);D.to_csv('extra_topshift.csv',index=False);print(D.round(2).T)
