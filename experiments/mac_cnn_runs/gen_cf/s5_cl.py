import json,numpy as np,pandas as pd,csv,matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from scipy.stats import spearmanr
O='~/AI/daker/cnn/runs/gen_cf';R='~/AI/daker'
d=np.load(f'{O}/cl_raw.npz',allow_pickle=True);meta=d['meta'];dnn=d['dnn'];p95=float(d['p95'])
HC=pd.read_csv(f'{O}/cl_hcfeat.csv');Hr=pd.read_csv(f'{R}/phase_sep/feat_train.csv')
y=np.array([float(r['hardness']) for r in csv.DictReader(open(f'{R}/open/train.csv'))])
P={k[2:]:d[k] for k in d.files if k.startswith('P_')}
MEAS={'gs_het':['u_dens_cv','g_Acv'],'ph_clust':['u_ph_sd','ph_maxcl','ph_conn'],'pore_clust':['po_in_ph','u_dens_cv']}
out=['var,model,dpred_L0_to_L6_med,q25,q75,sign_agree_bases,OOD_L0,OOD_L6']
fig,ax=plt.subplots(1,3,figsize=(15,4.5))
for j,v in enumerate(MEAS):
    idx=np.where([m[2]==v for m in meta])[0];bs=np.array([m[1] for m in meta[idx]]);li=np.array([m[3] for m in meta[idx]])
    for k,pv in P.items():
        pr=pv[:,idx].mean(0);dl=np.array([pr[(bs==b)&(li==6)].mean()-pr[(bs==b)&(li==0)].mean() for b in np.unique(bs)])
        out.append(f'{v},{k},{np.median(dl):.2f},{np.percentile(dl,25):.2f},{np.percentile(dl,75):.2f},{np.mean(np.sign(dl)==np.sign(np.median(dl))):.2f},{(dnn[idx][li==0]>p95).mean():.2f},{(dnn[idx][li==6]>p95).mean():.2f}')
        ax[j].plot(range(7),[pr[li==l].mean()-pr[li==0].mean() for l in range(7)],marker='o',label=k[:22])
    ax[j].set_title(f'{v}  OOD% '+' '.join('%.0f'%(100*(dnn[idx][li==l]>p95).mean()) for l in range(7)),fontsize=8);ax[j].set_xlabel('cluster level 0..1');ax[j].set_ylabel('pred change')
ax[0].legend(fontsize=6);plt.tight_layout();plt.savefig(f'{O}/cl_curves.png',dpi=110)
out.append('\nvar,feat,meas_L0,meas_L3,meas_L6,real_p5,real_p95,real_spearman_y')
for v,fs in MEAS.items():
    idx=np.where([m[2]==v for m in meta])[0];li=np.array([m[3] for m in meta[idx]])
    for f in fs: out.append(f'{v},{f},'+','.join(f'{HC[f].values[idx][li==l].mean():.3g}' for l in (0,3,6))+f',{Hr[f].quantile(.05):.3g},{Hr[f].quantile(.95):.3g},{spearmanr(Hr[f],y)[0]:.2f}')
open(f'{O}/s5_tables.csv','w').write('\n'.join(out));print('\n'.join(out))
