import json,numpy as np,pandas as pd,csv,matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from scipy.stats import spearmanr
O='~/AI/daker/cnn/runs/gen_cf';R='~/AI/daker'
d=np.load(f'{O}/cf_raw.npz',allow_pickle=True);meta=d['meta'];dnn=d['dnn'];p95=float(d['p95']);lev=json.loads(str(d['lev']))
HC=pd.read_csv(f'{O}/cf_hcfeat.csv');y=np.array([float(r['hardness']) for r in csv.DictReader(open(f'{R}/open/train.csv'))])
Hr=pd.read_csv(f'{R}/phase_sep/feat_train.csv')
VARS={'n_grains':'g_Amean','aspect':'e_ar','ph_frac':'ph_frac','pore_frac':'po_frac','bnd_c':'b_contrast','noise':'x_sig'}
P={k[2:]:d[k] for k in d.files if k.startswith('P_')}
from sklearn.model_selection import KFold
fo=np.zeros(500,int)
for f,(a,b) in enumerate(KFold(5,shuffle=True,random_state=42).split(y)): fo[b]=f
rep=np.array([m[0]=='rep' for m in meta]);ri=np.array([m[1] for m in meta])[rep]
out=['## 재현 세트: 실제 이미지 i의 적합 파라미터로 새로 생성한 이미지에 대한 예측 vs 실제 라벨 y_i','','| 모델 | Pearson r | Spearman | 예측 SD | 실제 OOF 대비 |','|---|---|---|---|---|']
for k,v in P.items():
    FL=json.load(open(f'{O}/folds_{k}.json')) if k.startswith('ViT') else list(range(5))
    pr=v[:,rep].mean(0)  # fold 평균(누수 주의: 재현 이미지는 합성이라 직접 누수 없음)
    out.append(f'| {k} | {np.corrcoef(pr,y)[0,1]:.3f} | {spearmanr(pr,y)[0]:.3f} | {pr.std():.1f} | - |')
# 실제 데이터 기준 기울기 (측정 특징 1SD당 경도)
rows=[];fig,ax=plt.subplots(2,3,figsize=(15,8));ax=ax.ravel()
for j,(v,feat) in enumerate(VARS.items()):
    xr=Hr[feat].values;sx=xr.std();rs=np.polyfit((xr-xr.mean())/sx,y,1)[0];rho=spearmanr(xr,y)[0]
    idx=np.where([m[0]=='cf' and m[2]==v for m in meta])[0];bs=np.array([m[1] for m in meta[idx]]);li=np.array([m[3] for m in meta[idx]])
    xm=HC[feat].values[idx];inside=(dnn[idx]<=p95)
    for k,pv in P.items():
        pr=pv[:,idx]  # (folds, n)
        sl_b=[];sl_bf=[]
        for b in np.unique(bs):
            mk=bs==b
            if np.ptp(xm[mk])<1e-9: continue
            s=[np.polyfit((xm[mk]-xr.mean())/sx,pr[f,mk],1)[0] for f in range(len(pr))];sl_bf+=s;sl_b.append(np.mean(s))
        sl_b=np.array(sl_b);sl_bf=np.array(sl_bf)
        # in-distribution only
        sl_in=[]
        for b in np.unique(bs):
            mk=(bs==b)&inside
            if mk.sum()>=3 and np.ptp(xm[mk])>1e-9: sl_in.append(np.polyfit((xm[mk]-xr.mean())/sx,pr[:,mk].mean(0),1)[0])
        rows.append(dict(var=v,feat=feat,model=k,real_slope=rs,real_rho=rho,model_slope_med=np.median(sl_b),q25=np.percentile(sl_b,25),q75=np.percentile(sl_b,75),
            sign_agree_bases=np.mean(np.sign(sl_b)==np.sign(np.median(sl_b))),sign_agree_basefold=np.mean(np.sign(sl_bf)==np.sign(np.median(sl_b))),
            slope_in_dist=np.median(sl_in) if sl_in else np.nan,frac_in=inside.mean()))
        cur=[pr.mean(0)[li==l].mean()-pr.mean(0).mean() for l in range(7)]
        ax[j].plot(range(7),cur,marker='o',label=k[:22])
    shift=[(dnn[idx][li==l]>p95).mean() for l in range(7)]
    ax[j].set_title(f'{v} (real slope {rs:+.1f}/SD, OOD {["%.0f"%(s*100) for s in shift]}%)',fontsize=8);ax[j].set_xlabel('level (fitted-param quantile 5..95%)');ax[j].set_ylabel('pred change')
ax[0].legend(fontsize=6);plt.tight_layout();plt.savefig(f'{O}/cf_curves.png',dpi=110)
df=pd.DataFrame(rows);df.to_csv(f'{O}/cf_slopes.csv',index=False)
# 변수별 측정 특징 실제 변화 확인
chk=[]
for v,feat in VARS.items():
    idx=np.where([m[0]=='cf' and m[2]==v for m in meta])[0];li=np.array([m[3] for m in meta[idx]])
    chk.append(f'| {v} | {feat} | '+' / '.join(f'{HC[feat].values[idx][li==l].mean():.3g}' for l in (0,3,6))+f' | {Hr[feat].quantile(.05):.3g}–{Hr[feat].quantile(.95):.3g} |')
open(f'{O}/s3_tables.md','w').write('\n'.join(out)+'\n\n## 기울기\n'+df.round(2).to_csv(index=False)+'\n\n## 측정 특징 (level 0/3/6 평균) vs 실제 5–95%\n| 변수 | 특징 | L0/L3/L6 | 실제 |\n|---|---|---|---|\n'+'\n'.join(chk))
print(open(f'{O}/s3_tables.md').read())
