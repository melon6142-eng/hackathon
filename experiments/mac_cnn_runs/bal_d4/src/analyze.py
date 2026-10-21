import numpy as np,glob,os,json
from sklearn.metrics import average_precision_score as ap
from sklearn.model_selection import KFold,cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,r2_score
from scipy.stats import spearmanr
from common import *
FC='~/AI/daker/final_clean';Y=y();N=len(Y)
dec=np.minimum((np.argsort(np.argsort(Y))*10)//N,9)
folds=list(KFold(5,shuffle=True,random_state=42).split(Y))
def met(p):
    r=dict(rmse=rm(p,Y),hi=hb(p,Y),lo=lb(p,Y),auprc=ap(Y>230,p),p18=float((Y[np.argsort(-p)[:18]]>230).mean()))
    r['dec_rmse']=[rm(p[dec==d],Y[dec==d]) for d in range(10)];r['dec_bias']=[float((p-Y)[dec==d].mean()) for d in range(10)];return r
bl1=np.load(f'{FC}/bl1/oof_fs42.npy');tr=np.mean([np.load(f'{FC}/vit/trans_s{s}/oof.npy') for s in (42,1,2)],0)
arms={'base(uni_d4b)':{s:np.load(f'{FC}/vit/orig_s{s}/oof.npy') for s in (42,1,2)}}
for d in sorted(glob.glob(f'{O}/arms/*_s*/oof.npy')):
    t=d.split('/')[-2];a,s=t.rsplit('_s',1);arms.setdefault(a,{})[int(s)]=np.load(d)
out=['# bal_d4 결과 (fold seed 42, KFold(5,rs=42), ViT-B orig, final_clean 파이프라인)\n','## 1. 단독 OOF (학습 seed별 / seed 평균)\n',
'| arm | seeds | 개별 RMSE | 평균모델 RMSE | hi bias | lo bias | AUPRC | P@18 | eq3(orig 교체) | eq4(추가) |','|---|---|---|---|---|---|---|---|---|---|']
eqb=(bl1+arms['base(uni_d4b)'][42]*0+np.mean(list(arms['base(uni_d4b)'].values()),0)+tr)/3
R={}
for a,d in arms.items():
    ss=sorted(d);mp=np.mean([d[s] for s in ss],0);m=met(mp);R[a]=dict(seeds=ss,ind=[rm(d[s],Y) for s in ss],**m)
    ob=np.mean([arms['base(uni_d4b)'][s] for s in ss],0)  # seed-matched baseline
    R[a]['base_matched']=rm(ob,Y);R[a]['eq3']=rm((bl1+mp+tr)/3,Y);R[a]['eq4']=rm((bl1+ob+tr+mp)/4,Y)
    out.append(f"| {a} | {ss} | {' / '.join(f'{v:.3f}' for v in R[a]['ind'])} | {m['rmse']:.3f} (기준 동일seed {R[a]['base_matched']:.3f}) | {m['hi']:+.1f} | {m['lo']:+.1f} | {m['auprc']:.3f} | {m['p18']:.3f} | {R[a]['eq3']:.3f} | {R[a]['eq4']:.3f} |")
out+=[f"\neq3 기준(BL1+orig3+trans3 균등) = {rm(eqb,Y):.3f}\n",'## 2. 경도 10분위별 RMSE / bias (seed 평균 모델)\n','| arm | '+' | '.join(f'D{d}' for d in range(10))+' |','|---'*11+'|']
for a in R: out.append(f'| {a} RMSE | '+' | '.join(f'{v:.1f}' for v in R[a]['dec_rmse'])+' |');out.append(f'| {a} bias | '+' | '.join(f'{v:+.1f}' for v in R[a]['dec_bias'])+' |')
out.append('\n분위 경계(HV): '+', '.join(f'{v:.1f}' for v in np.quantile(Y,np.linspace(0,1,11))))
# exposure audit
out+=['\n## 3. 학습 노출량 감사 (fold당 30 epoch 합계, 10분위별 sample-visit; loss 가중 모두 1)\n','| 설정 | '+' | '.join(f'D{d}' for d in range(10))+' |','|---'*11+'|']
uni=np.mean([np.bincount(dec[t],minlength=10)*30 for t,_ in folds],0);out.append('| uniform(permutation, 기준) | '+' | '.join(f'{v:.0f}' for v in uni)+' |')
for a in [x for x in arms if x.startswith('bal')]:
    I=[np.load(f) for f in glob.glob(f'{O}/arms/{a}_s*/fold*_idx.npy')]
    if I: e=np.mean([np.bincount(dec[i.ravel()],minlength=10) for i in I],0);u=np.mean([len(np.unique(i.ravel())) for i in I]);out.append(f'| {a} (실측, {len(I)} fold-run) | '+' | '.join(f'{v:.0f}' for v in e)+f' | (고유 원본 {u:.0f}/400)')
# embeddings
def ld(src):
    fs=[f'{O}/arms/{src}/fold{f}_emb.npz' for f in range(5)]
    return [np.load(f) for f in fs] if all(os.path.exists(f) for f in fs) else None
srcs={'pretrained DINOv2(미학습)':'emb_pre','base(uni_d4b) s42':'emb_base'}
for a in arms:
    if a!='base(uni_d4b)': srcs[f'{a} s42']=f'{a}_s42'
out+=['\n## 4. 임베딩 분석 (각 fold 모델이 학습에 안 쓴 held-out 100장 안에서만; 5 fold 평균)\n','| 임베딩 | kNN(k=10) R² | 선형probe PCA5 R² | PCA20 R² | Spearman(거리,|ΔHV|) | D4 cos sim | 경도방향 D4 일관성 r | 고경도18 silhouette k=2/3 |','|---|---|---|---|---|---|---|---|']
for nm,src in srcs.items():
    D=ld(src)
    if D is None: continue
    k=[];p5=[];p20=[];sp=[];cs=[];hc=[];H=[]
    for d in D:
        va=d['va'];E=d['E'].astype(np.float32);yy=Y[va];e0=E[:,0];mu,sd=e0.mean(0),e0.std(0)+1e-6;Z=(E-mu)/sd;z0=Z[:,0]
        cv=KFold(5,shuffle=True,random_state=0)
        k.append(r2_score(yy,cross_val_predict(KNeighborsRegressor(10),z0,yy,cv=cv)))
        p5.append(r2_score(yy,cross_val_predict(make_pipeline(PCA(5),RidgeCV(np.logspace(-3,3,13))),z0,yy,cv=cv)))
        p20.append(r2_score(yy,cross_val_predict(make_pipeline(PCA(20),RidgeCV(np.logspace(-3,3,13))),z0,yy,cv=cv)))
        iu=np.triu_indices(len(va),1);dm=np.linalg.norm(z0[:,None]-z0[None],axis=2)[iu];sp.append(spearmanr(dm,np.abs(yy[:,None]-yy[None])[iu])[0])
        nE=E/np.linalg.norm(E,axis=2,keepdims=True);cs.append(float((nE[:,1:]*nE[:,:1]).sum(2).mean()))
        w=RidgeCV(np.logspace(-1,4,11)).fit(z0,yy).coef_;pr=Z@w;hc.append(float(np.mean([np.corrcoef(pr[:,0],pr[:,j])[0,1] for j in range(1,8)])))
        H.append(z0[yy>230])
    H=np.concatenate(H);sil=[silhouette_score(H,KMeans(c,n_init=10,random_state=0).fit_predict(H)) for c in (2,3)]
    out.append(f'| {nm} | {np.mean(k):.3f} | {np.mean(p5):.3f} | {np.mean(p20):.3f} | {np.mean(sp):.3f} | {np.mean(cs):.4f} | {np.mean(hc):.3f} | {sil[0]:.3f}/{sil[1]:.3f} |')
out.append('\n주의: 경도방향 일관성은 view0에 in-sample 적합한 ridge 방향을 다른 D4 view에 투영한 상관(불변성 지표, 예측력 아님). 고경도 18장 군집은 fold별로 표준화 후 합친 것이라 근사이며 과해석 금지.')
open(f'{O}/analysis.md','w').write('\n'.join(out));json.dump(R,open(f'{O}/results.json','w'),indent=1);print('\n'.join(out))
