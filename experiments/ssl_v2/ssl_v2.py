# SSL v2 leak-free: inside each outer fold, fit on tr only → pseudo on test → refit → score va.
# Confidence: mid-band by fold-model pred + agreement with (optional) bagging std inside tr.
# Never uses full-data test.npy as pseudo targets (that leaked val via fold models included in TTA avg).
import os; os.environ['OMP_NUM_THREADS']='1'
import sys, json, time, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from scipy.optimize import nnls
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from joblib import Parallel, delayed
sys.path.insert(0,'./runs'); os.chdir('./runs')
from feats import HX, DINO
OUT='./runs/ssl_v2'; D='./runs'
y=pd.read_csv('./open/train.csv').hardness.values
hi=y>230; lo=y<170
rmse=lambda p: float(np.sqrt(((p-y)**2).mean()))
O3=['dino_ft/dinoB_orig','dino_ft2/dinoB_orig_s1','dino_ft2/dinoB_orig_s2']
T3=['dino_ft/dinoB_trans','dino_ft2/dinoB_trans_s1','dino_ft2/dinoB_trans_s2']
Mo=np.column_stack([np.load(f'{D}/oof_blend1.npy')]+[np.load(f'{D}/{t}/oof.npy') for t in O3+T3])
Bo=np.column_stack([Mo[:,0], Mo[:,1:4].mean(1), Mo[:,4:7].mean(1)])
SEL=[(L,k) for L in range(4) for k in (1,2)]
Xtr=np.hstack([HX('train'), DINO('train',sel=SEL)]).astype(np.float64)
Xte=np.hstack([HX('test'), DINO('test',sel=SEL)]).astype(np.float64)
ok=Xtr.std(0)>1e-8; Xtr,Xte=Xtr[:,ok],Xte[:,ok]
print(f'X {Xtr.shape} leak-free SSL {time.ctime()}', flush=True)
AL=np.logspace(-1,6,10)
def mk(): return make_pipeline(StandardScaler(), RidgeCV(AL))
def nn(Atr,Ava,ytr):
    w,_=nnls(np.column_stack([Atr,np.ones(len(Atr))]),ytr)
    return np.column_stack([Ava,np.ones(len(Ava))])@w

def bag_pred(Xfit,yfit,Xout,nbag=5):
    """mean/std of Ridge bagging for confidence"""
    ps=[]
    for b in range(nbag):
        rs=np.random.RandomState(10+b)
        idx=rs.randint(0,len(yfit),len(yfit))  # bootstrap
        ps.append(mk().fit(Xfit[idx],yfit[idx]).predict(Xout))
    P=np.stack(ps,0); return P.mean(0), P.std(0)

def select(mean_te, std_te, frac, band='mid'):
    if band=='mid': m=(mean_te>=170)&(mean_te<=220)
    elif band=='midw': m=(mean_te>=160)&(mean_te<=225)
    else: m=np.ones(len(mean_te),bool)
    idx=np.where(m)[0]
    if len(idx)==0: return np.array([],dtype=int)
    k=max(10,int(round(frac*len(idx))))
    return idx[np.argsort(std_te[idx])[:k]]

def one_seed(seed, frac, band, soft):
    n=len(y); P={k:np.zeros(n) for k in ['blend','ridge','ridge_pl','blend_plcol']}
    n_ps=[]
    for tr,va in KFold(5,shuffle=True,random_state=100+seed).split(y):
        P['blend'][va]=nn(Bo[tr],Bo[va],y[tr])
        P['ridge'][va]=mk().fit(Xtr[tr],y[tr]).predict(Xtr[va])
        # fold-safe pseudo from tr only
        mu_te, sd_te = bag_pred(Xtr[tr], y[tr], Xte)
        sel=select(mu_te, sd_te, frac, band); n_ps.append(len(sel))
        if len(sel)==0:
            P['ridge_pl'][va]=P['ridge'][va]; ri=np.zeros(len(tr)); rv=P['ridge'][va]
        else:
            if soft:
                med=np.median(sd_te[sel])+1e-6
                wp=np.clip(1-sd_te[sel]/med,0.15,1.0)
                sw=np.concatenate([np.ones(len(tr)),wp])
                Xs=np.vstack([Xtr[tr],Xte[sel]]); ys=np.concatenate([y[tr],mu_te[sel]])
                sc=StandardScaler().fit(Xs); Z=sc.transform(Xs)
                best=None
                for a in AL:
                    e=0
                    for a0,b0 in KFold(3,shuffle=True,random_state=0).split(Z):
                        if len(a0)<5: continue
                        rr=Ridge(a).fit(Z[a0],ys[a0],sample_weight=sw[a0])
                        e+=((rr.predict(Z[b0])-ys[b0])**2*sw[b0]).sum()
                    if best is None or e<best[0]: best=(e,a)
                rr=Ridge(best[1]).fit(Z,ys,sample_weight=sw)
                pred=lambda X: rr.predict(sc.transform(X))
            else:
                m=mk().fit(np.vstack([Xtr[tr],Xte[sel]]), np.concatenate([y[tr],mu_te[sel]]))
                pred=m.predict
            P['ridge_pl'][va]=pred(Xtr[va])
            # inner OOF on tr for blend column (also fold-safe)
            ri=np.zeros(len(tr))
            for a,b in KFold(3,shuffle=True,random_state=1).split(tr):
                tri,vai=tr[a],tr[b]
                mu2,sd2=bag_pred(Xtr[tri],y[tri],Xte,nbag=3)
                sel2=select(mu2,sd2,frac,band)
                if len(sel2)==0:
                    ri[b]=mk().fit(Xtr[tri],y[tri]).predict(Xtr[vai]); continue
                if soft:
                    med=np.median(sd2[sel2])+1e-6; wp=np.clip(1-sd2[sel2]/med,0.15,1.)
                    sw=np.concatenate([np.ones(len(tri)),wp])
                    Xs=np.vstack([Xtr[tri],Xte[sel2]]); ys=np.concatenate([y[tri],mu2[sel2]])
                    sc=StandardScaler().fit(Xs); Z=sc.transform(Xs)
                    best=None
                    for al in AL:
                        e=0
                        for a0,b0 in KFold(3,shuffle=True,random_state=0).split(Z):
                            if len(a0)<5: continue
                            rr=Ridge(al).fit(Z[a0],ys[a0],sample_weight=sw[a0])
                            e+=((rr.predict(Z[b0])-ys[b0])**2*sw[b0]).sum()
                        if best is None or e<best[0]: best=(e,al)
                    rr=Ridge(best[1]).fit(Z,ys,sample_weight=sw)
                    ri[b]=rr.predict(sc.transform(Xtr[vai]))
                else:
                    m3=mk().fit(np.vstack([Xtr[tri],Xte[sel2]]), np.concatenate([y[tri],mu2[sel2]]))
                    ri[b]=m3.predict(Xtr[vai])
            rv=pred(Xtr[va])
        P['blend_plcol'][va]=nn(np.column_stack([Bo[tr],ri]), np.column_stack([Bo[va],rv]), y[tr])
    P['_n_pseudo_mean']=float(np.mean(n_ps))
    return P

cfgs=[('mid30',0.30,'mid'),('mid50',0.50,'mid'),('midw40',0.40,'midw'),('lowstd20',0.20,'all')]
rows=[]; logs=[]
for soft in (False,True):
  tag='soft' if soft else 'hard'
  for cname,frac,band in cfgs:
    print(f'{tag}/{cname} ...', flush=True); t1=time.time()
    Ps=Parallel(n_jobs=6)(delayed(one_seed)(s,frac,band,soft) for s in range(10))
    n_ps=np.mean([P['_n_pseudo_mean'] for P in Ps])
    logs.append(f'{tag}/{cname}: mean_n_pseudo≈{n_ps:.0f}')
    print(f'  {logs[-1]} {time.time()-t1:.0f}s', flush=True)
    for k in ['blend','ridge','ridge_pl','blend_plcol']:
        rms=np.array([rmse(P[k]) for P in Ps])
        d=rms-np.array([rmse(P['blend']) for P in Ps]) if k!='blend' else np.zeros(10)
        hb=np.mean([(P[k]-y)[hi].mean() for P in Ps]); lb=np.mean([(P[k]-y)[lo].mean() for P in Ps])
        rows.append(dict(mode=tag,cfg=cname,model=k,n_pseudo=n_ps,ncv=rms.mean(),sd=rms.std(),
                         d_blend=d.mean(),d_sd=d.std(),hi=hb,lo=lb))
        print(f'  {k}: {rms.mean():.3f} Δ{d.mean():+.3f} (±{d.std():.3f}) hi{hb:.1f}', flush=True)

df=pd.DataFrame(rows); df.to_csv(f'{OUT}/ev.csv',index=False)
open(f'{OUT}/sel_log.txt','w').write('\n'.join(logs))
base=df[(df.model=='blend')&(df.cfg=='mid30')&(df.mode=='hard')].iloc[0]
sub=df[df.model=='blend_plcol'].sort_values('ncv'); best=sub.iloc[0]
# seed wins for best
Ps=Parallel(6)(delayed(one_seed)(s, dict(mid30=0.3,mid50=0.5,midw40=0.4,lowstd20=0.2)[best.cfg],
                                  dict(mid30='mid',mid50='mid',midw40='midw',lowstd20='all')[best.cfg],
                                  best.mode=='soft') for s in range(10))
improve=sum(1 for P in Ps if rmse(P['blend_plcol'])<rmse(P['blend']))
adopt=improve>=7 and best.d_blend<-0.02
L=['# SSL v2 — 준지도 (누수 없는 새 설계)','',
'## 이전 SSL 실패','- label_ssl: gen 특징 + 광범위 의사라벨 → 블렌드 가중≈0.','',
'## 새 설계 (누수 차단)','1. **각 outer fold의 train만**으로 Ridge+배깅 → test 의사라벨 생성.',
'2. 배깅 표준편차 낮고 **중간 경도**인 test만 채택.',
'3. hard/soft로 재학습 후 블렌드 추가 열. 전체학습 test.npy는 의사라벨로 쓰지 않음 (val 누수).',
'4. `runs/ssl_v2/`만 기록.','',
'## 선택']+[f'- {x}' for x in logs]+['',
'## nCV 10×5', f'기준 blend ≈ {base.ncv:.3f}','',
'| mode | cfg | model | n≈ | nCV | sd | Δblend | Δsd | hi | lo |','|---|---|---|---|---|---|---|---|---|---|']
for _,row in df.sort_values(['model','ncv']).iterrows():
    L.append(f"| {row.mode} | {row.cfg} | {row.model} | {row.n_pseudo:.0f} | {row.ncv:.3f} | {row.sd:.3f} | {row.d_blend:+.3f} | {row.d_sd:.3f} | {row.hi:.1f} | {row.lo:.1f} |")
L+=['','## 채택',
f'- 최선 blend_plcol: `{best.mode}/{best.cfg}` nCV={best.ncv:.3f} (Δ={best.d_blend:+.3f}±{best.d_sd:.3f}) hi={best.hi:.1f}',
f'- 기준 blend nCV={base.ncv:.3f} hi={base.hi:.1f}',
f'- blend보다 나은 seed: {improve}/10',
f'- **채택: {"예" if adopt else "아니오"}**',
'','## 고경도 편향', f'- {base.hi:.1f} → {best.hi:.1f}',
'','## 다음','- 미채택 시 의사라벨 중단. 진짜 라벨 추가가 우선.']
open(f'{OUT}/report.md','w').write('\n'.join(L))
json.dump(dict(best=best.to_dict(),base=float(base.ncv),improve=int(improve),adopt=bool(adopt),note='leak-free'),
          open(f'{OUT}/summary.json','w'), indent=2, default=float)
print('DONE adopt',adopt,'improve',improve, flush=True)
