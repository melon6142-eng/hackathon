#!/usr/bin/env python3
"""Evaluate Exp A/B against base BL1+Borig3+Btrans3 (nCV≈12.601). Write report.md"""
import os, json, time, numpy as np, csv, warnings
warnings.filterwarnings('ignore')
from scipy.optimize import nnls
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

R='~/AI/daker'; O=f'{R}/cnn/runs/exp_abc'; CR=f'{R}/cnn/runs'
os.makedirs(f'{O}/oof', exist_ok=True)
y=np.array([float(r['hardness']) for r in csv.DictReader(open(f'{R}/open/train.csv'))])
hi=y>230; lo=y<170
r=lambda p: float(np.sqrt(((p-y)**2).mean()))
mk=lambda: make_pipeline(StandardScaler(), RidgeCV(np.logspace(0,5,21)))

# Base blend members (same as overnight ev_d11)
ld=lambda tag,k: np.load(f'{CR}/{tag}/{k}.npy')
av=lambda tags,k: np.mean([ld(t,k) for t in tags],0)
O3=['dinoB_orig','dinoB_orig_s1','dinoB_orig_s2']
T3=['dinoB_trans','dinoB_trans_s1','dinoB_trans_s2']
BL1=np.load(f'{R}/cnn/oof_blend1.npy')
B=np.column_stack([BL1, av(O3,'oof'), av(T3,'oof')])
# test side
sub=list(csv.DictReader(open(f'{R}/cnn/sub_bl1.csv')))
BL1t=np.array([float(row[list(row.keys())[1]]) for row in sub])
BT=np.column_stack([BL1t, av(O3,'test'), av(T3,'test')])

def A1(X): return np.column_stack([X, np.ones(len(X))])
def ncv_cols(X, Rps=10):
    out=[]
    for s in range(Rps):
        pp=np.zeros(len(y))
        for tr,va in KFold(5,shuffle=True,random_state=100+s).split(X):
            pp[va]=A1(X[va])@nnls(A1(X[tr]), y[tr])[0]
        out.append((r(pp),(pp-y)[hi].mean(),(pp-y)[lo].mean()))
    return np.array(out).mean(0)

def nested_add(new_oof_fn, Rps=10):
    """Strict nested: ridge inner-OOF for new member inside each outer fold."""
    # new_oof_fn(tr_idx) -> (inner_oof_on_tr, pred_on_va) when given outer tr/va via closure
    res=[]; 
    # We'll pass feature matrix instead
    raise NotImplementedError

def ridge_oof(X, rs=42):
    o=np.zeros(len(y))
    for tr,va in KFold(5,shuffle=True,random_state=rs).split(X):
        o[va]=mk().fit(X[tr],y[tr]).predict(X[va])
    return o

def nested_blend_new(Xnew, Rps=10):
    res=[]; 
    for s in range(Rps):
        pp=np.zeros(len(y))
        for tr,va in KFold(5,shuffle=True,random_state=100+s).split(Xnew):
            io=np.zeros(len(tr))
            for i,j in KFold(5,shuffle=True,random_state=7).split(tr):
                io[j]=mk().fit(Xnew[tr[i]],y[tr[i]]).predict(Xnew[tr[j]])
            full=mk().fit(Xnew[tr],y[tr]).predict(Xnew[va])
            w=nnls(A1(np.column_stack([B[tr], io])), y[tr])[0]
            pp[va]=A1(np.column_stack([B[va], full]))@w
        res.append((r(pp),(pp-y)[hi].mean(),(pp-y)[lo].mean(), pp.copy()))
    return np.array([(a,b,c) for a,b,c,_ in res]).mean(0), np.mean([p for *_,p in res],0)

def sse_improve(base_pp, new_pp):
    """Fraction of samples where squared error improves; mean SSE delta."""
    e0=(base_pp-y)**2; e1=(new_pp-y)**2
    return float((e1<e0).mean()), float((e1-e0).mean())

base=ncv_cols(B)
print(f'BASE nCV={base[0]:.3f} hi={base[1]:.1f} lo={base[2]:.1f}', flush=True)
# optimistic full-fit base OOF for SSE compare (same as gated diag note)
wB=nnls(A1(B),y)[0]; base_full=A1(B)@wB

lines=['# Exp A/B/C report', f'', f'작성: {time.ctime()} (Asia/Shanghai / CST)', f'',
       f'기준 블렌드: BL1 + ViT-B orig 3시드 평균 + ViT-B trans 3시드 평균',
       f'- base nCV (10×5 NNLS+intercept) = **{base[0]:.3f}** (목표 레퍼런스 12.601)',
       f'- base hi>230 bias = {base[1]:.1f}, lo<170 = {base[2]:.1f}',
       f'', f'채택 기준: RMSE 한 번 0.05가 아니라 여러 fold/seed에서 샘플별 SSE 개선.',
       f'상관 ≥0.97이면 NNLS 사실상 제외. 개선 <0.05면 fine-tune 보류.', '']

results={}

def eval_member(tag, Xa, keys_desc):
    X=Xa[:500]; XT=Xa[500:] if len(Xa)>500 else None
    oof=ridge_oof(X,42)
    np.save(f'{O}/oof/{tag}_oof.npy', oof)
    if XT is not None:
        te=mk().fit(X,y).predict(XT); np.save(f'{O}/oof/{tag}_test.npy', te)
    m, pp = nested_blend_new(X, 10)
    cr=[float(np.corrcoef(oof,B[:,i])[0,1]) for i in range(3)]
    A=A1(np.column_stack([B,oof])); w=nnls(A,y)[0]
    # compare nested pp vs base nested — rebuild base nested same seeds for fair SSE
    base_pps=[]
    new_pps=[]
    for s in range(10):
        pb=np.zeros(len(y)); pn=np.zeros(len(y))
        for tr,va in KFold(5,shuffle=True,random_state=100+s).split(X):
            pb[va]=A1(B[va])@nnls(A1(B[tr]),y[tr])[0]
            io=np.zeros(len(tr))
            for i,j in KFold(5,shuffle=True,random_state=7).split(tr):
                io[j]=mk().fit(X[tr[i]],y[tr[i]]).predict(X[tr[j]])
            full=mk().fit(X[tr],y[tr]).predict(X[va])
            wn=nnls(A1(np.column_stack([B[tr],io])),y[tr])[0]
            pn[va]=A1(np.column_stack([B[va],full]))@wn
        base_pps.append(pb); new_pps.append(pn)
    # per-rep SSE win rate
    wins=[((nb-y)**2<(bb-y)**2).mean() for bb,nb in zip(base_pps,new_pps)]
    sse_d=[(((nb-y)**2).mean()-((bb-y)**2).mean()) for bb,nb in zip(base_pps,new_pps)]
    # hi bias of nested mean
    pn_mean=np.mean(new_pps,0)
    decide='채택 후보' if (m[0] <= base[0]-0.05 and np.mean(wins)>=0.52 and w[3]>0.05) else (
        '보류 (미세이득)' if m[0] < base[0]-0.01 else '보류/기각')
    if max(cr)>=0.97 and w[3]<0.05:
        decide='보류 (상관↑, NNLS≈0) — fine-tune 보류'
    info=dict(tag=tag, solo=r(oof), solo_hi=float((oof-y)[hi].mean()), solo_lo=float((oof-y)[lo].mean()),
              ncv=float(m[0]), dlt=float(m[0]-base[0]), ncv_hi=float(m[1]), ncv_lo=float(m[2]),
              corr=cr, w=w.tolist(), win_rate=float(np.mean(wins)), sse_delta=float(np.mean(sse_d)),
              decide=decide, keys=keys_desc)
    results[tag]=info
    print(json.dumps({k:info[k] for k in info if k!='w'}), flush=True)
    return info

# ---- A ----
fa=np.load(f'{O}/feats_a.npz')
# primary: cls+mean+std both layers
Xa=np.concatenate([fa[k] for k in ['cls_L0','mean_L0','std_L0','cls_L1','mean_L1','std_L1']],1)
print('A feat', Xa.shape, flush=True)
infoA=eval_member('A_clsmeanstd', Xa, 'DINOv3 Lmid/Llast CLS+mean+std')
# ablations
for sn,keys in [
    ('A_last_clsmean',['cls_L1','mean_L1']),
    ('A_last_all',['cls_L1','mean_L1','std_L1']),
    ('A_mid_all',['cls_L0','mean_L0','std_L0']),
]:
    eval_member(sn, np.concatenate([fa[k] for k in keys],1), '+'.join(keys))

# ---- B ----
fb=np.load(f'{O}/feats_b.npz')
Xb=np.concatenate([fb[k] for k in sorted(fb.keys())],1)
print('B feat', Xb.shape, flush=True)
infoB=eval_member('B_neighbor', Xb, 'DINOv3 patch neighbor cos/L2 dist (4dir×2×5stat ×2layers)')
# residual correlation with base blend residual
oB=np.load(f'{O}/oof/B_neighbor_oof.npy')
res_base=y-base_full; res_B=y-oB
print('B residual corr with base residual', float(np.corrcoef(res_B,res_base)[0,1]), flush=True)
results['B_neighbor']['res_corr_base']=float(np.corrcoef(res_B,res_base)[0,1])

# ---- C from json ----
cpath=f'{O}/diag_c.json'
Cblock=['## 실험 C — mild blur/noise 민감도 (진단)', '']
if os.path.exists(cpath):
    C=json.load(open(cpath))
    Cblock += [f"- clean frozen-DINOv2 Ridge OOF RMSE = {C['clean_oof_rmse']:.3f}, hi_bias={C['clean_hi_bias']:.1f}",
               '', '| 섭동 | mean|Δpred| | RMSE(Δ) | corr(clean) | OOF RMSE | ΔRMSE | hi |Δ| |',
               '|---|---|---|---|---|---|---|']
    max_d=0
    for row in C['rows']:
        Cblock.append(f"| {row['name']} | {row['mean_abs_d']:.3f} | {row['rmse_d']:.3f} | {row['corr_clean']:.4f} | "
                      f"{row['oof_rmse']:.3f} | {row['delta_rmse']:+.3f} | {row['hi_mean_abs_d']:.3f} |")
        max_d=max(max_d, row['mean_abs_d'])
    # decision heuristic
    # "크게 흔들림": mean|d| > ~2 hardness points OR corr < 0.99 OR ΔRMSE > 0.3
    shaky=any(row['mean_abs_d']>2.0 or row['corr_clean']<0.99 or row['delta_rmse']>0.3 for row in C['rows'])
    if shaky:
        cdec='민감함 → consistency/증강 실험 근거 있음 (우선순위는 A/B 이득 여부에 종속)'
    else:
        cdec='둔감함 → consistency/증강 경로 중단 권고'
    Cblock += ['', f'**판정:** {cdec}', '']
    results['C']=dict(decide=cdec, max_mean_abs_d=max_d, rows=C['rows'], clean=C['clean_oof_rmse'])
else:
    Cblock += ['- diag_c.json 없음 (미실행/실패)', '']
    results['C']=dict(decide='미실행')

# ---- assemble report ----
def fmt(info):
    cr='/'.join(f'{c:.3f}' for c in info['corr'])
    w=','.join(f'{x:.3f}' for x in info['w'])
    return (f"| {info['tag']} | {info['solo']:.3f} | {info['solo_hi']:.1f} | {info['ncv']:.3f} | "
            f"{info['dlt']:+.3f} | {info['ncv_hi']:.1f} | {cr} | win={info['win_rate']:.2f} | {info['decide']} |")

lines += ['## 실험 A — DINOv3 ViT-B/16 frozen', '',
          '| tag | solo OOF | hi bias | nCV(+base) | Δ | nCV hi | corr BL1/Bo/Bt | SSE win율 | 판정 |',
          '|---|---|---|---|---|---|---|---|---|']
for k,v in results.items():
    if k.startswith('A_'): lines.append(fmt(v))
bestA=min([v for k,v in results.items() if k.startswith('A_')], key=lambda z:z['ncv'])
lines += ['', f"- 특징: {bestA['keys']} (주력 `{bestA['tag']}`)",
          f"- solo OOF={bestA['solo']:.3f}, 블렌드 nCV={bestA['ncv']:.3f} (Δ{bestA['dlt']:+.3f})",
          f"- 상관={bestA['corr']}, NNLS w={np.round(bestA['w'],3).tolist()}",
          f"- 샘플 SSE 개선 비율(10 seeds 평균)={bestA['win_rate']:.3f}, SSEΔ={bestA['sse_delta']:+.4f}",
          f"- **판정: {bestA['decide']}**",
          f"- 다음: " + ('fine-tune 검토' if '채택' in bestA['decide'] else 'fine-tune 보류. 다른 표현(계층/해상도)만 저비용 스모크.'),
          '']

lines += ['## 실험 B — DINO patch 이웃 유사도 (분할 없음)', '',
          '| tag | solo OOF | hi bias | nCV(+base) | Δ | nCV hi | corr BL1/Bo/Bt | SSE win율 | 판정 |',
          '|---|---|---|---|---|---|---|---|---|']
lines.append(fmt(results['B_neighbor']))
b=results['B_neighbor']
lines += ['', f"- 특징: {b['keys']}",
          f"- 블렌드 잔차와 새 모델 잔차 상관={b.get('res_corr_base', float('nan')):.3f}",
          f"- **판정: {b['decide']}**",
          f"- 다음: " + ('GNN 확장 금지 (이득 없음)' if '기각' in b['decide'] or '보류' in b['decide'] else '이웃 그래프/GNN 소규모 파일럿 가능'),
          f"- grain_agree(분할 기반)와는 별개 — 해당 폴더 미변경",
          '']

lines += Cblock

# summary recommendations
lines += ['## 종합 다음 추천', '']
recs=[]
if '채택' in bestA['decide']:
    recs.append('A: DINOv3 fine-tune(last-4) 소규모 시드 1개 파일럿')
else:
    recs.append('A: DINOv3 fine-tune 보류')
if '채택' in b['decide']:
    recs.append('B: 이웃 유사도 유지·GNN 소규모만')
else:
    recs.append('B: GNN 확장 금지; grain_agree 결과만 비교')
recs.append('C: ' + results['C']['decide'])
recs.append('대기 중 dino12 multilayer 시드·grain_agree·fold_rs 검증과 숫자만 비교 (중복 실행 없음)')
for i,x in enumerate(recs,1):
    lines.append(f'{i}. {x}')

open(f'{O}/report.md','w').write('\n'.join(lines)+'\n')
json.dump(results, open(f'{O}/results.json','w'), indent=2, default=float)
print('WROTE', f'{O}/report.md', flush=True)
print('\n'.join(lines), flush=True)
