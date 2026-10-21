import json,glob,numpy as np,csv,matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
R='~/AI/daker';y=np.array([float(r['hardness']) for r in csv.DictReader(open(f'{R}/open/train.csv'))]);hi=y>230
d=json.load(open('frozen.json'));r=d['res'];lp=json.load(open('lp_sweep.json'))
c=np.array(r['clean']['retrain']);ct=np.array(r['clean']['transfer'])
V={}
for s in [42,1,2]:
    fs=sorted(glob.glob(f'vit_s{s}/fold*_preds.npz'))
    if not fs: continue
    o={}
    for f in fs:
        z=np.load(f)
        for k in z.files:
            if k=='va':continue
            o.setdefault(k,np.full(500,np.nan))[z['va']]=z[k]
    V[s]=o
vr={}
for n in r:
    if V:
        rm=[np.sqrt(np.nanmean((V[s][n]-y)**2)) for s in V];rc=[np.sqrt(np.nanmean((V[s]['clean']-y)**2)) for s in V]
        vr[n]=dict(d=np.array(rm)-np.array(rc),hb=np.mean([np.nanmean((V[s][n]-y)[hi]) for s in V]),abs=np.mean(rm))
L=['# 공간 주파수 대역 제거 진단 (freq_ablation)','',
f'- 대역(cycles/px → 주기): B1 <1/64 (>64px), B2 1/64–1/16 (16–64px), B3 1/16–1/4 (4–16px), B4 >1/4 (<4px). 경계는 log-주파수 cosine taper, DC(평균밝기)는 항상 유지.',
f'- 학습 이미지 자기상관 1/e 길이 ≈ {d["corr_len"]}px (구조의 대표 크기 ~수~10px 수준 → B3가 개별 결정립/경계 스케일, B2는 결정립 군집·2상 분포 스케일). 대역 에너지 비율: '+', '.join(f'{k} {v*100:.0f}%' for k,v in d['band_energy'].items()),
'- 모델: (a) frozen DINOv2 ViT-B (CLS+patch mean) + Ridge = BL1의 DINO 부분 대리(핸드크래프트 제외), nCV 5-fold×10 seed(rs 100+s). (b) ViT-B orig last-4 fine-tune, 기존 레시피 그대로 3 seed(42,1,2) 재학습(가중치 저장), fold rs42. 기존 OOF/ckpt는 건드리지 않음. test 미사용, 제출 없음.',
'- transfer = clean으로 학습한 모델에 필터 이미지를 넣은 OOF (분포 이탈+정보 손실이 섞임). retrain = 필터 이미지로 다시 학습/평가 (분포 이탈 제거 → 정보 손실만 남음, 결정적 시험).',
'- shift = clean 임베딩 PCA50-whiten 공간에서 같은 이미지 clean↔필터 거리 평균; MMD(RBF). 대조군: ps_* = 해당 대역 진폭 유지·위상만 무작위(에너지 동일), noise_* = 가우시안 노이즈 σ 단계.','',
'| 변형 | shift | MMD | frozen transfer ΔRMSE | frozen retrain ΔRMSE (±sd, 개선 seed/10) | retrain hi>230 bias | ViT-FT transfer ΔRMSE (3 seed) | ViT hi bias |','|---|---|---|---|---|---|---|---|']
for n,v in r.items():
    a=np.array(v['retrain'])-c;t=np.mean(v['transfer'])-ct.mean()
    vv=vr.get(n);vs=f'{vv["d"].mean():+.2f} ({", ".join(f"{x:+.2f}" for x in vv["d"])})' if vv else 'n/a';vh=f'{vv["hb"]:.1f}' if vv else 'n/a'
    L.append(f'| {n} | {v["dpair"]:.1f} | {v["mmd"]:.3f} | {t:+.2f} | {a.mean():+.2f} ±{a.std():.2f} ({int((a<0).sum())}/10) | {v["retrain_hibias"]:.1f} | {vs} | {vh} |')
L+=['',f'clean 기준: frozen RMSE {c.mean():.3f}, hi bias {r["clean"]["retrain_hibias"]:.1f}'+(f'; ViT-FT 3seed 평균 {vr["clean"]["abs"]:.3f}, hi bias {vr["clean"]["hb"]:.1f}' if vr else ''),'',
'## 저역통과 cutoff 스윕 (frozen retrain, 10 seed)','','| cutoff 주기 | RMSE | sd | hi bias |','|---|---|---|---|']
for k,v in lp.items(): L.append(f'| {"없음" if k=="None" else k+"px"} | {np.mean(v["rmse"]):.3f} | {np.std(v["rmse"]):.3f} | {v["hibias"]:.1f} |')
open('report_tables.md','w').write('\n'.join(L)+'\n')
fig,ax=plt.subplots(1,3,figsize=(17,5))
names=[n for n in r if n!='clean'];ra=[np.mean(np.array(r[n]['retrain'])-c) for n in names];rs=[np.std(np.array(r[n]['retrain'])-c) for n in names]
cols=['C0' if n.startswith('rm') else 'C1' if n.startswith('keep') else 'C2' if n.startswith('ps') else 'gray' for n in names]
ax[0].barh(names,ra,xerr=rs,color=cols);ax[0].axvline(0,c='k');ax[0].set_title('frozen DINO+Ridge: retrain ΔRMSE (info loss)');ax[0].invert_yaxis()
for n in names:
    x=r[n]['dpair'];ax[1].scatter(x,np.mean(r[n]['transfer'])-ct.mean(),c='gray' if n.startswith('noise') else ('C0' if n.startswith('rm') else 'C1' if n.startswith('keep') else 'C2'),marker='o');ax[1].annotate(n,(x,np.mean(r[n]['transfer'])-ct.mean()),fontsize=7)
    ax[1].scatter(x,np.mean(np.array(r[n]['retrain'])-c),marker='x',c='k')
ax[1].set_xlabel('embedding shift');ax[1].set_ylabel('ΔRMSE (o transfer, x retrain)');ax[1].set_title('shift vs damage (noise=gray control)')
if vr:
    ax[2].barh(names,[vr[n]['d'].mean() for n in names],xerr=[vr[n]['d'].std() for n in names],color=cols);ax[2].axvline(0,c='k');ax[2].invert_yaxis();ax[2].set_title('ViT-B FT (3 seeds): transfer ΔRMSE')
plt.tight_layout();plt.savefig('freq_ablation.png',dpi=110)
L2=[f'ViT 사용 fold 수: '+str({s:int((~np.isnan(V[s]["clean"])).sum()/100) for s in V})];open('vit_folds.txt','w').write(L2[0]);print(L2[0]);print(open('report_tables.md').read())
