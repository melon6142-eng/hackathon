import numpy as np,pandas as pd,csv,json,os
R='~/AI/daker';C=f'{R}/cnn/runs';H=f'{C}/repro_vit'
y=np.array([float(r['hardness']) for r in csv.DictReader(open(f'{R}/open/train.csv'))])
rm=lambda p:float(np.sqrt(((p-y)**2).mean()))
O=['dinoB_orig','dinoB_orig_s1','dinoB_orig_s2'];T=['dinoB_trans','dinoB_trans_s1','dinoB_trans_s2']
L=['| 멤버 | 원본 OOF | 재학습 OOF | OOF corr | test corr | test max|diff| | test RMS diff | ckpt재추론 max|diff| |','|---|---|---|---|---|---|---|---|']
def st(a,b): return float(np.corrcoef(a,b)[0,1]),float(np.abs(a-b).max()),float(np.sqrt(((a-b)**2).mean()))
old={};new={}
for t in O+T:
    old[t]=(np.load(f'{C}/{t}/oof.npy'),np.load(f'{C}/{t}/test.npy'))
    if not os.path.exists(f'{H}/{t}/test.npy'): L.append(f'| {t} | {rm(old[t][0]):.3f} | (미완) | | | | | |');continue
    new[t]=(np.load(f'{H}/{t}/oof.npy'),np.load(f'{H}/{t}/test.npy'))
    ck=json.load(open(f'{H}/{t}/ckpt_infer.json')) if os.path.exists(f'{H}/{t}/ckpt_infer.json') else {}
    cm=max([v['maxdiff_test'] for k,v in ck.items() if k!='oof_rmse'],default=float('nan'))
    c1=st(old[t][0],new[t][0]);c2=st(old[t][1],new[t][1])
    L.append(f'| {t} | {rm(old[t][0]):.3f} | {rm(new[t][0]):.3f} | {c1[0]:.4f} | {c2[0]:.4f} | {c2[1]:.2f} | {c2[2]:.2f} | {cm:.2e} |')
bt=pd.read_csv(f'{R}/cnn/sub_bl1.csv').iloc[:,1].values;bo=np.load(f'{H}/ref/oof_blend1.npy')
ref=pd.read_csv(f'{H}/ref/cand_BL1_Bavgorig-trans_cv12.61.csv');w=np.array([0.4398,0.2831,0.2773])
def blend(b,o,t,d): return w[0]*b+w[1]*np.mean([d[k][o] for k in O],0)+w[2]*np.mean([d[k][o] for k in T],0)
c=float((ref.iloc[:,1].values-blend(bt,1,None,old)).mean())
L+=['',f'블렌드 절편 c={c:.4f} (원본 후보 CSV에서 역산), 원본 재구성 잔차 max {np.abs(ref.iloc[:,1].values-blend(bt,1,None,old)-c).max():.2e}']
if len(new)==6:
    pn=blend(bt,1,None,new)+c;on=blend(bo,0,None,new)+c;oo=blend(bo,0,None,old)+c
    s=st(ref.iloc[:,1].values,pn)
    L+=[f'블렌드 OOF RMSE 원본 {rm(oo):.3f} → 재학습 {rm(on):.3f} (같은 가중치, in-sample)',
        f'블렌드 test: corr {s[0]:.5f}, max|diff| {s[1]:.3f}, RMS diff {s[2]:.3f}, mean {ref.iloc[:,1].mean():.2f}→{pn.mean():.2f}']
    out=ref.copy();out.iloc[:,1]=pn;out.to_csv(f'{H}/sub_repro_blend.csv',index=False);L.append('→ sub_repro_blend.csv 저장(원본 미변경)')
open(f'{H}/compare.md','w').write('\n'.join(L));print('\n'.join(L))
