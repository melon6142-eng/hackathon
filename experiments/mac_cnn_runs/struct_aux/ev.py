import numpy as np,pandas as pd,os,glob
from sklearn.metrics import average_precision_score
C='~/AI/daker/cnn';y=pd.read_csv('~/AI/daker/open/train.csv').hardness.values;hi=y>230;lo=y<170
av=lambda ts:np.mean([np.load(f'{C}/runs/{t}/oof.npy') for t in ts],0)
B=np.load(f'{C}/oof_blend1.npy');O=av(['dinoB_orig','dinoB_orig_s1','dinoB_orig_s2']);T=av(['dinoB_trans','dinoB_trans_s1','dinoB_trans_s2'])
def met(p):
    e=p-y;return dict(RMSE=np.sqrt((e**2).mean()),hi_bias=e[hi].mean(),lo_bias=e[lo].mean(),hi_RMSE=np.sqrt((e[hi]**2).mean()),mid_RMSE=np.sqrt((e[~hi&~lo]**2).mean()),AUPRC=average_precision_score(hi,p),P18=hi[np.argsort(-p)[:18]].mean())
R={};S={}
R['BL1']=met(B);R['ViT orig 3s (기존)']=met(O)
for s in ['42','1','2']: R[f'ViT orig s{s if s!="42" else ""}'.rstrip()]=met(np.load(f'{C}/runs/'+('dinoB_orig' if s=='42' else f'dinoB_orig_s{s}')+'/oof.npy'))
eq3=(B+O+T)/3;R['eq3']=met(eq3)
for W in ['0','0.3','1']:
    fs=[f'aux{W}_s{s}/oof.npy' for s in ['42','1','2'] if os.path.exists(f'aux{W}_s{s}/oof.npy')]
    for f in fs: R[f'aux{W} {f.split("/")[0][-3:]}']=met(np.load(f))
    if fs:
        A=np.mean([np.load(f) for f in fs],0);n=len(fs);R[f'aux{W} avg{n}s']=met(A)
        e4=(B+O+T+A)/4;R[f'eq4 (eq3+aux{W})']=met(e4);R[f'eq3 orig→aux{W}']=met((B+A+T)/3)
        S[W]=((e4-y)**2-(eq3-y)**2)
D=pd.DataFrame(R).T.round(3);print(D.to_string())
for W,d in S.items(): print(f'eq4 aux{W} per-image SSE delta: total {d.sum():.0f}, hi {d[hi].sum():.0f}, lo {d[lo].sum():.0f}, mid {d[~hi&~lo].sum():.0f}; improved {np.mean(d<0):.2f}')
