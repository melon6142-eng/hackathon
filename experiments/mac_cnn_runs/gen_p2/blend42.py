import numpy as np,csv
from sklearn.metrics import average_precision_score as ap
y=np.array([float(r['hardness']) for r in csv.DictReader(open('~/AI/daker/open/train.csv'))]);hi=y>230;lo=y<170;mid=~hi&~lo
bl1=np.load('../../oof_blend1.npy');vo=np.mean([np.load(f'../{d}/oof.npy') for d in ('dinoB_orig','dinoB_orig_s1','dinoB_orig_s2')],0)
vt=np.mean([np.load(f'../{d}/oof.npy') for d in ('dinoB_trans','dinoB_trans_s1','dinoB_trans_s2')],0);fp=np.load('oof_frozen_pre_rs42.npy');fd=np.load('oof_frozen_dino_rs42.npy')
def m(nm,o): e=o-y;print(f'{nm:28s} RMSE {np.sqrt((e**2).mean()):.3f} hi {e[hi].mean():+.1f} lo {e[lo].mean():+.1f} hiRMSE {np.sqrt((e[hi]**2).mean()):.1f} midRMSE {np.sqrt((e[mid]**2).mean()):.2f} loRMSE {np.sqrt((e[lo]**2).mean()):.2f} AUPRC {ap(hi,o):.3f} P@18 {hi[np.argsort(-o)[:18]].mean():.2f}')
for nm,o in (('BL1',bl1),('ViT orig3',vo),('ViT trans3',vt),('frozen DINO ridge',fd),('frozen PRE ridge',fp),('eq3 (현 최선)',(bl1+vo+vt)/3),('eq4 +PRE',(bl1+vo+vt+fp)/4),('eq3 BL1->PRE',(fp+vo+vt)/3)): m(nm,o)
print('corr PRE vs BL1/vo/vt',np.corrcoef([fp,bl1,vo,vt])[0,1:].round(3),' resid corr',np.corrcoef(fp-y,(bl1+vo+vt)/3-y)[0,1].round(3))
