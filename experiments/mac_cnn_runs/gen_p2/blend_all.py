# fold seed별: 기존 멤버 vs frozen PRE ridge, eq3 vs eq4. 고/저/중간 지표
import numpy as np,csv,os
from sklearn.metrics import average_precision_score as ap
y=np.array([float(r['hardness']) for r in csv.DictReader(open('~/AI/daker/open/train.csv'))]);hi=y>230;lo=y<170;mid=~hi&~lo
L=['| rs | 모델 | RMSE | 고편향 | 저편향 | 고RMSE | 중간RMSE | 저RMSE | AUPRC | P@18 |','|'+'---|'*10];D={}
for rs in ('42','7','123','2026'):
    if not os.path.exists(f'oof_frozen_pre_rs{rs}.npy'): continue
    if rs=='42': bl1=np.load('../../oof_blend1.npy');G=lambda c:[f'../dinoB_{c}'+s for s in ('','_s1','_s2')]
    else: bl1=np.load(f'../indepfold/bl1_rs{rs}.npy');G=lambda c:[f'../indepfold/rs{rs}_{c}_s{s}' for s in (42,1,2)]
    vo=np.mean([np.load(d+'/oof.npy') for d in G('orig')],0);vt=np.mean([np.load(d+'/oof.npy') for d in G('trans')],0)
    ftp=[f'ft/pre_rs{rs}_s{s}/oof.npy' for s in (42,1,2)];pv=np.mean([np.load(p) for p in ftp],0) if all(os.path.exists(p) for p in ftp) else None
    fp=np.load(f'oof_frozen_pre_rs{rs}.npy');fd=np.load(f'oof_frozen_dino_rs{rs}.npy')
    for nm,o in (('frozen DINO',fd),('frozen PRE',fp),('ViT orig3',vo),('eq3',(bl1+vo+vt)/3),('eq4(+PRE)',(bl1+vo+vt+fp)/4))+((('PRE-FT orig3',pv),('eq3 orig→PRE-FT',(bl1+pv+vt)/3),('eq4 PRE-FT+frozenPRE',(bl1+pv+vt+fp)/4)) if pv is not None else ()):
        e=o-y;r=[np.sqrt((e**2).mean()),e[hi].mean(),e[lo].mean(),np.sqrt((e[hi]**2).mean()),np.sqrt((e[mid]**2).mean()),np.sqrt((e[lo]**2).mean()),ap(hi,o),hi[np.argsort(-o)[:18]].mean()];D[(rs,nm)]=(r,o)
        L.append(f'| {rs} | {nm} | '+' | '.join(f'{v:.3f}' if i in(0,4,5,6,7) else f'{v:.1f}' for i,v in enumerate(r))+' |')
    a,b=D[(rs,'eq3')][1],D[(rs,'eq4(+PRE)')][1];d=(b-y)**2-(a-y)**2;L.append(f'| {rs} | ΔSSE eq4−eq3 | 전체 {d.mean():+.2f} | 고 {d[hi].mean():+.1f} | 저 {d[lo].mean():+.1f} | 중간 {d[mid].mean():+.2f} | 개선비율 {(d<0).mean():.2f} | | | |')
print('\n'.join(L));open('blend_all.md','w').write('\n'.join(L)+'\n')
