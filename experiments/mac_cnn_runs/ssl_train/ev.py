import sys,os,csv,json,numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score as ap
tag=sys.argv[1];rs=sys.argv[2];R='~/AI/daker';G='../gen_p2'
y=np.array([float(r['hardness']) for r in csv.DictReader(open(f'{R}/open/train.csv'))]);hi=y>230;lo=y<170;mid=~hi&~lo
def fit(Ftr,ytr,Fva):
    m,s=Ftr.mean(0),Ftr.std(0)+1e-6;r=RidgeCV(alphas=np.logspace(0,6,25)).fit((Ftr-m)/s,ytr);return r.predict((Fva-m)/s)
o=np.zeros(500)
for f,(tr,va) in enumerate(KFold(5,shuffle=True,random_state=int(rs)).split(y)):
    Fs=np.load(f'{tag}/rs{rs}_f{f}/feat_train.npy');o[va]=fit(Fs[tr],y[tr],Fs[va])
np.save(f'oof_{tag}_rs{rs}.npy',o)
def vit(kind): return [f'../dinoB_{kind}'+s for s in ('','_s1','_s2')] if rs=='42' else [f'../indepfold/rs{rs}_{kind}_s{s}' for s in (42,1,2)]
mo=lambda D:np.mean([np.load(d+'/oof.npy') for d in D],0)
bl1=np.load('../../oof_blend1.npy') if rs=='42' else np.load(f'../indepfold/bl1_rs{rs}.npy')
vo,vt=mo(vit('orig')),mo(vit('trans'));fd=np.load(f'{G}/oof_frozen_dino_rs{rs}.npy')
def M(o):
    e=o-y;return dict(rmse=np.sqrt((e**2).mean()),hb=e[hi].mean(),lb=e[lo].mean(),hr=np.sqrt((e[hi]**2).mean()),mr=np.sqrt((e[mid]**2).mean()),lr=np.sqrt((e[lo]**2).mean()),auprc=ap(hi,o),p18=hi[np.argsort(-o)[:18]].mean())
eq3=(bl1+vo+vt)/3;eq4=(bl1+vo+vt+o)/4
C={'frozen DINO':fd,'SSL frozen':o,'eq3':eq3,'eq4(+SSL)':eq4}
L=[f'## rs{rs} ({tag})','| 모델 | RMSE | 고편향 | 저편향 | 고RMSE | 중RMSE | 저RMSE | AUPRC | P@18 |','|'+'---|'*9]
res={}
for k,v in C.items():
    m=M(v);res[k]={a:float(b) for a,b in m.items()};L.append(f"| {k} | {m['rmse']:.3f} | {m['hb']:+.1f} | {m['lb']:+.1f} | {m['hr']:.1f} | {m['mr']:.2f} | {m['lr']:.2f} | {m['auprc']:.3f} | {m['p18']:.3f} |")
d=(eq4-y)**2-(eq3-y)**2;d2=(o-y)**2-(fd-y)**2
L.append(f'ΔSSE/샘플 eq4−eq3: 전체 {d.mean():+.2f} 고 {d[hi].mean():+.2f} 저 {d[lo].mean():+.2f} 중 {d[mid].mean():+.2f}, 개선비율 {(d<0).mean():.2f}; SSL−DINO 개선비율 {(d2<0).mean():.2f}')
res['dsse']=dict(all=float(d.mean()),hi=float(d[hi].mean()),lo=float(d[lo].mean()),mid=float(d[mid].mean()),frac=float((d<0).mean()))
np.save(f'dsse_{tag}_rs{rs}.npy',d)
print('\n'.join(L));open(f'ev_{tag}_rs{rs}.md','w').write('\n'.join(L)+'\n');json.dump(res,open(f'ev_{tag}_rs{rs}.json','w'),indent=1)
