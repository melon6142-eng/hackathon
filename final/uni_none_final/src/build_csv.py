# Build uni_none blend CSVs: BL1 + uni_none(orig ch, no aug, uniform) 3-seed + ViT trans 3-seed (final_clean), fold seed 42.
# usage: build_csv.py [src]   src=bal_d4 (existing preds, default) | ckpt (preds re-inferred from saved ckpts)
import sys,json,numpy as np,pandas as pd
from scipy.optimize import nnls
from sklearn.model_selection import KFold
from common import *
src=sys.argv[1] if len(sys.argv)>1 else 'bal_d4'
FC='~/AI/daker/final_clean'; BD='~/AI/daker/cnn/runs/bal_d4/arms'
Y=y(); ss=pd.read_csv(f'{R}/sample_submission.csv'); ids=test_ids()
def U(s,kind):
    if src=='bal_d4': return np.load(f'{BD}/uni_none_s{s}/{kind}.npy')
    return np.load(f'{O}/arms/ckpt/uni_none_s{s}/{"oof" if kind=="oof" else "test_from_ckpt"}.npy')
b,bt=np.load(f'{FC}/bl1/oof_fs42.npy'),np.load(f'{FC}/bl1/test.npy')
u,ut=[np.mean([U(s,k) for s in (42,1,2)],0) for k in ('oof','test')]
t,tt=[np.mean([np.load(f'{FC}/vit/trans_s{s}/{k}.npy') for s in (42,1,2)],0) for k in ('oof','test')]
X=np.column_stack([b,u,t,np.ones(500)]); w=nnls(X,Y)[0]
def ncv(cols,reps=20):
    Xc=np.column_stack(cols+[np.ones(500)]);o=[]
    for r in range(reps):
        p=np.zeros(500)
        for tr,va in KFold(5,shuffle=True,random_state=100+r).split(Xc): ww,_=nnls(Xc[tr],Y[tr]);p[va]=Xc[va]@ww
        o.append(rm(p,Y))
    return float(np.mean(o))
P={'eq3':(bt+ut+tt)/3,'nnls':np.column_stack([bt,ut,tt,np.ones(1000)])@w}
ref={'eq3':pd.read_csv(f'{FC}/submission_final_clean_eq3.csv'),'nnls':pd.read_csv(f'{FC}/submission_final_clean.csv')}
suf='' if src=='bal_d4' else '_ckpt'; out={'src':src,'w_bl1_uni_trans_c':w.tolist(),'oof_eq3':rm((b+u+t)/3,Y),'oof_nnls_insample':rm(X@w,Y),'ncv':ncv([b,u,t])}
for k,p in P.items():
    sub=pd.DataFrame({'ID':ids,'hardness':p})
    assert list(sub.ID)==list(ss.ID) and list(sub.columns)==list(ss.columns) and len(sub)==len(ss)==1000 and np.isfinite(p).all()
    fn=f'{O}/submission_uninone_{k}{suf}.csv'; sub.to_csv(fn,index=False)
    r=ref[k]; assert list(r.ID)==ids; q=r.hardness.values
    out[k]=dict(file=fn,sha256=sha(fn),mean=float(p.mean()),std=float(p.std()),min=float(p.min()),max=float(p.max()),
        vs_final_clean=dict(corr=float(np.corrcoef(p,q)[0,1]),mean_diff=float(p.mean()-q.mean()),rms_diff=float(np.sqrt(((p-q)**2).mean())),maxabs=float(np.abs(p-q).max())))
    for k2,r2 in ref.items(): out[k][f'corr_vs_fc_{k2}']=float(np.corrcoef(p,r2.hardness.values)[0,1])
json.dump(out,open(f'{O}/csv_{src}.json','w'),indent=1); print(json.dumps(out,indent=1))
