import os; os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import sys;sys.path.insert(0,'lib');sys.path.insert(0,'.')
import numpy as np,pandas as pd,json
from cv import y;from blend import *;from feats import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict,StratifiedKFold
from sklearn.metrics import roc_auc_score
R={};Y=y();n=len(Y);o=np.load('oof_blend1.npy');e=o-Y;rm=lambda a:np.sqrt((a**2).mean())
R['oof_rmse']=rm(e)
s=np.argsort(-e**2);tot=(e**2).sum()
for k in (5,10,18,25,50): R[f'top{k}_sse_share']=(e[s[:k]]**2).sum()/tot; R[f'rmse_wo_top{k}']=rm(e[s[k:]])
hi=Y>230;R['hi_n']=int(hi.sum());R['hi_sse_share']=(e[hi]**2).sum()/tot;R['rmse_wo_hi']=rm(e[~hi]);R['hi_bias']=e[hi].mean()
for lo,hi_ in [(0,170),(170,190),(190,210),(210,230),(230,999)]:
    m=(Y>=lo)&(Y<hi_);R[f'bin_{lo}_{hi_}']=(int(m.sum()),rm(e[m]),e[m].mean())
# public bootstrap
rng=np.random.default_rng(0)
for m in (300,500):
    b=[rm(e[rng.integers(0,n,m)]) for _ in range(20000)];R[f'boot{m}']=np.percentile(b,[2.5,10,50,90,97.5]).tolist()
    # finite: OOF 500 not resample but model; also robust: P(<=11.77)
    R[f'boot{m}_P<=11.77']=float(np.mean(np.array(b)<=11.77))
# test pred dist
sub=pd.read_csv('../out/submission_blend1_cv12.95.csv').hardness.values
R['train_y']=[Y.mean(),Y.std(),np.percentile(Y,[5,50,95]).tolist()]
R['oof_pred']=[o.mean(),o.std()];R['test_pred']=[sub.mean(),sub.std(),np.percentile(sub,[5,50,95]).tolist()]
R['pred>225_frac_oof_vs_test']=[float((o>225).mean()),float((sub>225).mean())]
R['pred<165_frac_oof_vs_test']=[float((o<165).mean()),float((sub<165).mean())]
# adversarial
SEL=[(L,k) for L in range(4) for k in (1,2)]
Htr,Hte=HX('train'),HX('test');Dtr,Dte=DINO('train',sel=SEL),DINO('test',sel=SEL)
h,_,_,_=hx('train');ht,_,_,_=hx('test')
def adv(A,B,C=0.1):
    X=np.vstack([A,B]);X=(X-X.mean(0))/(X.std(0)+1e-9);t=np.r_[np.zeros(len(A)),np.ones(len(B))]
    p=cross_val_predict(LogisticRegression(C=C,max_iter=3000),X,t,cv=StratifiedKFold(5,shuffle=True,random_state=0),method='predict_proba')[:,1]
    return roc_auc_score(t,p),p
R['adv_auc_HX'],ph=adv(Htr,Hte);R['adv_auc_DINO'],pd_=adv(Dtr,Dte,0.01)
# per-feature shift (hand)
from scipy.stats import ks_2samp
ks=sorted([(ks_2samp(h[c],ht[c]).statistic,c,h[c].mean(),ht[c].mean()) for c in h.columns],reverse=True)[:12]
R['top_ks_hand']=[(round(a,3),c,round(m1,3),round(m2,3)) for a,c,m1,m2 in ks]
# image stats: brightness, blur
from PIL import Image
def st(d,ids):
    out=[]
    for i in ids:
        a=np.asarray(Image.open(f'./open/{d}/{i}.png').convert('L'),float)
        lap=a[1:-1,1:-1]*4-a[:-2,1:-1]-a[2:,1:-1]-a[1:-1,:-2]-a[1:-1,2:]
        out.append((a.mean(),a.std(),lap.var()))
    return np.array(out)
tr_ids=pd.read_csv('./open/train.csv').ID;te_ids=sorted(x[:-4] for x in os.listdir('./open/test'))
Str,Ste=st('train',tr_ids),st('test',te_ids)
thr=np.percentile(Str[:,2],10)
R['img_mean_std_lapvar_train']=Str.mean(0).tolist();R['img_test']=Ste.mean(0).tolist()
R['blur_frac(lapvar<train p10)']=[0.10,float((Ste[:,2]<thr).mean())]
R['ks_img']=[ks_2samp(Str[:,j],Ste[:,j]).statistic for j in range(3)]
blur=Str[:,2]<thr;R['oof_rmse_blur_vs_sharp']=[rm(e[blur]),rm(e[~blur])]
# residual vs train-likeness (importance weighting estimate)
w=ph[:n]/(1-ph[:n]);w/=w.mean();R['weighted_oof_rmse_HXadv']=np.sqrt((w*e**2).mean())
w2=pd_[:n]/(1-pd_[:n]);w2/=w2.mean();R['weighted_oof_rmse_DINOadv']=np.sqrt((w2*e**2).mean())
json.dump(R,open('gap/res1.json','w'),indent=1,default=float);print(json.dumps(R,indent=1,default=float),flush=True)
# learning curve: holdout, repeated
from sklearn.model_selection import train_test_split
lc={}
for frac in (0.4,0.6,0.8):
    rs=[]
    for r in range(4):
        idx=np.random.default_rng(r).permutation(n);va=idx[:100];pool=idx[100:];tr=pool[:int(frac*500)]
        p,_=blend_fit_predict([Htr[tr],Dtr[tr]],Y[tr],[Htr[va],Dtr[va]]);rs.append(rm(p-Y[va]))
    lc[int(frac*500)]=float(np.mean(rs));print('lc',frac,lc,flush=True)
ns=np.array(list(lc));v=np.array(list(lc.values()))
# fit rmse^2 = a + b/n
A=np.c_[np.ones(3),1/ns];a,b=np.linalg.lstsq(A,v**2,rcond=None)[0]
R['lc']=lc;R['lc_extrap']={k:float(np.sqrt(max(a+b/k,0))) for k in (400,500,1000)}
json.dump(R,open('gap/res1.json','w'),indent=1,default=float);print(R['lc'],R['lc_extrap'])
