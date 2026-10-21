# 현실성: v4(기존 gen) vs v5(gen2) vs pool_all 10k 중 500. 동일 프로토콜(gen_cf real_emb PCA50, 로지스틱 AUC, NN 거리)
import sys,json,numpy as np
sys.path.insert(0,'.');sys.path.insert(0,'../gen_cf')
from multiprocessing import Pool
from s1_fit import stats,SN,auc2
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree
import emb,gen as G0,gen2
X=np.load('train_imgs.npy');RE=np.load('../gen_cf/real_emb.npz');Er=RE['E'];drr=RE['drr'];Sr=stats(X)
mu,sd=Er.mean(0),Er.std(0)+1e-6;pca=PCA(50,random_state=0).fit((Er-mu)/sd);Z=lambda E:pca.transform((E-mu)/sd);Zr=Z(Er);tr=cKDTree(Zr)
def hf(X): return np.array([[np.abs(np.diff(x,axis=1)).mean(),len(np.unique(np.round(x*255)))] for x in X])
def ev(Xg,tag):
    Eg=emb.embed(Xg);Zg=Z(Eg);dg=tr.query(Zg,k=1)[0];Sg=stats(Xg);ss=(Sr.mean(0),Sr.std(0)+1e-9)
    return dict(tag=tag,auc_dino=float(auc2(Zr,Zg)),auc_dino_s1=float(auc2(Zr,Zg,1)),auc_stats=float(auc2((Sr-ss[0])/ss[1],(Sg-ss[0])/ss[1])),
      nn_med=float(np.median(dg)),nn_ratio=float(np.median(dg)/np.median(drr)),frac_in_p95=float((dg<=np.percentile(drr,95)).mean()),
      real_cover=float(np.median(cKDTree(Zg).query(Zr,k=1)[0])/np.median(drr)),
      stat_smd={n:round(float((Sg[:,j].mean()-Sr[:,j].mean())/Sr[:,j].std()),3) for j,n in enumerate(SN)},
      grad_ratio=float(hf(Xg)[:,0].mean()/hf(X)[:,0].mean()),uniq_gen=float(hf(Xg)[:,1].mean()),uniq_real=float(hf(X)[:,1].mean())),Zg
def a(t): return t[0](*t[1:])
if __name__=='__main__':
    P4=json.load(open('../gen_cf/fitted_params_v4.json'));P5=json.load(open('fitted_params_v5.json'))
    with Pool(11) as pl:
        X4=np.stack(pl.map(a,[(G0.gen,P4[i],70000+i) for i in range(500)]))
        X5=np.stack(pl.map(a,[(gen2.gen,P5[i],70000+i) for i in range(500)]))
    R=[];r,_=ev(X4,'v4_old_gen');R.append(r);print(r,flush=True)
    r,_=ev(X5,'v5_tex_fitted');R.append(r);print(r,flush=True)
    import os
    if os.path.exists('pool_all.npz'):
        Xp=np.load('pool_all.npz')['X'][:500].astype(np.float32)/255;r,_=ev(Xp,'v5_pool10k_sample500');R.append(r);print(r,flush=True)
    json.dump(R,open('realism.json','w'),indent=1)
    import cv2;y=np.loadtxt('~/AI/daker/open/train.csv',delimiter=',',skiprows=1,usecols=1);o=np.argsort(y);ix=o[np.linspace(0,499,8).astype(int)]
    im=np.concatenate([np.concatenate([X[i] for i in ix],1),np.concatenate([X4[i] for i in ix],1),np.concatenate([X5[i] for i in ix],1)],0)
    cv2.imwrite('real_v4_v5.png',(im*255).astype(np.uint8))
