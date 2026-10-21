# Step3: 반사실 sweep. 기존 학습 모델(동결 DINO Ridge, 수작업특징 Ridge, ViT-B fold ckpt)로 예측. 학습은 train 500만, 합성은 예측만.
import sys,os,json,time,glob,numpy as np,torch,csv,timm,torch.nn as nn,torch.nn.functional as F
O=os.path.dirname(os.path.abspath(__file__));sys.path.insert(0,O);R='~/AI/daker'
sys.path.insert(0,f'{R}/phase_sep')
from gen import gen,KEYS
from multiprocessing import Pool
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from s1_fit import g1,G
import pandas as pd
def hcf(im):
    from feat import feats; from seg import P0
    return feats((np.clip(im,0,1)*255).astype(np.uint8),P0)
VARS={'n_grains':'g_Amean','aspect':'e_ar','ph_frac':'ph_frac','pore_frac':'po_frac','bnd_c':'b_contrast','noise':'x_sig'}
QL=[5,20,35,50,65,80,95]
if __name__=='__main__':
    import emb
    from scipy.spatial import cKDTree
    pool=Pool(10);L=open(f'{O}/s2.log','a')
    def P_(s): print(s,flush=True);L.write(s+'\n');L.flush()
    t0=time.time()
    y=np.array([float(r['hardness']) for r in csv.DictReader(open(f'{R}/open/train.csv'))])
    Xr,_=torch.load(f'{R}/cnn/imgs.pt');Xr=Xr[:,0].numpy()
    RE=np.load(f'{O}/real_emb.npz');Er=RE['E'];drr=RE['drr'];p95=np.percentile(drr,95)
    from sklearn.decomposition import PCA
    mu,sd=Er.mean(0),Er.std(0)+1e-6;pca=PCA(50,random_state=0).fit((Er-mu)/sd);Z=lambda E:pca.transform((E-mu)/sd);tr=cKDTree(Z(Er))
    best=json.load(open(f'{O}/fitted_params_v4.json'))
    folds=list(KFold(5,shuffle=True,random_state=42).split(y));fold_of=np.zeros(500,int)
    for f,(a,b) in enumerate(folds): fold_of[b]=f
    # ---- 이미지 세트: (a) 재현 세트 500(새 seed), (b) sweep
    o=np.argsort(y);bases=o[np.linspace(5,494,30).astype(int)]
    lev={v:np.percentile([p[v] for p in best],QL) for v in VARS}
    meta=[];plist=[];seeds=[]
    for i in range(500): meta.append(('rep',i,'',-1));plist.append(best[i]);seeds.append(40_000+i)
    for bi,i in enumerate(bases):
        for v in VARS:
            for li,val in enumerate(lev[v]):
                q=dict(best[i]);q[v]=float(val);meta.append(('cf',int(i),v,li));plist.append(q);seeds.append(50_000+bi)
    X=G(plist,seeds,pool);np.save(f'{O}/cf_X.npy',X);P_(f'gen {len(X)} {time.time()-t0:.0f}s')
    E=emb.embed(X);dnn=tr.query(Z(E),k=1)[0];P_(f'emb {time.time()-t0:.0f}s')
    import subprocess;subprocess.run([f'{R}/phase_sep/venv/bin/python',f'{O}/hc_feat.py',f'{O}/cf_X.npy',f'{O}/cf_hcfeat.csv'],check=True);HC=pd.read_csv(f'{O}/cf_hcfeat.csv');P_(f'hc feats {time.time()-t0:.0f}s')
    preds={}
    # ---- M1 동결 DINO Ridge (5 fold)
    oof=np.zeros(500);PF=[]
    for f,(a,b) in enumerate(folds):
        m=RidgeCV(alphas=np.logspace(0,5,21)).fit((Er[a]-mu)/sd,y[a]);oof[b]=m.predict((Er[b]-mu)/sd);PF.append(m.predict((E-mu)/sd))
    preds['FrozenDINO_Ridge']=np.array(PF);P_(f'FrozenRidge OOF {np.sqrt(((oof-y)**2).mean()):.3f}')
    # ---- M2 수작업 특징 Ridge
    Hr=pd.read_csv(f'{R}/phase_sep/feat_train.csv').drop(columns='ID');cols=Hr.columns;Hr=Hr.values;Hs=HC[cols].values
    hm,hs=Hr.mean(0),Hr.std(0)+1e-9;oof=np.zeros(500);PF=[]
    for f,(a,b) in enumerate(folds):
        m=RidgeCV(alphas=np.logspace(-2,4,25)).fit((Hr[a]-hm)/hs,y[a]);oof[b]=m.predict((Hr[b]-hm)/hs);PF.append(m.predict(np.clip((Hs-hm)/hs,-8,8)))
    preds['Handcrafted_Ridge']=np.array(PF);P_(f'HC Ridge OOF {np.sqrt(((oof-y)**2).mean()):.3f}')
    # ---- M3 ViT-B fine-tune ckpts (기존 학습본만 로드)
    dev='mps';MU,SD=Xr.mean(),Xr.std();ym,ys=y.mean(),y.std()
    class M(nn.Module):
        def __init__(s):
            super().__init__();s.b=timm.create_model('vit_base_patch14_dinov2.lvd142m',pretrained=False,num_classes=0,img_size=252)
            d=s.b.num_features;s.h=nn.Sequential(nn.Linear(2*d,256),nn.GELU(),nn.Dropout(0.1),nn.Linear(256,1))
        def forward(s,x):
            t=s.b.forward_features(x);pt=t[:,s.b.num_prefix_tokens:];return s.h(torch.cat([t[:,0],pt.mean(1)],1)).squeeze(1)
    Xt=torch.tensor((X-MU)/SD,dtype=torch.float32)[:,None].repeat(1,3,1,1)
    @torch.no_grad()
    def vpred(m):
        out=[]
        for i in range(0,len(Xt),32):
            xb=F.interpolate(Xt[i:i+32].to(dev),size=252,mode='bilinear',align_corners=False);acc=0
            for k in (0,1,4,5):
                xx=torch.rot90(xb,k%4,(2,3));xx=torch.flip(xx,(3,)) if k>=4 else xx;acc=acc+m(xx)
            out.append((acc/4).cpu())
        return torch.cat(out).numpy()*ys+ym
    fam={'ViT_mild':[f'{R}/cnn/runs/dinoB_orig_mild_s{s}' for s in (42,1,2)],'ViT_orig':sorted(glob.glob(f'{R}/cnn/runs/freq_ablation/vit_s*'))}
    for fn,dirs in fam.items():
        for dd in dirs:
            FL=[f for f in range(5) if os.path.exists(f'{dd}/fold{f}_ck.pt')]
            if not FL: continue
            PF=[]
            for f in FL:
                m=M().to(dev);m.load_state_dict(torch.load(f'{dd}/fold{f}_ck.pt',map_location=dev));m.eval();PF.append(vpred(m));del m
            preds[f'{fn}_{os.path.basename(dd)}']=np.array(PF);json.dump(FL,open(f'{O}/folds_{fn}_{os.path.basename(dd)}.json','w'));P_(f'{fn} {dd} {time.time()-t0:.0f}s')
    np.savez(f'{O}/cf_raw.npz',meta=np.array(meta,dtype=object),dnn=dnn,p95=p95,bases=bases,**{'P_'+k:v for k,v in preds.items()},lev=json.dumps({k:list(v) for k,v in lev.items()}))
    HC.to_csv(f'{O}/cf_hcfeat.csv',index=False);np.save(f'{O}/cf_emb.npy',E)
    P_('S2 DONE')
