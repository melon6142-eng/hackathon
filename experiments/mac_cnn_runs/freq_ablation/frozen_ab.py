import sys,os,time,json,numpy as np,torch,timm,torch.nn.functional as F,csv
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); from filt import VARIANTS,FR,BANDS
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
R='~/AI/daker';O=f'{R}/cnn/runs/freq_ablation';dev='mps'
y=np.array([float(r['hardness']) for r in csv.DictReader(open(f'{R}/open/train.csv'))]);hi=y>230
ARCH='vit_base_patch14_dinov2.lvd142m';S=252
m=timm.create_model(ARCH,pretrained=True,num_classes=0,img_size=S).eval().to(dev)
dc=timm.data.resolve_data_config({},model=m)
mu=torch.tensor(dc['mean'],device=dev).view(1,3,1,1);sd=torch.tensor(dc['std'],device=dev).view(1,3,1,1)
Xraw,_=torch.load(f'{R}/cnn/imgs.pt');Xraw=Xraw[:,:1].float()
@torch.no_grad()
def feats(X):
    out=[]
    for i in range(0,len(X),16):
        x=F.interpolate(X[i:i+16].repeat(1,3,1,1),size=S,mode='bilinear',align_corners=False).to(dev);x=(x-mu)/sd
        o=m.forward_intermediates(x,indices=[len(m.blocks)-1],intermediates_only=True,output_fmt='NLC',norm=True,return_prefix_tokens=True)
        p,pre=o[0];out.append(torch.cat([pre[:,0],p.mean(1)],1).float().cpu().numpy())
    return np.concatenate(out)
def pixstats(X):
    x=X[:,0].numpy();gx=np.diff(x,axis=2);gy=np.diff(x,axis=1)
    return dict(mean=float(x.mean()),std=float(x.std(axis=(1,2)).mean()),grad=float(np.sqrt((gx**2).mean()+(gy**2).mean())))
# PSD / grain scale
x=Xraw[:,0].double().numpy();x=x-x.mean((1,2),keepdims=True);P=(np.abs(np.fft.fft2(x))**2).mean(0)
rb=np.round(FR*256).astype(int);psd=np.bincount(rb.ravel(),P.ravel())/np.bincount(rb.ravel())
ac=np.real(np.fft.ifft2(P));ac/=ac[0,0];acr=np.bincount(rb.ravel(),ac.ravel()*0+0)  # placeholder
r1=np.array([ac[0,k] for k in range(128)]);r2=np.array([ac[k,0] for k in range(128)]);acl=(r1+r2)/2
corr_len=int(np.argmax(acl<np.exp(-1)));zero=int(np.argmax(acl<0)) if (acl<0).any() else -1
band_energy={b:float(P[(FR>=lo)&(FR<hi_)].sum()/P.sum()) for b,(lo,hi_) in BANDS.items()}
FE={};PS={};t0=time.time()
for n,fn in VARIANTS:
    Xv=fn(Xraw);FE[n]=feats(Xv);PS[n]=pixstats(Xv);print(n,f'{time.time()-t0:.0f}s',flush=True)
np.savez(f'{O}/frozen_feats.npz',**FE)
mk=lambda:make_pipeline(StandardScaler(),RidgeCV(np.logspace(0,5,21)))
rm=lambda p:float(np.sqrt(((p-y)**2).mean()))
Fc=FE['clean'];pca=PCA(50,whiten=True).fit(StandardScaler().fit(Fc).transform(Fc));scl=StandardScaler().fit(Fc)
Zc=pca.transform(scl.transform(Fc))
def mmd(A,B):
    Z=np.vstack([A,B]);d=((Z[:,None]-Z[None])**2).sum(-1);g=1/np.median(d[d>0]);K=np.exp(-g*d);n=len(A)
    return float(K[:n,:n].mean()+K[n:,n:].mean()-2*K[:n,n:].mean())
res={}
for n in FE:
    Zv=pca.transform(scl.transform(FE[n]))
    tr_r=[];re_r=[];tr_hb=[];re_hb=[]
    for s in range(10):
        pt=np.zeros(500);pr=np.zeros(500)
        for tr,va in KFold(5,shuffle=True,random_state=100+s).split(y):
            pt[va]=mk().fit(Fc[tr],y[tr]).predict(FE[n][va])
            pr[va]=mk().fit(FE[n][tr],y[tr]).predict(FE[n][va])
        tr_r.append(rm(pt));re_r.append(rm(pr));tr_hb.append(float((pt-y)[hi].mean()));re_hb.append(float((pr-y)[hi].mean()))
    res[n]=dict(transfer=tr_r,retrain=re_r,transfer_hibias=float(np.mean(tr_hb)),retrain_hibias=float(np.mean(re_hb)),
        maha=float(np.sqrt((Zv**2).sum(1)).mean()),dpair=float(np.sqrt(((Zv-Zc)**2).sum(1)).mean()),mmd=mmd(Zc,Zv),pix=PS[n])
    print(n,np.mean(tr_r),np.mean(re_r),res[n]['dpair'],flush=True)
json.dump(dict(res=res,psd=psd[:129].tolist(),acl=acl.tolist(),corr_len=corr_len,ac_zero=zero,band_energy=band_energy),open(f'{O}/frozen.json','w'),indent=1)
print('DONE')
