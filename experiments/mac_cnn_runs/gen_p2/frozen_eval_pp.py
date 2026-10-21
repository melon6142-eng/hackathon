# fold 정합 frozen 평가: fold f의 val은 pp_rs{rs}_f{f}(그 fold 학습 이미지 파라미터로만 만든 합성으로 사전학습) 특징 + fold-train ridge
import sys,os,csv,numpy as np,torch,timm,torch.nn.functional as F
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
rs=int(sys.argv[1]);R='~/AI/daker';y=np.array([float(r['hardness']) for r in csv.DictReader(open(f'{R}/open/train.csv'))]);hi=y>230;lo=y<170
X=np.load('train_imgs.npy');MU,SD=float(X.mean()),float(X.std());dev='mps'
def feats(sd=None):
    m=timm.create_model('vit_base_patch14_dinov2.lvd142m',pretrained=True,num_classes=0,img_size=252)
    if sd: m.load_state_dict(torch.load(sd),strict=False)
    m=m.eval().to(dev);out=[]
    with torch.no_grad():
        for i in range(0,500,32):
            x=torch.tensor((X[i:i+32]-MU)/SD)[:,None].repeat(1,3,1,1);x=F.interpolate(x,size=252,mode='bilinear',align_corners=False).to(dev);acc=0
            for k in range(8):
                xx=torch.rot90(x,k%4,(2,3));xx=torch.flip(xx,(3,)) if k>=4 else xx;t=m.forward_features(xx);acc=acc+torch.cat([t[:,0],t[:,1:].mean(1)],1)
            out.append((acc/8).cpu().numpy())
    return np.concatenate(out)
fd='feat_dino_tta.npy'
Fd=np.load(fd) if os.path.exists(fd) else feats();np.save(fd,Fd)
folds=list(KFold(5,shuffle=True,random_state=rs).split(y));od=np.zeros(500);on=np.zeros(500)
def fit(Ftr,ytr,Fva):
    m,s=Ftr.mean(0),Ftr.std(0)+1e-6;r=RidgeCV(alphas=np.logspace(0,6,25)).fit((Ftr-m)/s,ytr);return r.predict((Fva-m)/s)
for f,(tr,va) in enumerate(folds):
    fn=f'feat_pp_rs{rs}_f{f}.npy';Fs=np.load(fn) if os.path.exists(fn) else feats(f'pp_rs{rs}_f{f}.pt');np.save(fn,Fs)
    od[va]=fit(Fd[tr],y[tr],Fd[va]);on[va]=fit(Fs[tr],y[tr],Fs[va])
np.save(f'oof_frozen_dino_rs{rs}.npy',od);np.save(f'oof_frozen_pp_rs{rs}.npy',on)
def m(o): e=o-y;return f'RMSE {np.sqrt((e**2).mean()):.3f} hi {e[hi].mean():+.1f} lo {e[lo].mean():+.1f} hiRMSE {np.sqrt((e[hi]**2).mean()):.1f}'
print(f'rs{rs} frozen DINO ridge: {m(od)}\nrs{rs} frozen pair-pretrain(P3) ridge: {m(on)}')
