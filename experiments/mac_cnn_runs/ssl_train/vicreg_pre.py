# 규정준수: fold-train 실제 이미지 500장 중 학습 fold만으로 VICReg 자기지도 사전학습 (합성/외부/test 학습 없음)
# 시작 가중치 = 원본 DINOv2 ViT-B(lvd142m). 합성 사전학습 가중치(gen_cf/gen_p2) 미사용.
import argparse,os,time,csv,glob,numpy as np,torch,timm,torch.nn as nn,torch.nn.functional as F
from sklearn.model_selection import KFold
p=argparse.ArgumentParser();p.add_argument('--rs',type=int);p.add_argument('--fold',type=int);p.add_argument('--tag',default='v1')
p.add_argument('--ep',type=int,default=8);p.add_argument('--bs',type=int,default=16);p.add_argument('--lr',type=float,default=3e-5);p.add_argument('--nblk',type=int,default=4)
p.add_argument('--test',action='store_true');a=p.parse_args()
dev='mps';torch.manual_seed(a.fold);np.random.seed(a.fold)
R='~/AI/daker';y=np.array([float(r['hardness']) for r in csv.DictReader(open(f'{R}/open/train.csv'))])
X=np.load('train_imgs.npy');MU,SD=float(X.mean()),float(X.std())
if a.fold>=0: tr=list(KFold(5,shuffle=True,random_state=a.rs).split(y))[a.fold][0]
else: tr=np.arange(500)  # full-train (테스트 후보용)
Xt=torch.tensor(X[tr])
def norm(x): x=(x-MU)/SD;x=x[:,None].repeat(1,3,1,1);return F.interpolate(x,size=252,mode='bilinear',align_corners=False)
def d4(x,k): x=torch.rot90(x,k%4,(0,1));return torch.flip(x,(1,)) if k>=4 else x
def view(x):  # x: (256,256) in [0,1]; 촬영 조건 변화 + D4
    x=d4(x,int(np.random.randint(8)))
    g=np.exp(np.random.uniform(np.log(0.8),np.log(1.25)));x=x.clamp(1e-4,1)**g
    c=np.random.uniform(0.8,1.2);b=np.random.uniform(-0.08,0.08);m=x.mean();x=(x-m)*c+m+b
    if np.random.rand()<0.3:
        s=np.random.uniform(0.3,1.0);k=torch.arange(-3,4,dtype=torch.float32);w=torch.exp(-k**2/(2*s*s));w=w/w.sum()
        x=F.conv2d(F.conv2d(x[None,None],w.view(1,1,1,7),padding=(0,3)),w.view(1,1,7,1),padding=(3,0))[0,0]
    x=x+torch.randn_like(x)*np.random.uniform(0,0.03)
    return x.clamp(0,1)
b=timm.create_model('vit_base_patch14_dinov2.lvd142m',pretrained=True,num_classes=0,img_size=252)
for q in b.parameters(): q.requires_grad=False
for blk in b.blocks[-a.nblk:]:
    for q in blk.parameters(): q.requires_grad=True
for q in b.norm.parameters(): q.requires_grad=True
h=nn.Sequential(nn.Linear(1536,2048),nn.BatchNorm1d(2048),nn.ReLU(),nn.Linear(2048,2048));b,h=b.to(dev),h.to(dev)
def emb(x): t=b.forward_features(x);return torch.cat([t[:,0],t[:,1:].mean(1)],1)
def vicreg(z1,z2):
    inv=F.mse_loss(z1,z2);std=0;cov=0
    for z in (z1,z2):
        z=z-z.mean(0);sd=torch.sqrt(z.var(0)+1e-4);std=std+F.relu(1-sd).mean()
        c=(z.T@z)/(len(z)-1);off=c-torch.diag(torch.diag(c));cov=cov+(off**2).sum()/z.shape[1]
    return 25*inv+25*std/2+cov/2,inv.item()
opt=torch.optim.AdamW([{'params':[q for q in b.parameters() if q.requires_grad],'lr':a.lr},{'params':h.parameters(),'lr':1e-3}],weight_decay=0.05)
spe=len(tr)//a.bs;sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=[a.lr,1e-3],total_steps=spe*a.ep,pct_start=0.1)
od=f'{a.tag}/rs{a.rs}_f{a.fold}';os.makedirs(od,exist_ok=True);log=open(f'{od}/train.log','w');t0=time.time()
b.train();h.train()
for ep in range(a.ep):
    idx=np.random.permutation(len(tr))
    for s in range(spe):
        bi=idx[s*a.bs:(s+1)*a.bs];v1=torch.stack([view(Xt[i]) for i in bi]);v2=torch.stack([view(Xt[i]) for i in bi])
        x=norm(torch.cat([v1,v2])).to(dev);e=emb(x);z=h(e);l,inv=vicreg(z[:len(bi)],z[len(bi):])
        opt.zero_grad();l.backward();opt.step();sch.step()
    log.write(f'ep{ep} loss {l.item():.3f} inv {inv:.4f} t={time.time()-t0:.0f}\n');log.flush()
b.eval()
def feats(A):
    out=[]
    with torch.no_grad():
        for i in range(0,len(A),32):
            x=norm(torch.tensor(A[i:i+32])).to(dev);acc=0
            for k in range(8):
                xx=torch.rot90(x,k%4,(2,3));xx=torch.flip(xx,(3,)) if k>=4 else xx;t=b.forward_features(xx);acc=acc+torch.cat([t[:,0],t[:,1:].mean(1)],1)
            out.append((acc/8).cpu().numpy())
    return np.concatenate(out)
np.save(f'{od}/feat_train.npy',feats(X))
if a.test:  # test 이미지는 추론(특징 추출)에만 사용
    from PIL import Image
    fs=sorted(glob.glob(f'{R}/open/test/*.png'));T=np.stack([np.asarray(Image.open(f).convert('L').resize((256,256)),dtype=np.float32)/255 for f in fs])
    np.save(f'{od}/feat_test.npy',feats(T))
open(f'{od}/DONE','w').write(f'{time.time()-t0:.0f}s\n')
