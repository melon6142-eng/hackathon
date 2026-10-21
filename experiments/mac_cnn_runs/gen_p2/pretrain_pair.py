# Phase3: 파라미터 회귀(촬영 무관 구조 파라미터) + 같은 구조 쌍 특징 일치(코사인) + 촬영 파라미터는 예측하지 않음
import argparse,time,numpy as np,torch,timm,torch.nn as nn,torch.nn.functional as F
p=argparse.ArgumentParser();p.add_argument('--pool');p.add_argument('--out');p.add_argument('--lam',type=float,default=1.0);a=p.parse_args()
dev='mps';torch.manual_seed(0);np.random.seed(0);Xr=np.load('train_imgs.npy');MU,SD=float(Xr.mean()),float(Xr.std())
D=np.load(a.pool);X=torch.tensor(D['X']);T=D['T'][:,:8];T=torch.tensor((T-T.mean(0))/(T.std(0)+1e-6),dtype=torch.float32)  # tex_a(촬영) 제외
def prep(xb): x=(xb.float()/255-MU)/SD;x=x[:,None].repeat(1,3,1,1);return F.interpolate(x,size=252,mode='bilinear',align_corners=False)
def d4(x,k): x=torch.rot90(x,k%4,(2,3));return torch.flip(x,(3,)) if k>=4 else x
b=timm.create_model('vit_base_patch14_dinov2.lvd142m',pretrained=True,num_classes=0,img_size=252)
for q in b.parameters(): q.requires_grad=False
for blk in b.blocks[-4:]:
    for q in blk.parameters(): q.requires_grad=True
for q in b.norm.parameters(): q.requires_grad=True
h=nn.Sequential(nn.Linear(1536,256),nn.GELU(),nn.Linear(256,8));b,h=b.to(dev),h.to(dev)
def emb(x): t=b.forward_features(x);return torch.cat([t[:,0],t[:,1:].mean(1)],1)
opt=torch.optim.AdamW([{'params':[q for q in b.parameters() if q.requires_grad],'lr':5e-5},{'params':h.parameters(),'lr':5e-4}],weight_decay=0.05)
bs=8;tr=np.arange(len(X));spe=len(tr)//bs;E=2;sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=[5e-5,5e-4],total_steps=spe*E,pct_start=0.1)
log=open(a.out+'.log','w');t0=time.time()
for ep in range(E):
    idx=np.random.permutation(tr)
    for s in range(spe):
        bi=idx[s*bs:(s+1)*bs];x=prep(X[bi].reshape(-1,256,256));x=torch.cat([d4(x[i:i+1],int(np.random.randint(8))) for i in range(len(x))]).to(dev)
        e=emb(x);e1,e2=e[0::2],e[1::2];t=T[bi].to(dev)
        l=F.mse_loss(h(e1),t)+F.mse_loss(h(e2),t)+a.lam*(1-F.cosine_similarity(e1,e2)).mean()
        opt.zero_grad();l.backward();opt.step();sch.step()
        if s%200==0: log.write(f'ep{ep} s{s} loss {l.item():.4f} t={time.time()-t0:.0f}\n');log.flush()
torch.save({k:v.cpu() for k,v in b.state_dict().items() if k.startswith(('blocks.8','blocks.9','blocks.10','blocks.11','norm'))},a.out)
