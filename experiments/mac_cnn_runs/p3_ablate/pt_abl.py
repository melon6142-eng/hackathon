# P3 ablation pretrain: same as gen_p2/pretrain_pair.py, one change at a time. Steps fixed (=full run) so compute is constant.
import argparse,time,numpy as np,torch,timm,torch.nn as nn,torch.nn.functional as F
p=argparse.ArgumentParser();p.add_argument('--pool');p.add_argument('--out');p.add_argument('--lam',type=float,default=1.0)
p.add_argument('--wreg',type=float,default=1.0);p.add_argument('--frac',type=float,default=1.0);p.add_argument('--scramble',type=int,default=0);a=p.parse_args()
dev='mps';torch.manual_seed(0);np.random.seed(0);Xr=np.load('train_imgs.npy');MU,SD=float(Xr.mean()),float(Xr.std())
D=np.load(a.pool);Xn=D['X'];T=D['T'][:,:8];T=(T-T.mean(0))/(T.std(0)+1e-6)
n=len(Xn);keep=np.random.RandomState(1).permutation(n)[:int(round(n*a.frac))];Xn=Xn[keep];T=T[keep]
if a.scramble:  # degraded generator: phase-randomize (same random phase for both views of a pair) -> keeps power spectrum, destroys Voronoi/morphology
    rg=np.random.RandomState(2);out=np.empty_like(Xn)
    for i in range(len(Xn)):
        ph=np.angle(np.fft.fft2(rg.rand(256,256)))
        for v in range(2):
            x=Xn[i,v].astype(np.float32);m=x.mean();Fx=np.fft.fft2(x-m);y=np.real(np.fft.ifft2(np.abs(Fx)*np.exp(1j*ph)))+m
            out[i,v]=np.clip(y,0,255).round().astype(np.uint8)
    Xn=out
X=torch.tensor(Xn);T=torch.tensor(T,dtype=torch.float32)
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
bs=8;TOT=(n//bs)*2;sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=[5e-5,5e-4],total_steps=TOT,pct_start=0.1)
log=open(a.out+'.log','w');t0=time.time();s=0;N=len(X)
while s<TOT:
    idx=np.random.permutation(N)
    for j in range(N//bs):
        if s>=TOT: break
        bi=idx[j*bs:(j+1)*bs];x=prep(X[bi].reshape(-1,256,256));x=torch.cat([d4(x[i:i+1],int(np.random.randint(8))) for i in range(len(x))]).to(dev)
        e=emb(x);e1,e2=e[0::2],e[1::2];t=T[bi].to(dev)
        l=a.wreg*(F.mse_loss(h(e1),t)+F.mse_loss(h(e2),t))+a.lam*(1-F.cosine_similarity(e1,e2)).mean()
        opt.zero_grad();l.backward();opt.step();sch.step();s+=1
        if s%200==0: log.write(f's{s}/{TOT} loss {l.item():.4f} std {e.std(0).mean().item():.4f} t={time.time()-t0:.0f}\n');log.flush()
torch.save({k:v.cpu() for k,v in b.state_dict().items() if k.startswith(('blocks.8','blocks.9','blocks.10','blocks.11','norm'))},a.out)
