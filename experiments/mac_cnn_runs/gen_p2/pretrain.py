# 합성 풀에서 생성 파라미터 회귀로 DINOv2 ViT-B(마지막 4블록+norm) 사전학습. 경도 라벨 없음.
import argparse,os,time,numpy as np,torch,timm,torch.nn as nn,torch.nn.functional as F
p=argparse.ArgumentParser();p.add_argument('--pool',required=True);p.add_argument('--out',required=True)
p.add_argument('--epochs',type=int,default=2);p.add_argument('--lr',type=float,default=5e-5);p.add_argument('--bs',type=int,default=16)
a=p.parse_args();dev='mps';torch.manual_seed(0);np.random.seed(0)
Xr=np.load('train_imgs.npy');MU,SD=float(Xr.mean()),float(Xr.std())
D=np.load(a.pool);X=torch.tensor(D['X']);T=D['T'];tm,ts=T.mean(0),T.std(0)+1e-6;T=torch.tensor((T-tm)/ts,dtype=torch.float32)
nv=500;tr=np.arange(nv,len(X));va=np.arange(nv)
def prep(xb): x=(xb.float()/255-MU)/SD;x=x[:,None].repeat(1,3,1,1);return F.interpolate(x,size=252,mode='bilinear',align_corners=False)
def d4(x,k): x=torch.rot90(x,k%4,(2,3));return torch.flip(x,(3,)) if k>=4 else x
b=timm.create_model('vit_base_patch14_dinov2.lvd142m',pretrained=True,num_classes=0,img_size=252)
for q in b.parameters(): q.requires_grad=False
for blk in b.blocks[-4:]:
    for q in blk.parameters(): q.requires_grad=True
for q in b.norm.parameters(): q.requires_grad=True
h=nn.Sequential(nn.Linear(1536,256),nn.GELU(),nn.Linear(256,T.shape[1]));b,h=b.to(dev),h.to(dev)
def fwd(x): t=b.forward_features(x);return h(torch.cat([t[:,0],t[:,b.num_prefix_tokens:].mean(1)],1))
bp=[q for q in b.parameters() if q.requires_grad]
opt=torch.optim.AdamW([{'params':bp,'lr':a.lr},{'params':h.parameters(),'lr':a.lr*10}],weight_decay=0.05)
spe=len(tr)//a.bs;sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=[a.lr,a.lr*10],total_steps=spe*a.epochs,pct_start=0.1)
t0=time.time();log=open(a.out+'.log','w')
@torch.no_grad()
def val():
    b.eval();h.eval();P=torch.cat([fwd(prep(X[va[i:i+32]]).to(dev)).cpu() for i in range(0,nv,32)]);b.train();h.train()
    return 1-((P-T[va])**2).mean(0)/T[va].var(0)
for ep in range(a.epochs):
    idx=np.random.permutation(tr)
    for s in range(spe):
        bi=idx[s*a.bs:(s+1)*a.bs];x=prep(X[bi]);x=torch.cat([d4(x[i:i+1],int(np.random.randint(8))) for i in range(len(x))]).to(dev)
        l=F.mse_loss(fwd(x),T[bi].to(dev));opt.zero_grad();l.backward();opt.step();sch.step()
        if s%200==0: log.write(f'ep{ep} s{s} loss {l.item():.4f} t={time.time()-t0:.0f}\n');log.flush()
r2=val().numpy();log.write('VAL R2 '+' '.join(f'{v:.3f}' for v in r2)+f' mean {r2.mean():.3f} t={time.time()-t0:.0f}\n');log.close()
torch.save({k:v.cpu() for k,v in b.state_dict().items() if k.startswith('blocks.8') or k.startswith('blocks.9') or k.startswith('blocks.10') or k.startswith('blocks.11') or k.startswith('norm')},a.out)
np.save(a.out+'.r2.npy',r2)
