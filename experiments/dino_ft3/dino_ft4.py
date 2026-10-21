# DINOv2 ViT-B trans, unfreeze4. options: --res {resize,pad,crop}, --loss {huber,binreg,rank,lds}
import argparse,os,time,numpy as np,torch,timm,torch.nn as nn,torch.nn.functional as F,csv,cv2
from sklearn.model_selection import KFold
from scipy.ndimage import gaussian_filter1d
p=argparse.ArgumentParser()
p.add_argument('--arch',default='vit_base_patch14_dinov2.lvd142m');p.add_argument('--size',type=int,default=252)
p.add_argument('--res',default='resize');p.add_argument('--crop',type=int,default=140)
p.add_argument('--loss',default='huber');p.add_argument('--chan',default='trans');p.add_argument('--unfreeze',type=int,default=4)
p.add_argument('--epochs',type=int,default=30);p.add_argument('--bs',type=int,default=16)
p.add_argument('--lr',type=float,default=5e-5);p.add_argument('--tag',required=True);p.add_argument('--seed',type=int,default=42)
a=p.parse_args()
R='~/AI/daker';O=f'{R}/cnn/runs/{a.tag}';os.makedirs(O,exist_ok=True);dev='mps'
torch.manual_seed(a.seed);np.random.seed(a.seed)
rows=list(csv.DictReader(open(f'{R}/open/train.csv')));y=np.array([float(r['hardness']) for r in rows])
Xtr,Xte=torch.load(f'{R}/cnn/imgs.pt')
def build(X,lo,hi):
    g=(X[:,0].numpy()*255)
    cl=np.stack([cv2.createCLAHE(clipLimit=2.5,tileGridSize=(8,8)).apply(np.clip(i,0,255).astype(np.uint8)) for i in g])/255.
    cs=np.power(np.clip((g-lo)/(hi-lo),0,1),0.85);o=g/255.
    C=np.stack([o,o,o],1) if a.chan=='orig' else np.stack([o,cl,cs],1)
    return torch.tensor(((C-C.mean((0,2,3),keepdims=True))/C.std((0,2,3),keepdims=True)).astype(np.float32))
pix=(Xtr.numpy().ravel()*255)[np.random.default_rng(0).choice(Xtr.numel(),2_000_000)];lo,hi=np.percentile(pix,[1,99])
Xtr,Xte=build(Xtr,lo,hi),build(Xte,lo,hi)
mu,sd=y.mean(),y.std();z=(y-mu)/sd
NB=10;edges=np.quantile(z,np.linspace(0,1,NB+1));edges[0]-=1e-3;edges[-1]+=1e-3
cent=torch.tensor([(z[(z>=edges[i])&(z<edges[i+1])]).mean() for i in range(NB)],dtype=torch.float32)
ybin=torch.tensor(np.clip(np.searchsorted(edges,z,side='right')-1,0,NB-1))
# LDS weights: inverse of gaussian-smoothed label density, normalized
hist,be=np.histogram(z,bins=40);dens=gaussian_filter1d(hist.astype(float),2)+1e-3
wl=1/np.sqrt(dens[np.clip(np.digitize(z,be)-1,0,39)]);wl=wl/wl.mean()
W=torch.tensor(wl if a.loss=='lds' else np.ones(len(y)),dtype=torch.float32)
IMG=a.crop if a.res=='crop' else (266 if a.res=='pad' else a.size)
def d4(x,k):
    x=torch.rot90(x,k%4,(2,3));return torch.flip(x,(3,)) if k>=4 else x
def prep(x):
    if a.res=='pad': return F.pad(x,(5,5,5,5),mode='reflect')
    if a.res=='crop': return x
    return F.interpolate(x,size=a.size,mode='bilinear',align_corners=False)
class M(nn.Module):
    def __init__(s):
        super().__init__();s.b=timm.create_model(a.arch,pretrained=True,num_classes=0,img_size=IMG)
        for q in s.b.parameters(): q.requires_grad=False
        for blk in s.b.blocks[-a.unfreeze:]:
            for q in blk.parameters(): q.requires_grad=True
        for q in s.b.norm.parameters(): q.requires_grad=True
        d=s.b.num_features;s.f=nn.Sequential(nn.Linear(2*d,256),nn.GELU(),nn.Dropout(0.1));s.r=nn.Linear(256,1);s.c=nn.Linear(256,NB)
    def forward(s,x,both=False):
        t=s.b.forward_features(x);h=s.f(torch.cat([t[:,0],t[:,s.b.num_prefix_tokens:].mean(1)],1))
        r=s.r(h).squeeze(1);lg=s.c(h)
        if a.loss=='binreg': pr=0.5*r+0.5*(lg.softmax(1)*cent.to(x.device)).sum(1)
        else: pr=r
        return (pr,r,lg) if both else pr
def crops(x):
    c=a.crop;r,s=np.random.randint(0,256-c+1,2);return x[:,:,r:r+c,s:s+c]
def aug(x):
    x=torch.cat([d4(x[i:i+1],int(np.random.randint(8))) for i in range(len(x))])
    if a.res=='crop': x=torch.cat([crops(x[i:i+1]) for i in range(len(x))])
    return x+(torch.rand(len(x),1,1,1,device=x.device)-.5)*0.2
@torch.no_grad()
def predict(m,X):
    m.eval();P=[];bs=8
    for i in range(0,len(X),bs):
        xb=X[i:i+bs].to(dev);acc=0;n=0
        for k in range(8):
            xk=d4(xb,k)
            if a.res=='crop':
                c=a.crop;st=(256-c)//2
                for r_ in range(0,256-c+1,st):
                    for s_ in range(0,256-c+1,st): acc=acc+m(xk[:,:,r_:r_+c,s_:s_+c]);n+=1
            else: acc=acc+m(prep(xk));n+=1
        P.append((acc/n).cpu())
    return torch.cat(P).numpy()*sd+mu
def lossf(m,xb,bi):
    pr,r,lg=m(prep(xb),both=True);t=zt[bi].to(dev);w=W[bi].to(dev)
    if a.loss=='binreg': return (F.huber_loss(pr,t,reduction='none')*w).mean()+0.5*F.huber_loss(r,t)+0.3*F.cross_entropy(lg,ybin[bi].to(dev))
    l=(F.huber_loss(pr,t,reduction='none')*w).mean()
    if a.loss=='rank':
        d=pr[:,None]-pr[None];s=torch.sign(t[:,None]-t[None]);msk=(s!=0).float()
        l=l+0.5*(F.softplus(-s*d*3)*msk).sum()/msk.sum().clamp(min=1)
    return l
folds=list(KFold(5,shuffle=True,random_state=42).split(y))
log=open(f'{O}/log.txt','a')
def L(s): print(s,flush=True);log.write(s+'\n');log.flush()
L(f'== start {time.ctime()} {vars(a)} IMG={IMG}')
zt=torch.tensor(z,dtype=torch.float32);rep=4 if a.res=='crop' else 1
for f,(tr,va) in enumerate(folds):
    res=f'{O}/fold{f}.npz'
    if os.path.exists(res): L(f'fold{f} done, skip');continue
    t0=time.time();torch.manual_seed(a.seed+f*1000);np.random.seed(a.seed+f*1000);m=M().to(dev)
    bp=[q for q in m.b.parameters() if q.requires_grad];hp=list(m.f.parameters())+list(m.r.parameters())+list(m.c.parameters())
    opt=torch.optim.AdamW([{'params':bp,'lr':a.lr},{'params':hp,'lr':a.lr*10}],weight_decay=0.05)
    spe=int(np.ceil(len(tr)*rep/a.bs));sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=[a.lr,a.lr*10],total_steps=spe*a.epochs,pct_start=0.1)
    for ep in range(a.epochs):
        m.train();idx=np.concatenate([np.random.permutation(tr) for _ in range(rep)]);tl=0
        for b in range(spe):
            bi=idx[b*a.bs:(b+1)*a.bs]
            if len(bi)<2: continue
            l=lossf(m,aug(Xtr[bi].to(dev)),bi);opt.zero_grad();l.backward();opt.step();sch.step();tl+=l.item()
        if ep%10==9: pv=predict(m,Xtr[va]);L(f'fold{f} ep{ep} loss {tl/spe:.4f} va {np.sqrt(((pv-y[va])**2).mean()):.3f} t={time.time()-t0:.0f}')
    pv=predict(m,Xtr[va]);pt=predict(m,Xte);np.savez(res,va=va,pv=pv,pt=pt)
    L(f'FOLD{f} RMSE {np.sqrt(((pv-y[va])**2).mean()):.3f} {time.time()-t0:.0f}s')
oof=np.zeros(len(y));pt=0
for f in range(5): d=np.load(f'{O}/fold{f}.npz');oof[d['va']]=d['pv'];pt=pt+d['pt']/5
np.save(f'{O}/oof.npy',oof);np.save(f'{O}/test.npy',pt)
L(f'ALL OOF RMSE {np.sqrt(((oof-y)**2).mean()):.3f} hi_bias {(oof-y)[y>230].mean():.1f} lo_bias {(oof-y)[y<170].mean():.1f}')
