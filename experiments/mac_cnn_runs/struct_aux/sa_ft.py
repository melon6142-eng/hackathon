# DINOv2 partial fine-tune. channels: orig | orig+CLAHE(2.5,8x8)+global p1-p99 gamma0.85
import argparse,os,time,numpy as np,torch,timm,torch.nn as nn,torch.nn.functional as F,csv,cv2
from sklearn.model_selection import KFold
p=argparse.ArgumentParser()
p.add_argument('--arch',default='vit_small_patch14_dinov2.lvd142m');p.add_argument('--size',type=int,default=224)
p.add_argument('--chan',default='orig');p.add_argument('--unfreeze',type=int,default=4)
p.add_argument('--epochs',type=int,default=30);p.add_argument('--bs',type=int,default=32)
p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--weight',type=float,default=0.0)
p.add_argument('--aux_w',type=float,default=0.3);p.add_argument('--fold_rs',type=int,default=42);p.add_argument('--tag',required=True);p.add_argument('--seed',type=int,default=42)
a=p.parse_args()
R='~/AI/daker';O=f'{R}/cnn/runs/struct_aux/{a.tag}';os.makedirs(O,exist_ok=True);dev='mps'
torch.manual_seed(a.seed);np.random.seed(a.seed)
rows=list(csv.DictReader(open(f'{R}/open/train.csv')));y=np.array([float(r['hardness']) for r in rows])
Xtr,Xte=torch.load(f'{R}/cnn/imgs.pt');AUX=np.load(f'{R}/cnn/runs/struct_aux/aux_train.npy')
def build(X,lo,hi):
    g=(X[:,0].numpy()*255)
    cl=np.stack([cv2.createCLAHE(clipLimit=2.5,tileGridSize=(8,8)).apply(np.clip(i,0,255).astype(np.uint8)) for i in g])/255.
    cs=np.power(np.clip((g-lo)/(hi-lo),0,1),0.85)
    o=g/255.
    if a.chan=='orig': C=np.stack([o,o,o],1)
    else: C=np.stack([o,cl,cs],1)
    return torch.tensor(((C-C.mean((0,2,3),keepdims=True))/C.std((0,2,3),keepdims=True)).astype(np.float32))
pix=(Xtr.numpy().ravel()*255)[np.random.default_rng(0).choice(Xtr.numel(),2_000_000)];lo,hi=np.percentile(pix,[1,99])
Xtr,Xte=build(Xtr,lo,hi),build(Xte,lo,hi)
mu,sd=y.mean(),y.std();z=(y-mu)/sd
def d4(x,k):
    x=torch.rot90(x,k%4,(2,3));return torch.flip(x,(3,)) if k>=4 else x
def prep(x): return F.interpolate(x,size=a.size,mode='bilinear',align_corners=False)
class M(nn.Module):
    def __init__(s):
        super().__init__();s.b=timm.create_model(a.arch,pretrained=True,num_classes=0,img_size=a.size)
        for q in s.b.parameters(): q.requires_grad=False
        for blk in s.b.blocks[-a.unfreeze:]:
            for q in blk.parameters(): q.requires_grad=True
        for q in s.b.norm.parameters(): q.requires_grad=True
        d=s.b.num_features;s.ha=nn.Linear(2*d,AUX.shape[1]);s.h=nn.Sequential(nn.Linear(2*d,256),nn.GELU(),nn.Dropout(0.1),nn.Linear(256,1))
    def forward(s,x):
        t=s.b.forward_features(x);pt=t[:,s.b.num_prefix_tokens:]
        f=torch.cat([t[:,0],pt.mean(1)],1);return s.h(f).squeeze(1),s.ha(f)
def aug(x):
    x=torch.cat([d4(x[i:i+1],int(np.random.randint(8))) for i in range(len(x))])
    b=(torch.rand(len(x),1,1,1,device=x.device)-.5)*0.2;return x+b
@torch.no_grad()
def predict(m,X):
    m.eval();P=[]
    for i in range(0,len(X),32):
        xb=X[i:i+32].to(dev);P.append((sum(m(prep(d4(xb,k)))[0] for k in range(8))/8).cpu())
    return torch.cat(P).numpy()*sd+mu
# fold split fixed at 42 so multi-seed OOF aligns for averaging
folds=list(KFold(5,shuffle=True,random_state=a.fold_rs).split(y))
log=open(f'{O}/log.txt','a')
def L(s): print(s,flush=True);log.write(s+'\n');log.flush()
L(f'== start {time.ctime()} {vars(a)} lo={lo:.1f} hi={hi:.1f}')
w=torch.tensor(1+a.weight*np.clip(z,0,None),dtype=torch.float32);zt=torch.tensor(z,dtype=torch.float32)
for f,(tr,va) in enumerate(folds):
    res=f'{O}/fold{f}.npz'
    if os.path.exists(res): L(f'fold{f} done, skip');continue
    t0=time.time();torch.manual_seed(a.seed+f*1000);np.random.seed(a.seed+f*1000)
    m=M().to(dev)
    am,asd=AUX[tr].mean(0),AUX[tr].std(0)+1e-6;At=torch.tensor((AUX-am)/asd,dtype=torch.float32)  # train-fold stats only; va rows never used
    bp=[q for q in m.b.parameters() if q.requires_grad]
    opt=torch.optim.AdamW([{'params':bp,'lr':a.lr},{'params':list(m.h.parameters())+list(m.ha.parameters()),'lr':a.lr*10}],weight_decay=0.05)
    spe=int(np.ceil(len(tr)/a.bs));sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=[a.lr,a.lr*10],total_steps=spe*a.epochs,pct_start=0.1)
    for ep in range(a.epochs):
        m.train();idx=np.random.permutation(tr);tl=0
        for b in range(spe):
            bi=idx[b*a.bs:(b+1)*a.bs]
            if len(bi)<2: continue
            out,ao=m(prep(aug(Xtr[bi].to(dev))));l=(F.huber_loss(out,zt[bi].to(dev),reduction='none')*w[bi].to(dev)).mean()+a.aux_w*F.mse_loss(ao,At[bi].to(dev))
            opt.zero_grad();l.backward();opt.step();sch.step();tl+=l.item()
        if ep%10==9: pv=predict(m,Xtr[va]);L(f'fold{f} ep{ep} loss {tl/spe:.4f} va {np.sqrt(((pv-y[va])**2).mean()):.3f} t={time.time()-t0:.0f}')
    pv=predict(m,Xtr[va]);pt=predict(m,Xte);np.savez(res,va=va,pv=pv,pt=pt)
    L(f'FOLD{f} RMSE {np.sqrt(((pv-y[va])**2).mean()):.3f} {time.time()-t0:.0f}s')
oof=np.zeros(len(y));pt=0
for f in range(5): d=np.load(f'{O}/fold{f}.npz');oof[d['va']]=d['pv'];pt=pt+d['pt']/5
np.save(f'{O}/oof.npy',oof);np.save(f'{O}/test.npy',pt);hi_=y>230
L(f'ALL OOF RMSE {np.sqrt(((oof-y)**2).mean()):.3f} hi_bias {(oof-y)[hi_].mean():.1f} lo_bias {(oof-y)[y<170].mean():.1f}')
