import argparse,os,time,json,numpy as np,torch,timm,torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import KFold
p=argparse.ArgumentParser()
p.add_argument('--arch',default='convnext_tiny.fb_in1k');p.add_argument('--mode',default='full')
p.add_argument('--size',type=int,default=256);p.add_argument('--crop',type=int,default=128)
p.add_argument('--epochs',type=int,default=50);p.add_argument('--bs',type=int,default=32)
p.add_argument('--lr',type=float,default=2e-4);p.add_argument('--weight',type=float,default=0.0)
p.add_argument('--loss',default='huber');p.add_argument('--tag',required=True);p.add_argument('--seed',type=int,default=42)
a=p.parse_args()
R='~/AI/daker';O=f'{R}/cnn/runs/{a.tag}';os.makedirs(O,exist_ok=True)
dev='mps';torch.manual_seed(0)
import csv
rows=list(csv.DictReader(open(f'{R}/open/train.csv')));ids=[r['ID'] for r in rows];y=np.array([float(r['hardness']) for r in rows])
tids=[r['ID'] for r in csv.DictReader(open(f'{R}/open/sample_submission.csv'))]
def load(d,ii):
    X=np.stack([np.array(Image.open(f'{R}/open/{d}/{i}.png').convert('L'),dtype=np.float32)/255 for i in ii])
    return torch.tensor(X)[:,None]
cache=f'{R}/cnn/imgs.pt'
if os.path.exists(cache): Xtr,Xte=torch.load(cache)
else: Xtr,Xte=load('train',ids),load('test',tids);torch.save((Xtr,Xte),cache)
mu,sd=y.mean(),y.std();z=(y-mu)/sd
MEAN,STD=0.45,0.25
def d4(x,k):
    x=torch.rot90(x,k%4,(2,3))
    return torch.flip(x,(3,)) if k>=4 else x
def prep(x):
    x=(x-MEAN)/STD
    if a.size!=x.shape[-1] and a.mode=='full': x=F.interpolate(x,size=a.size,mode='bilinear',align_corners=False)
    return x.expand(-1,3,-1,-1)
def aug(x):
    B=x.shape[0];out=torch.empty_like(x) if a.mode=='full' else None
    xs=[]
    for i in range(B):
        xi=d4(x[i:i+1],int(np.random.randint(8)))
        if a.mode=='patch':
            c=a.crop;r,s=np.random.randint(0,256-c+1,2);xi=xi[:,:,r:r+c,s:s+c]
        xs.append(xi)
    x=torch.cat(xs)
    b=(torch.rand(B,1,1,1,device=x.device)-.5)*0.1;c=1+(torch.rand(B,1,1,1,device=x.device)-.5)*0.1
    m=x.mean((2,3),keepdim=True);return (x-m)*c+m+b
@torch.no_grad()
def predict(model,X):
    model.eval();P=[]
    for i in range(0,len(X),16):
        xb=X[i:i+16].to(dev);acc=0;n=0
        for k in range(8):
            xk=d4(xb,k)
            if a.mode=='full': acc=acc+model(prep(xk)).squeeze(1);n+=1
            else:
                c=a.crop;st=(256-c)//2
                for r in range(0,256-c+1,st):
                    for s in range(0,256-c+1,st): acc=acc+model(prep(xk[:,:,r:r+c,s:s+c])).squeeze(1);n+=1
        P.append((acc/n).cpu())
    return torch.cat(P).numpy()*sd+mu
folds=list(KFold(5,shuffle=True,random_state=a.seed).split(ids))
log=open(f'{O}/log.txt','a')
def L(s): print(s,flush=True);log.write(s+'\n');log.flush()
L(f'== start {time.ctime()} {vars(a)}')
for f,(tr,va) in enumerate(folds):
    res=f'{O}/fold{f}.npz'
    if os.path.exists(res): L(f'fold{f} done, skip');continue
    t0=time.time()
    model=timm.create_model(a.arch,pretrained=True,num_classes=1).to(dev)
    head=[p for n,p in model.named_parameters() if any(k in n for k in ['head','fc','classifier'])]
    hid={id(p) for p in head};body=[p for p in model.parameters() if id(p) not in hid]
    opt=torch.optim.AdamW([{'params':body,'lr':a.lr},{'params':head,'lr':a.lr*10}],weight_decay=0.05)
    spe=int(np.ceil(len(tr)*(4 if a.mode=='patch' else 1)/a.bs));tot=spe*a.epochs
    sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=[a.lr,a.lr*10],total_steps=tot,pct_start=0.1)
    ck=f'{O}/fold{f}_ck.pt';ep0=0
    if os.path.exists(ck):
        s=torch.load(ck,weights_only=False);model.load_state_dict(s['m']);opt.load_state_dict(s['o']);sch.load_state_dict(s['s']);ep0=s['ep']+1;L(f'resume fold{f} ep{ep0}')
    zt=torch.tensor(z,dtype=torch.float32);w=torch.tensor(1+a.weight*np.abs(z),dtype=torch.float32)
    for ep in range(ep0,a.epochs):
        model.train();idx=np.concatenate([np.random.permutation(tr) for _ in range(4 if a.mode=='patch' else 1)]);tl=0
        for b in range(spe):
            bi=idx[b*a.bs:(b+1)*a.bs]
            if len(bi)<2: continue
            xb=aug(Xtr[bi].to(dev));out=model(prep(xb)).squeeze(1);tb=zt[bi].to(dev)
            l=F.huber_loss(out,tb,reduction='none',delta=1.0) if a.loss=='huber' else (out-tb)**2
            l=(l*w[bi].to(dev)).mean();opt.zero_grad();l.backward();opt.step();sch.step();tl+=l.item()
        if ep%5==4 or ep==a.epochs-1:
            pv=predict(model,Xtr[va]) if (ep%10==9 or ep==a.epochs-1) else None
            L(f'fold{f} ep{ep} loss {tl/spe:.4f} '+(f'va_rmse {np.sqrt(((pv-y[va])**2).mean()):.3f}' if pv is not None else '')+f' t={time.time()-t0:.0f}s')
            torch.save({'m':model.state_dict(),'o':opt.state_dict(),'s':sch.state_dict(),'ep':ep},ck)
    pv=predict(model,Xtr[va]);pt=predict(model,Xte)
    np.savez(res,va=va,pv=pv,pt=pt);os.remove(ck)
    r=np.sqrt(((pv-y[va])**2).mean());L(f'FOLD{f} RMSE {r:.3f} time {time.time()-t0:.0f}s')
oof=np.zeros(len(y));pt=0
for f in range(5):
    d=np.load(f'{O}/fold{f}.npz');oof[d['va']]=d['pv'];pt=pt+d['pt']/5
np.save(f'{O}/oof.npy',oof);np.save(f'{O}/test.npy',pt)
hi=y>230;lo=y<170
L(f'ALL OOF RMSE {np.sqrt(((oof-y)**2).mean()):.3f} hi_bias {(oof-y)[hi].mean():.1f} lo_bias {(oof-y)[lo].mean():.1f}')
