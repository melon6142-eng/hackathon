import numpy as np,torch,timm,torch.nn.functional as F
dev='mps'
_m=None
def model():
    global _m
    if _m is None:
        _m=timm.create_model('vit_base_patch14_dinov2.lvd142m',pretrained=True,num_classes=0,img_size=252).eval().to(dev)
    return _m
MU=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1);SD=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
@torch.no_grad()
def embed(X,bs=32,tta=(0,5)):
    """X: (N,256,256) float [0,1] -> (N,1536) cls+mean, d4 TTA 평균"""
    m=model();out=[]
    for i in range(0,len(X),bs):
        x=torch.tensor(np.asarray(X[i:i+bs]),dtype=torch.float32)[:,None].repeat(1,3,1,1)
        x=F.interpolate(x,size=252,mode='bilinear',align_corners=False);x=((x-MU)/SD).to(dev);acc=0
        for k in tta:
            xx=torch.rot90(x,k%4,(2,3)); xx=torch.flip(xx,(3,)) if k>=4 else xx
            t=m.forward_features(xx); acc=acc+torch.cat([t[:,0],t[:,m.num_prefix_tokens:].mean(1)],1)
        out.append((acc/len(tta)).float().cpu().numpy())
    return np.concatenate(out)
