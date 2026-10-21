# DINOv2 (Apache-2.0) multi-layer, multi-scale features with D4 TTA. usage: python feat_dino.py vitb14 252
import torch, numpy as np, os, sys, cv2
torch.set_num_threads(6); torch.manual_seed(0)
arch, size = sys.argv[1], int(sys.argv[2]); NL=int(sys.argv[3]); torch.set_num_threads(3)
m=torch.hub.load('facebookresearch/dinov2',f'dinov2_{arch}').eval()
R=os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../open'); out=os.path.dirname(os.path.abspath(__file__))
mean=np.array([0.485,0.456,0.406]);std=np.array([0.229,0.224,0.225])
def feats(x):
    L=m.get_intermediate_layers(x,n=NL,return_class_token=True)
    fs=[]
    for p,c in L:  # p: B,N,D
        g=p.clamp(min=1e-6).pow(3).mean(1).pow(1/3)
        fs+= [p.mean(1), p.std(1)]
    return torch.cat(fs,1)
for sp in ['train','test']:
    ids=sorted(x[:-4] for x in os.listdir(f'{R}/{sp}')); F=[]
    for k in range(0,len(ids),8):
        ims=[cv2.resize(cv2.imread(f'{R}/{sp}/{i}.png',0),(size,size),interpolation=cv2.INTER_AREA)/255. for i in ids[k:k+8]]
        x=torch.tensor(np.stack([(np.stack([a]*3,-1)-mean)/std for a in ims]).transpose(0,3,1,2),dtype=torch.float32)
        acc=0
        with torch.no_grad():
            for r in range(4):
                for fl in [False,True]:
                    xx=torch.rot90(x,r,(2,3)); xx=torch.flip(xx,(3,)) if fl else xx
                    acc=acc+feats(xx)
        F.append((acc/8).numpy())
        if k%200==0: print(sp,k,flush=True)
    np.save(f'{out}/{arch}_{size}_L{NL}ms_{sp}.npy',np.concatenate(F).astype(np.float32))
print('DONE')
