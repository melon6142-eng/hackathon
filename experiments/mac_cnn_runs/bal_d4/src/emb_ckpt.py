# embeddings of held-out fold images from baseline final_clean ckpts (orig_s42) and from untuned pretrained DINOv2
import sys,json,numpy as np,torch
sys.argv=['x','--tag','_emb','--chan','orig','--seed','42','--fs','42']
exec(open('~/AI/daker/cnn/runs/bal_d4/src/vit_arm.py').read().split("folds=list(")[0])
import os
folds=list(KFold(5,shuffle=True,random_state=42).split(yv))
for src in ['base','pre']:
    od=f'{O}/arms/emb_{src}';os.makedirs(od,exist_ok=True)
    for f,(tr,va) in enumerate(folds):
        if os.path.exists(f'{od}/fold{f}_emb.npz'): continue
        st=json.load(open(f'~/AI/daker/final_clean/vit/orig_s42/fold{f}_norm.json'))
        m=M().to(dev)
        if src=='base': m.load_state_dict(torch.load(f'~/AI/daker/final_clean/vit/orig_s42/fold{f}.pt',map_location='cpu',weights_only=False)['state_dict'])
        np.savez(f'{od}/fold{f}_emb.npz',va=va,E=embed(m,apply(Gtr[va],st)));print(src,f,flush=True)
