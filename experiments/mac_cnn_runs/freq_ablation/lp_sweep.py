import sys,os,json,numpy as np,torch
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import filt; exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'frozen_ab.py')).read().split('# PSD')[0])
from sklearn.model_selection import KFold
R='~/AI/daker';Xraw,_=torch.load(f'{R}/cnn/imgs.pt');Xraw=Xraw[:,:1].float()
mk=lambda:make_pipeline(StandardScaler(),RidgeCV(np.logspace(0,5,21)));rm=lambda p:float(np.sqrt(((p-y)**2).mean()))
out={}
for per in [None,2.5,3,4,5,6,8]:
    if per is None: Xv=Xraw
    else:
        M=filt.soft(0,1/per);Xv=filt._apply(Xraw,lambda F:F*M)
    Fe=feats(Xv);rs=[];hb=[]
    for s in range(10):
        p=np.zeros(500)
        for tr,va in KFold(5,shuffle=True,random_state=100+s).split(y): p[va]=mk().fit(Fe[tr],y[tr]).predict(Fe[va])
        rs.append(rm(p));hb.append(float((p-y)[hi].mean()))
    out[str(per)]=dict(rmse=rs,hibias=float(np.mean(hb)));np.save(f'{O}/lp_feats_{per}.npy',Fe)
    print(per,np.mean(rs),np.std(rs),np.mean(hb),flush=True)
json.dump(out,open(f'{O}/lp_sweep.json','w'),indent=1)
