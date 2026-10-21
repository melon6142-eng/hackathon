import numpy as np,csv,os
from scipy.optimize import nnls
from sklearn.model_selection import KFold
R='~/AI/daker';C=f'{R}/cnn/runs'
y=np.array([float(r['hardness']) for r in csv.DictReader(open(f'{R}/open/train.csv'))])
rm=lambda p:np.sqrt(((p-y)**2).mean());hb=lambda p:(p-y)[y>230].mean()
bl=np.load(f'{R}/cnn/oof_blend1.npy')
L=lambda ts:[np.load(f'{C}/{t}/oof.npy') for t in ts]
orig=L(['dinoB_orig','dinoB_orig_s1','dinoB_orig_s2']);trans=L(['dinoB_trans','dinoB_trans_s1','dinoB_trans_s2'])
print('member seeds (RMSE, >230 bias):')
for n,o in [('orig',orig),('trans',trans)]: print(n,[f'{rm(x):.3f}' for x in o],f'avg {rm(np.mean(o,0)):.3f} hb {hb(np.mean(o,0)):.1f}')
V={}
for v in ['ms_s126','ms_s168','ms_two']:
    o=[f'{C}/multiscale/{v}_s{s}/oof.npy' for s in (42,1,2)]
    if all(map(os.path.exists,o)):
        V[v]=[np.load(p) for p in o];a=np.mean(V[v],0)
        print(v,[f'{rm(x):.3f}' for x in V[v]],f'avg {rm(a):.3f} hb {hb(a):.1f}','per-seed d vs orig',[f'{rm(x)-rm(z):+.3f}' for x,z in zip(V[v],orig)],'corr w orig',f'{np.corrcoef(a-y,np.mean(orig,0)-y)[0,1]:.3f}')
def ncv(cols,reps=10):
    X=np.column_stack(cols+[np.ones(len(y))]);sc=[]
    for r in range(reps):
        o=np.zeros(len(y))
        for tr,va in KFold(5,shuffle=True,random_state=100+r).split(X):
            w,_=nnls(X[tr],y[tr]);o[va]=X[va]@w
        sc.append(rm(o))
    return np.mean(sc)
O,T=np.mean(orig,0),np.mean(trans,0)
print(f'BASE nnls {ncv([bl,O,T]):.3f} eq {rm((bl+O+T)/3):.3f} hb {hb((bl+O+T)/3):.1f}')
for v,o in V.items():
    M=np.mean(o,0)
    print(f'{v} replace: nnls {ncv([bl,M,T]):.3f} eq {rm((bl+M+T)/3):.3f} | add: nnls {ncv([bl,O,T,M]):.3f} eq4 {rm((bl+O+T+M)/4):.3f} eq3(orig+v avg) {rm((bl+(O+M)/2+T)/3):.3f}')
