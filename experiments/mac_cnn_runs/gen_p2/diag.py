import sys,json,time,numpy as np
sys.path.insert(0,'.');sys.path.insert(0,'../gen_cf')
from s1_fit import stats,SN
import gen2
X=np.load('train_imgs.npy');P=json.load(open('../gen_cf/fitted_params_v4.json'))
t=time.time();g=[gen2.gen(P[i],40000+i) for i in range(40)];print('t/img',(time.time()-t)/40)
Sr=stats(X[:40]);Sg=stats(np.stack(g))
def hf(X):
    d=np.diff(X,axis=2);return np.array([[np.abs(a).mean(),((a-a.mean())**4).mean()/a.var()**2] for a in d])
print(np.round((Sg-Sr).mean(0)/Sr.std(0),2));print(SN)
print('diff absmean/kurt real',hf(X[:40]).mean(0),'gen',hf(np.stack(g)).mean(0))
print('uniq real',np.mean([len(np.unique(x)) for x in X[:40]]))
