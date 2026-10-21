exec(open('s.py').read().split('res={}')[0])
import lightgbm as lgb
P={'lgb':np.zeros(n),'lgb_pl':np.zeros(n)}
for tr,va in KFold(5,shuffle=True,random_state=0).split(np.arange(n)):
  p=fit(F[tr],y[tr]).predict(F);u=np.arange(n,N)
  kw=dict(n_estimators=600,learning_rate=.02,num_leaves=15,subsample=.8,subsample_freq=1,colsample_bytree=.3,n_jobs=2,verbose=-1)
  P['lgb'][va]=lgb.LGBMRegressor(**kw).fit(F[tr],y[tr]).predict(F[va])
  P['lgb_pl'][va]=lgb.LGBMRegressor(**kw).fit(np.vstack([F[tr],F[u]]),np.r_[y[tr],p[u]],sample_weight=np.r_[np.ones(len(tr)),.5*np.ones(len(u))]).predict(F[va])
b=np.load('oof_blend1.npy');M=lambda t:np.mean([np.load(f'dino_ft2/{x}/oof.npy') if os.path.exists(f'dino_ft2/{x}/oof.npy') else np.load(f'dino_ft/{x}/oof.npy') for x in t],0)
bo=M(['dinoB_orig','dinoB_orig_s1','dinoB_orig_s2']);bt=M(['dinoB_trans','dinoB_trans_s1','dinoB_trans_s2'])
from sklearn.model_selection import KFold as K
def ncv(cols,reps=20):
    X=np.column_stack(cols+[np.ones(n)]);o=[]
    for s_ in range(reps):
        q=np.zeros(n)
        for a_,v_ in K(5,shuffle=True,random_state=100+s_).split(X): w,_=nnls(X[a_],y[a_]);q[v_]=X[v_]@w
        o.append(r(q))
    return np.mean(o)
L=[f'| {k} (1회) | {r(v):.3f} | {hb(v):.1f} | {lb(v):.1f} | {ncv([b,bo,bt,v]):.3f} |' for k,v in P.items()]
open('label_ssl/interim.md','a').write('\n'.join(L)+'\n');print('\n'.join(L))
