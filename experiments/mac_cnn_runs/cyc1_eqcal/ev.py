import glob,numpy as np,pandas as pd
from scipy.optimize import nnls;from sklearn.model_selection import KFold
from sklearn.isotonic import IsotonicRegression
R='~/AI/daker/cnn/runs';I=R+'/indepfold';G=R+'/bag'
y=pd.read_csv(R+'/lcurve/w/open/train.csv').hardness.values;hi=y>230
r_=lambda p:np.sqrt(((p-y)**2).mean());A1=lambda X:np.column_stack([X,np.ones(len(X))])
def nest(X,fit,R=10):
    o=[]
    for s in range(R):
        pp=np.zeros(len(y))
        for tr,va in KFold(5,shuffle=True,random_state=100+s).split(X): pp[va]=fit(X[tr],y[tr],X[va])
        o.append((r_(pp),(pp-y)[hi].mean()))
    return np.array(o).mean(0)
def m_nnls(Xt,yt,Xv): return A1(Xv)@nnls(A1(Xt),yt)[0]
def m_eq(Xt,yt,Xv): return Xv.mean(1)
def m_eqlin(Xt,yt,Xv):  # equal mean + nested linear stretch
    a,b=np.polyfit(Xt.mean(1),yt,1);return a*Xv.mean(1)+b
def m_eqiso(Xt,yt,Xv):
    ir=IsotonicRegression(out_of_bounds='clip').fit(Xt.mean(1),yt);p=ir.predict(Xv.mean(1));return 0.5*p+0.5*Xv.mean(1)
def m_nnlslin(Xt,yt,Xv):
    w=nnls(Xt,yt)[0];return m_eqlin((Xt@w)[:,None],yt,(Xv@w)[:,None])
M=dict(nnls=m_nnls,eq=m_eq,eq_lin=m_eqlin,eq_iso50=m_eqiso,nnls_noint_lin=m_nnlslin)
rows=[]
for rs in (7,123,2026):
    b=np.load(f'{I}/bl1_rs{rs}.npy')
    ld=lambda c:[np.load(f) for f in sorted(glob.glob(f'{I}/rs{rs}_{c}_s*/oof.npy'))+sorted(glob.glob(f'{G}/rs{rs}_{c}_s*/oof.npy'))]
    oo,tt=ld('orig'),ld('trans')
    for k in (3,6):
        X=np.column_stack([b,np.mean(oo[:k],0),np.mean(tt[:k],0)])
        for n,f in M.items():
            m,h=nest(X,f);rows.append(dict(rs=rs,k=k,method=n,ncv=round(m,3),hi=round(h,1)))
d=pd.DataFrame(rows);d.to_csv('res.csv',index=False)
p=d.pivot_table(index=['k','method'],columns='rs',values='ncv');p['mean']=p.mean(1);p['sd']=p[[7,123,2026]].std(1)
base=p.xs('nnls',level='method');
for k in (3,6):
  for mth in M: p.loc[(k,mth),'d_vs_nnls_all']=all(p.loc[(k,mth),c]<base.loc[k,c] for c in (7,123,2026))
print(p.round(3).to_string());print(d.pivot_table(index=['k','method'],columns='rs',values='hi').to_string())
