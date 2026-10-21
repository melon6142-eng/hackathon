import numpy as np,pandas as pd,sys,warnings;warnings.filterwarnings('ignore')
from scipy.optimize import nnls
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
D='./runs';y=pd.read_csv('./open/train.csv')['hardness'].values
ld=lambda t,k:np.load(f'{D}/{t}/{k}.npy');av=lambda ts,k:np.mean([ld(t,k) for t in ts],0)
O=['dino_ft/dinoB_orig','dino_ft2/dinoB_orig_s1','dino_ft2/dinoB_orig_s2'];T=['dino_ft/dinoB_trans','dino_ft2/dinoB_trans_s1','dino_ft2/dinoB_trans_s2']
B=np.column_stack([np.load(f'{D}/oof_blend1.npy'),av(O,'oof'),av(T,'oof')])
X=pd.read_pickle(f'{D}/hi_rule/X_train.pkl').reset_index(drop=True);f3=pd.read_csv(f'{D}/joaegi/feat3.csv').iloc[:500].reset_index(drop=True);G=pd.read_csv('geo_train.csv')
M=pd.DataFrame({'ws_A_mean(grain)':X.grain_A_mean,'ws_n(grain)':X.grain_ng,'ws_A_mean(new)':G.ws_Amean,'ws_n(new)':G.ws_n,
 'coh(hand)':X.hand_coh,'coh8_mean':f3.st_coh8_mean,'dark_frac_z-1':X['hand_dark1.0'],'otsu_dark_frac':X.grain_otsu_frac,
 'psd_slope':G.psd_slope,'psd_lowfrac':G.psd_lowfrac,'acf_Lmaj':X.acf_Lmaj,'acf_Lmin':X.acf_Lmin,'maxdark_cc_z-1':f3['cc-1_max'],'maxdark_cc_z-1.5':f3['cc-1.5_max']})
M=M.fillna(M.median());Mv=M.values;names=list(M.columns)
def blend(tr,va):
    A=np.column_stack([B,np.ones(len(B))]);w=nnls(A[tr],y[tr])[0];return A[tr]@w,A[va]@w
r=lambda p,t:np.sqrt(((p-t)**2).mean())
# --- 2) correlations, 1x 5fold blend OOF
oof=np.zeros(500)
for tr,va in KFold(5,shuffle=True,random_state=100).split(B): oof[va]=blend(tr,va)[1]
res=y-oof;L=['## 2) 구간별 Spearman (측정값 vs y / vs 잔차 y-pred)','','| 측정 | y>210 ρy | ρres | y>230 ρy | ρres | pred상위20% ρy | ρres |','|---|---|---|---|---|---|---|']
Z={'y>210':y>210,'y>230':y>230,'top20':oof>=np.quantile(oof,.8)}
for j,n in enumerate(names):
    s=[]
    for m in Z.values(): s+=[spearmanr(Mv[m,j],y[m])[0],spearmanr(Mv[m,j],res[m])[0]]
    L.append(f'| {n} | '+' | '.join(f'{v:+.2f}' for v in s)+' |')
L.append(f'\nn: y>210={Z["y>210"].sum()}, y>230={Z["y>230"].sum()}, top20={Z["top20"].sum()}; blend OOF RMSE={r(oof,y):.3f}')
# --- 3) band reorder
QS=[.1,.2,.3];WS=[2,4,6,8,12]
def scores(tr,va,ptr):
    """structure-score candidates fit on tr: ridge on residual-in-band + singles signed by train corr"""
    out={}
    mu,sd=Mv[tr].mean(0),Mv[tr].std(0)+1e-9;Zs=(Mv-mu)/sd
    band=ptr>=np.quantile(ptr,.7)
    rg=Ridge(30).fit(Zs[tr][band],y[tr][band]);out['ridge']=rg.predict(Zs)
    for j,n in enumerate(names):
        c=spearmanr(Zs[tr][band,j],y[tr][band])[0];out[n]=np.sign(c)*Zs[:,j]
    return out
def apply(ptr,pva,sva,sc_tr,q,kind,w):
    thr=np.quantile(ptr,1-q);b=pva>=thr;p=pva.copy()
    if b.sum()<2: return p
    if kind=='iso':
        vals=np.sort(p[b]);order=np.argsort(np.argsort(sva[b]));p[b]=vals[order]
    else:
        bt=ptr>=thr;m,s=sc_tr[bt].mean(),sc_tr[bt].std()+1e-9;p[b]+=w*(sva[b]-m)/s
    return p
CF=[('none',None,None,0)]+[(k,q,'iso',0) for k in ['ridge']+names for q in QS]+[(k,q,'add',w) for k in ['ridge']+names for q in QS for w in WS]
def method(tr,va):
    e=np.zeros(len(CF))
    for i,j in KFold(5,shuffle=True,random_state=7).split(tr):
        a,b=tr[i],tr[j];pa,pb=blend(a,b);S=scores(a,b,pa)
        for c,(k,q,kind,w) in enumerate(CF):
            pp=pb if k=='none' else apply(pa,pb,S[k][b],S[k][a],q,kind,w);e[c]+=((pp-y[b])**2).sum()
    k,q,kind,w=CF[e.argmin()];pa,pv=blend(tr,va)
    if k=='none': return pv,CF[e.argmin()]
    S=scores(tr,va,pa);return apply(pa,pv,S[k][va],S[k][tr],q,kind,w),CF[e.argmin()]
# fixed-config oracles for reference (not selectable cheating, reported separately)
R=int(sys.argv[1]);hi=y>230;lo=y<170;res3={'base':[],'geo_reorder':[]};ch=[]
for s in range(R):
    pp={k:np.zeros(500) for k in res3}
    for tr,va in KFold(5,shuffle=True,random_state=100+s).split(B):
        pp['base'][va]=blend(tr,va)[1];p,c=method(tr,va);pp['geo_reorder'][va]=p;ch.append(str(c))
    for k in res3: res3[k].append((r(pp[k],y),(pp[k]-y)[hi].mean(),(pp[k]-y)[lo].mean()))
    print(s,{k:round(res3[k][-1][0],3) for k in res3},flush=True)
L+=['','## 3) 상위밴드 재정렬 (구조점수·q·방식·w 모두 내부CV 선택)','',f'| 방법 | nCV({R}회) | std | hi>230 편향 | lo<170 편향 |','|---|---|---|---|---|']
for k,v in res3.items():
    a=np.array(v);L.append(f'| {k} | {a[:,0].mean():.3f} | {a[:,0].std():.3f} | {a[:,1].mean():.1f} | {a[:,2].mean():.1f} |')
L.append('\n선택된 설정 상위: '+str(pd.Series(ch).value_counts().head(6).to_dict()))
open('ev3.md','w').write('\n'.join(L));print('\n'.join(L))
