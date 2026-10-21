import numpy as np,pandas as pd,glob
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
D='./runs';y=pd.read_csv('./open/train.csv').hardness.values
Xtr=np.load(f'{D}/dino/vitb14_252_train.npy');Xte=np.load(f'{D}/dino/vitb14_252_test.npy')
L=['# label_ssl interim','## (2) 라벨 구조']
d=np.diff(np.sort(y));L.append(f'고유값 {len(np.unique(y.round(4)))}/500, 최소간격 {d.min():.4f}, 간격중앙 {np.median(d):.3f}, 소수부 균일성: 모두 연속 실수')
k=gaussian_kde(y,bw_method=.15);g=np.linspace(140,270,1000);pk,_=find_peaks(k(g),prominence=1e-4);L.append(f'KDE(bw .15) 봉우리: {g[pk].round(1).tolist()}')
bic=[GaussianMixture(c,random_state=0,n_init=3).fit(y[:,None]).bic(y[:,None]) for c in range(1,7)];L.append(f'GMM BIC k=1..6: {np.round(bic,1).tolist()}')
A=np.vstack([Xtr,Xte]);Z=PCA(64,random_state=0).fit_transform(StandardScaler().fit_transform(A));Zt,Zs=Z[:500],Z[500:]
for c in [4,8,16,32]:
  lab=KMeans(c,n_init=5,random_state=0).fit_predict(Z);lt=lab[:500]
  ss_w=sum(((y[lt==i]-y[lt==i].mean())**2).sum() for i in set(lt));r2=1-ss_w/((y-y.mean())**2).sum()
  L.append(f'KMeans k={c}: 클러스터가 설명하는 경도분산 R2(in-sample) {r2:.3f}; 클러스터 내 std 중앙 {np.median([y[lt==i].std() for i in set(lt) if (lt==i).sum()>2]):.1f}; test만 있는 클러스터 {len(set(lab[500:])-set(lt))}')
from sklearn.metrics import pairwise_distances
Dm=pairwise_distances(Zt);np.fill_diagonal(Dm,1e9);nn=Dm.argmin(1);dy=np.abs(y-y[nn])
L.append(f'최근접쌍 |Δ경도| 중앙 {np.median(dy):.1f}, RMS/√2 (노이즈 바닥 상한추정) {np.sqrt((dy**2).mean()/2):.2f}; 가장 가까운 10% 쌍 RMS/√2 {np.sqrt((dy[Dm.min(1)<np.quantile(Dm.min(1),.1)]**2).mean()/2):.2f}')
Ds=pairwise_distances(Zs,Zt);L.append(f'test→train 최근접거리 중앙 {np.median(Ds.min(1)):.2f} vs train→train {np.median(Dm.min(1)):.2f}')
gf=pd.read_csv(f'{D}/gen/gen_feats.csv');L.append('gen 컬럼: '+','.join(gf.columns[:30]))
open('interim.md','w').write('\n'.join(L)+'\n');print('\n'.join(L))
np.save('Z64.npy',Z)
