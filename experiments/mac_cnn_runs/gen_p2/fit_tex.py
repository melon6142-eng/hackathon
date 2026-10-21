# 각 train 이미지별(그 이미지만 사용) 후처리 질감 파라미터 적합 -> fitted_params_v5.json (500, train index 정렬)
import sys,json,itertools,numpy as np
sys.path.insert(0,'.');sys.path.insert(0,'../gen_cf')
from multiprocessing import Pool
import gen2
X=np.load('train_imgs.npy');P=json.load(open('../gen_cf/fitted_params_v4.json'))
fy=np.fft.fftfreq(256)[:,None];fx=np.fft.fftfreq(256)[None,:];FR=np.sqrt(fx**2+fy**2);EDG=np.geomspace(1/256,0.5,13)
M=[(FR>=EDG[j])&(FR<EDG[j+1]) for j in range(4,12)]
def feat(x):
    Pw=np.abs(np.fft.fft2(x-x.mean()))**2
    return np.r_[[np.log(Pw[m].mean()+1e-12) for m in M],np.log(x.std()),np.log(np.abs(np.diff(x,axis=1)).mean()+np.abs(np.diff(x,axis=0)).mean())]
W=np.array([1]*8+[3,3.])
GRID=list(itertools.product([0,.008,.016,.03,.05],[.6,1.,1.6],[0,.6,1.2],[.6,1.,1.4],[.5,1.,1.5]))
def fit(i):
    p=P[i];b=gen2.base(p,50000+i).astype(np.float32);fr=feat(X[i]);best=(1e9,None)
    for ta,ts,sh,bs,ns in GRID:
        q=dict(p,tex_a=ta,tex_s=ts,sharp=sh,blur=p['blur']*bs,noise=p['noise']*ns)
        e=(((feat(gen2.post(b,q,50000+i))-fr)*W)**2).sum()
        if e<best[0]: best=(e,q)
    return best[1]
if __name__=='__main__':
    with Pool(11) as pl: R=pl.map(fit,range(500),chunksize=4)
    json.dump(R,open('fitted_params_v5.json','w'));print('done')
