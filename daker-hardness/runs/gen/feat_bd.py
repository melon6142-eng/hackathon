import os; os.environ['OMP_NUM_THREADS']='1'
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, cv2, sys
from skimage import filters, segmentation, measure, morphology, restoration, feature
from scipy import stats, ndimage as ndi
from sklearn.ensemble import RandomForestClassifier
from multiprocessing import Pool
R='./open'
def st(pref,v,d):
    v=np.asarray(v,float); v=v[np.isfinite(v)]
    if len(v)==0: v=np.zeros(1)
    d[pref+'_mean']=v.mean(); d[pref+'_med']=np.median(v); d[pref+'_q90']=np.quantile(v,.9); d[pref+'_q99']=np.quantile(v,.99); d[pref+'_std']=v.std()
def segstats(lab,den,g,tag,d):
    # boundary pixels between pairs of labels
    b=segmentation.find_boundaries(lab,mode='inner')&(lab>0)
    # pair id via max of neighbor label
    mx=ndi.maximum_filter(lab,3); mn=ndi.minimum_filter(np.where(lab>0,lab,1<<30),3)
    pair=(mn.astype(np.int64)<<20)+mx; bb=b&(mx!=mn)&(mn<(1<<30))
    pid=pair[bb]; up,inv=np.unique(pid,return_inverse=True)
    cellI=ndi.median(den,lab,index=np.arange(lab.max()+1)) if lab.max()<4000 else None
    I=den[bb]; G=g[bb]
    n=len(up); cnt=np.bincount(inv,minlength=n)
    segI=np.bincount(inv,I,n)/np.maximum(cnt,1); segG=np.bincount(inv,G,n)/np.maximum(cnt,1)
    a=(up>>20).astype(int); c=(up&((1<<20)-1)).astype(int)
    nbr=(np.asarray(cellI)[a]+np.asarray(cellI)[c])/2
    dark=nbr-segI; step=np.abs(np.asarray(cellI)[a]-np.asarray(cellI)[c])
    st(f'{tag}_bdark',dark,d); st(f'{tag}_bgrad',segG,d); st(f'{tag}_bstep',step,d); st(f'{tag}_blen',cnt,d)
    d[f'{tag}_lowc_frac']=(dark<3).mean(); d[f'{tag}_nseg']=n; d[f'{tag}_ncell']=lab.max()
    return b
def f(args):
    s,i=args; im=cv2.imread(f'{R}/{s}/{i}.png',0).astype(np.float32); d={'ID':i}
    v=im.ravel(); d['I_mean']=v.mean(); d['I_skew']=stats.skew(v); d['I_kurt']=stats.kurtosis(v)
    for q in (1,5,25,50,75,95,99): d[f'I_q{q}']=np.percentile(v,q)
    sig=restoration.estimate_sigma(im/255)*255
    den=cv2.fastNlMeansDenoising(im.astype(np.uint8),None,h=max(3,sig*1.2),templateWindowSize=5,searchWindowSize=15).astype(np.float32)
    h,_=np.histogram(den,64,(0,256)); hs=ndi.gaussian_filter1d(h.astype(float),1.5)
    pk=feature.peak_local_max(hs,min_distance=3,threshold_rel=0.05); d['I_nmodes']=len(pk)
    g=np.hypot(cv2.Sobel(den,cv2.CV_32F,1,0),cv2.Sobel(den,cv2.CV_32F,0,1))
    d['g_q99']=np.percentile(g,99); d['g_q999']=np.percentile(g,99.9)
    rid=filters.sato(den,sigmas=[1,1.5],black_ridges=True); thr=filters.threshold_otsu(rid); bd=rid>thr
    # width: local thickness of ridge mask via distance transform on skeleton
    sk=morphology.skeletonize(bd); dt=ndi.distance_transform_edt(bd); st('bwidth',2*dt[sk],d)
    # continuity: skeleton endpoints / length
    nb=ndi.convolve(sk.astype(int),np.ones((3,3)),mode='constant')-1
    d['b_endfrac']=((nb==1)&sk).sum()/max(sk.sum(),1); d['b_comp']=measure.label(sk).max()/max(sk.sum(),1)*100
    # sharpness: 10-90 edge width proxy = ridge depth / max grad
    st('rid',rid[sk],d)
    # watershed
    interior=~bd; dist=cv2.distanceTransform(interior.astype(np.uint8),cv2.DIST_L2,3)
    mk=measure.label(morphology.h_maxima(dist,1.5)); ws=segmentation.watershed(rid,mk)
    segstats(ws,den,g,'ws',d)
    # random walker on 2x downsample
    sm=cv2.resize(den,(128,128),interpolation=cv2.INTER_AREA); mk2=cv2.resize(mk.astype(np.float32),(128,128),interpolation=cv2.INTER_NEAREST).astype(int)
    try:
        rw=segmentation.random_walker((sm-sm.mean())/(sm.std()+1e-6),mk2,beta=130,mode='cg',tol=1e-2)
        rwu=cv2.resize(rw.astype(np.float32),(256,256),interpolation=cv2.INTER_NEAREST).astype(int)
        segstats(rwu,den,g,'rw',d)
        # stability: ARI-ish = boundary agreement
        b1=segmentation.find_boundaries(ws); b2=segmentation.find_boundaries(rwu)
        b1d=ndi.binary_dilation(b1,iterations=2); d['stab_ws_rw']=(b2&b1d).sum()/max(b2.sum(),1)
    except Exception as e: d['stab_ws_rw']=np.nan
    # ilastik-like: multiscale filters + RF self-training (seeds: confident ridge vs confident interior)
    F=[]
    for sgm in (0.7,1.6,3.5):
        gg=ndi.gaussian_filter(den,sgm); F+=[gg,ndi.gaussian_laplace(den,sgm),ndi.gaussian_gradient_magnitude(den,sgm)]
        He=feature.hessian_matrix_eigvals(feature.hessian_matrix(den,sgm,order='rc',use_gaussian_derivatives=False)); F+=[He[0],He[1]]
    F=np.stack(F,-1).reshape(-1,len(F))
    pos=(rid>np.percentile(rid,95)).ravel(); neg=(dist>4).ravel()
    rng=np.random.default_rng(0); ip=rng.choice(np.where(pos)[0],min(1500,pos.sum()),False); ineg=rng.choice(np.where(neg)[0],min(1500,neg.sum()),False)
    rf=RandomForestClassifier(40,max_depth=10,n_jobs=1,random_state=0).fit(F[np.r_[ip,ineg]],np.r_[np.ones(len(ip)),np.zeros(len(ineg))])
    P=rf.predict_proba(F)[:,1].reshape(256,256)
    d['rf_bfrac']=(P>0.5).mean(); d['rf_unc']=((P>0.3)&(P<0.7)).mean()
    bd2=P>0.5; dist2=cv2.distanceTransform((~bd2).astype(np.uint8),cv2.DIST_L2,3)
    mk3=measure.label(morphology.h_maxima(dist2,1.5)); ws2=segmentation.watershed(P,mk3)
    segstats(ws2,den,g,'rf',d)
    d['stab_ws_rf']=abs(ws2.max()-ws.max())/max(ws.max(),1)
    return d
if __name__=='__main__':
    jobs=[]
    for s in ['train','test']:
        jobs+=[(s,x[:-4]) for x in sorted(os.listdir(f'{R}/{s}'))]
    if len(sys.argv)>1: jobs=jobs[:int(sys.argv[1])]
    with Pool(3) as p: out=p.map(f,jobs,chunksize=5)
    pd.DataFrame(out).to_csv('bd_feats.csv' if len(sys.argv)==1 else 'bd_tmp.csv',index=False); print('done',len(out))
