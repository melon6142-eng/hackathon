import os;os.environ['OMP_NUM_THREADS']='1'
import numpy as np,pandas as pd,cv2,glob
from skimage import measure
from scipy.spatial import cKDTree
from multiprocessing import Pool
O='./open'
# 기존 stage_v1/featA 규약: blur σ2, z=(g-median)/std. 어두운쪽 단계: S1(최암) z<-1.0 (=L2+L3+pore), S2 -1.0<=z<-0.5 (=L1)
def zimg(p,glob_=None):
    im=cv2.imread(p,0).astype(np.float32);g=cv2.GaussianBlur(im,(0,0),2)
    return (g-np.median(g))/(g.std()+1e-6) if glob_ is None else (g-glob_[0])/glob_[1]
def mf(m,p,d):
    n=m.size;d[p+'frac']=m.mean()
    d[p+'euler']=measure.euler_number(m,connectivity=2);d[p+'perim']=measure.perimeter(m)/n
    lab=measure.label(m,connectivity=2);keys=['n','A_mean','A_med','A_q90','A_cv','ar','ecc','align','nn_mean','nn_cv','CE','L5','L10','L20','g10','iso3','iso6','iso10','perc3','perc6','perc10','ncomp6']
    if lab.max()<3:
        for k in keys:d[p+k]=0
        return lab
    pr=measure.regionprops_table(lab,properties=('label','area','axis_major_length','axis_minor_length','eccentricity','orientation','centroid'))
    A=pr['area'].astype(float);k=A>=4
    if k.sum()<3:k=A>=1
    A=A[k];w=A/A.sum();d[p+'n']=k.sum();d[p+'A_mean']=A.mean();d[p+'A_med']=np.median(A);d[p+'A_q90']=np.percentile(A,90);d[p+'A_cv']=A.std()/A.mean()
    d[p+'ar']=(w*pr['axis_major_length'][k]/(pr['axis_minor_length'][k]+1)).sum();d[p+'ecc']=(w*pr['eccentricity'][k]).sum()
    o=pr['orientation'][k];d[p+'align']=np.hypot((w*np.cos(2*o)).sum(),(w*np.sin(2*o)).sum())
    c=np.c_[pr['centroid-0'][k],pr['centroid-1'][k]];lam=len(c)/n;tr=cKDTree(c)
    nn=tr.query(c,2)[0][:,1];d[p+'nn_mean']=nn.mean();d[p+'nn_cv']=nn.std()/nn.mean();d[p+'CE']=nn.mean()/(0.5/np.sqrt(lam))
    for r in [5,10,20]:
        K=2*len(tr.query_pairs(r))/len(c)/lam;d[p+f'L{r}']=np.sqrt(K/np.pi)-r
    d[p+'g10']=(2*(len(tr.query_pairs(12))-len(tr.query_pairs(8)))/len(c)/lam)/(np.pi*(144-64))
    keep=np.isin(lab,pr['label'][k]);L0=lab*keep
    for kk in [3,6,10]:
        dl=cv2.dilate(keep.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kk,kk)))
        L2=measure.label(dl,connectivity=2);s=pd.DataFrame({'c':L2[keep],'l':L0[keep]}).drop_duplicates()
        cnt=s.groupby('c').size();per=s.c.map(cnt)
        d[p+f'iso{kk}']=(per==1).mean();d[p+f'perc{kk}']=cnt.max()/len(s)
        if kk==6:d[p+'ncomp6']=len(cnt)/len(s)
    return lab
def one(args):
    p,G=args;d={'ID':os.path.basename(p)[:-4]}
    for tag,gl in [('i',None),('g',G)]:
        z=zimg(p,gl);s1=z<-1.0;s2=(z>=-1.0)&(z<-0.5);s12=s1|s2
        mf(s1,f'{tag}S1_',d);mf(s2,f'{tag}S2_',d);lab12=mf(s12,f'{tag}S12_',d)
        l1=measure.label(s1,connectivity=2)
        if l1.max()>0:
            ring=cv2.dilate(s1.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)&~s1
            d[f'{tag}rel_s1ring_in_s2']=s2[ring].mean() if ring.any() else 0
            hit=np.unique(lab12[s1]);d[f'{tag}rel_s12comp_with_s1']=len(hit)/max(lab12.max(),1)
            d[f'{tag}rel_s1_over_s12']=s1.sum()/max(s12.sum(),1)
        else:
            for k in ['rel_s1ring_in_s2','rel_s12comp_with_s1','rel_s1_over_s12']:d[f'{tag}{k}']=0
    return d
if __name__=='__main__':
    P=sorted(glob.glob(f'{O}/train/*.png'))+sorted(glob.glob(f'{O}/test/*.png'))
    v=np.concatenate([cv2.GaussianBlur(cv2.imread(p,0).astype(np.float32),(0,0),2).ravel()[::97] for p in P[:500]])
    G=(np.median(v),v.std());print('global',G,flush=True)
    with Pool(2) as pool:R=pool.map(one,[(p,G) for p in P],chunksize=10)
    pd.DataFrame(R).to_csv('dm_feats.csv',index=False);print('done',len(R))
