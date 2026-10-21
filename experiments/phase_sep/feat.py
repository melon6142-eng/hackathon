# 6개 변수 군 특징 (train 500장만, 이미지별 독립). 민감도 변형 포함.
import os,sys; os.environ['OMP_NUM_THREADS']='1'
import numpy as np,pandas as pd,cv2
from skimage import measure
from scipy import ndimage as ndi
from multiprocessing import Pool
from seg import segment,P0
O='../open'
def wavg(x,w): return float((x*w).sum()/(w.sum()+1e-9))
def feats(im,P):
    S=segment(im,P); lab=S['lab']; n=lab.max(); A=S['A']; ok=S['ok']; dark=S['dark']; ph=S['ph']; pore=S['pore']
    d={}
    # 결정립 크기·분포
    Ag=A[ok]; d['g_n']=ok.sum(); d['g_Amean']=Ag.mean(); d['g_Amed']=np.median(Ag); d['g_Acv']=Ag.std()/Ag.mean(); d['g_Aq90']=np.percentile(Ag,90)
    L=S['skel'].sum(); d['b_len_dens']=L/im.size; d['g_bnd_spacing']=2*im.size/(L+1)  # 평균 절편 길이 근사
    d['b_contrast']=float(S['ridge'][S['skel']].mean()/(S['ridge'].mean()+1e-9))
    # 신장·정렬 (셀)
    pr=measure.regionprops_table(lab,properties=('label','area','axis_major_length','axis_minor_length','orientation','centroid'))
    idx=pr['label']-1; k=ok[idx]
    ar=pr['axis_major_length'][k]/(pr['axis_minor_length'][k]+1); w=pr['area'][k].astype(float); o=pr['orientation'][k]
    d['e_ar']=wavg(ar,w); d['e_ar_med']=np.median(ar); d['e_align']=np.hypot(wavg(np.cos(2*o),w),wavg(np.sin(2*o),w))
    dk=dark[idx][k]
    d['e_ar_ph']=wavg(ar[dk],w[dk]) if dk.sum()>0 else d['e_ar']
    d['e_ar_mat']=wavg(ar[~dk],w[~dk]) if (~dk).sum()>0 else d['e_ar']
    # 2상
    d['ph_frac']=ph.mean(); d['ph_n']=int(dk.sum()); d['ph_sep']=S['sep']
    Ap=w[dk]; d['ph_Amean']=Ap.mean() if len(Ap) else 0; d['ph_Acv']=Ap.std()/Ap.mean() if len(Ap)>1 else 0
    d['ph_size_ratio']=d['ph_Amean']/d['g_Amean']
    phd=cv2.dilate(ph.astype(np.uint8),np.ones((5,5),np.uint8))  # 입계 넘어 인접 2상 셀 병합
    cl=measure.label(phd); cs=np.bincount(cl.ravel())[1:]
    d['ph_ncl']=len(cs); d['ph_maxcl']=(cs.max()/im.size) if len(cs) else 0
    d['ph_conn']=(d['ph_n']/max(len(cs),1))  # 클러스터당 셀 수
    d['ph_contrast']=(np.nanmedian(S['Ic'][ok])-np.nanmean(S['Ic'][ok&dark]))/(S['csd']) if dk.sum()>0 else 0
    # 기공
    pl=measure.label(pore); pa=np.bincount(pl.ravel())[1:].astype(float)
    d['po_n']=len(pa); d['po_frac']=pore.mean(); d['po_Amean']=pa.mean() if len(pa) else 0; d['po_Amax']=pa.max() if len(pa) else 0
    d['po_Acv']=pa.std()/pa.mean() if len(pa)>1 else 0
    d['po_in_ph']=(pore&cv2.dilate(ph.astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)).sum()/(pore.sum()+1)
    # 균일성: 4x4 타일
    T=lambda m: m.reshape(4,64,4,64).mean((1,3))
    cen=np.zeros(im.shape,bool); cy=pr['centroid-0'][k].astype(int); cx=pr['centroid-1'][k].astype(int); cen[cy,cx]=True
    tc=T(cen.astype(float)); d['u_dens_cv']=tc.std()/(tc.mean()+1e-9)
    tp=T(ph.astype(float)); d['u_ph_sd']=tp.std()
    tb=T(S['skel'].astype(float)); d['u_bnd_cv']=tb.std()/(tb.mean()+1e-9)
    # 이미지 조건(교란 통제용)
    d['x_sig']=S['sig']
    return d
def one(a):
    i,P=a; return feats(cv2.imread(f'{O}/train/{i}.png',0),P)
VAR={'base':{},'rp75':{'rp':75},'rp85':{'rp':85},'h1.5':{'h':1.5},'h3':{'h':3.0},'tp2':{'tp':2.0},'tp3':{'tp':3.0},
     'tph1.0':{'tph':1.0},'tph2.0':{'tph':2.0},'gs0.75':{'gs':0.75},'gs1.25':{'gs':1.25},'circ0.65':{'circ':0.65},'circ0.85':{'circ':0.85}}
if __name__=='__main__':
    ids=pd.read_csv(f'{O}/train.csv').ID.tolist(); os.makedirs('sens',exist_ok=True)
    for nm,ch in VAR.items():
        f='feat_train.csv' if nm=='base' else f'sens/{nm}.csv'
        if os.path.exists(f): continue
        P=dict(P0,**ch)
        with Pool(10) as p: R=p.map(one,[(i,P) for i in ids],chunksize=5)
        df=pd.DataFrame(R).fillna(0); df.insert(0,'ID',ids); df.to_csv(f,index=False); print(nm,'done',flush=True)
