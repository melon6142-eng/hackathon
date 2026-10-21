import numpy as np,pandas as pd,cv2,matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
y=pd.read_csv('./open/train.csv');b=np.load('../hi_rule/base_oof_fullw.npy');h=y.hardness.values
hi=np.where(h>230)[0];hi=hi[np.argsort(b[hi])][::-1][:6];used=set();ct=[]
for i in hi:
    c=[j for j in np.argsort(np.abs(b-b[i])) if h[j]<210 and j not in used][0];used.add(c);ct.append(c)
f,ax=plt.subplots(4,6,figsize=(18,12))
def ov(j):
    im=cv2.imread(f'./open/train/{y.ID[j]}.png',0).astype(np.float32);g=cv2.GaussianBlur(im,(0,0),2);z=(g-np.median(g))/(g.std()+1e-6)
    o=np.ones((256,256,3));o[(z>=-1)&(z<-.5)]=[1,.65,0];o[z<-1]=[.6,0,0];return o
for k,(i,c) in enumerate(zip(hi,ct)):
    for r,j,t in [(0,i,'HI'),(1,c,'CTRL')]:
        a=ax[(k//3)*2+r][k%3*2];a.imshow(cv2.imread(f'./open/train/{y.ID[j]}.png',0),cmap='gray');a.set_title(f'{t} y={h[j]:.0f} pred={b[j]:.0f}',fontsize=9)
        a=ax[(k//3)*2+r][k%3*2+1];a.imshow(ov(j));a.set_title('dark mask S1=red S2=orange',fontsize=9)
for a in ax.ravel():a.axis('off')
plt.tight_layout();plt.savefig('fig_mask_hi_vs_ctrl.png',dpi=70)
