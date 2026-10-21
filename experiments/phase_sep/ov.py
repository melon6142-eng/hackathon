import sys,numpy as np,pandas as pd,cv2; from seg import segment,overlay
d=pd.read_csv('../open/train.csv').sort_values('hardness').reset_index(drop=True)
ix=[0,40,100,170,250,330,400,470,482,490,495,499]
tiles=[]
for i in ix:
    r=d.iloc[i]; im=cv2.imread(f'../open/train/{r.ID}.png',0); S=segment(im)
    o=overlay(im,S); g=cv2.cvtColor(im,cv2.COLOR_GRAY2BGR); t=np.hstack([g,o])
    cv2.putText(t,f'{r.ID[-4:]} H={r.hardness:.0f} ph={S["ph"].mean():.2f} pore={S["pore"].mean():.3f} sep={S["sep"]:.1f}',(4,14),cv2.FONT_HERSHEY_SIMPLEX,0.42,(0,255,0),1)
    tiles.append(t)
rows=[np.hstack(tiles[k:k+2]) for k in range(0,12,2)]
cv2.imwrite(sys.argv[1] if len(sys.argv)>1 else 'overlay12.png',np.vstack(rows))
