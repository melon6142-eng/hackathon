import os,numpy as np
os.chdir('./runs')
n=500
ld=lambda p:np.load(p)
av=lambda ts:np.mean([ld(f'dino_ft{x}/oof.npy') for x in ts],0)
M={'BL1':ld('oof_blend1.npy'),'ViT-B 원본(3seed평균)':av(['/dinoB_orig','2/dinoB_orig_s1','2/dinoB_orig_s2']),
 'ViT-B 변환(3seed평균)':av(['/dinoB_trans','2/dinoB_trans_s1','2/dinoB_trans_s2']),
 'ViT-B 변환 u6':av(['2/dinoB_trans_u6']),'ViT-B 변환 u8':av(['2/dinoB_trans_u8']),
 'ViT-S 원본':av(['/dino_orig']),'ViT-S 변환':av(['/dino_trans']),'ViT-S 변환(가중)':av(['/dino_trans_w']),
 'ConvNeXt 미세조정':av(['/cnx_full_w']),'EffNet-B0 미세조정':av(['/effb0_full']),'LDS(d4_lds)':ld('dino_ft3/d4_lds/oof.npy'),
 'gen LGB':ld('gen/oof_lgb_2.npy'),'gen Ridge':ld('gen/oof_ridge_2.npy')}
z=np.load('label_ssl/oof_ssl.npz')
for k in z: M[f'SSL {k}']=z[k]
for k in ['isotonic','qmap','qreg','stretch']: M[f'보정 {k}']=ld(f'cal/oof_{k}.npy')
for k in M: assert M[k].shape==(n,),k
os.chdir('./runs/ablation_all')
