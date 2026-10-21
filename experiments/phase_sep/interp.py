import os; os.environ['OMP_NUM_THREADS']='1'
import numpy as np,pandas as pd,json,warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from interpret.glassbox import ExplainableBoostingRegressor
from scipy.stats import spearmanr
import statsmodels.api as sm
R='~/AI/daker'
y=pd.read_csv(f'{R}/open/train.csv').hardness.values
F=pd.read_csv('feat_train.csv'); cols=[c for c in F.columns if c not in ('ID','x_sig')]; X=F[cols]
m=ExplainableBoostingRegressor(interactions=10,outer_bags=8,random_state=0,n_jobs=8).fit(X,y)
imp=sorted(zip(m.term_names_,m.term_importances()),key=lambda t:-t[1])
json.dump(imp,open('ebm_full_imps.json','w'),indent=1)
G=m.explain_global()
names=m.term_names_
mains=[t for t,_ in imp if ' & ' not in t][:6]; inters=[t for t,_ in imp if ' & ' in t][:3]
fig,ax=plt.subplots(2,3,figsize=(13,7))
for a,t in zip(ax.ravel(),mains):
    d=G.data(names.index(t)); e=np.array(d['names'],float); s=np.array(d['scores'])
    a.step(e[:-1],s,where='post'); a.set_title(t); a.axhline(0,c='k',lw=.5)
    q=np.percentile(X[t],[1,99]); a.set_xlim(q)
plt.suptitle('EBM top main effects (full-train fit, interpretation only)'); plt.tight_layout(); plt.savefig('ebm_main_effects.png',dpi=110)
for j,t in enumerate(inters):
    d=G.data(names.index(t)); s=np.array(d['scores'])
    l=np.array(d['left_names'],float); r=np.array(d['right_names'],float)
    a,b=t.split(' & ')
    fig=plt.figure(figsize=(6,5)); plt.pcolormesh(l,r,s.T,cmap='RdBu_r',shading='flat',vmin=-np.abs(s).max(),vmax=np.abs(s).max())
    plt.colorbar(label='hardness contribution'); plt.xlabel(a); plt.ylabel(b)
    plt.xlim(np.percentile(X[a],[1,99])); plt.ylim(np.percentile(X[b],[1,99]))
    plt.scatter(X[a],X[b],c=y,s=6,cmap='viridis',alpha=.5,edgecolors='none'); plt.title(f'Interaction #{j+1}: {t} (dots: samples colored by hardness)')
    plt.tight_layout(); plt.savefig(f'ebm_inter{j+1}.png',dpi=110); plt.close()
# 역 Hall-Petch 교란 점검
out={}
Z=(X-X.mean())/X.std()
for gv in ('g_Amean','g_Amed','g_bnd_spacing'):
    out[gv]={'spearman_raw':float(spearmanr(X[gv],y)[0]),'spearman_hi_vs_rest':float(X[gv][y>230].median()/X[gv][y<=230].median())}
    ctrl_sets={'none':[],'ph':['ph_frac','ph_maxcl'],'pore':['po_frac','po_n'],'elong':['e_ar','e_align'],'unif':['u_dens_cv','u_ph_sd','g_Acv'],
               'all':['ph_frac','ph_maxcl','po_frac','po_n','e_ar','e_align','u_dens_cv','u_ph_sd','g_Acv','b_contrast'],'all+sig':['ph_frac','ph_maxcl','po_frac','po_n','e_ar','e_align','u_dens_cv','u_ph_sd','g_Acv','b_contrast','x_sig']}
    for k,cs in ctrl_sets.items():
        D=Z[[gv]+[c for c in cs if c!='x_sig']].copy()
        if 'x_sig' in cs: D['x_sig']=(F.x_sig-F.x_sig.mean())/F.x_sig.std()
        r=sm.OLS(y,sm.add_constant(D)).fit(cov_type='HC3')
        out[gv][k]=dict(beta=float(r.params[gv]),lo=float(r.conf_int().loc[gv,0]),hi=float(r.conf_int().loc[gv,1]),p=float(r.pvalues[gv]))
json.dump(out,open('hallpetch.json','w'),indent=1)
# 상관 행렬 일부(교란 경로)
cc=X.corrwith(X['g_Amean'],method='spearman').sort_values(); cc.to_csv('corr_with_gAmean.csv')
X.corrwith(pd.Series(y),method='spearman').sort_values().to_csv('spearman_y.csv')
print(imp[:15]); print(json.dumps(out,indent=1))
