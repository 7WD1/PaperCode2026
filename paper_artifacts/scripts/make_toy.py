"""Reproduce the explicitly constructed game in Method 4.3 (not a benchmark)."""
from pathlib import Path
import sys, json, csv
ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / 'tools' / 'python-packages', ROOT.parent / 'tools' / 'python-packages'):
    if candidate.exists():
        sys.path.insert(0, str(candidate))
PAPER = ROOT
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = PAPER / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
(PAPER/'data').mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
 'axes.labelsize':9,'xtick.labelsize':8.5,'ytick.labelsize':8.5,
 'legend.fontsize':8.5,'pdf.fonttype':42,'ps.fonttype':42})
lam = np.linspace(0,1,101)
scores = np.array([.5+.5*np.log(7/3),.5])
p = np.exp(scores/.5-np.max(scores/.5)); p /= p.sum()
assert np.allclose(p,[.7,.3])
share = (1-lam[:,None])*.5+lam[:,None]*p
psi = np.column_stack([4+4*share[:,0]-2*share[:,1],
                       3+4*share[:,1],2-2*share[:,0]])
assert np.allclose(psi.sum(axis=1),11)
assert np.allclose(psi[50],[5.6,4.6,.8])
assert np.allclose(psi[0],[5,5,1])
delta = {frozenset([0]):4.,frozenset([1]):3.,frozenset([2]):2.,
         frozenset([0,1]):4.,frozenset([0,2]):-2.}
v = {s:sum(x for t,x in delta.items() if t <= s)
     for s in [frozenset(i for i in range(3) if mask & (1<<i)) for mask in range(8)]}
for s,value in v.items():
 recovered=sum((-1)**(len(s)-len(u))*vv for u,vv in v.items() if u <= s)
 assert abs(recovered-delta.get(s,0))<1e-12

fig,axs=plt.subplots(1,2,figsize=(7.15,2.6),layout='constrained')
colors=['#D55E00','#0072B2','#009E73']
for i,(name,style) in enumerate(zip('ABC',['-','--','-.'])):
 axs[0].plot(lam,psi[:,i],style,color=colors[i],label='Route '+name,lw=1.8)
 axs[0].scatter([.5],[psi[50,i]],c=colors[i],s=23,zorder=3)
axs[0].set(xlabel=r'Modification strength $\lambda$',ylabel='Allocated contribution',
           title='(a) Resource-dependent redistribution',xlim=(0,1))
axs[0].legend(ncol=1,loc='center right',frameon=False)
axs[0].grid(alpha=.18)
labels=['A','B','C','Total']; x=np.arange(4); width=.35
axs[1].bar(x-width/2,[5,5,1,11],width,label='Uniform sharing',color='#0072B2',hatch='//',alpha=.85)
axs[1].bar(x+width/2,[5.6,4.6,.8,11],width,label=r'RC sharing ($\lambda=0.5$)',color='#D55E00',alpha=.9)
axs[1].set(xticks=x,xticklabels=labels,ylabel='Allocated contribution',
           title='(b) Exact efficiency at fixed game')
axs[1].legend(frameon=False,loc='upper left')
axs[1].set_ylim(0,14)
for a in axs:
 a.spines[['top','right']].set_visible(False)
fig.savefig(OUT/'toy_allocation.pdf')
fig.savefig(OUT/'toy_allocation.png',dpi=600,facecolor='white')
with (OUT/'toy_values.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.writer(f); w.writerow(['lambda','psi_A','psi_B','psi_C','sum'])
 for a,row in zip(lam,psi):w.writerow([a,*row,row.sum()])
(PAPER/'data'/'toy_checks.json').write_text(json.dumps({
 'kind':'constructed deterministic game; no trained model and no benchmark claim',
 'players':3,'enumerated_coalitions':8,'tau':.5,'eta':0,
 'singleton_dividends':[4,3,2],'pair_dividends':{'AB':4,'AC':-2},
 'standard_shapley':[5,5,1],'rc_at_lambda_half':psi[50].tolist(),
 'maximum_efficiency_residual':float(abs(psi.sum(axis=1)-11).max()),
 'all_checks_passed':True},indent=2),encoding='utf-8')
print('Toy game: enumeration, reduction and efficiency checks passed.')
