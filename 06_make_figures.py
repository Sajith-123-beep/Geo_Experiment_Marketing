import sys; sys.path.insert(0,"src")
from common import *
import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, warnings; warnings.filterwarnings('ignore')
import placebo_tbr as S
import matplotlib.dates as mdates
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
raw=pd.read_excel(DATA_XLSX,sheet_name='Daily metrics'); raw['state']=raw.state.str.strip().str.upper()
d=pd.read_pickle(INTERIM/'clean.pkl'); m=pd.read_pickle(INTERIM/'markets.pkl').set_index('state')
BL='#1F4E79';OR='#D9822B';GR='#7F7F7F';RD='#C0392B';GN='#2E8B57'
# ---- Fig 1: data quality / state patterns (2x2)
fig,ax=plt.subplots(2,2,figsize=(10,6.4))
a=ax[0,0]; ga=raw[(raw.state=='GA')].drop_duplicates('date').sort_values('date')
a.plot(ga.date,ga.net_revenue_usd,color=BL,lw=1); 
bad=ga[(ga.net_revenue_usd.isna())|(ga.revenue_feed_complete==0)]
a.scatter(bad.date,bad.net_revenue_usd.fillna(0),color=RD,s=14,zorder=3,label='12 missing / zero-filled days')
a.set_title('A. Georgia: 12 unusable revenue days',loc='left',fontsize=9.5,fontweight='bold'); a.set_ylabel('Net revenue (USD)'); a.legend(frameon=False,fontsize=8,loc='lower left')
a=ax[0,1]
idx=d.pivot(index='date',columns='state',values='net_revenue_clean'); idx7=idx.rolling(7).mean()
base=idx7.loc['2026-03-30':'2026-04-24'].mean()
for s in idx7.columns:
    if s in('TX','CA','FL','NV'): continue
    a.plot(idx7.index,idx7[s]/base[s]*100,color='#BBBBBB',lw=.8)
a.plot(idx7.index,idx7['TX']/base['TX']*100,color=RD,lw=1.8,label='TX (fulfilment expansion 27 Apr)')
a.axvline(pd.Timestamp('2026-04-27'),color=RD,ls=':'); a.axvspan(pd.Timestamp('2026-03-23'),pd.Timestamp('2026-03-29'),color='#DDDDDD',alpha=.5)
a.text(pd.Timestamp('2026-03-30'),118,'spring sale',fontsize=7.5,color=GR)
a.set_ylim(80,120); a.set_title('B. Revenue index (7-day avg; 30 Mar-24 Apr = 100)',loc='left',fontsize=9.5,fontweight='bold'); a.legend(frameon=False,fontsize=8,loc='upper left')
a=ax[1,0]
co=d[d.state=='CO'].set_index('date'); ohd=d[d.state=='OH'].set_index('date')
r=lambda x:(x.platform_attributed_revenue_usd/x.paid_social_spend_usd).rolling(7).mean()
a.plot(co.index,r(co),color=BL,lw=1.6,label='CO platform-attributed ROAS'); a.plot(ohd.index,r(ohd),color=GR,lw=1.2,label='OH (typical state)')
a.plot(co.index,(co.net_revenue_clean/co.paid_social_spend_usd).rolling(7).mean()/2.4,color=GN,lw=1.2,ls='--',label='CO finance revenue / spend (scaled)')
a.axvline(pd.Timestamp('2026-05-04'),color=RD,ls=':'); a.text(pd.Timestamp('2026-05-05'),8.4,'attribution window\n7d -> 28d click',fontsize=7.5,color=RD)
a.set_ylim(3,10); a.set_title('C. Colorado: platform ROAS jumps, finance does not',loc='left',fontsize=9.5,fontweight='bold'); a.legend(frameon=False,fontsize=7.5,loc='upper left')
a=ax[1,1]
sh=(m.audited_last_28d_revenue_usd/m.audited_last_28d_revenue_usd.sum()*100).sort_values()
cols=[GR if s=='CA' else (OR if s in('FL','TX') else BL) for s in sh.index]
a.barh(sh.index,sh.values,color=cols); a.set_xlabel('% of audited 28-day revenue (total $46.2M)')
a.set_title('D. Revenue share by state',loc='left',fontsize=9.5,fontweight='bold')
from matplotlib.patches import Patch
a.legend(handles=[Patch(color=GR,label='CA: cannot be treated'),Patch(color=OR,label='FL/TX: June / structural confounds'),Patch(color=BL,label='Eligible')],frameon=False,fontsize=7.5,loc='lower right')
for a_ in (ax[0,0],ax[0,1],ax[1,0]):
    a_.xaxis.set_major_locator(mdates.MonthLocator()); a_.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax[0,1].legend(frameon=False,fontsize=8,loc='upper left')
plt.tight_layout(); plt.savefig(FIG/'fig1.png',dpi=170); plt.close()
# ---- Fig 2: feasible designs scatter
A=pd.read_pickle(INTERIM/'scored_gx.pkl')
fig,ax=plt.subplots(figsize=(7.2,3.9))
ax.scatter(A.cost/1000,A.mde*100,s=14,color='#BBBBBB',label=f'{len(A)} feasible designs (share 12-25%, cost <= $300k)')
ax.axhline(2.5,color=RD,ls='--',lw=1); ax.text(207,2.6,'2.5% planning effect',color=RD,fontsize=8)
OFF={'D1 OH+PA+MO':(-30,-16),'D2 WI+TN+KY':(-20,-16),'D6 OH+PA+TN':(6,6),'D4 GA+NC+SC':(-30,8)}
pts={'D1 OH+PA+MO':('OH+PA+MO',BL),'D2 WI+TN+KY':('WI+TN+KY',GN),'D6 OH+PA+TN':('OH+PA+TN',OR),'D4 GA+NC+SC':('GA+NC+SC',GR)}
for lab,(t,c) in pts.items():
    r=A[A.treat==t].iloc[0]; ax.scatter(r.cost/1000,r.mde*100,s=70,color=c,zorder=3,edgecolor='k'); ax.annotate(t,(r.cost/1000,r.mde*100),xytext=OFF[lab],textcoords='offset points',fontsize=8,color=c,fontweight='bold')
ax.scatter([330.1],[1.04],marker='x',color=RD,s=60); ax.annotate('KY+PA+TN+WI\n(GeoX top pick)\n$330k > cap',(330.1,1.04),xytext=(-30,-38),textcoords='offset points',fontsize=8,color=RD)
ax.axvline(300,color=RD,lw=.8,ls=':'); ax.set_xlim(195,345); ax.set_ylim(0,4.9)
ax.set_xlabel('Additional spend (USD thousands)'); ax.set_ylabel('MDE, % of treated revenue\n(in-time placebo, TBR, alpha 0.10, power 0.80)')
ax.legend(frameon=False,fontsize=8,loc='upper center',bbox_to_anchor=(0.5,1.13)); plt.tight_layout(); plt.savefig(FIG/'fig2.png',dpi=170); plt.close()
# ---- Fig 3: fit D1 and placebo distribution + power curves
t=['OH','PA','MO']; c=['CA','CO','GA','IN','KY','MI','NC','NV','SC','TN','WI']
ti=[S.cols.index(s) for s in t]; ci=[S.cols.index(s) for s in c]
y=S.X[:,ti].sum(1); x=S.X[:,ci].mean(1); dates=S.rev.index
msk=np.arange(len(y))<len(y)-28; a_,b_=S.fit(x[msk],y[msk]); pred=a_+b_*x
rel,_,_=S.placebo(t,c); se=rel.std(ddof=1)
fig,ax=plt.subplots(1,3,figsize=(11,3.4),gridspec_kw={'width_ratios':[1.5,1,1]})
a=ax[0]; a.plot(dates,y/1000,color=BL,lw=1.2,label='Treated (OH+PA+MO) actual'); a.plot(dates,pred/1000,color=OR,lw=1.2,ls='--',label='TBR prediction from 11 controls')
a.axvspan(dates[-28],dates[-1],color='#EEEEEE'); a.text(dates[-27],y.min()/1000*1.0,'held-out\n28 days',fontsize=7.5,color=GR)
a.xaxis.set_major_locator(mdates.MonthLocator()); a.xaxis.set_major_formatter(mdates.DateFormatter('%b')); a.set_ylabel('Daily revenue (USD k)'); a.set_title('E. Control-based fit (R2 0.95)',loc='left',fontsize=9.5,fontweight='bold'); a.legend(frameon=False,fontsize=7.5,loc='upper left'); 
a=ax[1]; a.hist((rel-rel.mean())*100,bins=14,color=BL,alpha=.85); a.axvline(0,color='k',lw=.8)
a.axvline(1.645*se*100,color=RD,ls='--'); a.axvline(-1.645*se*100,color=RD,ls='--'); a.set_xlabel('28-day placebo lift error (%)'); a.set_title('F. Noise: SE = %.2f%%'%(se*100),loc='left',fontsize=9.5,fontweight='bold')
a=ax[2]; Ls=np.linspace(0,.03,31)
for k,tt,cc,col in (('D1 OH+PA+MO',t,c,BL),('D2 WI+TN+KY',['WI','TN','KY'],['CA','CO','GA','IN','MI','MO','NC','NV','OH','PA','SC'],GN)):
    r_,_,_=S.placebo(tt,cc); s_=r_.std(ddof=1); b=r_.mean()
    a.plot(Ls*100,[np.mean(np.abs(r_-b+L)>1.645*s_)*100 for L in Ls],color=col,label=k)
a.axvline(2.5,color=RD,ls='--',lw=1); a.axhline(80,color=GR,ls=':',lw=1); a.set_xlabel('True lift (%)'); a.set_ylabel('Power (%)'); a.set_title('G. Power curve',loc='left',fontsize=9.5,fontweight='bold'); a.legend(frameon=False,fontsize=7.5,loc='lower right')
plt.tight_layout(); plt.savefig(FIG/'fig3.png',dpi=170); plt.close()
print('ok')
