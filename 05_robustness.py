import sys; sys.path.insert(0,"src")
from common import *
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
import placebo_tbr as S
from scipy import stats
m=S.m; rev=S.rev
D={'D1 OH+PA+MO':['OH','PA','MO'],'D2 WI+TN+KY':['WI','TN','KY']}
base={'D1 OH+PA+MO':['CA','CO','GA','IN','KY','MI','NC','NV','SC','TN','WI'],'D2 WI+TN+KY':['CA','CO','GA','IN','MI','MO','NC','NV','OH','PA','SC']}
rows=[]
def run(name,t,c,label,alpha_z=None):
    r=S.evaluate(t,c); r.update(design=name,scenario=label,n_ctrl=len(c)); rows.append(r)
for k,t in D.items():
    c=base[k]
    run(k,t,c,'Base control pool')
    for x in ['NV','GA','CO','CA']:
        run(k,t,[s for s in c if s!=x],f'Drop {x} from controls')
    run(k,t,[s for s in c if s not in('NV','GA')],'Drop NV+GA (noisy/imputed)')
    run(k,t,c+['TX'],'Add TX (post-expansion drift)')
    run(k,t,c+['TX','FL'],'Add TX + FL (promo-contaminated)')
    # exclude spring sale week
    old=S.rev, S.X
    keep=~S.rev.index.isin(pd.date_range('2026-03-23','2026-03-29'))
    S.X=S.rev.values[keep]; S.T=len(S.X)
    run(k,t,c,'Exclude spring-sale week (Mar 23-29)')
    S.X=S.rev.values; S.T=len(S.X)
    # last-6-weeks fit only
    S.X=S.rev.values[-70:]; S.T=70
    run(k,t,c,'Post-Mar-23 data only (last 70d)')
    S.X=S.rev.values; S.T=len(S.X)
R=pd.DataFrame(rows); R['mde_alpha05']=R.mde*(1.96+.8416)/(1.645+.8416)
pd.set_option('display.width',250); print(R[['design','scenario','n_ctrl','se_pct','bias_pct','mde','mde_alpha05','power25','r2']].round(4).to_string())
R.to_pickle(INTERIM/'robust.pkl')
# power curve for finalists: share of placebo windows detected at true lift L
for k,t in D.items():
    rel,_,_=S.placebo(t,base[k]); se=rel.std(ddof=1); b=rel.mean()
    print(k,[ (L,round(float(np.mean(np.abs(rel-b+L)>1.645*se)),2)) for L in (0.005,0.01,0.015,0.02,0.025,0.03)])
# economics
d=pd.read_pickle(INTERIM/'clean.pkl')
tot=d[d.date>='2026-05-04'].groupby('state')[['net_revenue_clean','paid_social_spend_usd','platform_attributed_revenue_usd']].sum()
tot['rev_per_spend']=tot.net_revenue_clean/tot.paid_social_spend_usd; tot['plat_roas']=tot.platform_attributed_revenue_usd/tot.paid_social_spend_usd
print(tot.round(2))
# CO platform attribution shift
c=d[d.state=='CO'].copy(); c['roas']=c.platform_attributed_revenue_usd/c.paid_social_spend_usd
print(c.groupby(c.date>='2026-05-04').roas.mean())
oh=d[d.state=='OH'].copy(); print((oh.platform_attributed_revenue_usd/oh.paid_social_spend_usd).mean())
