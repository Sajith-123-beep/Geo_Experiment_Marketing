import sys; sys.path.insert(0,"src")
from common import *
import pandas as pd, numpy as np
raw=pd.read_excel(DATA_XLSX,sheet_name='Daily metrics')
m=pd.read_excel(DATA_XLSX,sheet_name='Markets')
log=[]
d=raw.copy()
d['state']=d.state.str.strip().str.upper()
log.append(('state label case/space fixes',int((raw.state!=d.state).sum())))
n0=len(d); d=d.drop_duplicates(['date','state']).copy()
log.append(('exact duplicate rows removed',n0-len(d)))
# GA missing revenue (8 NaN + 4 zeros flagged incomplete) -> impute
bad=d.net_revenue_usd.isna()|(d.revenue_feed_complete==0)
log.append(('revenue days imputed (GA)',int(bad.sum())))
d['rev_clean']=d.net_revenue_usd.where(~bad)
d['aov']=d.rev_clean/d.orders.where(d.orders>0)
# impute = orders * state median AOV of +/-14d neighbours
d=d.sort_values(['state','date'])
for s,g in d.groupby('state'):
    idx=g.index[g.rev_clean.isna()]
    for i in idx:
        dt=d.at[i,'date']
        w=g[(abs(g.date-dt).dt.days<=14)&g.aov.notna()]
        d.at[i,'rev_clean']=d.at[i,'orders']*w.aov.median() if d.at[i,'orders']>0 else np.nan
# GA Apr13-16 orders=0 -> use spend-scaled interpolation of revenue on DOW-adjusted neighbours
ga=d.state=='GA'
z=d[ga&(d.date.between('2026-04-13','2026-04-16'))].index
loc=d[ga&(d.date.between('2026-03-30','2026-04-30'))&~d.index.isin(z)]
for i in z:
    dow=d.at[i,'date'].dayofweek
    d.at[i,'rev_clean']=loc[loc.date.dt.dayofweek==dow].rev_clean.median()
# NV negative spend -> replace with state median spend; negative orders -> abs
negsp=d.paid_social_spend_usd<=0
log.append(('negative spend rows set to median',int(negsp.sum())))
d.loc[negsp,'paid_social_spend_usd']=d[d.state=='NV'].paid_social_spend_usd[d.paid_social_spend_usd>0].median()
negor=d.orders<0
log.append(('negative order rows (orders=abs)',int(negor.sum())))
d.loc[negor,'orders']=d.loc[negor,'orders'].abs()
d['net_revenue_clean']=d.rev_clean
print(log)
print(d.net_revenue_clean.isna().sum(), len(d))
d.to_pickle(INTERIM/'clean.pkl'); m.to_pickle(INTERIM/'markets.pkl')
# show imputed GA vs neighbours
print(d[ga&d.date.between('2026-04-10','2026-04-19')][['date','orders','net_revenue_clean']])
print(d[ga&d.date.between('2026-02-01','2026-02-06')][['date','orders','net_revenue_clean']])
