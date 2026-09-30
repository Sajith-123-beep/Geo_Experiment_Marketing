import sys; sys.path.insert(0,"src")
from common import *
import pandas as pd, numpy as np, itertools, warnings
warnings.filterwarnings('ignore')
from scipy import stats
d=pd.read_pickle(INTERIM/'clean.pkl'); m=pd.read_pickle(INTERIM/'markets.pkl').set_index('state')
rev=d.pivot(index='date',columns='state',values='net_revenue_clean').sort_index()
X=rev.values; cols=list(rev.columns); T=len(X); W=28
tot=m.audited_last_28d_revenue_usd.sum(); bau=m.planned_bau_daily_spend_usd
zA=stats.norm.ppf(.95); zP=stats.norm.ppf(.8)
def fit(x,y):   # GeoX TBR: non-negative slope OLS
    xm,ym=x.mean(),y.mean(); sxx=((x-xm)**2).sum(); b=max(0,((x-xm)*(y-ym)).sum()/sxx) if sxx>1e-10 else 0
    return ym-b*xm,b
def placebo(t,c):
    ti=[cols.index(s) for s in t]; ci=[cols.index(s) for s in c]
    y=X[:,ti].sum(1); x=X[:,ci].mean(1)
    eff=[];base=[]
    for s in range(0,T-W+1,3):
        msk=np.ones(T,bool); msk[s:s+W]=False
        a,b=fit(x[msk],y[msk]); pred=a+b*x[s:s+W]
        eff.append((y[s:s+W]-pred).mean()); base.append(y[s:s+W].mean())
    eff=np.array(eff); base=np.array(base)
    rel=eff/base
    # in-sample fit quality
    a,b=fit(x,y); r2=1-((y-(a+b*x))**2).sum()/((y-y.mean())**2).sum()
    return rel,r2,y[-W:].mean()
def evaluate(t,c):
    rel,r2,base=placebo(t,c)
    se=rel.std(ddof=1); bias=rel.mean()
    mde=(zA+zP)*se
    pw=np.mean(np.abs(rel-bias+0.025)>zA*se)   # power at +2.5% lift after de-biasing
    return dict(se_pct=se,bias_pct=bias,mde=mde,power25=pw,r2=r2,base_daily=base)
if __name__=='__main__':
    units={s:[s] for s in m.index if s not in('CA','FL','TX','NC','SC')}; units['NC_SC']=['NC','SC']
    overlap={'OH':'PA','PA':'OH','MI':'IN','IN':'MI','TN':'KY','KY':'TN'}
    rows=[]
    for k in (1,2,3,4):
        for combo in itertools.combinations(units,k):
            t=[s for c in combo for s in units[c]]
            if not 2<=len(t)<=4: continue
            share=m.loc[t,'audited_last_28d_revenue_usd'].sum()/tot; cost=bau[t].sum()*.4*28
            if not .12<=share<=.25 or cost>300000: continue
            part={overlap[s] for s in t if s in overlap}-set(t)
            for pol,extra in (('A',{'FL','TX'}),('B',{'FL'})):
                ex=set(t)|extra|part; ctrl=[s for s in m.index if s not in ex]
                if len(ctrl)<4: continue
                r=evaluate(t,ctrl); r.update(treat='+'.join(t),n=len(t),share=share,cost=cost,pol=pol,n_ctrl=len(ctrl),ctrl=','.join(ctrl),
                    req_iroas=.025*m.loc[t,'audited_last_28d_revenue_usd'].sum()/cost,partial_overlap=','.join(part)); rows.append(r)
    R=pd.DataFrame(rows); R.to_pickle(INTERIM/'scored.pkl')
    pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
    print(len(R)); 
    print(R[R.pol=='A'].sort_values('mde').head(15).drop(columns='ctrl').round(4).to_string())
    print(R[R.pol=='B'].sort_values('mde').head(8).drop(columns='ctrl').round(4).to_string())
