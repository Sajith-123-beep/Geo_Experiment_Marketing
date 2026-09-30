import sys; sys.path.insert(0,"src")
from common import *
import pandas as pd, numpy as np, datetime, warnings, jax.numpy as jnp, pickle
warnings.filterwarnings('ignore')
import meridian_geox as geox
from meridian_geox import api, util
from meridian_geox.methodology import tbr
from scipy import stats
d=pd.read_pickle(INTERIM/'clean.pkl'); m=pd.read_pickle(INTERIM/'markets.pkl').set_index('state')
g=d.rename(columns={'state':'location','net_revenue_clean':'conversions','paid_social_spend_usd':'spend'})[['date','location','conversions','spend']].copy()
g['date']=pd.to_datetime(g.date)
D={'D1 OH+PA+MO':(['OH','PA','MO']),'D2 WI+TN+KY':(['WI','TN','KY']),'D3 MI+WI+IN':(['MI','WI','IN']),'D4 GA+NC+SC':(['GA','NC','SC']),'D5 KY+PA+TN+WI (GeoX top)':(['KY','PA','TN','WI']),'D6 OH+PA+TN':(['OH','PA','TN'])}
overlap={'OH':'PA','PA':'OH','MI':'IN','IN':'MI','TN':'KY','KY':'TN'}
def ctrl_for(t):
    part={overlap[s] for s in t if s in overlap}-set(t)
    return [s for s in m.index if s not in set(t)|{'FL','TX'}|part]
def mkdesign(t,c,data,dur=28):
    cfg=geox.DesignConfig(experiment_duration=datetime.timedelta(days=dur),experiment_types=geox.ExperimentType.HEAVY_UP,methodology=geox.Methodology.TBR,geo_assignment_rule=geox.GeoAssignmentRule.RANDOM,cell_count=1,alpha=0.1,power=0.8)
    con=geox.Constraints(excluded_geos={'FL','TX'},budget_constraint={'cell_1':geox.Budget(budget_pct=0.4)},max_conversions_percent=0.25)
    pc=api.PerCellDesign(treatment_geos=set(t),minimum_detectable_effect=np.nan,design_implied_cpic=np.nan,p_value=np.nan,budget=float(m.loc[t,'planned_bau_daily_spend_usd'].sum()*.4*28))
    return api.Design(designs={'cell_1':pc},control_geos=set(c),excluded_geos={'FL','TX'},design_config=cfg,constraints=con,data=data)
# 1) GeoX R2 + AA p on mask, family-based MDE (std across all feasible-design masks, GeoX method)
R=pd.read_pickle(INTERIM/'scored.pkl'); A=R[R.pol=='A']
geos=sorted(set(m.index)-{'FL','TX'})
piv=g[~g.location.isin(['FL','TX'])].pivot(index='date',columns='location',values='conversions')[geos]
n=len(piv); t2=n-28; t1=t2-28
masks=[]
for _,r in A.iterrows():
    masks.append([1.0 if s in r.treat.split('+') else 0.0 for s in geos])
masks=jnp.array(masks)
z=stats.norm.ppf(.95)+stats.norm.ppf(.8)
res=tbr.get_mde(jnp.array(piv.iloc[:t2].values),jnp.array(piv.iloc[t2:].values),masks,z,api.TestType.TWO_SIDED,cell_ids=jnp.array([1.0]))
r2=tbr.get_r2(jnp.array(piv.iloc[:t1].values),jnp.array(piv.iloc[t1:t2].values),masks,jnp.array([1.0]))
A=A.copy(); A['gx_mde']=np.array(res.mde_pct[:,0]); A['gx_aa_p']=np.array(res.p_value[:,0]); A['gx_r2']=np.array(r2[:,0])
A.to_pickle(INTERIM/'scored_gx.pkl')
pd.set_option('display.width',250)
rows=[]
for k,t in D.items():
    key='+'.join(t) 
    tt=[s for s in ['OH','PA','MO'] if False]
    c=ctrl_for(t)
    # GeoX stats need a matching row: find by set
    hit=A[A.treat.apply(lambda x:set(x.split('+'))==set(t))]
    row=dict(design=k,cost=float(m.loc[t,'planned_bau_daily_spend_usd'].sum()*.4*28),share=m.loc[t,'audited_last_28d_revenue_usd'].sum()/m.audited_last_28d_revenue_usd.sum(),n_ctrl=len(c),ctrl=','.join(c))
    if len(hit):
        h=hit.iloc[0]; row.update(gx_mde=h.gx_mde,gx_r2=h.gx_r2,gx_aa_p=h.gx_aa_p,placebo_mde=h.mde,power25=h.power25)
    rows.append(row)
S=pd.DataFrame(rows); print(S.drop(columns='ctrl').round(4).to_string()); S.to_pickle(INTERIM/'shortlist.pkl')
# 2) GeoX analyze pseudo-tests: fake test window 2026-05-04..05-31 (last 28d), pre = before; A/A and +2.5% injected
out=[]
for k,t in D.items():
    c=ctrl_for(t)
    for lift in (0.0,0.025,0.015):
        gg=g.copy(); msk=gg.location.isin(t)&(gg.date>='2026-05-04')
        gg.loc[msk,'conversions']*=1+lift
        gg.loc[msk,'spend']*=1.4
        gg=gg[gg.location.isin(t+c)]
        des=mkdesign(t,c,gg)
        cfg=geox.AnalysisConfig(design=des,analysis_start_date=pd.Timestamp('2026-05-04'),analysis_end_date=pd.Timestamp('2026-05-31'),n_placebo_candidates=20000,n_top_placebos=200,min_placebo_r2=0.5)
        try:
            ar=geox.analyze(gg,cfg,geox.QualityCheckConfig(exclude_outlier_dates=False))
            r=ar.results['cell_1']; pl=r.percent_lift
            out.append(dict(design=k,injected=lift,est=pl.point_estimate,lo=pl.lower_bound,hi=pl.upper_bound,p=pl.p_value))
        except Exception as e:
            out.append(dict(design=k,injected=lift,err=repr(e)[:200]))
O=pd.DataFrame(out); print(O.round(4).to_string()); O.to_pickle(INTERIM/'analyze_pseudo.pkl')
