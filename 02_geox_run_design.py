import sys; sys.path.insert(0,"src")
from common import *
import pandas as pd, numpy as np, datetime, warnings, pickle
warnings.filterwarnings('ignore')
import meridian_geox as geox
d=pd.read_pickle(INTERIM/'clean.pkl'); m=pd.read_pickle(INTERIM/'markets.pkl')
g=d.rename(columns={'state':'location','net_revenue_clean':'conversions','paid_social_spend_usd':'spend'})[['date','location','conversions','spend']]
g['date']=pd.to_datetime(g.date)
out={}
def go(tag,excl,inc,rule,min_r2=0.5,maxconv=0.25,budget=300000):
    cfg=geox.DesignConfig(experiment_duration=datetime.timedelta(days=28),experiment_types=geox.ExperimentType.HEAVY_UP,
        methodology=geox.Methodology.TBR,geo_assignment_rule=rule,cell_count=1,alpha=0.1,power=0.8,
        design_output_count=100,n_candidates=200000,n_ranked_candidates=2000,min_r2=min_r2,seed=42)
    con=geox.Constraints(excluded_geos=set(excl),included_control_geos=set(inc),budget_constraint=geox.Budget(budget_pct=0.4),max_conversions_percent=maxconv)
    try:
        ds=geox.run_design(g,cfg,con,geox.QualityCheckConfig(exclude_geos_no_response=False,exclude_outlier_dates=True))
    except Exception as e:
        print(tag,'ERR',repr(e)[:300]);return None
    dm=ds.design_metrics.copy()
    dm['treat']=[ '+'.join(sorted(ds.designs[i].designs['cell_1'].treatment_geos)) for i in dm.design_id]
    dm['control']=[ ','.join(sorted(ds.designs[i].control_geos)) for i in dm.design_id]
    dm['tag']=tag
    out[tag]=(ds,dm); print(tag,len(dm)); return dm
for rule in (geox.GeoAssignmentRule.STRATIFIED_SAMPLING,geox.GeoAssignmentRule.RANDOM):
    go('excl FL,TX|'+rule.name,{'FL','TX'},{'CA'},rule)
pickle.dump({k:v[1] for k,v in out.items()},open(INTERIM/'geox_dm.pkl','wb'))
pd.set_option('display.width',250)
for k,(ds,dm) in out.items():
    print(k);print(dm[['treat','r2','mde','budget','treatment_conversions_pct','p_value (AA)']].head(15).to_string())
    ds0=ds
pickle.dump({k:{i:v[0].designs[i] for i in v[1].design_id} for k,v in out.items()},open(INTERIM/'geox_designs.pkl','wb')) 
