"""Rebuild only the app queue from the newest chronological holdout, preserving synthetic labels."""
import json, os, sys
import numpy as np, pandas as pd
ROOT=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.join(ROOT,'src'))
from features import FEATURE_COLUMNS,TARGET_COLUMN
from enriched_features import EXTRA_COLUMNS
from simulation import simulate_logistics,region
from clustering import assign
from categories import tier_of
PROC=os.path.join(ROOT,'data','processed'); OUT=os.path.join(ROOT,'backend','app_data')
ready=pd.read_csv(os.path.join(PROC,'ontime_model_ready.csv'),parse_dates=['order_purchase_timestamp'])
ready=ready.sort_values('order_purchase_timestamp',kind='stable').reset_index(drop=True)
cut=int(len(ready)*.80); test=ready.iloc[cut:].copy().reset_index(drop=True)
joined=pd.read_csv(os.path.join(ROOT,'data','interim','olist_joined.csv'),parse_dates=['order_purchase_timestamp','order_estimated_delivery_date','order_delivered_customer_date'])
join_cols=['order_id','primary_seller_id','customer_city','seller_city','order_estimated_delivery_date','order_delivered_customer_date','delay_days']
extra=joined[join_cols].rename(columns={'primary_seller_id':'seller_id','order_estimated_delivery_date':'promised_by','order_delivered_customer_date':'delivered_at'})
q=test.merge(extra,on='order_id',how='left',validate='1:1').rename(columns={'order_purchase_timestamp':'purchased_at'})
q['actual_days']=(pd.to_datetime(q['delivered_at'])-q['purchased_at']).dt.total_seconds()/86400
q['customer_region']=q['customer_state'].map(region); q['seller_region']=q['seller_state'].map(region)
# Synthetic logistics remain a clearly marked demo overlay, never a feature in the real-data model.
d=ready.merge(joined[['order_id','primary_seller_id','customer_city','seller_city','delay_days','order_delivered_customer_date']].rename(columns={'primary_seller_id':'seller_id'}),on='order_id',how='left',validate='1:1')
d['actual_days']=(pd.to_datetime(d['order_delivered_customer_date'])-d['order_purchase_timestamp']).dt.total_seconds()/86400
d['customer_region']=d['customer_state'].map(region); d['seller_region']=d['seller_state'].map(region)
sim=simulate_logistics(d)
if not set(sim.columns).issubset(d.columns): d=pd.concat([d,sim],axis=1)
simcols=[c for c in d.columns if c.startswith('sim_')]
q=q.merge(d[['order_id']+simcols],on='order_id',how='left',validate='1:1')
pat=json.load(open(os.path.join(OUT,'patterns.json'),encoding='utf-8'))
cparams=pat['params']; id_to_rank={int(k):int(v) for k,v in pat['id_to_rank'].items()}
q['pattern']=[id_to_rank[int(c)] for c in assign(q,cparams)]
q['focus_tier']=q['product_category'].map(tier_of)
q['priority']=0.0
# Explicit placeholders until the new chronological model fills them.
q['probability']=np.nan; q['risk_level']=''; q['flagged']=False; q['suggested_action']=''; q['confidence']=''; q['reasons_json']='[]'; q['main_driver']=''
q.to_csv(os.path.join(OUT,'queue.csv'),index=False,float_format='%.7g')
print(f'Rebuilt queue: {len(q):,} newest chronological orders; {q.purchased_at.min()} to {q.purchased_at.max()}; late rate {q[TARGET_COLUMN].mean():.3%}; synthetic overlays labelled by sim_ columns.')
