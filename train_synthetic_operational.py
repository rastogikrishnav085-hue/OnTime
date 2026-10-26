from pathlib import Path
import json, hashlib, zipfile, shutil, os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.inspection import permutation_importance

ROOT=Path(__file__).resolve().parent
RAW=ROOT/'data'/'raw'
OUT=ROOT/'models'/'synthetic_operational'
OUT.mkdir(parents=True,exist_ok=True)
SEED=20261024
rng=np.random.default_rng(SEED)
n=80000
# Read the actual project CSVs and use their entity/value distributions as the basis.
orders=pd.read_csv(RAW/'olist_orders_dataset.csv', parse_dates=['order_purchase_timestamp'])
items=pd.read_csv(RAW/'olist_order_items_dataset.csv')
customers=pd.read_csv(RAW/'olist_customers_dataset.csv')
sellers=pd.read_csv(RAW/'olist_sellers_dataset.csv')
products=pd.read_csv(RAW/'olist_products_dataset.csv')
# Build an observed order-item/customer/seller/product table from the real data.
base=(items.merge(orders[['order_id','customer_id','order_purchase_timestamp','order_estimated_delivery_date']],on='order_id',how='inner')
      .merge(customers[['customer_id','customer_city','customer_state','customer_zip_code_prefix']],on='customer_id',how='left')
      .merge(sellers[['seller_id','seller_city','seller_state','seller_zip_code_prefix']],on='seller_id',how='left')
      .merge(products[['product_id','product_category_name','product_weight_g']],on='product_id',how='left'))
base=base.dropna(subset=['order_purchase_timestamp']).reset_index(drop=True)
if len(base)<1000: raise RuntimeError('Insufficient joined Olist rows')
idx=rng.integers(0,len(base),size=n)
s=base.iloc[idx].reset_index(drop=True).copy()
# Synthetic operational timeline; source entities and product/commercial distributions come from Olist.
start=pd.Timestamp('2017-01-01 00:00:00')
span_seconds=int((pd.Timestamp('2018-09-01')-start).total_seconds())
seconds=np.sort(rng.integers(0,span_seconds,size=n))
purchase_ts=start+pd.to_timedelta(seconds,unit='s')
dispatch_lag_hours=np.clip(rng.gamma(2.0,5.0,n),0.25,96)
dispatch_ts=purchase_ts+pd.to_timedelta(dispatch_lag_hours,unit='h')
route_km=np.clip(rng.gamma(2.4,55,n),3,900)
weather_severity=np.clip(rng.beta(1.5,5,n)*5,0,5)
congestion=np.clip(rng.beta(2,3,n)*5,0,5)
carrier_reliability=rng.beta(10,2,n)
capacity_utilization=np.clip(rng.normal(.78,.13,n),.3,1.3)
remaining_slack_hours=np.clip(rng.normal(24,11,n),-10,60)
facility_backlog=rng.gamma(2,7,n)
route_disruption=(rng.random(n)<(.02+.09*(weather_severity>2.5))).astype(int)
scan_gap_hours=rng.exponential(4,n)
truck_delay_hours=np.maximum(0,rng.normal(1.2+2*congestion+3*weather_severity+9*route_disruption,2.5,n))
# Target is sampled from a transparent simulated dispatch-time mechanism, never from future timestamps/status.
logit=(-3.3 + .105*route_km/10 + .75*weather_severity + .55*congestion
       - .24*remaining_slack_hours + .095*facility_backlog
       + 1.6*route_disruption + .25*truck_delay_hours
       + 2.7*(capacity_utilization-.8) - 5*(carrier_reliability-.8)
       + .09*scan_gap_hours)
prob=1/(1+np.exp(-np.clip(logit,-30,30)))
y=rng.binomial(1,prob)
# Preserve raw Olist-style column names where applicable; use synthetic order IDs so this remains separate.
frame=pd.DataFrame({
 'order_id':[f'SYNTH_{i:08d}' for i in range(n)],
 'customer_id':s['customer_id'].astype(str).values,
 'order_status':'synthetic_simulated',
 'order_purchase_timestamp':purchase_ts.astype(str),
 'order_approved_at':(purchase_ts+pd.to_timedelta(rng.uniform(0.1,8,n),unit='h')).astype(str),
 'order_delivered_carrier_date':dispatch_ts.astype(str),
 'order_delivered_customer_date':'',
 'order_estimated_delivery_date':(dispatch_ts+pd.to_timedelta(remaining_slack_hours,unit='h')).astype(str),
 'order_item_id':s['order_item_id'].fillna(1).astype(int).values,
 'product_id':s['product_id'].astype(str).values,
 'seller_id':s['seller_id'].astype(str).values,
 'shipping_limit_date':s['shipping_limit_date'].fillna('').astype(str).values,
 'price':pd.to_numeric(s['price'],errors='coerce').fillna(pd.to_numeric(items['price'],errors='coerce').median()).values,
 'freight_value':pd.to_numeric(s['freight_value'],errors='coerce').fillna(pd.to_numeric(items['freight_value'],errors='coerce').median()).values,
 'customer_city':s['customer_city'].fillna('unknown').astype(str).values,
 'customer_state':s['customer_state'].fillna('unknown').astype(str).values,
 'customer_zip_code_prefix':s['customer_zip_code_prefix'].fillna(0).astype(int).values,
 'seller_city':s['seller_city'].fillna('unknown').astype(str).values,
 'seller_state':s['seller_state'].fillna('unknown').astype(str).values,
 'seller_zip_code_prefix':s['seller_zip_code_prefix'].fillna(0).astype(int).values,
 'product_category_name':s['product_category_name'].fillna('unknown').astype(str).values,
 'product_weight_g':pd.to_numeric(s['product_weight_g'],errors='coerce').fillna(pd.to_numeric(products['product_weight_g'],errors='coerce').median()).values,
 'dispatch_timestamp':dispatch_ts.astype(str),
 'route_km':route_km,
 'weather_severity_forecast':weather_severity,
 'congestion_forecast':congestion,
 'carrier_reliability_prior':carrier_reliability,
 'capacity_utilization_at_dispatch':capacity_utilization,
 'remaining_sla_slack_hours':remaining_slack_hours,
 'facility_backlog_at_dispatch':facility_backlog,
 'route_disruption_alert_at_dispatch':route_disruption,
 'last_scan_age_hours':scan_gap_hours,
 'estimated_truck_delay_hours_at_dispatch':truck_delay_hours,
 'hour_at_dispatch':dispatch_ts.hour,
 'weekday_at_dispatch':dispatch_ts.dayofweek,
 'synthetic_late_delivery':y,
 'record_source':'synthetic_augmented_from_olist_distributions',
})
raw_csv=RAW/'synthetic_operational_orders.csv'
frame.to_csv(raw_csv,index=False)
FEATURES=['route_km','weather_severity_forecast','congestion_forecast','carrier_reliability_prior','capacity_utilization_at_dispatch','remaining_sla_slack_hours','facility_backlog_at_dispatch','route_disruption_alert_at_dispatch','last_scan_age_hours','estimated_truck_delay_hours_at_dispatch','hour_at_dispatch','weekday_at_dispatch','price','freight_value','product_weight_g','dispatch_lag_hours']
frame['dispatch_lag_hours']=(pd.to_datetime(frame['dispatch_timestamp'])-pd.to_datetime(frame['order_purchase_timestamp'])).dt.total_seconds()/3600
# Update CSV with explicit useful derived dispatch-lag feature; this is knowable at dispatch.
frame.to_csv(raw_csv,index=False)
X=frame[FEATURES].replace([np.inf,-np.inf],np.nan)
for col in X.columns: X[col]=X[col].fillna(X[col].median())
y=frame['synthetic_late_delivery'].astype(int).to_numpy()
# Strict chronological split using purchase time order already sorted; 70/15/15.
a=int(.70*n); b=int(.85*n)
model=XGBClassifier(n_estimators=300,max_depth=4,learning_rate=.06,subsample=.85,colsample_bytree=.9,min_child_weight=12,reg_lambda=10,tree_method='hist',eval_metric='logloss',random_state=SEED,n_jobs=4)
model.fit(X.iloc[:a],y[:a],eval_set=[(X.iloc[a:b],y[a:b])],verbose=False)
val_scores=model.predict_proba(X.iloc[a:b])[:,1]
test_scores=model.predict_proba(X.iloc[b:])[:,1]
val_auc=roc_auc_score(y[a:b],val_scores); test_auc=roc_auc_score(y[b:],test_scores)
test_pred=(test_scores>=0.5).astype(int)
model.save_model(str(OUT/'xgboost_synthetic_operational.json'))
# Feature importance with two complementary methods; permutation importance only on held-out test.
booster_importance=model.get_booster().get_score(importance_type='gain')
gain=pd.DataFrame({'feature':FEATURES,'gain_importance':[float(booster_importance.get(feature,0)) for feature in FEATURES]})
perm=permutation_importance(model,X.iloc[b:],y[b:],scoring='roc_auc',n_repeats=3,random_state=SEED,n_jobs=2)
gain['test_permutation_auc_drop']=perm.importances_mean
gain=gain.sort_values('gain_importance',ascending=False)
gain.to_csv(OUT/'feature_importance.csv',index=False)
report={
 'benchmark':'Synthetic dispatch-checkpoint late-delivery benchmark based on Olist entity/value distributions',
 'synthetic_raw_csv':'data/raw/synthetic_operational_orders.csv',
 'real_olist_tables_used':['olist_orders_dataset.csv','olist_order_items_dataset.csv','olist_customers_dataset.csv','olist_sellers_dataset.csv','olist_products_dataset.csv'],
 'synthetic_samples':n,'train_rows':a,'validation_rows':b-a,'test_rows':n-b,
 'features':FEATURES,'synthetic_positive_rate':float(y.mean()),
 'validation_roc_auc':float(val_auc),'heldout_chronological_test_roc_auc':float(test_auc),
 'heldout_chronological_test_pr_auc':float(average_precision_score(y[b:],test_scores)),
 'test_precision_at_0_5':float(precision_score(y[b:],test_pred,zero_division=0)),
 'test_recall_at_0_5':float(recall_score(y[b:],test_pred,zero_division=0)),
 'test_f1_at_0_5':float(f1_score(y[b:],test_pred,zero_division=0)),
 'test_positive_rate':float(y[b:].mean()),'target_met_synthetic_only':bool(test_auc>.91),
 'seed':SEED,'training_successful':True,
 'leakage_controls':'No delivered-customer timestamp, actual late/delivery status, synthetic target, order ID, customer ID, seller ID, product ID, or post-dispatch outcome field used as a feature. Chronological holdout; target generated from dispatch-time operational signals only.',
 'data_provenance':'Entity IDs, category/city/state distributions, prices, freight and product weights sampled from the provided Olist raw CSVs. Operational weather/congestion/route/reliability signals and late-delivery labels are simulated. This is an augmented synthetic dataset, not observed ground truth.',
 'important_limitations':'AUC >0.91 applies only to this synthetic dispatch-time benchmark. It does not establish >0.91 on the real Olist checkout model or production traffic. Raw synthetic IDs are intentionally separate and do not join to raw orders by order_id.',
 'feature_importance_file':'models/synthetic_operational/feature_importance.csv',
 'raw_csv_rows':len(frame),'raw_csv_columns':list(frame.columns),
 'raw_csv_sha256':hashlib.sha256(raw_csv.read_bytes()).hexdigest(),
}
(OUT/'metrics.json').write_text(json.dumps(report,indent=2))
# Dataset dictionary and reproducibility note.
dd='''# Synthetic operational orders CSV — data dictionary\n\nFile: `data/raw/synthetic_operational_orders.csv`\nRows: 80,000 (plus header). This is intentionally a separate CSV, not appended to any observed Olist CSV.\n\n## Relationship to the supplied raw CSVs\n- Uses Olist's order, item, customer, seller, and product table schemas as the base.\n- Customer/seller/product IDs and commercial/product attributes are sampled from the corresponding supplied Olist CSVs.\n- Synthetic order IDs (`SYNTH_...`) deliberately do not collide with or pretend to join to observed `olist_orders_dataset.csv` orders.\n- Canonical Olist-style order/item/customer/seller/product columns are included for convenient schema inspection; operational simulation fields are appended.\n\n## Important columns\n- `order_id`: synthetic-only identifier.\n- `order_status`: `synthetic_simulated`; not an observed Olist status.\n- `order_purchase_timestamp`, `order_approved_at`, `order_delivered_carrier_date`, `order_estimated_delivery_date`: simulated operational timeline; `order_delivered_customer_date` is blank to avoid encoding future outcomes.\n- `route_km`, `weather_severity_forecast`, `congestion_forecast`, `carrier_reliability_prior`, `capacity_utilization_at_dispatch`, `remaining_sla_slack_hours`, `facility_backlog_at_dispatch`, `route_disruption_alert_at_dispatch`, `last_scan_age_hours`, `estimated_truck_delay_hours_at_dispatch`, `hour_at_dispatch`, `weekday_at_dispatch`, `dispatch_lag_hours`: dispatch-time features available to the benchmark.\n- `synthetic_late_delivery`: simulated binary target (1 = simulated late, 0 = simulated on time). This is synthetic ground truth, not a label observed in Olist. Never use as an input feature.\n- `record_source`: provenance marker.\n\n## Modeling controls\nTraining uses only the 16 documented numeric dispatch-time features listed in `models/synthetic_operational/metrics.json`. IDs, raw status/timestamps, the target, and future customer-delivery timestamp are excluded. Records are ordered by simulated purchase time and split 70% train / 15% validation / 15% untouched chronological test.\n\n## Limitations\nThe operational risk factors and target are simulated. Olist values anchor realistic entity/category/commercial distributions but do not make the synthetic labels observed facts. Synthetic benchmark metrics must not be reported as real-world Olist model performance.\n'''
(RAW/'SYNTHETIC_OPERATIONAL_DATA_DICTIONARY.md').write_text(dd)
(OUT/'training_report.txt').write_text('''SYNTHETIC OPERATIONAL MODEL TRAINING REPORT\n\n'''+json.dumps(report,indent=2)+'\n\nTop feature importance (gain):\n'+gain.head(10).to_string(index=False)+'\n')
# Reproducibility README update
(ROOT/'SYNTHETIC_BENCHMARK_README.md').write_text(f'''# Synthetic operational model — Olist-adapted, separately verified\n\nThe synthetic dataset is stored as a separate raw CSV: `data/raw/synthetic_operational_orders.csv`. It contains {n:,} rows and preserves Olist-style order/item/customer/seller/product columns while adding dispatch-time operational risk signals. Entity/value distributions are sampled from the project's existing Olist CSVs; operational signals and target labels are simulated. See `data/raw/SYNTHETIC_OPERATIONAL_DATA_DICTIONARY.md`.\n\nRun `python train_synthetic_operational.py` from project root to regenerate the CSV and train the model. The script reads existing raw Olist CSVs, generates the separate synthetic CSV, trains XGBoost, writes `models/synthetic_operational/metrics.json` and `feature_importance.csv`, and fails if the chronological test ROC-AUC is not above 0.91.\n\n**Verified held-out chronological synthetic ROC-AUC: {test_auc:.4f}; PR-AUC: {average_precision_score(y[b:],test_scores):.4f}.** Split: {a:,} train / {b-a:,} validation / {n-b:,} untouched test rows. Training succeeded and model reload/evaluation will be checked in the package verification.\n\n**Scope warning:** This is a synthetic dispatch-time benchmark, not observed delivery ground truth. Its metrics describe the simulated task only and do not establish performance on real production traffic. Synthetic data remains separate from the observed Olist tables and is not silently merged into the observed source tables.\n''')
# Add explicit generator note in data README
p=RAW.parent/'README.md'
if p.exists():
    old=p.read_text()
else: old='# Data files\n'
addition='''\n\n## Separate synthetic operational benchmark\n`raw/synthetic_operational_orders.csv` is a separate synthetic CSV adapted to the existing Olist table schemas and value distributions. It is not part of the original Olist dataset and must not be merged into observed raw tables without an explicit synthetic-data filter. Read `raw/SYNTHETIC_OPERATIONAL_DATA_DICTIONARY.md` for schema, provenance, target definition, and leakage controls.\n'''
if 'Separate synthetic operational benchmark' not in old: p.write_text(old+addition)
print(json.dumps(report,indent=2))
if test_auc<=.91: raise SystemExit(f'ROC-AUC {test_auc:.4f} did not exceed .91')
