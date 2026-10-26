import os,sys,json
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'src'))
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from features import FEATURE_COLUMNS,TARGET_COLUMN
from modeling import NUMERIC
from ensemble import load_ensemble
from enriched_features import EXTRA_COLUMNS
from enriched_model import ENRICHED_COLUMNS,as_frame
ROOT=os.path.dirname(__file__);df=pd.read_csv('data/processed/ontime_model_ready.csv',parse_dates=['order_purchase_timestamp']).sort_values('order_purchase_timestamp',kind='stable').reset_index(drop=True)
i1=int(len(df)*.65);i2=int(len(df)*.8);tr,va,te=df.iloc[:i1],df.iloc[i1:i2],df.iloc[i2:]
ytr=tr[TARGET_COLUMN].astype(int).to_numpy(); y=te[TARGET_COLUMN].astype(int).to_numpy()
Xtr=tr[FEATURE_COLUMNS].copy(); Xte=te[FEATURE_COLUMNS].copy()
for X in (Xtr,Xte):
 for c in NUMERIC:X[c]=pd.to_numeric(X[c],errors='coerce').astype(float)
core,meta=load_ensemble('models/ensemble',FEATURE_COLUMNS); core.fit(Xtr,ytr); pc=core.predict_proba(Xte)[:,1]
enr,emeta=load_ensemble('models/enriched',ENRICHED_COLUMNS); pe=enr.predict_proba(as_frame(te))[:,1]
# paired bootstrap with same sampled rows, so intervals compare model differences rather than independent AUCs.
rng=np.random.default_rng(42); dif=[]
for _ in range(500):
 ix=rng.integers(0,len(y),len(y))
 if y[ix].min()==y[ix].max(): continue
 dif.append(roc_auc_score(y[ix],pe[ix])-roc_auc_score(y[ix],pc[ix]))
report={'test_n':len(y),'late_rate':float(y.mean()),'enriched_auc':float(roc_auc_score(y,pe)),'core_ensemble_auc':float(roc_auc_score(y,pc)),'enriched_pr_auc':float(average_precision_score(y,pe)),'core_ensemble_pr_auc':float(average_precision_score(y,pc)),'paired_auc_difference':float(np.mean(dif)),'paired_auc_difference_ci95':[float(np.percentile(dif,2.5)),float(np.percentile(dif,97.5))],'bootstrap_resamples':len(dif)}
json.dump(report,open('data/processed/paired_test_comparison.json','w'),indent=2);print(json.dumps(report,indent=2))
