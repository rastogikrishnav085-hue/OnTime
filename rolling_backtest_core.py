import os,sys
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'src'))
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from features import FEATURE_COLUMNS,TARGET_COLUMN
from modeling import NUMERIC
from ensemble import build_xgb
ROOT=os.path.dirname(__file__); df=pd.read_csv('data/processed/ontime_model_ready.csv',parse_dates=['order_purchase_timestamp']).sort_values('order_purchase_timestamp',kind='stable').reset_index(drop=True)
params={'n_estimators':200,'max_depth':8,'learning_rate':.05,'subsample':.9,'colsample_bytree':.9,'min_child_weight':5}
rows=[]
for month in ('2018-06','2018-07','2018-08'):
 mask=df.order_purchase_timestamp.dt.to_period('M')<month; te=df[df.order_purchase_timestamp.dt.to_period('M')==month]; tr=df[mask]
 Xtr,Xte=tr[FEATURE_COLUMNS].copy(),te[FEATURE_COLUMNS].copy()
 for X in (Xtr,Xte):
  for c in NUMERIC:X[c]=pd.to_numeric(X[c],errors='coerce').astype(float)
 ytr=tr[TARGET_COLUMN].astype(int); yte=te[TARGET_COLUMN].astype(int).to_numpy(); spw=float((ytr==0).sum()/max((ytr==1).sum(),1))
 m=build_xgb(params,spw).fit(Xtr,ytr); p=m.predict_proba(Xte)[:,1]
 row={'month':month,'n_train':len(tr),'n_test':len(te),'late_rate':round(float(yte.mean()),5),'roc_auc':round(float(roc_auc_score(yte,p)),5),'pr_auc':round(float(average_precision_score(yte,p)),5)}
 rows.append(row);print(row,flush=True)
pd.DataFrame(rows).to_csv('data/processed/rolling_backtest_core.csv',index=False)
print('mean',pd.DataFrame(rows).roc_auc.mean())
