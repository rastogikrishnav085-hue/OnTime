import os,sys,json,time
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'src'))
import pandas as pd,numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from features import TARGET_COLUMN
from enriched_model import ENRICHED_COLUMNS,NUMERIC_E,CATEGORICAL_E,as_frame
ROOT=os.path.dirname(__file__); df=pd.read_csv('data/processed/ontime_model_ready.csv',parse_dates=['order_purchase_timestamp']).sort_values('order_purchase_timestamp',kind='stable').reset_index(drop=True)
# Fixed hyperparameters and tree count selected using the earlier validation block; no month is used to tune.
params=dict(n_estimators=225,max_depth=5,learning_rate=.03,min_child_weight=10,subsample=.8,colsample_bytree=.8,reg_lambda=10,reg_alpha=.1,objective='binary:logistic',eval_metric='auc',tree_method='hist',random_state=42,n_jobs=1)
rows=[]; months=sorted(df.order_purchase_timestamp.dt.to_period('M').unique())
for month in months:
 if str(month) not in ('2018-06','2018-07','2018-08'): continue
 train=df[df.order_purchase_timestamp.dt.to_period('M')<month]; test=df[df.order_purchase_timestamp.dt.to_period('M')==month]
 if len(test)<100 or test[TARGET_COLUMN].nunique()<2: continue
 Xtr,Xte=as_frame(train),as_frame(test); ytr=train[TARGET_COLUMN].astype(int).to_numpy(); yte=test[TARGET_COLUMN].astype(int).to_numpy()
 cats=list(CATEGORICAL_E); nums=[c for c in ENRICHED_COLUMNS if c not in cats]
 pre=ColumnTransformer([('num',SimpleImputer(strategy='median'),nums),('cat',OneHotEncoder(handle_unknown='ignore'),cats)],sparse_threshold=.3)
 A=pre.fit_transform(Xtr[ENRICHED_COLUMNS],ytr); B=pre.transform(Xte[ENRICHED_COLUMNS])
 m=XGBClassifier(**params); t=time.time(); m.fit(A,ytr,verbose=False); p=m.predict_proba(B)[:,1]
 row={'month':str(month),'n_train':len(train),'n_test':len(test),'late_rate':round(float(yte.mean()),5),'roc_auc':round(float(roc_auc_score(yte,p)),5),'pr_auc':round(float(average_precision_score(yte,p)),5),'seconds':round(time.time()-t,1)}
 rows.append(row); print(row,flush=True)
out='data/processed/rolling_backtest_enriched.csv'; pd.DataFrame(rows).to_csv(out,index=False); print('mean auc',float(pd.DataFrame(rows).roc_auc.mean()),'saved',out)
