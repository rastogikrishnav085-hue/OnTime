import os,sys,json
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'src'))
import pandas as pd,numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from features import FEATURE_COLUMNS,TARGET_COLUMN
from modeling import NUMERIC
from ensemble import load_ensemble,build_xgb,shift_odds
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import clone
ROOT=os.path.dirname(__file__)
df=pd.read_csv('data/processed/ontime_model_ready.csv',parse_dates=['order_purchase_timestamp']).sort_values('order_purchase_timestamp',kind='stable').reset_index(drop=True)
i1=int(len(df)*.65); i2=int(len(df)*.8); tr,va,te=df.iloc[:i1],df.iloc[i1:i2],df.iloc[i2:]
ytr=tr[TARGET_COLUMN].astype(int).to_numpy(); yva=va[TARGET_COLUMN].astype(int).to_numpy(); yte=te[TARGET_COLUMN].astype(int).to_numpy()
Xtr,Xva,Xte=[x[FEATURE_COLUMNS].copy() for x in (tr,va,te)]
for X in (Xtr,Xva,Xte):
 for c in NUMERIC:X[c]=pd.to_numeric(X[c],errors='coerce')
spw=float((ytr==0).sum()/max((ytr==1).sum(),1))
print('fit core xgb')
m=build_xgb({'n_estimators':200,'max_depth':8,'learning_rate':.05,'subsample':.9,'colsample_bytree':.9,'min_child_weight':5},spw).fit(Xtr,ytr)
p=m.predict_proba(Xte)[:,1]
print('core xgb test auc',roc_auc_score(yte,p),'pr',average_precision_score(yte,p))
print('load core ensemble')
b,meta=load_ensemble('models/ensemble',FEATURE_COLUMNS)
b.fit(Xtr,ytr); pb=b.predict_proba(Xte)[:,1]
print('core ensemble test auc',roc_auc_score(yte,pb),'pr',average_precision_score(yte,pb))
print('enriched test',json.load(open('dashboard/verified_training_report.json'))['production_artifact']['test_metrics']['roc_auc'])
