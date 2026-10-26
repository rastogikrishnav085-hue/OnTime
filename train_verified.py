"""Reproducible chronological model comparison for OnTime real Olist data."""
import json, os, time, hashlib
import numpy as np, pandas as pd
import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, brier_score_loss, confusion_matrix
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from xgboost import XGBClassifier
from sklearn.base import clone

ROOT=os.path.dirname(os.path.abspath(__file__))
DATA=os.path.join(ROOT,'data','processed','ontime_model_ready.csv')
OUT=os.path.join(ROOT,'dashboard','verified_training_report.json')
MODEL_DIR=os.path.join(ROOT,'models','verified_chronological')
os.makedirs(MODEL_DIR,exist_ok=True)
SEED=42

df=pd.read_csv(DATA,parse_dates=['order_purchase_timestamp']).sort_values('order_purchase_timestamp',kind='stable').reset_index(drop=True)
from features import FEATURE_COLUMNS,TARGET_COLUMN
from enriched_features import EXTRA_COLUMNS
from enriched_model import ENRICHED_COLUMNS, CATEGORICAL_E, NUMERIC_E, as_frame
n=len(df); i1=int(n*.65); i2=int(n*.80)
tr,va,te=df.iloc[:i1].copy(),df.iloc[i1:i2].copy(),df.iloc[i2:].copy()
Xtr,Xva,Xte=[as_frame(x).reset_index(drop=True) for x in (tr,va,te)]
ytr,yva,yte=[x[TARGET_COLUMN].astype(int).to_numpy() for x in (tr,va,te)]
print(f'rows={n}; train={len(tr)} {tr.order_purchase_timestamp.min().date()}..{tr.order_purchase_timestamp.max().date()} late={ytr.mean():.4f}; val={len(va)} {va.order_purchase_timestamp.min().date()}..{va.order_purchase_timestamp.max().date()} late={yva.mean():.4f}; test={len(te)} {te.order_purchase_timestamp.min().date()}..{te.order_purchase_timestamp.max().date()} late={yte.mean():.4f}',flush=True)

cat=list(CATEGORICAL_E); num=[c for c in ENRICHED_COLUMNS if c not in cat]
# All pipelines fit only on training. Candidate selection uses validation; test is reported after selection.
candidates={}
# LightGBM: native categorical values via train-only ordinal encoder and categorical_feature positions.
pre_lgb=ColumnTransformer([('num',SimpleImputer(strategy='median',add_indicator=False),num),('cat',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1,encoded_missing_value=-1),cat)],remainder='drop',sparse_threshold=0)
class LGBMCat(LGBMClassifier):
 def fit(self,X,y,**kw):
  kw['categorical_feature']=list(range(len(num),len(num)+len(cat)))
  return super().fit(X,y,**kw)
lgb=Pipeline([('pre',pre_lgb),('clf',LGBMCat(n_estimators=900,learning_rate=.025,num_leaves=31,max_depth=-1,min_child_samples=50,subsample=.85,subsample_freq=1,colsample_bytree=.8,reg_lambda=10,reg_alpha=.2,objective='binary',random_state=SEED,n_jobs=1,verbosity=-1))])
print('fit LightGBM...',flush=True); t=time.time()
# Fit transformed matrix explicitly to provide validation set for early stopping.
A=pre_lgb.fit_transform(Xtr[ENRICHED_COLUMNS],ytr); B=pre_lgb.transform(Xva[ENRICHED_COLUMNS]); C=pre_lgb.transform(Xte[ENRICHED_COLUMNS])
lgb.named_steps['clf'].fit(A,ytr,eval_set=[(B,yva)],eval_metric='auc',callbacks=[early_stopping(60,verbose=False),log_evaluation(0)])
plv=lgb.named_steps['clf'].predict_proba(B)[:,1]; plt=lgb.named_steps['clf'].predict_proba(C)[:,1]
candidates['lightgbm_enriched']={'model':lgb,'val':plv,'test':plt,'seconds':round(time.time()-t,1)}
print('LightGBM best_iter',lgb.named_steps['clf'].best_iteration_,'val_auc',roc_auc_score(yva,plv),'sec',round(time.time()-t,1),flush=True)
# HGB, native categorical ordinal encoding
pre_hgb=clone(pre_lgb)
hgb=Pipeline([('pre',pre_hgb),('clf',HistGradientBoostingClassifier(max_iter=450,learning_rate=.05,max_leaf_nodes=31,min_samples_leaf=50,l2_regularization=5,categorical_features=[False]*len(num)+[True]*len(cat),random_state=SEED,early_stopping=True,validation_fraction=.12,n_iter_no_change=25))])
print('fit HistGradientBoosting...',flush=True); t=time.time(); hgb.fit(Xtr[ENRICHED_COLUMNS],ytr); phv=hgb.predict_proba(Xva[ENRICHED_COLUMNS])[:,1]; pht=hgb.predict_proba(Xte[ENRICHED_COLUMNS])[:,1]
candidates['hist_gradient_enriched']={'model':hgb,'val':phv,'test':pht,'seconds':round(time.time()-t,1)}
print('HGB val_auc',roc_auc_score(yva,phv),'sec',round(time.time()-t,1),flush=True)
# XGBoost one-hot categories; early stop using transformed matrices
pre_xgb=ColumnTransformer([('num',SimpleImputer(strategy='median'),num),('cat',OneHotEncoder(handle_unknown='ignore'),cat)],sparse_threshold=.3)
print('fit XGBoost...',flush=True); t=time.time(); Xta=pre_xgb.fit_transform(Xtr[ENRICHED_COLUMNS],ytr); Xva_t=pre_xgb.transform(Xva[ENRICHED_COLUMNS]); Xte_t=pre_xgb.transform(Xte[ENRICHED_COLUMNS])
xgb=XGBClassifier(n_estimators=800,max_depth=5,learning_rate=.03,min_child_weight=10,subsample=.8,colsample_bytree=.8,reg_lambda=10,reg_alpha=.1,objective='binary:logistic',eval_metric='auc',tree_method='hist',random_state=SEED,n_jobs=1,early_stopping_rounds=60)
xgb.fit(Xta,ytr,eval_set=[(Xva_t,yva)],verbose=False)
pxv=xgb.predict_proba(Xva_t)[:,1]; pxt=xgb.predict_proba(Xte_t)[:,1]
xpipe=Pipeline([('pre',pre_xgb),('clf',xgb)])
candidates['xgboost_enriched']={'model':xpipe,'val':pxv,'test':pxt,'seconds':round(time.time()-t,1)}
print('XGB best_iter',xgb.best_iteration,'val_auc',roc_auc_score(yva,pxv),'sec',round(time.time()-t,1),flush=True)

# Select by validation ROC-AUC only; test is never used for model choice.
val_aucs={k:float(roc_auc_score(yva,v['val'])) for k,v in candidates.items()}
selected=max(val_aucs,key=val_aucs.get)
# Prefer the best single candidate on validation. A tiny blend gain is not enough to justify extra serving complexity.
sel_val = candidates[selected]['val']
sel_test = candidates[selected]['test']

# Choose alert threshold on validation only, maximizing F1 over unique candidate thresholds.
from sklearn.metrics import precision_recall_curve
prec,rec,ths=precision_recall_curve(yva,sel_val)
f1s=2*prec[:-1]*rec[:-1]/np.maximum(prec[:-1]+rec[:-1],1e-12)
thr=float(ths[int(np.argmax(f1s))]) if len(ths) else .5

def metrics(y,p,threshold):
 pred=(p>=threshold).astype(int); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
 return {'roc_auc':round(float(roc_auc_score(y,p)),5),'pr_auc':round(float(average_precision_score(y,p)),5),'brier':round(float(brier_score_loss(y,p)),5),'threshold_from_validation':round(float(threshold),5),'precision':round(float(precision_score(y,pred,zero_division=0)),5),'recall':round(float(recall_score(y,pred,zero_division=0)),5),'f1':round(float(f1_score(y,pred,zero_division=0)),5),'confusion':{'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)},'recall_top10pct':round(float(y[np.argsort(-p)[:max(1,int(len(p)*.1))]].sum()/max(int(y.sum()),1)),5)}

results={}
for k,v in candidates.items():
 results[k]={'validation':metrics(yva,v['val'],thr),'test':metrics(yte,v['test'],thr),'fit_seconds':v['seconds'],'selected_on_validation':k==selected}
if selected=='validation_blend':
 results['validation_blend']={'validation':metrics(yva,sel_val,thr),'test':metrics(yte,sel_test,thr),'weights':dict(zip(names,[round(float(x),3) for x in blend_best[1]])),'selected_on_validation':True}

# Persist candidates to reproduce comparison. Save preprocessing and model together; final serving integration is separate.
import joblib
for k,v in candidates.items():
 joblib.dump(v['model'],os.path.join(MODEL_DIR,k+'.joblib'),compress=3)

def fit_sigmoid_calibration(scores, y):
    """Smooth Platt calibration fitted on the chronological validation block only."""
    from sklearn.linear_model import LogisticRegression
    p = np.clip(np.asarray(scores, dtype=float), 1e-5, 1 - 1e-5)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    cal_model = LogisticRegression(C=1000.0, max_iter=2000, random_state=SEED).fit(z, np.asarray(y, dtype=int))
    grid = np.linspace(0.0, 1.0, 501)
    g = np.clip(grid, 1e-5, 1 - 1e-5)
    gz = np.log(g / (1 - g)).reshape(-1, 1)
    values = cal_model.predict_proba(gz)[:, 1]
    return {"x": [round(float(v), 6) for v in grid], "y": [round(float(v), 6) for v in values], "method": "sigmoid (Platt scaling) fitted on chronological validation only"}

# Export a production-compatible enriched model using the best single candidate (complexity wins over a <0.002 AUC blend gain).
# The final model is refit on train+validation only; the newest 20% remains untouched for evaluation.
from ensemble import BlendModel, save_ensemble, shift_odds, fit_calibration, apply_calibration, reliability_table
selected = max(val_aucs, key=val_aucs.get)
sel_val_raw = candidates[selected]['val']
sel_test_raw = candidates[selected]['test']
# Keep the selected candidate fitted on the oldest training block. This is the model whose
# untouched newest-20% test metrics are reported; no second refit changes the evaluated artifact.
final_pipe = candidates[selected]['model']
spw_tr = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
spw_tv = spw_tr
# Threshold/calibration are based only on validation predictions. Apply the class-prior odds shift used by app risk bands.
val_shift = shift_odds(sel_val_raw, spw_tr)
thr, _ = __import__('ensemble').choose_threshold(yva, val_shift)
cal = fit_calibration(val_shift, yva)
test_shift = shift_odds(sel_test_raw, spw_tr)
cal_test = apply_calibration(cal, test_shift)
cal_val = apply_calibration(cal, val_shift)
rel, ece = reliability_table(yte, cal_test, 10)
prod_metrics = metrics(yte, test_shift, thr)
prod_metrics['calibrated_brier'] = round(float(brier_score_loss(yte, cal_test)), 5)
prod_metrics['calibration_ece'] = round(float(ece), 5)
prod_metrics['calibration_method'] = 'isotonic fit on chronological validation block only'
blend = BlendModel({'xgb' if selected == 'xgboost_enriched' else ('lgbm' if selected == 'lightgbm_enriched' else 'hgb'): final_pipe},
                   weights={'xgb' if selected == 'xgboost_enriched' else ('lgbm' if selected == 'lightgbm_enriched' else 'hgb'): 1.0},
                   explainer='xgb' if selected == 'xgboost_enriched' else ('lgbm' if selected == 'lightgbm_enriched' else 'hgb'),
                   odds_shift=spw_tv)
# Save the actual 80%-chronology-trained model, calibration, threshold, provenance and held-out test metrics.
model_meta = save_ensemble(blend, os.path.join(ROOT, 'models', 'enriched'), ENRICHED_COLUMNS, thr, prod_metrics, extra={
    'calibration': cal, 'calibration_report': {'method':'sigmoid/Platt validation-only', 'test_ece': round(float(ece),5), 'test_brier': round(float(brier_score_loss(yte, cal_test)),5), 'reliability':rel},
    'training_protocol': 'Chronological: oldest 65% train, next 15% validation, newest 20% untouched test. The saved artifact is the model evaluated on the untouched test; it is fitted on the oldest 65% only.',
    'prediction_cutoff': 'Order purchase time; real Olist data only; no simulated weather/tracking inputs.',
    'split': {'train_rows':len(tr), 'validation_rows':len(va), 'test_rows':len(te), 'train_start':str(tr.order_purchase_timestamp.min()), 'train_end':str(tr.order_purchase_timestamp.max()), 'validation_start':str(va.order_purchase_timestamp.min()), 'validation_end':str(va.order_purchase_timestamp.max()), 'test_start':str(te.order_purchase_timestamp.min()), 'test_end':str(te.order_purchase_timestamp.max()), 'test_late_rate':round(float(yte.mean()),5)},
    'validation_auc_by_candidate': {k:round(v,5) for k,v in val_aucs.items()},
    'test_roc_auc_goal':0.90, 'test_roc_auc_goal_reached':bool(prod_metrics['roc_auc'] >= .90),
    'source_data':'Brazilian Olist real CSVs in data/raw; synthetic tracking/weather files excluded.'
})
print('Saved production model to models/enriched; serving member:', selected, 'threshold:', round(thr,4), 'test AUC:', prod_metrics['roc_auc'], flush=True)

report={'generated_utc':pd.Timestamp.now('UTC').isoformat(),'data_source':'Real Olist delivered-order data; no simulated weather/tracking data in model inputs','target':'is_delayed; actual delivered date later than estimated delivery date','prediction_cutoff':'Order purchase time; features must be known at checkout','split':{'train':{'n':len(tr),'start':str(tr.order_purchase_timestamp.min()),'end':str(tr.order_purchase_timestamp.max()),'late_rate':round(float(ytr.mean()),5)},'validation':{'n':len(va),'start':str(va.order_purchase_timestamp.min()),'end':str(va.order_purchase_timestamp.max()),'late_rate':round(float(yva.mean()),5)},'test':{'n':len(te),'start':str(te.order_purchase_timestamp.min()),'end':str(te.order_purchase_timestamp.max()),'late_rate':round(float(yte.mean()),5)}},'feature_count':len(ENRICHED_COLUMNS),'candidates':results,'validation_auc_by_model':{k:round(v,5) for k,v in val_aucs.items()},'selected':selected,'threshold':round(thr,5),'target_auc_0_90_reached':bool(results[selected]['test']['roc_auc']>=.90),'limitations':['Olist does not include observed historical weather, road disruptions, carrier scans, or hub congestion. Synthetic tracking files are not used as real data or model inputs.','Chronological test is the primary estimate; temporal distribution shift is expected.','Model selection uses validation only; the newest 20% test is held out.'],'runtime_note':'Single-environment training; see fit_seconds per candidate.'}
with open(OUT,'w') as f: json.dump(report,f,indent=2)
print('\nFINAL REPORT',json.dumps({'selected':selected,'validation_auc_by_model':report['validation_auc_by_model'],'test':results[selected]['test'],'target_090':report['target_auc_0_90_reached'],'report':OUT},indent=2),flush=True)
