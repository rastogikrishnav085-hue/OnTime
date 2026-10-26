import os,sys,json,time
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'src'))
import numpy as np,pandas as pd,joblib
from sklearn.metrics import roc_auc_score,average_precision_score,precision_score,recall_score,f1_score,brier_score_loss,confusion_matrix
from enriched_model import ENRICHED_COLUMNS,as_frame
from features import TARGET_COLUMN
from ensemble import BlendModel,save_ensemble,shift_odds,fit_calibration,apply_calibration,reliability_table,choose_threshold
ROOT=os.path.dirname(os.path.abspath(__file__)); DATA=os.path.join(ROOT,'data','processed','ontime_model_ready.csv')
print('read table',flush=True)
df=pd.read_csv(DATA,parse_dates=['order_purchase_timestamp']).sort_values('order_purchase_timestamp',kind='stable').reset_index(drop=True)
n=len(df); i1=int(n*.65); i2=int(n*.80); tr,va,te=df.iloc[:i1],df.iloc[i1:i2],df.iloc[i2:]
ytr=tr[TARGET_COLUMN].astype(int).to_numpy(); yva=va[TARGET_COLUMN].astype(int).to_numpy(); yte=te[TARGET_COLUMN].astype(int).to_numpy()
spw=float((ytr==0).sum()/max((ytr==1).sum(),1)); spw_tv=float(((pd.concat([tr,va])[TARGET_COLUMN]==0).sum())/max((pd.concat([tr,va])[TARGET_COLUMN]==1).sum(),1))
print('load fitted train candidate',flush=True)
cand=joblib.load(os.path.join(ROOT,'models','verified_chronological','xgboost_enriched.joblib'))
Xva=as_frame(va); Xte=as_frame(te)
pva_raw=cand.predict_proba(Xva)[:,1]; pte_raw=cand.predict_proba(Xte)[:,1]
pva=shift_odds(pva_raw,spw_tv); pte_candidate=shift_odds(pte_raw,spw_tv)
thr,_=choose_threshold(yva,pva); cal=fit_calibration(pva,yva)
print('validation threshold',thr,'validation auc',roc_auc_score(yva,pva),'calibration done',flush=True)
final=joblib.load(os.path.join(ROOT,'models','verified_chronological','selected_model.joblib'))
pte_raw_final=final.predict_proba(as_frame(te))[:,1]; pte=shift_odds(pte_raw_final,spw_tv); pcal=apply_calibration(cal,pte)
print('final refit test auc',roc_auc_score(yte,pte),'calibration done',flush=True)
print('loaded final refit; saving blend',flush=True)
blend=BlendModel({'xgb':final},weights={'xgb':1.0},explainer='xgb',odds_shift=spw_tv)
pred=(pte>=thr).astype(int); tn,fp,fn,tp=confusion_matrix(yte,pred,labels=[0,1]).ravel()
metrics={'roc_auc':float(roc_auc_score(yte,pte)),'pr_auc':float(average_precision_score(yte,pte)),'precision':float(precision_score(yte,pred,zero_division=0)),'recall':float(recall_score(yte,pred,zero_division=0)),'f1':float(f1_score(yte,pred,zero_division=0)),'brier_raw':float(brier_score_loss(yte,pte)),'brier_calibrated':float(brier_score_loss(yte,pcal)),'threshold':float(thr),'confusion':{'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}}
rel,ece=reliability_table(yte,pcal,10); metrics['calibration_ece']=ece
extra={'calibration':cal,'calibration_report':{'method':'isotonic regression fitted on chronological validation block only','test_ece':ece,'test_brier':metrics['brier_calibrated'],'reliability':rel},'training_protocol':'Chronological 65% train / 15% validation / 20% held-out test; final refit on train+validation; test used only for final report.','prediction_cutoff':'Order purchase time. Real Olist data only; no synthetic weather/tracking inputs.','split':{'train_rows':len(tr),'validation_rows':len(va),'test_rows':len(te),'train_start':str(tr.order_purchase_timestamp.min()),'train_end':str(tr.order_purchase_timestamp.max()),'validation_start':str(va.order_purchase_timestamp.min()),'validation_end':str(va.order_purchase_timestamp.max()),'test_start':str(te.order_purchase_timestamp.min()),'test_end':str(te.order_purchase_timestamp.max()),'test_late_rate':float(yte.mean())},'test_roc_auc_goal':0.90,'test_roc_auc_goal_reached':bool(metrics['roc_auc']>=.90),'source_data':'data/raw: nine original Brazilian Olist CSV files; synthetic tracking/weather excluded','note':'Chronological generalization is below the 0.90 aspiration; do not claim it was reached.'}
print('save ensemble starting',flush=True)
meta=save_ensemble(blend,os.path.join(ROOT,'models','enriched'),ENRICHED_COLUMNS,thr,metrics,extra=extra)
print('saved ensemble',meta['members'],flush=True)
# report enriched metrics to verified report
report=json.load(open(os.path.join(ROOT,'dashboard','verified_training_report.json'))) if os.path.exists(os.path.join(ROOT,'dashboard','verified_training_report.json')) else {}
report['production_artifact']={'directory':'models/enriched','member':'xgb','threshold':thr,'test_metrics':metrics,'model_selection':'XGBoost selected on chronological validation; blend gains were below 0.002 AUC and not considered meaningful.'}
report['selected']='xgboost_enriched'
report['target_auc_0_90_reached']=bool(metrics['roc_auc']>=.90)
with open(os.path.join(ROOT,'dashboard','verified_training_report.json'),'w') as f:json.dump(report,f,indent=2,default=float)
print('FINAL',json.dumps({'threshold':thr,'metrics':metrics,'model_dir':'models/enriched'},indent=2),flush=True)
