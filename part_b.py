"""Part B: RFM segmentation, leakage-aware lapse prediction and persistence."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)
orders=pd.read_csv(ROOT/'data/clean/orders.csv',parse_dates=['order_ts'])
customers=pd.read_csv(ROOT/'data/clean/customers.csv',parse_dates=['signup_date'])
products=pd.read_csv(ROOT/'data/clean/products.csv')
orders=orders.merge(products[['product_id','category']],on='product_id',how='left',validate='many_to_one')
completed=orders[orders.status.eq('completed')].copy()

# Task 1: whole-window RFM as of 2025-10-01.
snapshot=pd.Timestamp('2025-10-01 00:00:00')
rfm=(completed.groupby('customer_id').agg(last_order_ts=('order_ts','max'),frequency=('order_id','nunique'),monetary=('order_amount','sum')).reset_index())
rfm['recency']=(snapshot-rfm.last_order_ts).dt.days
rfm=rfm[['customer_id','recency','frequency','monetary']]
print('\nPART B — WHOLE-WINDOW RFM')
print('Customers:',len(rfm),'recency median/min/max:',rfm.recency.median(),rfm.recency.min(),rfm.recency.max())
print('Frequency mean/max:',round(rfm.frequency.mean(),2),rfm.frequency.max(),'Monetary mean/max:',round(rfm.monetary.mean(),2),round(rfm.monetary.max(),2))

# Task 2: standardise RFM and evaluate K=2..8.
rfm_scaler=StandardScaler()
rfm_scaled=rfm_scaler.fit_transform(rfm[['recency','frequency','monetary']])
k_rows=[]
for k in range(2,9):
    km=KMeans(n_clusters=k,n_init=10,random_state=42)
    labels=km.fit_predict(rfm_scaled)
    k_rows.append({'k':k,'inertia':float(km.inertia_),'silhouette':float(silhouette_score(rfm_scaled,labels))})
    print(f"K={k}: inertia={km.inertia_:.2f}, silhouette={k_rows[-1]['silhouette']:.3f}")
pd.DataFrame(k_rows).to_csv(OUT/'kmeans_evaluation.csv',index=False)

# K=4 is selected for elbow, strong silhouette and useful four-segment actionability.
kmeans=KMeans(n_clusters=4,n_init=10,random_state=42)
rfm['segment_id']=kmeans.fit_predict(rfm_scaled)
profiles=rfm.groupby('segment_id').agg(customer_count=('customer_id','size'),mean_recency=('recency','mean'),mean_frequency=('frequency','mean'),mean_monetary=('monetary','mean')).reset_index()
# Assign business names based on profile characteristics, not numeric cluster ordering.
monetary_leader=int(profiles.loc[profiles.mean_monetary.idxmax(),'segment_id'])
frequency_leader=int(profiles.loc[profiles.mean_frequency.idxmax(),'segment_id'])
recency_laggard=int(profiles.loc[profiles.mean_recency.idxmax(),'segment_id'])
segment_names={}
for _,row in profiles.iterrows():
    sid=int(row.segment_id)
    if sid==monetary_leader: segment_names[sid]='Premium Customers'
    elif sid==frequency_leader: segment_names[sid]='Loyal Customers'
    elif sid==recency_laggard: segment_names[sid]='At-Risk Customers'
    else: segment_names[sid]='Occasional Customers'
rfm['segment_name']=rfm.segment_id.map(segment_names)
rfm['is_premium']=rfm.segment_id.eq(monetary_leader)
rfm[['customer_id','recency','frequency','monetary','segment_id','segment_name','is_premium']].to_csv(OUT/'customer_segments.csv',index=False)
profiles['segment_name']=profiles.segment_id.map(segment_names)
profiles['is_premium']=profiles.segment_id.eq(monetary_leader)
profiles.to_csv(OUT/'cluster_profiles.csv',index=False)
pca=PCA(n_components=2,random_state=42)
pca.fit(rfm_scaled)
print('\nSEGMENT PROFILES\n',profiles.round(2).to_string(index=False))
print('PCA explained variance:',pca.explained_variance_ratio_.round(3).tolist(),'total:',round(float(pca.explained_variance_ratio_.sum()),3))

# Task 3: lapse cohort. Features use only completed/final orders before 2025-09-01.
cutoff=pd.Timestamp('2025-09-01 00:00:00')
sep_end=pd.Timestamp('2025-10-01 00:00:00')
hist=orders[orders.order_ts < cutoff].copy()
hist_completed=hist[hist.status.eq('completed')].copy()
cohort_ids=sorted(hist_completed.customer_id.unique())
sep_completed=set(completed.loc[(completed.order_ts>=cutoff)&(completed.order_ts<sep_end),'customer_id'])
cohort=pd.DataFrame({'customer_id':cohort_ids})
cohort['lapsed']=(~cohort.customer_id.isin(sep_completed)).astype(int)

# Aggregate behaviour from pre-cutoff completed orders only.
agg=hist_completed.groupby('customer_id').agg(
    last_order_ts=('order_ts','max'),
    frequency=('order_id','nunique'),
    monetary=('order_amount','sum'),
    avg_discount_pct=('discount_pct','mean'),
    avg_delivery_minutes=('delivery_minutes','mean')
).reset_index()
agg['recency']=(cutoff-agg.last_order_ts).dt.days
# Cancellation rate denominator: final completed + cancelled orders before cutoff.
final_hist=hist[hist.status.isin(['completed','cancelled'])]
all_final=final_hist.groupby('customer_id').size().rename('all_final')
cancelled=final_hist[final_hist.status.eq('cancelled')].groupby('customer_id').size().rename('cancelled')
rate=pd.concat([all_final,cancelled],axis=1).fillna(0)
rate['cancellation_rate']=rate.cancelled/rate.all_final
# Most frequent category, deterministic lexical tie-break.
fav=(hist_completed.groupby(['customer_id','category']).size().reset_index(name='n')
     .sort_values(['customer_id','n','category'],ascending=[True,False,True])
     .drop_duplicates('customer_id').rename(columns={'category':'favorite_category'}))
features=cohort.merge(agg,on='customer_id',how='left').merge(rate[['cancellation_rate']],left_on='customer_id',right_index=True,how='left').merge(fav[['customer_id','favorite_category']],on='customer_id',how='left')
features=features.merge(customers[['customer_id','city','age','gender','signup_date']],on='customer_id',how='left')
features['tenure_days']=(cutoff-features.signup_date).dt.days
features=features.drop(columns=['last_order_ts','signup_date'])
num_cols=['recency','frequency','monetary','avg_discount_pct','cancellation_rate','avg_delivery_minutes','age','tenure_days']
cat_cols=['favorite_category','city','gender']
for c in num_cols: features[c]=pd.to_numeric(features[c],errors='coerce')
for c in cat_cols: features[c]=features[c].fillna('Unknown').astype(str)
# Data is expected to be complete; explicit train-only imputation keeps pipeline robust.
X=features[num_cols+cat_cols].copy(); y=features['lapsed'].astype(int)
assert not X.isna().any().any(), 'Feature matrix contains missing values.'
print('\nLAPSE COHORT:',len(features),'lapsed:',int(y.sum()),'rate:',round(y.mean()*100,2),'%')
X_train,X_test,y_train,y_test,idx_train,idx_test=train_test_split(
    X,y,features.index,test_size=0.2,stratify=y,random_state=42
)
print('Train/test:',len(X_train),len(X_test),'train class counts:',y_train.value_counts().to_dict(),'test:',y_test.value_counts().to_dict())

numeric_pipe=Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())])
category_pipe=Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(drop='first',handle_unknown='ignore'))])
preprocess=ColumnTransformer([('numeric',numeric_pipe,num_cols),('categorical',category_pipe,cat_cols)])

# Baseline: always predict not-lapsed.
baseline_pred=np.zeros(len(y_test),dtype=int)
baseline={'accuracy':float(accuracy_score(y_test,baseline_pred)),'precision':float(precision_score(y_test,baseline_pred,zero_division=0)),'recall':float(recall_score(y_test,baseline_pred,zero_division=0)),'f1':float(f1_score(y_test,baseline_pred,zero_division=0))}
print('Baseline (always active):',baseline)

# Fit preprocessing only inside each CV fold / training fold.
lr_pipe=Pipeline([('preprocess',preprocess),('model',LogisticRegression(max_iter=1000))])
lr_grid=GridSearchCV(lr_pipe,{'model__C':[0.01,0.1,1,10,100]},cv=5,scoring='f1',n_jobs=-1)
lr_grid.fit(X_train,y_train)
rf_pipe=Pipeline([('preprocess',preprocess),('model',RandomForestClassifier(n_estimators=100,random_state=42))])
rf_grid=GridSearchCV(rf_pipe,{'model__max_depth':[3,5,8],'model__min_samples_leaf':[5,10,20]},cv=5,scoring='f1',n_jobs=-1)
rf_grid.fit(X_train,y_train)
print('Logistic best params:',lr_grid.best_params_,'CV F1:',round(lr_grid.best_score_,4))
print('Random forest best params:',rf_grid.best_params_,'CV F1:',round(rf_grid.best_score_,4))

# Evaluate only after tuning on the untouched test set.
def eval_pipe(model, name):
    pred=model.predict(X_test)
    return {'model':name,'confusion_matrix':confusion_matrix(y_test,pred).tolist(),
            'precision':float(precision_score(y_test,pred,zero_division=0)),
            'recall':float(recall_score(y_test,pred,zero_division=0)),
            'f1':float(f1_score(y_test,pred,zero_division=0)),
            'accuracy':float(accuracy_score(y_test,pred))}
lr_eval=eval_pipe(lr_grid.best_estimator_,'Logistic Regression')
rf_eval=eval_pipe(rf_grid.best_estimator_,'Random Forest')
print('Test logistic:',lr_eval)
print('Test random forest:',rf_eval)

# Class-balance experiment: oversample positive rows in TRAIN ONLY, with replacement.
train_df=X_train.copy(); train_df['_y']=y_train.values
n_major=int(y_train.value_counts().max()); n_minor=int(y_train.value_counts().min())
minor=train_df[train_df['_y']==1]
extra=minor.sample(n=n_major-n_minor,replace=True,random_state=42)
balanced=pd.concat([train_df,extra],ignore_index=True)
X_bal=balanced.drop(columns='_y'); y_bal=balanced['_y']
bal_lr=Pipeline([('preprocess',preprocess),('model',LogisticRegression(max_iter=1000,C=lr_grid.best_params_['model__C']))])
bal_lr.fit(X_bal,y_bal)
bal_pred=bal_lr.predict(X_test)
balanced_metrics={'precision':float(precision_score(y_test,bal_pred,zero_division=0)),'recall':float(recall_score(y_test,bal_pred,zero_division=0)),'f1':float(f1_score(y_test,bal_pred,zero_division=0))}
print('Balanced LR metrics:',balanced_metrics,'balanced train counts:',y_bal.value_counts().to_dict())

# Interpret best RF importances and LR coefficient signs.
rf_best=rf_grid.best_estimator_
rf_names=rf_best.named_steps['preprocess'].get_feature_names_out()
rf_imp=rf_best.named_steps['model'].feature_importances_
feature_importance=sorted([{'feature':str(n),'importance':float(v)} for n,v in zip(rf_names,rf_imp)],key=lambda z:z['importance'],reverse=True)
lr_best=lr_grid.best_estimator_
lr_names=lr_best.named_steps['preprocess'].get_feature_names_out()
lr_coef=lr_best.named_steps['model'].coef_[0]
coefficients=[{'feature':str(n),'coefficient':float(v),'sign':'positive' if v>0 else 'negative' if v<0 else 'zero'} for n,v in zip(lr_names,lr_coef)]

# Select higher test F1; persist the full fitted preprocessing+model pipeline.
selected=rf_best if rf_eval['f1']>=lr_eval['f1'] else lr_best
selected_name='Random Forest' if selected is rf_best else 'Logistic Regression'
selected_path=OUT/'lapse_model.joblib'
joblib.dump(selected,selected_path)
reloaded=joblib.load(selected_path)
assert np.array_equal(selected.predict(X_test),reloaded.predict(X_test))
print('Selected model:',selected_name,'reload predictions identical: True')

# Premium retention list: test-set rows with Premium flag from whole-window RFM.
segment_lookup=rfm[['customer_id','is_premium']]
test_customers=features.loc[idx_test,['customer_id','lapsed']].copy().reset_index(drop=True)
test_customers['probability']=selected.predict_proba(X_test)[:,1]
test_customers=test_customers.merge(segment_lookup,on='customer_id',how='left')
retention=test_customers[test_customers.is_premium.eq(True)].sort_values('probability',ascending=False).head(10)
retention=retention.rename(columns={'probability':'lapse_probability','lapsed':'actually_lapsed'})[['customer_id','lapse_probability','actually_lapsed']]
retention.to_csv(OUT/'retention_list.csv',index=False)

summary={
    'rfm':{'customers':int(len(rfm)),'recency_median':float(rfm.recency.median()),'recency_min':int(rfm.recency.min()),'recency_max':int(rfm.recency.max()),'frequency_mean':float(rfm.frequency.mean()),'frequency_max':int(rfm.frequency.max()),'monetary_mean':float(rfm.monetary.mean()),'monetary_max':float(rfm.monetary.max())},
    'kmeans':k_rows,'profiles':profiles.to_dict(orient='records'),'pca_variance':pca.explained_variance_ratio_.tolist(),
    'cohort':{'customers':int(len(features)),'lapsed':int(y.sum()),'active':int((y==0).sum()),'lapse_rate':float(y.mean()),'train':int(len(X_train)),'test':int(len(X_test))},
    'baseline':baseline,'logistic':{'best_params':lr_grid.best_params_,'cv_f1':float(lr_grid.best_score_),'test':lr_eval},
    'random_forest':{'best_params':rf_grid.best_params_,'cv_f1':float(rf_grid.best_score_),'test':rf_eval},
    'balanced_logistic':balanced_metrics,'rf_feature_importance':feature_importance[:5],'lr_coefficients':coefficients,
    'scaler_frequency_mean_train':float(lr_best.named_steps['preprocess'].named_transformers_['numeric'].named_steps['scaler'].mean_[1]),
    'frequency_mean_full_cohort':float(X['frequency'].mean()),
    'selected_model':selected_name,'retention_list_rows':int(len(retention)),'retention_true_lapsed':int(retention.actually_lapsed.sum())
}
(OUT/'part_b_summary.json').write_text(json.dumps(summary,indent=2,default=str),encoding='utf-8')
print('Premium test retention list:',len(retention),'actually lapsed:',int(retention.actually_lapsed.sum()))
print('\nPART B COMPLETE')