from pathlib import Path
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parent
orders=pd.read_csv(ROOT/'data/clean/orders.csv',parse_dates=['order_ts'])
customers=pd.read_csv(ROOT/'data/clean/customers.csv')
products=pd.read_csv(ROOT/'data/clean/products.csv')
completed=orders[orders.status.eq('completed')].merge(customers[['customer_id','city']],on='customer_id').merge(products[['product_id','category']],on='product_id')
pivot=pd.pivot_table(completed,index='city',columns='category',values='order_amount',aggfunc='sum',fill_value=0,margins=True)
print('PIVOT TABLE — REVENUE BY CITY AND CATEGORY\n',pivot.round(2).to_string())
city=completed.groupby('city').agg(total_revenue=('order_amount','sum'),mean_order_amount=('order_amount','mean'),order_count=('order_amount','count'))
print('\nCITY COMPLETED ORDER STATISTICS\n',city.round(2).to_string())
s=completed.order_amount
q1,q3=s.quantile(.25),s.quantile(.75); iqr=q3-q1; lower=q1-1.5*iqr; upper=q3+1.5*iqr
print('\nCOMPLETED ORDER AMOUNT DISTRIBUTION')
print('Mean',round(s.mean(),2),'Median',round(s.median(),2),'Skewness',round(s.skew(),2))
print('Q1',round(q1,2),'Q2',round(s.median(),2),'Q3',round(q3,2),'IQR',round(iqr,2))
print('Lower fence',round(lower,2),'Upper fence',round(upper,2))
print('Outliers below',int((s<lower).sum()),'above',int((s>upper).sum()))
cor=completed[['order_amount','discount_pct','quantity','delivery_minutes']].corr()
print('\nCORRELATION MATRIX\n',cor.round(3).to_string())
print('\nPANDAS ANALYSIS COMPLETED')