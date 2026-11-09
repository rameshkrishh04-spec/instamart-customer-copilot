from pathlib import Path
import sqlite3, pandas as pd
ROOT=Path(__file__).resolve().parent
sql=(ROOT/'sql'/'queries.sql').read_text(encoding='utf-8')
queries=[q.strip() for q in sql.split(';') if q.strip()]
with sqlite3.connect(ROOT/'data'/'instamart.db') as conn:
    for i,q in enumerate(queries,1):
        df=pd.read_sql_query(q,conn)
        print(f'\nQUERY {i}\nNumber of rows: {len(df)}')
        print(df.head(10).to_string(index=False))
        if i==2: print('Q2 total revenue:',round(df.total_revenue.sum(),2),'largest:',df.loc[df.total_revenue.idxmax()].to_dict())
        if i==5: print('Q5 average spend:',df.average_spend.iloc[0] if len(df) else 'n/a')
        if i==6: print('Q6 rank vs row number differences:',int((df.rank_value!=df.row_number_value).sum()))
        if i==7: print('Q7 final cumulative revenue:',df.cumulative_revenue.iloc[-1])
print('\nSQL ANALYSIS COMPLETED')