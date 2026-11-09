"""Independent Part A data validation checkpoints."""
from pathlib import Path
import pandas as pd
import sqlite3
ROOT=Path(__file__).resolve().parent
c=pd.read_csv(ROOT/'data/clean/customers.csv')
p=pd.read_csv(ROOT/'data/clean/products.csv')
o=pd.read_csv(ROOT/'data/clean/orders.csv')
assert len(c)==1200 and len(p)==40 and len(o)==12309
assert c.city.nunique()==6 and c.age.isna().sum()==0
assert set(c.gender.unique())=={'male','female','unknown'}
assert o.order_id.nunique()==12309 and not o.status.eq('pending').any()
assert o.loc[o.status.eq('completed'),'delivery_minutes'].isna().sum()==0
assert o.loc[o.status.eq('cancelled'),'delivery_minutes'].isna().sum()==1199
assert round(o.loc[o.status.eq('completed'),'order_amount'].sum(),2)==2998668.80
with sqlite3.connect(ROOT/'data/instamart.db') as conn:
    names={r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {'customers','products','orders'}.issubset(names)
print('ALL CHECKPOINTS PASSED')