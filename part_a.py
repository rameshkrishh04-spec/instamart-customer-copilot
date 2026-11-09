"""Part A: raw audit, deterministic cleaning, SQLite storage and validation."""
from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
CLEAN.mkdir(parents=True, exist_ok=True)

customers_raw = pd.read_csv(RAW / "customers.csv", dtype={"age": "string"})
products_raw = pd.read_csv(RAW / "products.csv")
orders_raw = pd.read_csv(RAW / "orders.csv", dtype={"order_amount": "string", "delivery_minutes": "string"})
tickets_raw = pd.read_csv(RAW / "tickets.csv", keep_default_na=False)

print("PART A — RAW DATA AUDIT")
for name, frame in [("customers", customers_raw), ("products", products_raw), ("orders", orders_raw), ("tickets", tickets_raw)]:
    print(f"\n{name.upper()} shape={frame.shape}")
    print(frame.dtypes.to_string())
    missing = frame.isna().sum()
    pct = (missing / len(frame) * 100).round(2)
    print(pd.DataFrame({"missing_count": missing, "missing_pct": pct}).to_string())

order_amount_numeric = pd.to_numeric(orders_raw["order_amount"], errors="coerce")
raw_audit = {
    "raw_order_rows": len(orders_raw),
    "exact_duplicate_rows": int(orders_raw.duplicated().sum()),
    "pending_rows": int(orders_raw["status"].eq("pending").sum()),
    "distinct_order_ids": int(orders_raw["order_id"].nunique()),
    "non_numeric_order_amounts": int(order_amount_numeric.isna().sum()),
    "negative_delivery_minutes": int(pd.to_numeric(orders_raw["delivery_minutes"], errors="coerce").lt(0).sum()),
    "blank_delivery_minutes": int(orders_raw["delivery_minutes"].fillna("").astype(str).str.strip().eq("").sum()),
    "blank_ages": int(customers_raw["age"].fillna("").astype(str).str.strip().eq("").sum()),
    "ages_outside_10_100": int((pd.to_numeric(customers_raw["age"], errors="coerce").notna() & ~pd.to_numeric(customers_raw["age"], errors="coerce").between(10, 100)).sum()),
    "raw_city_spellings": int(customers_raw["city"].nunique()),
    "raw_nonblank_gender_spellings": int(customers_raw.loc[customers_raw["gender"].fillna("").astype(str).str.strip().ne(""), "gender"].nunique()),
    "blank_genders": int(customers_raw["gender"].fillna("").astype(str).str.strip().eq("").sum()),
}
print("\nRAW AUDIT CHECKPOINTS")
for k, v in raw_audit.items(): print(f"{k}: {v}")

# Customers: standardise, map, clean age and impute within city.
customers = customers_raw.copy()
customers["city"] = customers["city"].astype(str).str.strip().str.title()
customers["gender"] = customers["gender"].fillna("").astype(str).str.strip().str.lower().map(
    {"male": "male", "m": "male", "female": "female", "f": "female"}
).fillna("unknown")
customers["age"] = pd.to_numeric(customers["age"], errors="coerce").astype(float)
customers.loc[~customers["age"].between(10, 100), "age"] = np.nan
customers["age"] = customers["age"].fillna(customers.groupby("city")["age"].transform("median"))
customers["signup_date"] = pd.to_datetime(customers["signup_date"], errors="raise")
assert customers["city"].nunique() == 6
assert customers["age"].isna().sum() == 0

# Products.
products = products_raw.copy()
products["price_inr"] = pd.to_numeric(products["price_inr"], errors="raise")

# Orders: apply the required rules in the stated order.
orders = orders_raw.copy()
orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="raise")
raw_n = len(orders)
dup_n = int(orders.duplicated().sum())
orders = orders.drop_duplicates().copy()
orders = orders.sort_values(["order_id", "order_ts"], kind="stable").drop_duplicates("order_id", keep="last").copy()
orders["quantity"] = pd.to_numeric(orders["quantity"], errors="raise").astype(int)
orders["discount_pct"] = pd.to_numeric(orders["discount_pct"], errors="raise")
orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")
orders["delivery_minutes"] = pd.to_numeric(orders["delivery_minutes"], errors="coerce").astype(float)
orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = np.nan
orders = orders.merge(products[["product_id", "category", "price_inr"]], on="product_id", how="left", validate="many_to_one")
missing_amount_before = int(orders["order_amount"].isna().sum())
orders["order_amount"] = orders["order_amount"].fillna(
    (orders["price_inr"] * orders["quantity"] * (1 - orders["discount_pct"] / 100)).round(2)
)
completed = orders["status"].eq("completed")
category_medians = orders.loc[completed].groupby("category")["delivery_minutes"].median()
missing_completed_before = int(orders.loc[completed, "delivery_minutes"].isna().sum())
orders.loc[completed, "delivery_minutes"] = orders.loc[completed].apply(
    lambda r: category_medians.loc[r["category"]] if pd.isna(r["delivery_minutes"]) else r["delivery_minutes"], axis=1
)
# Cancelled orders intentionally retain missing delivery time.
orders = orders.drop(columns=["price_inr", "category"])
orders["delivery_minutes"] = orders["delivery_minutes"].round().astype("Int64")

# Save clean tables. Datetimes are ISO strings for portability in CSV/SQLite.
customers_out = customers.copy(); customers_out["signup_date"] = customers_out["signup_date"].dt.strftime("%Y-%m-%d")
orders_out = orders.copy(); orders_out["order_ts"] = orders_out["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
customers_out.to_csv(CLEAN / "customers.csv", index=False)
products.to_csv(CLEAN / "products.csv", index=False)
orders_out.to_csv(CLEAN / "orders.csv", index=False)

# SQLite database.
db_path = ROOT / "data" / "instamart.db"
with sqlite3.connect(db_path) as conn:
    customers_out.to_sql("customers", conn, if_exists="replace", index=False)
    products.to_sql("products", conn, if_exists="replace", index=False)
    orders_out.to_sql("orders", conn, if_exists="replace", index=False)

# Reference audit and cleaning summary.
summary = {
    "customers_rows": len(customers), "products_rows": len(products), "orders_rows": len(orders),
    "completed_orders": int(orders["status"].eq("completed").sum()),
    "cancelled_orders": int(orders["status"].eq("cancelled").sum()),
    "exact_duplicates_removed": dup_n, "duplicate_order_records_removed_after_latest": int(raw_n - dup_n - len(orders)),
    "amounts_imputed_after_dedup": missing_amount_before,
    "completed_delivery_times_imputed": missing_completed_before,
    "cancelled_delivery_times_left_missing": int(orders.loc[orders["status"].eq("cancelled"), "delivery_minutes"].isna().sum()),
}
print("\nCLEANING SUMMARY")
for k, v in summary.items(): print(f"{k}: {v}")
print("\nCLEAN CUSTOMER CITY COUNTS")
print(customers["city"].value_counts().to_string())
print("\nCLEAN CUSTOMER GENDER COUNTS")
print(customers["gender"].value_counts().to_string())
print("\nDATABASE TABLES AND ROW COUNTS")
with sqlite3.connect(db_path) as conn:
    for table in ["customers", "products", "orders"]:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"{table}: {count}")

# Core exact checkpoints.
assert raw_audit["raw_order_rows"] == 13084
assert raw_audit["exact_duplicate_rows"] == 119
assert raw_audit["pending_rows"] == 656
assert raw_audit["distinct_order_ids"] == 12309
assert raw_audit["non_numeric_order_amounts"] == 120
assert raw_audit["negative_delivery_minutes"] == 106
assert raw_audit["blank_delivery_minutes"] == 2541
assert raw_audit["blank_ages"] == 61 and raw_audit["ages_outside_10_100"] == 6
assert raw_audit["raw_city_spellings"] == 24 and raw_audit["raw_nonblank_gender_spellings"] == 10 and raw_audit["blank_genders"] == 30
assert len(customers) == 1200 and len(products) == 40 and len(orders) == 12309
assert int(orders["status"].eq("completed").sum()) == 11110
assert int(orders["status"].eq("cancelled").sum()) == 1199
print("\nALL CHECKPOINTS PASSED")