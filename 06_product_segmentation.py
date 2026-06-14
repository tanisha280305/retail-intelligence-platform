import pandas as pd
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Load clean data
df = pd.read_parquet("data/cleaned_online_retail.parquet")

# 2. Compute per-SKU metrics
today = df.InvoiceDate.max() + pd.Timedelta(days=1)
sku = df.groupby("StockCode").agg(
    Description    = ("Description", lambda x: x.mode()[0]),
    TotalRevenue   = ("TotalSales", "sum"),
    UnitsSold      = ("Quantity", "sum"),
    FirstSale      = ("InvoiceDate", "min"),
    LastSale       = ("InvoiceDate", "max"),
    DaysWithSales  = ("InvoiceDate", lambda x: x.dt.date.nunique()),
    TotalOrders    = ("InvoiceNo", "nunique"),
).reset_index()

# 3. Feature engineering
sku["AvgPrice"]    = sku.TotalRevenue / sku.UnitsSold
sku["Recency"]     = (today - sku.LastSale).dt.days
sku["SalesFreq"]   = sku.DaysWithSales
# (Optional) compute return rate if you have returns data importaaaaaaaant

features = sku[["TotalRevenue","UnitsSold","AvgPrice","Recency","SalesFreq"]].fillna(0)

# 4. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# 5. K-Means clustering (k=4)
kmeans = KMeans(n_clusters=4, random_state=42)
sku["Segment"] = kmeans.fit_predict(X_scaled)

# 6. Save results
sku.to_csv("data/product_segments.csv", index=False)

# 7. Build a segment profile
profile = sku.groupby("Segment").agg(
    Count          = ("StockCode","count"),
    AvgRevenue     = ("TotalRevenue","mean"),
    AvgUnits       = ("UnitsSold","mean"),
    AvgRecency     = ("Recency","mean"),
    AvgSalesDays   = ("SalesFreq","mean")
).round(2).reset_index()

profile.to_csv("data/product_segment_profile.csv", index=False)
print(profile)
