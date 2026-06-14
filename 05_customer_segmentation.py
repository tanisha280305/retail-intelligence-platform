import pandas as pd
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Load cleaned transactions
df = pd.read_parquet("data/cleaned_online_retail.parquet")

# 2. Compute RFM for each customer
#    Recency: days since last purchase (relative to max date in data)
max_date = df.InvoiceDate.max()
rfm = df.groupby("CustomerID").agg(
    Recency  = ("InvoiceDate", lambda x: (max_date - x.max()).days),
    Frequency= ("InvoiceNo",  "nunique"),
    Monetary = ("TotalSales", "sum"),
).reset_index().dropna()

# 3. Scale RFM features
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[["Recency","Frequency","Monetary"]])

# 4. K-Means clustering (k=4)
k = 4
km = KMeans(n_clusters=k, random_state=42)
rfm["Cluster"] = km.fit_predict(rfm_scaled)

# 5. Save the customer-level data
rfm.to_csv("data/customer_rfm_clusters.csv", index=False)

# 6. Build a cluster profile summary
profile = rfm.groupby("Cluster").agg(
    Count       = ("CustomerID", "count"),
    AvgRecency  = ("Recency",  "mean"),
    AvgFrequency= ("Frequency","mean"),
    AvgMonetary = ("Monetary", "mean")
).round(2).reset_index()

profile.to_csv("data/cluster_profile.csv", index=False)

print("Saved: customer_rfm_clusters.csv and cluster_profile.csv")
print(profile)
