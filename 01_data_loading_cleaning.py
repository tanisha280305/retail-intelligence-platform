import pandas as pd

# 1. Load raw data
df = pd.read_excel("data/Online Retail.xlsx", sheet_name="Online Retail")

# 2. Quick inspect
print(df.shape, df.dtypes)

# 3. Drop rows with no InvoiceNo or StockCode
df = df.dropna(subset=["InvoiceNo", "StockCode"])

# 4. Remove negative or zero quantities and unit prices
df = df[(df.Quantity > 0) & (df.UnitPrice > 0)]

# 5. Convert InvoiceDate to datetime
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# 6. Remove obvious duplicates
df = df.drop_duplicates()

# 7. Add useful columns
df["TotalSales"] = df.Quantity * df.UnitPrice
df["InvoiceDay"]  = df.InvoiceDate.dt.date

# ─── NEW: Force InvoiceNo (and StockCode) to string ───
df["InvoiceNo"]  = df["InvoiceNo"].astype(str)
df["StockCode"]  = df["StockCode"].astype(str)

# 8. Save cleaned data as Parquet
df.to_parquet("data/cleaned_online_retail.parquet", index=False)

print("Cleaned data saved:", df.shape)
