import pandas as pd
import matplotlib.pyplot as plt

# 1. Load cleaned data
df = pd.read_parquet("data/cleaned_online_retail.parquet")

# 2. Time-series: daily total sales
daily = df.groupby("InvoiceDay")["TotalSales"].sum().reset_index()

plt.figure(figsize=(10,4))
plt.plot(daily.InvoiceDay, daily.TotalSales)
plt.title("Daily Total Sales")
plt.xlabel("Date")
plt.ylabel("Sales (GBP)")
plt.tight_layout()
plt.show()

# 3. Top 10 products by revenue
top_products = (
    df.groupby("Description")["TotalSales"]
      .sum()
      .sort_values(ascending=False)
      .head(10)
)
print("Top 10 products by revenue:")
print(top_products)

# 4. Sales heatmap: weekday vs hour
df["Weekday"] = df.InvoiceDate.dt.day_name()
df["Hour"]    = df.InvoiceDate.dt.hour
pivot = df.pivot_table(index="Weekday",
                       columns="Hour",
                       values="TotalSales",
                       aggfunc="sum"
                       ).fillna(0)

plt.figure(figsize=(12,6))
plt.imshow(pivot, aspect="auto")
plt.yticks(range(len(pivot.index)), pivot.index)
plt.xticks(range(0,24,2), range(0,24,2))
plt.title("Sales Heatmap by Weekday and Hour")
plt.colorbar(label="Total Sales")
plt.show()
