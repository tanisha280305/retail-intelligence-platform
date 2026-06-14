import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from pyod.models.iforest import IForest

# ─── 1) Load cleaned data ──────────────────────────────────────────────────────
# 1) Load cleaned data and build daily sales
df = pd.read_parquet("data/cleaned_online_retail.parquet")
daily = (
    df.groupby("InvoiceDay")["TotalSales"]
      .sum()
      .reset_index()
      .rename(columns={"InvoiceDay": "ds", "TotalSales": "y"})
)

# ─── NEW: ensure `ds` is a timestamp, not a plain date ───
daily["ds"] = pd.to_datetime(daily["ds"])

# ─── 2) Fit & Forecast with Prophet ───────────────────────────────────────────
m = Prophet(
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=True  # set True if >1 year of data
)
m.fit(daily)

future = m.make_future_dataframe(periods=30)
forecast = m.predict(future)

# ─── 3) Plot Forecast (history vs. future) ───────────────────────────────────
cutoff = daily['ds'].max()
hist_fc  = forecast[forecast['ds'] <= cutoff]
future_fc = forecast[forecast['ds'] >  cutoff]

fig, ax = plt.subplots(figsize=(12, 6))

# a) Fitted on history
ax.plot(hist_fc['ds'], hist_fc['yhat'],
        color='tab:blue', linestyle='-', label='Fitted (history)')

# b) Forecast horizon
ax.plot(future_fc['ds'], future_fc['yhat'],
        color='tab:orange', linestyle='--', label='Forecast (future)')

# c) Uncertainty bands
ax.fill_between(
    hist_fc['ds'],
    hist_fc['yhat_lower'], hist_fc['yhat_upper'],
    color='tab:blue', alpha=0.2
)
ax.fill_between(
    future_fc['ds'],
    future_fc['yhat_lower'], future_fc['yhat_upper'],
    color='tab:orange', alpha=0.2
)

# d) Actual sales dots
ax.scatter(daily['ds'], daily['y'],
           color='black', s=15, alpha=0.6, label='Actual sales')

# e) Cutoff line
ax.axvline(cutoff, color='gray', linestyle=':', linewidth=1)
ax.text(cutoff, ax.get_ylim()[1]*0.95, 'Cutoff',
        rotation=90, va='top', ha='right', color='gray')

# Labels, legend
ax.set_title("Prophet Forecast of Daily Sales")
ax.set_xlabel("Date")
ax.set_ylabel("Sales (GBP)")
ax.legend(loc='upper left')

plt.tight_layout()
plt.show()

# ─── 4) Anomaly Detection on Residuals ────────────────────────────────────────
# Compute residuals on history
hist = (
    forecast[['ds', 'yhat']]
    .merge(daily, on='ds', how='inner')
)
hist['residual'] = hist['y'] - hist['yhat']

# Fit Isolation Forest
clf = IForest(contamination=0.05)
clf.fit(hist[['residual']])
hist['anomaly'] = clf.predict(hist[['residual']])  # 1 = outlier

anoms = hist[hist['anomaly'] == 1]

# Plot residuals & anomalies
fig2, ax2 = plt.subplots(figsize=(12, 4))
ax2.plot(hist['ds'], hist['residual'],
         label='Residual (actual - predicted)')
ax2.scatter(anoms['ds'], anoms['residual'],
            color='red', label='Anomaly', s=30)
ax2.set_title("Daily Sales Residuals & Anomalies")
ax2.set_xlabel("Date")
ax2.set_ylabel("Residual")
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()

# ─── 5) Print anomaly summary ─────────────────────────────────────────────────
print("\nAnomalous Dates (Isolation Forest):")
print(anoms[['ds','y','yhat','residual']].to_string(index=False))
