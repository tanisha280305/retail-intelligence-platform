import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1) Load daily data
df = pd.read_parquet("data/cleaned_online_retail.parquet")
daily = (
    df.groupby("InvoiceDay")["TotalSales"]
      .sum()
      .reset_index()
      .rename(columns={"InvoiceDay":"ds","TotalSales":"y"})
)
daily["ds"] = pd.to_datetime(daily["ds"])

# 2) Split into train/test (last 30 days held out)
holdout_days = 30
cut_date = daily['ds'].max() - pd.Timedelta(days=holdout_days)
train = daily[daily['ds'] <= cut_date].copy()
test  = daily[daily['ds'] >  cut_date].copy()

print(f"Training on {train.shape[0]} days; testing on {test.shape[0]} days\n")

# 3) Fit Prophet on train set
m = Prophet(
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=False
)
m.fit(train)

# 4) Forecast for entire period (train+test)
future = m.make_future_dataframe(periods=holdout_days)
forecast = m.predict(future)

# 5) Merge predictions with test set
pred = forecast[['ds','yhat']].merge(test[['ds','y']], on='ds')

# 6) Compute accuracy metrics
mae  = mean_absolute_error(pred['y'], pred['yhat'])
rmse = np.sqrt(mean_squared_error(pred['y'], pred['yhat']))

print("Forecast Accuracy on last {} days:".format(holdout_days))
print(f"  • MAE  = {mae:,.2f} GBP")
print(f"  • RMSE = {rmse:,.2f} GBP")
