"""
Daily processing job for XInsights.
- Reads raw transactions from data/cleaned_online_retail.parquet
- Builds Prophet forecast, computes residuals
- Detects anomalies with IForest
- Writes processed parquet files under processed/
- If GCS_BUCKET is set and credentials available, uploads the processed files to GCS
"""

import os
import io
import argparse
from datetime import date, timedelta
import pandas as pd
from prophet import Prophet
from pyod.models.iforest import IForest
from google.cloud import storage

# Config
GCS_BUCKET = os.environ.get("GCS_BUCKET")  # if set, upload processed files there
OUT_PREFIX = "processed"  # local folder or gcs prefix
LOCAL_PROCESSED_DIR = "processed"

os.makedirs(LOCAL_PROCESSED_DIR, exist_ok=True)


def read_raw_data(local_path="app/data/cleaned_online_retail.parquet"):
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Raw data not found at {local_path}")
    df = pd.read_parquet(local_path)
    # ensure datetime cols
    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    if "InvoiceDay" in df.columns:
        # unify InvoiceDay as date
        df["InvoiceDay"] = pd.to_datetime(df["InvoiceDay"]).dt.date
    return df


def build_forecast_and_anomalies(df, history_days=None):
    # daily aggregated series
    daily = (
        df.groupby("InvoiceDay")["TotalSales"]
        .sum()
        .reset_index()
        .rename(columns={"InvoiceDay": "ds", "TotalSales": "y"})
    )
    daily["ds"] = pd.to_datetime(daily["ds"])

    # fit Prophet
    m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
    m.fit(daily)

    # make future
    future = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)

    cutoff = daily["ds"].max()
    hist_fc = forecast[forecast["ds"] <= cutoff].copy()
    future_fc = forecast[forecast["ds"] > cutoff].copy()

    # residuals & anomalies
    hist = forecast[["ds", "yhat"]].merge(daily, on="ds", how="inner")
    hist["residual"] = hist["y"] - hist["yhat"]

    clf = IForest(contamination=0.05)
    clf.fit(hist[["residual"]])
    hist["anomaly"] = clf.predict(hist[["residual"]])
    anoms = hist[hist["anomaly"] == 1].copy()

    # return dataframes
    return daily, hist_fc, future_fc, hist, anoms


def write_parquet_local(df, path):
    df.to_parquet(path, index=False)
    print("Wrote:", path)


def upload_to_gcs(local_path, bucket_name, dest_blob_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(dest_blob_name)
    blob.upload_from_filename(local_path)
    print(f"Uploaded {local_path} -> gs://{bucket_name}/{dest_blob_name}")


def main(process_date=None):
    # We process whole dataset (recompute forecast using all data) but
    # you could change this to only process recent window if desired.
    print("Loading raw data...")
    df = read_raw_data()

    # Build forecasting and anomaly detection
    print("Building forecast + anomalies...")
    daily, hist_fc, future_fc, hist, anoms = build_forecast_and_anomalies(df)

    # file names
    local_daily = os.path.join(LOCAL_PROCESSED_DIR, "daily.parquet")
    local_hist_fc = os.path.join(LOCAL_PROCESSED_DIR, "hist_fc.parquet")
    local_future_fc = os.path.join(LOCAL_PROCESSED_DIR, "future_fc.parquet")
    local_hist = os.path.join(LOCAL_PROCESSED_DIR, "hist.parquet")
    local_anoms = os.path.join(LOCAL_PROCESSED_DIR, "anoms.parquet")

    # write locally
    write_parquet_local(daily, local_daily)
    write_parquet_local(hist_fc, local_hist_fc)
    write_parquet_local(future_fc, local_future_fc)
    write_parquet_local(hist, local_hist)
    write_parquet_local(anoms, local_anoms)

    # upload to GCS if bucket configured
    if GCS_BUCKET:
        print("GCS_BUCKET set, uploading processed files...")
        upload_to_gcs(local_daily, GCS_BUCKET, f"{OUT_PREFIX}/daily.parquet")
        upload_to_gcs(local_hist_fc, GCS_BUCKET, f"{OUT_PREFIX}/hist_fc.parquet")
        upload_to_gcs(local_future_fc, GCS_BUCKET, f"{OUT_PREFIX}/future_fc.parquet")
        upload_to_gcs(local_hist, GCS_BUCKET, f"{OUT_PREFIX}/hist.parquet")
        upload_to_gcs(local_anoms, GCS_BUCKET, f"{OUT_PREFIX}/anoms.parquet")
    else:
        print("GCS_BUCKET not set — kept processed files locally in ./processed/")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="Optional date string to process (not used in current flow)")
    args = parser.parse_args()
    main(process_date=args.date)
