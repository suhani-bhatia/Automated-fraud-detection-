import pandas as pd
import numpy as np

# Load original fraud dataset
df = pd.read_csv("data/creditcard.csv")

# Keep only required columns
df = df[["Time", "Class"]].copy()

# Rename actual label
df.rename(columns={"Class": "actual"}, inplace=True)

# Create 20 sequential monitoring periods from the dataset
n_periods = 20
df = df.sort_values("Time").reset_index(drop=True)
df["period"] = pd.qcut(df.index, q=n_periods, labels=False) + 1
df["date"] = pd.to_datetime("2026-01-01") + pd.to_timedelta(df["period"] - 1, unit="D")

# Simulate model predictions with slightly worse performance in later periods
np.random.seed(42)

threshold_period = int(n_periods * 0.7)
error_prob = np.where(
    df["period"] <= threshold_period,
    np.random.uniform(0.03, 0.07, len(df)),
    np.random.uniform(0.08, 0.15, len(df))
)

flip_mask = np.random.rand(len(df)) < error_prob
df["predicted"] = np.where(flip_mask, 1 - df["actual"], df["actual"])

# Simulate confidence
correct_mask = df["actual"] == df["predicted"]
df["confidence"] = np.where(
    correct_mask,
    np.random.uniform(0.75, 0.98, len(df)),
    np.random.uniform(0.45, 0.70, len(df))
)

# Simulate latency with gradual increase over periods
base_latency = np.linspace(110, 145, n_periods)
latency_map = {period: base_latency[period - 1] for period in range(1, n_periods + 1)}
df["latency_ms"] = df["period"].map(latency_map) + np.random.normal(0, 5, len(df))

# Create drift-like score using fraud-rate variation across periods
daily_fraud = df.groupby("date")["actual"].mean().reset_index(name="fraud_rate")
daily_fraud["drift_score"] = daily_fraud["fraud_rate"].diff().abs().fillna(0)

df = df.merge(daily_fraud[["date", "drift_score"]], on="date", how="left")

# Keep final columns
final_df = df[["date", "actual", "predicted", "confidence", "latency_ms", "drift_score"]].copy()

# Save processed monitoring dataset
final_df.to_csv("data/model_predictions.csv", index=False)

print("Prepared monitoring dataset: data/model_predictions.csv")
print(final_df.head())
print(final_df["date"].nunique(), "monitoring periods created")