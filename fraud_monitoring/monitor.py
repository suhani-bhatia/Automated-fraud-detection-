import pandas as pd
import matplotlib.pyplot as plt
import os

# Create output folder if it doesn't exist
os.makedirs("output", exist_ok=True)

# Load data
df = pd.read_csv("data/model_predictions.csv")
df["date"] = pd.to_datetime(df["date"])

# Calculate correctness and error
df["correct"] = (df["actual"] == df["predicted"]).astype(int)
df["error"] = 1 - df["correct"]

# Daily summary metrics
daily_metrics = df.groupby("date").agg(
    accuracy=("correct", "mean"),
    error_rate=("error", "mean"),
    fraud_rate=("actual", "mean"),
    avg_confidence=("confidence", "mean"),
    avg_latency_ms=("latency_ms", "mean"),
    avg_drift_score=("drift_score", "mean")
).reset_index()

# Detect sudden drops in accuracy
daily_metrics["accuracy_drop"] = daily_metrics["accuracy"].diff()
daily_metrics["sudden_drop_alert"] = daily_metrics["accuracy_drop"] < -0.10

# Convert to percentages
daily_metrics["accuracy"] = daily_metrics["accuracy"] * 100
daily_metrics["error_rate"] = daily_metrics["error_rate"] * 100
daily_metrics["fraud_rate"] = daily_metrics["fraud_rate"] * 100
daily_metrics["accuracy_drop"] = daily_metrics["accuracy_drop"] * 100

# Add alert flags
daily_metrics["accuracy_alert"] = daily_metrics["accuracy"] < 85
daily_metrics["latency_alert"] = daily_metrics["avg_latency_ms"] > 140
daily_metrics["drift_alert"] = daily_metrics["avg_drift_score"] > 0.02

# Save daily summary CSV
daily_metrics.to_csv("output/daily_metrics.csv", index=False)

# Plot Accuracy
plt.figure(figsize=(10, 5))
plt.plot(daily_metrics["date"], daily_metrics["accuracy"], marker="o")
plt.axhline(85, linestyle="--")
plt.title("Daily Fraud Detection Accuracy")
plt.xlabel("Date")
plt.ylabel("Accuracy (%)")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig("output/accuracy_trend.png")
plt.close()

# Plot Latency
plt.figure(figsize=(10, 5))
plt.plot(daily_metrics["date"], daily_metrics["avg_latency_ms"], marker="o")
plt.axhline(140, linestyle="--")
plt.title("Daily Average Latency")
plt.xlabel("Date")
plt.ylabel("Latency (ms)")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig("output/latency_trend.png")
plt.close()

# Plot Drift
plt.figure(figsize=(10, 5))
plt.plot(daily_metrics["date"], daily_metrics["avg_drift_score"], marker="o")
plt.axhline(0.02, linestyle="--")
plt.title("Daily Drift Score")
plt.xlabel("Date")
plt.ylabel("Drift Score")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig("output/drift_trend.png")
plt.close()

# Dashboard summary
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].plot(daily_metrics["date"], daily_metrics["accuracy"], marker="o")
axes[0, 0].axhline(85, linestyle="--")
axes[0, 0].set_title("Accuracy (%)")
axes[0, 0].tick_params(axis="x", rotation=45)
axes[0, 0].grid(True)

axes[0, 1].plot(daily_metrics["date"], daily_metrics["error_rate"], marker="o")
axes[0, 1].set_title("Error Rate (%)")
axes[0, 1].tick_params(axis="x", rotation=45)
axes[0, 1].grid(True)

axes[1, 0].plot(daily_metrics["date"], daily_metrics["avg_latency_ms"], marker="o")
axes[1, 0].axhline(140, linestyle="--")
axes[1, 0].set_title("Average Latency (ms)")
axes[1, 0].tick_params(axis="x", rotation=45)
axes[1, 0].grid(True)

axes[1, 1].plot(daily_metrics["date"], daily_metrics["fraud_rate"], marker="o")
axes[1, 1].set_title("Fraud Rate (%)")
axes[1, 1].tick_params(axis="x", rotation=45)
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig("output/dashboard_summary.png")
plt.close()

# Create text report
with open("output/report.txt", "w") as f:
    f.write("FRAUD DETECTION MODEL MONITORING REPORT\n")
    f.write("=======================================\n\n")

    latest = daily_metrics.iloc[-1]

    f.write("LATEST METRICS\n")
    f.write("--------------\n")
    f.write(f"Date: {latest['date'].date()}\n")
    f.write(f"Accuracy: {latest['accuracy']:.2f}%\n")
    f.write(f"Error Rate: {latest['error_rate']:.2f}%\n")
    f.write(f"Fraud Rate: {latest['fraud_rate']:.2f}%\n")
    f.write(f"Average Confidence: {latest['avg_confidence']:.3f}\n")
    f.write(f"Average Latency: {latest['avg_latency_ms']:.2f} ms\n")
    f.write(f"Average Drift Score: {latest['avg_drift_score']:.4f}\n")
    f.write(f"Accuracy Change from Previous Day: {latest['accuracy_drop']:.2f}%\n\n")

    f.write("ALERT STATUS\n")
    f.write("------------\n")
    alert_found = False

    if latest["accuracy_alert"]:
        f.write("- Accuracy below threshold (85%)\n")
        alert_found = True
    if latest["latency_alert"]:
        f.write("- Latency above threshold (140 ms)\n")
        alert_found = True
    if latest["drift_alert"]:
        f.write("- Drift score above threshold (0.02)\n")
        alert_found = True
    if latest["sudden_drop_alert"]:
        f.write("- Sudden drop in accuracy detected\n")
        alert_found = True

    if not alert_found:
        f.write("- No alerts detected\n")

    f.write("\nINTERPRETATION\n")
    f.write("--------------\n")
    f.write("This report monitors fraud detection model health using daily performance, fraud-rate, latency, and drift metrics. ")
    f.write("Alerts indicate possible degradation in model reliability or shifts in transaction behavior that may require investigation.\n")

print("Monitoring complete. Outputs saved in /output")