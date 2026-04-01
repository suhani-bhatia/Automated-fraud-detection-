# Fraud Detection Model Monitoring System

This project uses a real-world credit card fraud dataset to simulate monitoring of a fraud detection model in a banking environment.

## Objective
To build an automated monitoring pipeline that tracks model performance and operational health over time.

## Features
- Uses real transaction fraud labels from a credit card fraud dataset
- Simulates model predictions for monitoring analysis
- Tracks:
  - Accuracy
  - Error rate
  - Fraud rate
  - Confidence
  - Latency
  - Drift score
- Generates:
  - Daily metrics CSV
  - Trend plots
  - Dashboard summary
  - Monitoring report

## Tools Used
- Python
- Pandas
- NumPy
- Matplotlib

## Workflow
Raw fraud dataset → preprocessing → monitoring dataset creation → metric calculation → alert generation → visualization → report output

## Output Files
- daily_metrics.csv
- accuracy_trend.png
- latency_trend.png
- drift_trend.png
- dashboard_summary.png
- report.txt