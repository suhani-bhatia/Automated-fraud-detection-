# Automated-fraud-detection
Automated pipeline for fraud detection monitoring, tracking model performance metrics (accuracy, error rate, latency) and visualizing anomalies through dashboards.

# Automated Fraud Detection Monitoring and Analytics Pipeline

## Overview
This project implements an automated monitoring pipeline for evaluating fraud detection system performance on transactional data. It focuses on tracking key performance metrics, identifying anomalies, and visualizing trends through dashboards.

The system enables structured monitoring of model behavior, helping detect performance degradation and operational issues in near real-time.

---

## Dataset
- ~280K+ transaction records
- Binary classification: Fraud / Non-Fraud
- Highly imbalanced dataset (real-world scenario)

---

## Key Features

### 1. Data Processing
- Preprocessing of transactional data
- Handling missing values and inconsistencies
- Feature preparation for analysis

### 2. Monitoring Metrics
The pipeline computes and tracks:
- Accuracy (~95%)
- Error Rate (~5%)
- Fraud Rate trends
- Latency (~100–150 ms)

### 3. Performance Monitoring
- Daily metric tracking
- Trend analysis over time
- Detection of anomalies such as:
  - Sudden drops in accuracy
  - Spikes in latency

### 4. Visualization
- Interactive Tableau dashboard
- Key views:
  - Accuracy trend
  - Error rate trend
  - Latency monitoring
  - Fraud rate behavior

---

## Tech Stack
- **Python**
- Pandas, NumPy
- Matplotlib
- Tableau (for dashboarding)

---


---

## How It Works

1. **Data Preparation**
   - Clean and preprocess raw transaction data

2. **Metric Computation**
   - Calculate accuracy, error rate, fraud rate, latency

3. **Monitoring Pipeline**
   - Track metrics over time
   - Store results in structured format

4. **Visualization**
   - Load output into Tableau
   - Analyze trends and anomalies

---

## Key Insights
- Model maintains ~95% accuracy under normal conditions
- Detectable performance drops under anomaly scenarios
- Latency trends highlight system efficiency changes
- Fraud rate fluctuations provide behavioral insights

---

## Future Improvements
- Real-time streaming integration
- Alerting system for anomaly detection
- Model retraining triggers
- Deployment as a monitoring dashboard

---

## Author
Suhani Bhatia

## Project Structure
