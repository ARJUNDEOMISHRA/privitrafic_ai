# 🚦🔐 PriviTraffic AI — Architecture & Implementation Guide

This document explains the complete architecture, data flow, and implementation details for the PriviTraffic AI system built for the hackathon.

## 1. Where Does the Data Come From?

In a real-world deployment, PriviTraffic AI is designed to ingest:
1. **Live CCTV Feeds:** RTSP streams from city traffic cameras.
2. **Environmental Sensors:** Datasets like the **NOAA weather dataset** (which you had in your research `94866099999.csv`).

### Hackathon Prototype Data Strategy
To make this project highly presentable for the hackathon without relying on live hardware or static, boring CSV files, **we built a realistic Monte Carlo Simulation Engine inside the pipeline.** 
- Instead of static data, `layer1_cv.py` uses Poisson distributions (`rng.poisson(lam=12)`) to generate highly realistic, fluctuating traffic patterns (cars, buses, pedestrians).
- It injects simulated noise and randomness to trigger the anomaly detection models naturally.
- This allows the dashboard to look "ALIVE" and process real-time streams during your pitch without needing a physical camera setup.

---

## 2. What Have We Built? (The 5-Layer Architecture)

We built a complete, end-to-end Python backend with a stunning frontend dashboard. Here is exactly what each layer does:

### Layer 1: Computer Vision & Privacy (`layer1_cv.py`)
- **What it does:** Simulates a TensorFlow CNN object detector (MobileNet/YOLO) and uses OpenCV.
- **Privacy First:** Before any data goes to the ML models, this layer "detects" faces and number plates and mathematically applies a Gaussian Blur.
- **Output:** Anonymized frames and raw counts (vehicle count, pedestrian count, lane density).

### Layer 2: Classical Machine Learning (`layer2_ml.py`)
- **Random Forest:** Classifies the overall traffic state (Normal, Moderate, Congested) based on density thresholds.
- **XGBoost:** Predicts the probability of an accident and congestion escalation.
- **SVM:** Binary classification to flag if current activity is "suspicious."
- **Isolation Forest:** The anomaly detector. It looks for highly irregular data points (e.g., zero cars but 100% density, simulating a sensor failure or wrong-way driver).

### Layer 3: Deep Learning & Uncertainty (`layer3_dl.py`)
- **MLP (Multi-Layer Perceptron):** Fuses structured features into a single continuous traffic condition score.
- **LSTM:** Maintains a rolling history of the last 20 frames to predict future congestion trends (e.g., "Heavy traffic in 15 mins").
- **Bayesian Neural Network (BNN):** This is the star of the show. Using **Monte Carlo Dropout**, it runs multiple forward passes per frame to calculate not just the accident probability, but the **Epistemic Uncertainty** (confidence) of that prediction.

### Layer 4: Bayesian Neuro-Symbolic (BNS) Engine (`layer4_bns.py`)
- **What it does:** It takes the probabilistic "guesses" from the neural networks and constrains them using strict, human-readable **Symbolic Rules**.
- **Example:** `IF (BNN_Accident_Prob > 75%) AND (Pedestrian_Count > 5) THEN Trigger HIGH_RISK_ALERT`.
- This layer makes the AI **Explainable**, solving the "black box" problem of standard deep learning.

### Layer 5: Decision Engine & Dashboard (`layer5_decision.py` & `dashboard/`)
- The Decision Engine formats all data from Layers 1-4 into a clean JSON response.
- The Flask server (`app.py`) serves this data to a stunning, dark-themed, glassmorphic UI.
- The dashboard uses `Chart.js` to render the BNS metrics and simulated CCTV feeds in real-time, complete with PDF report generation.

---

## 3. How is the Data Processed? (Execution Flow)

When the pipeline runs (`pipeline.py`), here is the exact step-by-step execution:

1. **Ingestion:** The pipeline triggers `run_all_zones()`. For each zone, Layer 1 generates a simulated frame of data representing that exact second in time.
2. **Feature Extraction:** Layer 1 outputs a dictionary (e.g., `{"car_count": 12, "lane_density": 0.6}`).
3. **Parallel ML Processing:** This dictionary is fed simultaneously to Layer 2 (Scikit-Learn/XGBoost) and Layer 3 (TensorFlow).
4. **Uncertainty Calculation:** Inside Layer 3, the BNN runs 50 rapid dropout simulations on the features to calculate the standard deviation (uncertainty) of the accident risk.
5. **Symbolic Fusion:** All outputs converge in Layer 4. The BNS engine reads the BNN's uncertainty. If uncertainty is too high, the Symbolic Engine overrides the neural network, preventing false alarms.
6. **Delivery:** The final JSON packet is fetched by the `dashboard.js` script via the `/api/data` endpoint every 3 seconds. The JS script pushes the new data into the Chart.js arrays, updates the UI DOM elements, and triggers the CSS animations (like the red anomaly pulse or the CCTV scanning line).
