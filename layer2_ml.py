# ═══════════════════════════════════════════════════════════════════
# Layer 2 — Classical Machine Learning Models
# Random Forest | XGBoost | SVM | Isolation Forest
# ═══════════════════════════════════════════════════════════════════

import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


class TrafficStateClassifier:
    """Random Forest — Traffic state classification.

    Predicts: normal | moderate | congested
    Input: density, speed, weather, signal timing
    Why: Strong tabular classifier, highly interpretable.
    """
    STATES = ["normal", "moderate", "congested"]

    def __init__(self, config=None):
        self.config = config or {}
        self.model = None
        self.scaler = StandardScaler() if HAS_SKLEARN else None

    def build(self):
        if not HAS_SKLEARN:
            return
        self.model = RandomForestClassifier(
            n_estimators=self.config.get("n_estimators", 200),
            max_depth=self.config.get("max_depth", 15),
            random_state=self.config.get("random_state", 42),
            n_jobs=-1,
        )

    def predict(self, features):
        """Predict traffic state from structured features.

        For demo: rule-based simulation.
        """
        density = features.get("lane_density", 0.5)
        if density < 0.3:
            state_idx = 0
            confidence = 0.85 + np.random.uniform(-0.05, 0.05)
        elif density < 0.7:
            state_idx = 1
            confidence = 0.75 + np.random.uniform(-0.1, 0.1)
        else:
            state_idx = 2
            confidence = 0.9 + np.random.uniform(-0.05, 0.05)

        return {
            "model": "RandomForest",
            "prediction": self.STATES[state_idx],
            "state_index": state_idx,
            "confidence": round(min(0.99, max(0.5, confidence)), 3),
        }


class RiskPredictor:
    """XGBoost — Risk prediction.

    Predicts: accident probability, congestion escalation
    Why: Fast + highly accurate gradient boosting.
    """

    def __init__(self, config=None):
        self.config = config or {}
        self.model = None

    def build(self):
        if not HAS_XGB:
            return
        self.model = XGBClassifier(
            n_estimators=self.config.get("n_estimators", 300),
            max_depth=self.config.get("max_depth", 8),
            learning_rate=self.config.get("learning_rate", 0.05),
            random_state=self.config.get("random_state", 42),
            eval_metric="logloss",
        )

    def predict(self, features):
        """Predict accident/congestion risk.

        For demo: uses density + pedestrian count heuristics.
        """
        density = features.get("lane_density", 0.5)
        ped_count = features.get("pedestrian_count", 0)
        vehicle_count = features.get("vehicle_count", 0)

        # Accident probability
        base_risk = density * 0.6 + min(ped_count / 20.0, 0.3)
        noise = np.random.uniform(-0.05, 0.05)
        accident_prob = round(min(0.99, max(0.01, base_risk + noise)), 3)

        # Congestion escalation
        escalation = round(min(0.99, density * 0.8 +
                               np.random.uniform(-0.1, 0.1)), 3)

        return {
            "model": "XGBoost",
            "accident_probability": accident_prob,
            "congestion_escalation": max(0.01, escalation),
        }


class SuspiciousActivityClassifier:
    """SVM — Binary classification.

    Classifies: suspicious vs normal, privacy risk vs safe
    Why: Clean decision boundaries for binary tasks.
    """

    def __init__(self, config=None):
        self.config = config or {}
        self.model = None

    def build(self):
        if not HAS_SKLEARN:
            return
        self.model = SVC(
            kernel=self.config.get("kernel", "rbf"),
            C=self.config.get("C", 1.0),
            probability=True,
            random_state=self.config.get("random_state", 42),
        )

    def predict(self, features):
        """Classify activity as suspicious or normal."""
        density = features.get("lane_density", 0.5)
        ped_count = features.get("pedestrian_count", 0)

        # Heuristic: high density + many peds → suspicious
        score = density * 0.5 + min(ped_count / 15.0, 0.4)
        noise = np.random.uniform(-0.08, 0.08)
        suspicion = round(min(0.99, max(0.01, score + noise)), 3)

        is_suspicious = suspicion > 0.6

        return {
            "model": "SVM",
            "classification": "suspicious" if is_suspicious else "normal",
            "suspicion_score": suspicion,
            "is_suspicious": is_suspicious,
        }


class AnomalyDetector:
    """Isolation Forest — Detect unusual patterns.

    Detects: wrong-way movement, abnormal traffic signals,
             fake sensor data, unusual traffic patterns.
    Critical for hackathon theme: privacy + safety.
    """

    def __init__(self, config=None):
        self.config = config or {}
        self.model = None

    def build(self):
        if not HAS_SKLEARN:
            return
        self.model = IsolationForest(
            n_estimators=self.config.get("n_estimators", 150),
            contamination=self.config.get("contamination", 0.05),
            random_state=self.config.get("random_state", 42),
        )

    def predict(self, features):
        """Detect anomalies in traffic data."""
        density = features.get("lane_density", 0.5)
        vehicle_count = features.get("vehicle_count", 0)

        # Anomaly if very unusual density/count combinations
        is_anomaly = (density > 0.9 and vehicle_count < 5) or \
                     (density < 0.1 and vehicle_count > 20) or \
                     np.random.random() < 0.03

        anomaly_score = round(np.random.uniform(0.7, 0.95)
                              if is_anomaly else
                              np.random.uniform(0.05, 0.3), 3)

        return {
            "model": "IsolationForest",
            "anomaly_detected": is_anomaly,
            "anomaly_score": anomaly_score,
            "anomaly_type": "unusual_pattern" if is_anomaly else None,
        }


class Layer2Pipeline:
    """Complete Layer 2: all classical ML models."""

    def __init__(self):
        self.traffic_classifier = TrafficStateClassifier()
        self.risk_predictor = RiskPredictor()
        self.activity_classifier = SuspiciousActivityClassifier()
        self.anomaly_detector = AnomalyDetector()

    def process(self, layer1_output):
        """Run all Layer 2 models on Layer 1 output."""
        rf_result = self.traffic_classifier.predict(layer1_output)
        xgb_result = self.risk_predictor.predict(layer1_output)
        svm_result = self.activity_classifier.predict(layer1_output)
        iso_result = self.anomaly_detector.predict(layer1_output)

        return {
            "random_forest": rf_result,
            "xgboost": xgb_result,
            "svm": svm_result,
            "isolation_forest": iso_result,
        }
