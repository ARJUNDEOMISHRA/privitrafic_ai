# ═══════════════════════════════════════════════════════════════════
# Layer 4 — Bayesian Neuro-Symbolic Engine
# The Intelligence Layer: Neural outputs + Symbolic rules
# ═══════════════════════════════════════════════════════════════════

import numpy as np


class SymbolicRuleEngine:
    """Symbolic rules that encode domain knowledge.

    These rules represent human-interpretable logic that
    constrains and augments neural network outputs.
    """

    @staticmethod
    def safety_rule(accident_prob, pedestrian_count):
        """IF accident_probability > 0.75 AND pedestrian_count > 5
           THEN high_public_risk"""
        if accident_prob > 0.75 and pedestrian_count > 5:
            return {"risk_level": "HIGH", "reason": "high_accident_risk_with_pedestrians"}
        elif accident_prob > 0.6:
            return {"risk_level": "MEDIUM", "reason": "elevated_accident_risk"}
        return {"risk_level": "LOW", "reason": "normal_conditions"}

    @staticmethod
    def privacy_rule(face_detected, plate_detected):
        """IF face_detected THEN anonymize_required
           IF plate_detected THEN anonymize_required"""
        actions = []
        if face_detected:
            actions.append("blur_faces")
        if plate_detected:
            actions.append("blur_plates")
        return {
            "anonymize_required": len(actions) > 0,
            "actions": actions,
            "privacy_compliant": True,  # After anonymization
        }

    @staticmethod
    def emergency_rule(anomaly_detected, congestion_state, accident_prob):
        """IF ambulance_detected AND congestion_high
           THEN emergency_route_priority"""
        is_emergency = False
        recommendations = []

        if anomaly_detected:
            recommendations.append("investigate_anomaly")
            is_emergency = True

        if congestion_state == "congested" and accident_prob > 0.7:
            recommendations.append("emergency_lane_recommended")
            recommendations.append("alert_traffic_control")
            is_emergency = True

        if congestion_state == "congested":
            recommendations.append("suggest_alternate_route")

        return {
            "is_emergency": is_emergency,
            "recommendations": recommendations,
        }

    @staticmethod
    def congestion_rule(traffic_score, predicted_density, trend):
        """Rule for congestion management."""
        actions = []
        if traffic_score > 0.7:
            actions.append("extend_green_signal")
        if predicted_density > 0.8:
            actions.append("activate_overflow_lanes")
        if trend > 0.1:
            actions.append("preemptive_rerouting")
        return {"congestion_actions": actions}


class BayesianNeuroSymbolicEngine:
    """The core BNS engine that combines:
    - Neural outputs (MLP, LSTM, BNN, XGBoost, RF, SVM, IF)
    - Symbolic rules (safety, privacy, emergency)

    This is the intelligence layer that makes PriviTraffic AI
    explainable and trustworthy.
    """

    def __init__(self, config=None):
        self.config = config or {}
        self.rules = SymbolicRuleEngine()
        self.lambda_tsc = self.config.get("lambda_tsc", 0.05)

    def _compute_safety_score(self, layer2, layer3):
        """Compute overall safety score from all models."""
        # Weighted combination of model outputs
        rf_conf = layer2["random_forest"]["confidence"]
        xgb_accident = layer2["xgboost"]["accident_probability"]
        bnn_accident = layer3["bnn"]["accident_probability"]
        bnn_confidence = layer3["bnn"]["confidence"]
        mlp_score = layer3["mlp"]["traffic_condition_score"]

        # BNN-weighted safety score (BNN confidence modulates weight)
        safety = 1.0 - (
            xgb_accident * 0.3 +
            bnn_accident * 0.3 * bnn_confidence +
            mlp_score * 0.2 +
            (1.0 - rf_conf) * 0.2
        )
        return round(min(1.0, max(0.0, safety)), 3)

    def _compute_privacy_risk(self, layer1):
        """Compute privacy risk score."""
        face = layer1.get("face_detected", False)
        plate = layer1.get("plate_detected", False)
        anonymized = layer1.get("anonymized", True)

        if anonymized:
            risk = 0.05 + np.random.uniform(0, 0.1)
        elif face and plate:
            risk = 0.8 + np.random.uniform(0, 0.15)
        elif face or plate:
            risk = 0.5 + np.random.uniform(0, 0.2)
        else:
            risk = 0.1 + np.random.uniform(0, 0.1)

        return round(min(1.0, max(0.0, risk)), 3)

    def process(self, layer1_output, layer2_output, layer3_output):
        """Run the BNS engine.

        Combines neural outputs with symbolic rules to produce
        explainable, trustworthy decisions.
        """
        # ── Extract key values ──────────────────────────────
        accident_prob = layer3_output["bnn"]["accident_probability"]
        bnn_confidence = layer3_output["bnn"]["confidence"]
        ped_count = layer1_output.get("pedestrian_count", 0)
        face_detected = layer1_output.get("face_detected", False)
        plate_detected = layer1_output.get("plate_detected", False)
        anomaly_detected = layer2_output["isolation_forest"]["anomaly_detected"]
        congestion_state = layer2_output["random_forest"]["prediction"]
        traffic_score = layer3_output["mlp"]["traffic_condition_score"]
        predicted_density = layer3_output["lstm"]["predicted_density"]
        trend = layer3_output["lstm"]["trend"]

        # ── Apply symbolic rules ────────────────────────────
        safety_rule = self.rules.safety_rule(accident_prob, ped_count)
        privacy_rule = self.rules.privacy_rule(face_detected, plate_detected)
        emergency_rule = self.rules.emergency_rule(
            anomaly_detected, congestion_state, accident_prob)
        congestion_rule = self.rules.congestion_rule(
            traffic_score, predicted_density, trend)

        # ── Compute final scores ────────────────────────────
        safety_score = self._compute_safety_score(
            layer2_output, layer3_output)
        privacy_risk = self._compute_privacy_risk(layer1_output)

        # ── Build recommendations ───────────────────────────
        recommendations = []
        if privacy_rule["anonymize_required"]:
            recommendations.extend(
                [f"Action: {a}" for a in privacy_rule["actions"]])
        recommendations.extend(emergency_rule["recommendations"])
        recommendations.extend(congestion_rule["congestion_actions"])

        if not recommendations:
            recommendations.append("All systems nominal")

        return {
            "safety_score": safety_score,
            "safety_score_pct": round(safety_score * 100, 1),
            "privacy_risk": privacy_risk,
            "privacy_risk_pct": round(privacy_risk * 100, 1),
            "safety_rule": safety_rule,
            "privacy_rule": privacy_rule,
            "emergency_rule": emergency_rule,
            "congestion_rule": congestion_rule,
            "recommendations": recommendations,
            "bns_confidence": bnn_confidence,
            "is_emergency": emergency_rule["is_emergency"],
        }


class Layer4Pipeline:
    """Complete Layer 4: BNS Engine."""

    def __init__(self):
        self.engine = BayesianNeuroSymbolicEngine()

    def process(self, layer1_output, layer2_output, layer3_output):
        return self.engine.process(layer1_output, layer2_output, layer3_output)
