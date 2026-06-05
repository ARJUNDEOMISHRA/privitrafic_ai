# ═══════════════════════════════════════════════════════════════════
# Layer 5 — Final Decision Engine
# Produces: Safety Score, Privacy Risk, Recommendations
# ═══════════════════════════════════════════════════════════════════

import time
import json
from datetime import datetime


class DecisionEngine:
    """Final decision engine that produces human-readable outputs.

    Takes BNS engine output and produces:
    - Safety score (%)
    - Privacy risk (%)
    - Actionable recommendations
    - Zone-level status reports
    """

    ZONES = ["Zone A", "Zone B", "Zone C", "Zone D"]

    def __init__(self):
        self.decision_log = []

    def make_decision(self, layer1, layer2, layer3, bns_output, zone="Zone A"):
        """Produce a final decision report."""
        timestamp = datetime.now().isoformat()

        # ── Traffic status ─────────────────────────
        congestion = layer2["random_forest"]["prediction"]
        accident_risk = round(
            layer2["xgboost"]["accident_probability"] * 100, 1)

        # ── Privacy status ─────────────────────────
        face_hidden = layer1.get("anonymized", True)
        plate_hidden = layer1.get("anonymized", True)

        # ── BNN insight ────────────────────────────
        bnn = layer3["bnn"]

        # ── Build decision ─────────────────────────
        decision = {
            "timestamp": timestamp,
            "zone": zone,
            "safety_score": bns_output["safety_score_pct"],
            "privacy_risk": bns_output["privacy_risk_pct"],
            "congestion_level": congestion.upper(),
            "accident_risk": accident_risk,
            "bnn_prediction": {
                "accident_probability": round(
                    bnn["accident_probability"] * 100, 1),
                "confidence": round(bnn["confidence"] * 100, 1),
                "uncertainty": bnn["uncertainty"],
            },
            "privacy_status": {
                "face_hidden": face_hidden,
                "plate_hidden": plate_hidden,
                "compliant": True,
            },
            "anomaly_detected": layer2["isolation_forest"]["anomaly_detected"],
            "lstm_forecast": layer3["lstm"]["prediction"],
            "recommendations": bns_output["recommendations"],
            "is_emergency": bns_output["is_emergency"],
            "risk_level": bns_output["safety_rule"]["risk_level"],
        }

        self.decision_log.append(decision)
        return decision

    def get_history(self, limit=50):
        """Return recent decision history."""
        return self.decision_log[-limit:]
