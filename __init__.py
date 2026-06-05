# ═══════════════════════════════════════════════════════════════════
# PriviTraffic AI — Global Configuration
# Privacy-Preserving Smart Traffic Intelligence
# ═══════════════════════════════════════════════════════════════════

import os

# ── Paths ────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ── Reproducibility ─────────────────────────────────────────────
SEED = 42

# ── Layer 1: Computer Vision ────────────────────────────────────
CV_CONFIG = {
    "input_shape": (224, 224, 3),
    "detection_classes": ["car", "bus", "bike", "pedestrian"],
    "confidence_threshold": 0.5,
    "face_blur_kernel": (31, 31),
    "plate_blur_kernel": (31, 31),
}

# ── Layer 2: Classical ML ───────────────────────────────────────
RF_CONFIG = {
    "n_estimators": 200,
    "max_depth": 15,
    "random_state": SEED,
    "n_jobs": -1,
    "traffic_states": ["normal", "moderate", "congested"],
}

XGBOOST_CONFIG = {
    "n_estimators": 300,
    "max_depth": 8,
    "learning_rate": 0.05,
    "random_state": SEED,
    "use_label_encoder": False,
    "eval_metric": "logloss",
}

SVM_CONFIG = {
    "kernel": "rbf",
    "C": 1.0,
    "gamma": "scale",
    "random_state": SEED,
}

ISOLATION_FOREST_CONFIG = {
    "n_estimators": 150,
    "contamination": 0.05,
    "random_state": SEED,
}

# ── Layer 3: Deep Learning ──────────────────────────────────────
MLP_CONFIG = {
    "hidden_layers": [128, 64, 32],
    "dropout": 0.3,
    "learning_rate": 1e-3,
    "epochs": 30,
    "batch_size": 256,
}

LSTM_CONFIG = {
    "units": [128, 64],
    "dropout": 0.3,
    "sequence_length": 20,
    "learning_rate": 1e-3,
    "epochs": 30,
    "batch_size": 128,
}

BNN_CONFIG = {
    "hidden_layers": [128, 64],
    "mc_samples": 50,
    "kl_weight": None,   # set to 1/N_train at runtime
    "learning_rate": 1e-3,
    "epochs": 30,
    "batch_size": 256,
}

# ── Layer 4: BNS Engine ─────────────────────────────────────────
BNS_CONFIG = {
    "lambda_tsc": 0.05,
    "safety_threshold": 0.75,
    "privacy_threshold": 0.5,
    "emergency_threshold": 0.8,
}

# ── Layer 5: Dashboard ──────────────────────────────────────────
DASHBOARD_CONFIG = {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": True,
}
