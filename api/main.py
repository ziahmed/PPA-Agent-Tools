import os
import json
import zipfile
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any

app = FastAPI(title="Boltchip PPA Prediction API")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZIP_PATH = os.path.join(BASE_DIR, "models.zip")
MANIFEST_PATH = os.path.join(BASE_DIR, "models", "model_manifest.json")

# Auto-extract models if they don't exist
if not os.path.exists(os.path.join(BASE_DIR, "models", "v3_production")):
    if os.path.exists(ZIP_PATH):
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(BASE_DIR)

MODEL_DIR = os.path.join(BASE_DIR, "models", "v3_production")

with open(MANIFEST_PATH) as f:
    manifest = json.load(f)

models = {}
for label in ["area", "setup", "dynamic", "leakage"]:
    model_file = manifest[label]["file"]
    models[label] = joblib.load(os.path.join(MODEL_DIR, model_file))

class FeatureInput(BaseModel):
    design_id: str = Field(default="unnamed_design")
    features: Dict[str, float]

@app.post("/api/v1/predict")
def predict(payload: FeatureInput):
    try:
        df = pd.DataFrame([payload.features])
        
        FEATURES_ALL = [
            'x_clock_period_ns', 'x_synth_cell_area_um2', 'x_synth_leaf_instances',
            'x_synth_seq_instances', 'x_synth_comb_instances', 'x_seq_ratio',
            'x_area_per_instance', 'x_synth_wns_ps', 'x_synth_tns_ps',
            'x_synth_violating_paths', 'x_clock_skew_ns', 'x_clock_insertion_latency_ns',
            'x_core_util_h_pct', 'x_core_util_v_pct', 'x_buffer_area_um2',
            'x_inverter_area_um2', 'x_comb_area_um2', 'x_flop_area_um2', 'x_total_overflow'
        ]
        FEATURES_SETUP = [f for f in FEATURES_ALL if not f.startswith('x_max_')]
        FEATURES_LEAKAGE = [f for f in FEATURES_ALL if f != 'x_synth_cell_area_um2']
        
        missing = [f for f in FEATURES_ALL if f not in payload.features]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing features: {missing}")

        def _predict(label, feats):
            raw = models[label]["model"].predict(df[feats])[0]
            if models[label].get("log_transform"):
                return float(np.expm1(raw))
            return float(raw)

        preds = {
            "area_um2": max(_predict("area", FEATURES_ALL), 10.0),
            "setup_wns_ns": _predict("setup", FEATURES_SETUP),
            "dynamic_power_mw": max(_predict("dynamic", FEATURES_ALL), 0.0),
            "leakage_power_mw": max(_predict("leakage", FEATURES_LEAKAGE), 0.0)
        }

        return {
            "status": "success",
            "design_id": payload.design_id,
            "predictions": preds,
            "metadata": {
                "model_version": manifest["version"],
                "accuracy_bounds": {
                    "area_mape": manifest["area"]["mape"],
                    "setup_r2": manifest["setup"]["r2"],
                    "dynamic_mape": manifest["dynamic"]["mape"],
                    "leakage_mape": manifest["leakage"]["mape"]
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
