"""Erode Stormwater & Flood Risk Explorer - Flask backend.

Prediction is ONLY made when the user submits input (rainfall + lat/lon).
Terrain inputs for slope, flow accumulation, low-elevation flag and drainage come from the
civil team's real QGIS rasters/polygons (gis.py). Elevation, impervious surface and soil
infiltration are not available from QGIS yet, so they still come from the nearest synthetic
sample points. Rainfall comes from the user (later: from the rain-gauge hardware).
"""
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, request, Response
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sqlalchemy import select, desc, func
from database import (SessionLocal, TrainingSample, RainfallReading, Prediction, VerifiedFloodObservation,
                      init_database, load_training_frame, iso, engine)
import gis

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "erode_synthetic_flood_dataset.csv"
GEO_DIR = ROOT / "data" / "geojson"

TERRAIN = [
    "elevation_m_synthetic", "slope_deg_synthetic", "flow_accumulation_synthetic",
    "distance_to_drainage_m_synthetic", "drainage_candidate_synthetic",
    "low_elevation_flag_synthetic", "impervious_surface_fraction_synthetic",
    "soil_infiltration_mm_per_hr_synthetic",
]
RAIN = ["rainfall_1h_mm_synthetic", "rainfall_24h_mm_synthetic", "antecedent_rainfall_3d_mm_synthetic"]
FEATURES = TERRAIN + RAIN
TARGET = "flood_occurrence_synthetic"
CENTER = [(gis.BOUNDS[0][0] + gis.BOUNDS[1][0]) / 2, (gis.BOUNDS[0][1] + gis.BOUNDS[1][1]) / 2]
FROM_SYNTH = ["elevation_m_synthetic", "impervious_surface_fraction_synthetic", "soil_infiltration_mm_per_hr_synthetic"]

app = Flask(__name__)
init_database(DATA_PATH)
df = load_training_frame()

# ---- train model (labels are synthetic) ----
X_train, X_test, y_train, y_test = train_test_split(
    df[FEATURES], df[TARGET], test_size=0.2, random_state=42, stratify=df[TARGET])
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)
demo_accuracy = float(accuracy_score(y_test, model.predict(X_test)))
importance = sorted(zip(FEATURES, model.feature_importances_), key=lambda t: -t[1])

# ---- nearest synthetic samples (only for inputs QGIS does not provide) ----
LAT0 = df["latitude"].mean()
M_LAT, M_LON = 111_320.0, 111_320.0 * np.cos(np.radians(LAT0))
to_xy = lambda lat, lon: np.column_stack([np.asarray(lat) * M_LAT, np.asarray(lon) * M_LON])
nn = NearestNeighbors(n_neighbors=4).fit(to_xy(df["latitude"], df["longitude"]))

def band(p): return "Low" if p < 0.35 else ("Moderate" if p < 0.65 else "High")

def terrain_rows(lats, lons):
    """Model terrain inputs for many points (DataFrame) + per-point GIS info."""
    dist, idx = nn.kneighbors(to_xy(lats, lons))
    w = 1.0 / np.maximum(dist, 1.0); w /= w.sum(axis=1, keepdims=True)
    rows = {c: (df[c].values[idx] * w).sum(axis=1) for c in FROM_SYNTH}
    g = [gis.at(a, b) for a, b in zip(lats, lons)]
    rows["slope_deg_synthetic"] = [x["slope"] for x in g]
    rows["flow_accumulation_synthetic"] = [gis.flow_to_training_scale(x["flow_raw"], df["flow_accumulation_synthetic"]) for x in g]
    rows["distance_to_drainage_m_synthetic"] = [x["dist_drain"] for x in g]
    rows["drainage_candidate_synthetic"] = [x["drain"] for x in g]
    rows["low_elevation_flag_synthetic"] = [x["low"] for x in g]
    return pd.DataFrame(rows), g, dist[:, 0], df["sample_id"].values[idx[:, 0]]

@app.route("/")
def home():
    return render_template("index.html")

def db_count(model_cls):
    with SessionLocal() as session:
        return int(session.query(func.count(model_cls.id)).scalar() or 0)

@app.route("/api/info")
def info():
    return jsonify({
        "center": CENTER, "bounds": gis.BOUNDS, "samples": int(len(df)), "rasters": gis.RASTERS,
        "demo_accuracy": round(demo_accuracy * 100, 1),
        "importance": [{"feature": f.replace("_synthetic", ""), "value": round(float(v) * 100, 1)} for f, v in importance[:6]],
        "cell": [(gis.BOUNDS[1][0] - gis.BOUNDS[0][0]) / gis.H, (gis.BOUNDS[1][1] - gis.BOUNDS[0][1]) / gis.W],
        "database": {"training_samples": db_count(TrainingSample), "predictions": db_count(Prediction),
                     "rainfall_readings": db_count(RainfallReading), "engine": "PostgreSQL" if engine.dialect.name == "postgresql" else "SQLite (local development)"},
    })

def parse_rain(d):
    try:
        lat, lon, r1 = float(d["latitude"]), float(d["longitude"]), float(d["rainfall_1h"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("latitude, longitude and rainfall_1h are required numbers.")
    r24, ant = d.get("rainfall_24h"), d.get("antecedent_3d")
    r24 = float(r24) if r24 not in (None, "") else r1   # assumption: at least the last hour
    ant = float(ant) if ant not in (None, "") else 0.0
    if not (-90 <= lat <= 90 and -180 <= lon <= 180): raise ValueError("Latitude/longitude out of range.")
    if r1 < 0 or r24 < 0 or ant < 0: raise ValueError("Rainfall cannot be negative.")
    if r24 < r1: raise ValueError("24-hour rainfall cannot be less than 1-hour rainfall.")
    return lat, lon, r1, r24, ant

@app.route("/api/predict", methods=["POST"])
def predict():
    """Single-location prediction. A rain-gauge device can POST the same JSON here."""
    try:
        lat, lon, r1, r24, ant = parse_rain(request.get_json(force=True, silent=True) or {})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    T, g, dist, sid = terrain_rows([lat], [lon])
    T[RAIN[0]], T[RAIN[1]], T[RAIN[2]] = r1, r24, ant
    p = float(model.predict_proba(T[FEATURES])[0, 1]); t = T.iloc[0]; g = g[0]
    result = {
        "latitude": lat, "longitude": lon, "probability": round(p * 100, 1), "risk": band(p),
        "inputs": {"rainfall_1h": r1, "rainfall_24h": r24, "antecedent_3d": ant},
        "gis": {
            "slope": round(g["slope"], 2), "flow_accumulation": round(g["flow_raw"], 1),
            "distance_to_drainage": round(g["dist_drain"], 0), "drainage_candidate": bool(g["drain"]),
            "low_elevation": bool(g["low"]), "priority_area": g["priority"]},
        "synthetic": {
            "elevation": round(float(t[FROM_SYNTH[0]]), 1), "impervious": round(float(t[FROM_SYNTH[1]]) * 100),
            "infiltration": round(float(t[FROM_SYNTH[2]]), 1)},
        "outside_area": not g["inside"],
        "status": "Prototype using synthetic training labels - not a validated flood forecast",
    }
    payload = request.get_json(force=True, silent=True) or {}
    source = str(payload.get("source", "website"))[:30]
    device_id = str(payload.get("device_id", ""))[:120] or None
    with SessionLocal() as session:
        reading = RainfallReading(latitude=lat, longitude=lon, rainfall_1h_mm=r1,
                                  rainfall_24h_mm=r24, antecedent_3d_mm=ant,
                                  source=source, device_id=device_id)
        session.add(reading)
        session.flush()
        prediction = Prediction(latitude=lat, longitude=lon, rainfall_1h_mm=r1,
                                rainfall_24h_mm=r24, antecedent_3d_mm=ant,
                                probability_pct=result["probability"], risk=result["risk"],
                                rainfall_reading_id=reading.id, details_json=result)
        session.add(prediction)
        session.commit()
        result["prediction_id"] = prediction.id
        result["predicted_at"] = iso(prediction.predicted_at)
    return jsonify(result)

@app.route("/api/history")
def prediction_history():
    """Return recent saved point predictions; limit is capped to avoid unbounded queries."""
    try:
        limit = max(1, min(int(request.args.get("limit", 50)), 200))
    except ValueError:
        limit = 50
    with SessionLocal() as session:
        rows = session.scalars(select(Prediction).order_by(desc(Prediction.predicted_at)).limit(limit)).all()
        return jsonify([{
            "id": x.id, "predicted_at": iso(x.predicted_at), "latitude": x.latitude,
            "longitude": x.longitude, "rainfall_1h": x.rainfall_1h_mm,
            "rainfall_24h": x.rainfall_24h_mm, "antecedent_3d": x.antecedent_3d_mm,
            "probability": x.probability_pct, "risk": x.risk,
        } for x in rows])

@app.route("/api/health")
def health():
    try:
        with SessionLocal() as session:
            session.execute(select(1))
        return jsonify({"status": "ok", "database": engine.dialect.name})
    except Exception:
        app.logger.exception("Database health check failed")
        return jsonify({"status": "error", "database": "unavailable"}), 503

@app.route("/api/simulate", methods=["POST"])
def simulate():
    """One rainfall scenario applied to every 30 m raster cell of the study area."""
    d = request.get_json(force=True, silent=True) or {}
    try:
        _, _, r1, r24, ant = parse_rain({**d, "latitude": CENTER[0], "longitude": CENTER[1]})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    cells = gis.grid_cells()
    lats = [c[2] for c in cells]; lons = [c[3] for c in cells]
    T, g, _, _ = terrain_rows(lats, lons)
    T[RAIN[0]], T[RAIN[1]], T[RAIN[2]] = r1, r24, ant
    p = model.predict_proba(T[FEATURES])[:, 1]
    bands = pd.Series(p).map(band)
    return jsonify({
        "cells": [[round(a, 6), round(b, 6), round(float(c) * 100, 1), int(x["priority"])] for a, b, c, x in zip(lats, lons, p, g)],
        "counts": {k: int((bands == k).sum()) for k in ("Low", "Moderate", "High")},
        "mean_probability": round(float(p.mean()) * 100, 1),
        "inputs": {"rainfall_1h": r1, "rainfall_24h": r24, "antecedent_3d": ant},
    })

@app.route("/api/layers")
def layers():
    return jsonify(sorted(p.name for p in GEO_DIR.glob("*") if p.suffix.lower() in (".geojson", ".json")))

@app.route("/geojson/<path:name>")
def geojson(name):
    """Serve QGIS vector layers, converting EPSG:32643 -> lon/lat automatically."""
    path = (GEO_DIR / name).resolve()
    if GEO_DIR.resolve() not in path.parents or not path.exists():
        return jsonify({"error": "not found"}), 404
    return Response(json.dumps(gis.reproject_geojson(json.loads(path.read_text()))), mimetype="application/json")

@app.route("/api/raster/<name>.png")
def raster(name):
    png = gis.raster_png(name)
    return Response(png, mimetype="image/png") if png else (jsonify({"error": "unknown raster"}), 404)

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
