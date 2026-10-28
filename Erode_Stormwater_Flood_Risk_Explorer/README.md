# Erode Stormwater & Flood Risk Explorer

A Flask academic prototype combining QGIS terrain layers with a Random Forest classifier and a persistent database. The included flood dataset and target labels are synthetic and must not be treated as a validated flood forecast.

## Three-tier architecture

1. **Frontend:** HTML/CSS/JavaScript + Leaflet map (`templates/index.html`).
2. **Backend:** Flask API, Random Forest model and QGIS raster/vector processing (`app.py`, `gis.py`).
3. **Database:** SQLAlchemy models (`database.py`) with PostgreSQL for deployment and SQLite as a local development fallback.

The browser communicates with Flask APIs; it does not connect directly to the database. A hosted deployment uses one shared PostgreSQL database, so people opening the public site from different computers see the same saved prediction history.

## Folder layout

```text
Erode_Stormwater_Flood_Risk_Explorer/
├── app.py
├── database.py
├── gis.py
├── requirements.txt
├── Procfile
├── README.md
├── templates/
│   └── index.html
└── data/
    ├── erode_synthetic_flood_dataset.csv
    ├── flood_explorer.db             # created automatically for local SQLite use
    ├── gis/
    │   ├── slope.tif
    │   ├── flow_accumulation.tif
    │   └── low_elevation_mask_.tif
    └── geojson/
        ├── Erode_Drainage_Candidate.geojson
        └── Erode_Priority_Areas.geojson
```

## Run locally in Windows / VS Code

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000. The first run creates `data/flood_explorer.db` and imports the included synthetic CSV into the database. Later runs reuse the saved records. Internet access is needed for the OpenStreetMap basemap and externally hosted Leaflet/font assets.

If PowerShell blocks activation, use Command Prompt and run:

```bat
.venv\Scripts\activate.bat
```

## Database tables

- `training_samples`: training rows imported from the included CSV. The model loads its training frame from this table.
- `rainfall_readings`: rainfall, location, time, source (`website` or `sensor`) and optional device ID submitted with point predictions.
- `predictions`: prediction probability, risk category, input values, timestamp and associated rainfall reading.
- `verified_flood_observations`: schema reserved for confirmed real-world flood observations. Add records only when supported by a reliable observation/source; predictions are not labels.

The app creates tables automatically. For a production project, use database migrations and a managed backup policy before making schema changes.

## APIs

### Point prediction (also saves a rainfall reading and prediction)

`POST /api/predict` with JSON:

```json
{
  "rainfall_1h": 12.5,
  "rainfall_24h": 20,
  "antecedent_3d": 35,
  "latitude": 11.3328,
  "longitude": 77.7245,
  "source": "sensor",
  "device_id": "rain-gauge-01"
}
```

`source` and `device_id` are optional. For a web form, `source` defaults to `website`. Protect this endpoint with authentication or a device API key before connecting publicly accessible hardware; do not embed a secret API key in browser JavaScript.

### Saved prediction history

- `GET /api/history?limit=20` returns recent point predictions saved by all users in the shared database (maximum 200 rows).
- `GET /api/info` includes database counts and whether the app is using PostgreSQL or local SQLite.
- `GET /api/health` checks database connectivity.

### Area simulation

`POST /api/simulate` calculates a rainfall scenario across the study area. It is not saved as an individual point prediction in this version.

## Deploy publicly with PostgreSQL on Railway

1. Put this entire project folder in a GitHub repository. Include the code, CSV, TIFF rasters and GeoJSON files; do not commit `.env`, passwords or local `*.db` files.
2. Create a project on [Railway](https://railway.com/) and deploy the repository as a web service.
3. Add a PostgreSQL database service to the same Railway project.
4. In the Flask web service, set `DATABASE_URL` to the PostgreSQL service's **private** connection URL/reference. Railway usually exposes a `DATABASE_URL` variable for the database service; use Railway's variable reference UI so credentials are not hard-coded.
5. Set the start command to `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 180` if the platform does not automatically use the included `Procfile`.
6. Deploy and test `/api/health`, `/api/info`, a point prediction and `/api/history` on the public domain.
7. Keep the database private to the app service. Enable backups if your hosting plan supports them. Never expose the database password in frontend code or source control.

Railway documentation: [PostgreSQL](https://docs.railway.com/databases/postgresql) and [deployment](https://docs.railway.com/).

The public website and the database are hosted services, so the project does not need to run on your laptop for other people to access it. All code and GIS assets must be included in the deployment repository; PostgreSQL stores records, not the Python/HTML/GIS application files. If you later upload user-managed files or large datasets, use object/file storage rather than putting the files themselves in ordinary database rows.

## Important limitations and security

- `erode_synthetic_flood_dataset.csv` has synthetic target labels. The model's test accuracy is measured against synthetic labels, not real-world flood outcomes.
- New website/sensor readings and predictions are persisted, but they do **not** automatically retrain the model. Real model training requires confirmed flood/non-flood observations with matching input features, data-quality checks and an explicit retraining/evaluation workflow.
- `verified_flood_observations` is separate from predictions to avoid treating a model's own output as ground truth.
- The public prediction API currently has no user login, rate limiting or device authentication. Add these before accepting untrusted internet traffic or connecting hardware.
- Keep PostgreSQL private; use environment variables for secrets and configure regular backups.
- Slope, flow accumulation, low-elevation mask, drainage-candidate polygons and priority-area polygons are sourced from the included QGIS files. Elevation, impervious-surface fraction and soil infiltration are still estimated from nearby synthetic samples.
- Locations outside the QGIS raster extent use nearest-edge terrain cells and are flagged as outside the study area. Do not use this academic prototype as an official warning or for emergency decisions.
