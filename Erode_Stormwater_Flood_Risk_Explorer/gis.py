"""Real GIS inputs from the civil team's QGIS outputs (EPSG:32643, UTM zone 43N).

Reads slope / flow-accumulation / low-elevation rasters and the drainage-candidate and
priority-area polygons, and answers: "what are the terrain values at this lat/lon?"
Pure Python + numpy/scipy/Pillow/tifffile (no GDAL or pyproj needed).
"""
import io, json, math
from pathlib import Path
import numpy as np
import tifffile
from PIL import Image, ImageDraw
from scipy import ndimage

GIS_DIR = Path(__file__).resolve().parent / "data" / "gis"
GEO_DIR = Path(__file__).resolve().parent / "data" / "geojson"

# ---------- WGS84 <-> UTM 43N (Krueger series, mm accuracy) ----------
_a, _f, _k0, _lon0 = 6378137.0, 1 / 298.257223563, 0.9996, math.radians(75.0)
_n = _f / (2 - _f)
_A = _a / (1 + _n) * (1 + _n**2 / 4 + _n**4 / 64)
_alpha = [_n/2 - 2*_n**2/3 + 5*_n**3/16, 13*_n**2/48 - 3*_n**3/5, 61*_n**3/240]
_beta = [_n/2 - 2*_n**2/3 + 37*_n**3/96, _n**2/48 + _n**3/15, 17*_n**3/480]
_delta = [2*_n - 2*_n**2/3 - 2*_n**3, 7*_n**2/3 - 8*_n**3/5, 56*_n**3/15]

def latlon_to_utm(lat, lon):
    phi, lam = math.radians(lat), math.radians(lon) - _lon0
    e = math.sqrt(_f * (2 - _f))
    t = math.sinh(math.atanh(math.sin(phi)) - e * math.atanh(e * math.sin(phi)))
    xi_p = math.atan2(t, math.cos(lam)); eta_p = math.atanh(math.sin(lam) / math.sqrt(1 + t * t))
    xi = xi_p + sum(_alpha[j] * math.sin(2*(j+1)*xi_p) * math.cosh(2*(j+1)*eta_p) for j in range(3))
    eta = eta_p + sum(_alpha[j] * math.cos(2*(j+1)*xi_p) * math.sinh(2*(j+1)*eta_p) for j in range(3))
    return 500000 + _k0 * _A * eta, _k0 * _A * xi

def utm_to_latlon(E, N):
    xi, eta = N / (_k0 * _A), (E - 500000) / (_k0 * _A)
    xp = xi - sum(_beta[j] * math.sin(2*(j+1)*xi) * math.cosh(2*(j+1)*eta) for j in range(3))
    ep = eta - sum(_beta[j] * math.cos(2*(j+1)*xi) * math.sinh(2*(j+1)*eta) for j in range(3))
    chi = math.asin(math.sin(xp) / math.cosh(ep))
    lat = chi + sum(_delta[j] * math.sin(2*(j+1)*chi) for j in range(3))
    return math.degrees(lat), math.degrees(_lon0 + math.atan2(math.sinh(ep), math.cos(xp)))

def reproject_geojson(gj):
    """Return GeoJSON in lon/lat. If it declares EPSG:32643 convert it; otherwise leave as is."""
    crs = (((gj.get("crs") or {}).get("properties") or {}).get("name") or "")
    if "32643" not in crs:
        return gj
    def conv(c):
        if isinstance(c[0], (int, float)):
            lat, lon = utm_to_latlon(c[0], c[1]); return [lon, lat]
        return [conv(x) for x in c]
    out = {k: v for k, v in gj.items() if k != "crs"}
    out["features"] = [{**f, "geometry": {**f["geometry"], "coordinates": conv(f["geometry"]["coordinates"])}} for f in gj["features"]]
    return out

# ---------- rasters ----------
def _read(name):
    path = next(GIS_DIR.glob(name))
    with tifffile.TiffFile(path) as t:
        p = t.pages[0]; arr = p.asarray().astype("float64")
        sx, sy = p.tags["ModelPixelScaleTag"].value[:2]
        x0, y0 = p.tags["ModelTiepointTag"].value[3:5]
    return arr, x0, y0, sx, sy

def _fill(arr, bad):
    """Replace invalid cells with the nearest valid cell."""
    if not bad.any(): return arr
    idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
    return arr[tuple(idx)]

slope, X0, Y0, RES, RES_Y = _read("slope.tif")
flow, *_ = _read("flow_accumulation*.tif")
low, *_ = _read("low_elevation*.tif")
H, W = slope.shape
BAD = (slope < -9000) | np.isnan(slope) | np.isnan(flow) | (low > 1e30) | np.isnan(low)
slope = _fill(np.where(BAD, np.nan, slope), BAD)
flow = _fill(np.where(BAD, np.nan, flow), BAD)
low = _fill(np.where(BAD, np.nan, low), BAD)
flow = np.maximum(flow, 1.0)
_flow_sorted = np.sort(flow.ravel())

def _rasterize(filename, only_dn=1):
    """Polygon GeoJSON (UTM) -> boolean grid. Uses DN==1 polygons only (DN==0 = background)."""
    path = GEO_DIR / filename
    mask = np.zeros((H, W), bool)
    if not path.exists(): return mask
    gj = json.loads(path.read_text()); S = 8
    img = Image.new("L", (W * S, H * S), 0); d = ImageDraw.Draw(img)
    for f in gj["features"]:
        if f["properties"].get("DN", 1) != only_dn: continue
        g = f["geometry"]; polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        for poly in polys:
            d.polygon([((x - X0) / RES * S, (Y0 - y) / RES_Y * S) for x, y in poly[0]], fill=255)
    return np.asarray(img.resize((W, H), Image.BOX)) > 127

drain = _rasterize("Erode_Drainage_Candidate.geojson")
priority = _rasterize("Erode_Priority_Areas.geojson")
dist_drain = (ndimage.distance_transform_edt(~drain) * RES) if drain.any() else np.full((H, W), 500.0)
dist_drain = np.minimum(dist_drain, 500.0)  # same cap as the training data

BOUNDS = [list(utm_to_latlon(X0, Y0 - H * RES_Y)), list(utm_to_latlon(X0 + W * RES, Y0))]  # [[S,W],[N,E]]

def cell(lat, lon):
    E, N = latlon_to_utm(lat, lon)
    c, r = int((E - X0) // RES), int((Y0 - N) // RES_Y)
    return r, c, (0 <= r < H and 0 <= c < W)

def flow_to_training_scale(v, train_flow):
    """Real flow accumulation spans 1-2363 but the model was trained on 0.1-73: map by rank."""
    q = np.searchsorted(_flow_sorted, v) / len(_flow_sorted)
    return float(np.quantile(train_flow, min(max(q, 0), 1)))

def at(lat, lon):
    r, c, inside = cell(lat, lon)
    r, c = min(max(r, 0), H - 1), min(max(c, 0), W - 1)
    return {"slope": float(slope[r, c]), "flow_raw": float(flow[r, c]), "low": float(low[r, c] > 0.5),
            "drain": float(drain[r, c]), "dist_drain": float(dist_drain[r, c]),
            "priority": bool(priority[r, c]), "inside": inside, "row": r, "col": c}

def grid_cells():
    """Centre lat/lon of every raster cell plus its row/col."""
    out = []
    for r in range(H):
        for c in range(W):
            lat, lon = utm_to_latlon(X0 + (c + .5) * RES, Y0 - (r + .5) * RES_Y)
            out.append((r, c, lat, lon))
    return out

# ---------- PNG overlays for the map ----------
def _ramp(t, stops):
    t = np.clip(t, 0, 1); xs = np.linspace(0, 1, len(stops)); st = np.array(stops, float)
    return np.stack([np.interp(t, xs, st[:, i]) for i in range(3)], -1)

def raster_png(name):
    if name == "slope":
        rgb = _ramp(slope / max(slope.max(), 1e-6), [(68,1,84),(49,104,142),(33,145,140),(94,201,98),(253,231,37)]); a = np.full((H, W), 170)
        label = "Slope (degrees)"
    elif name == "flow_accumulation":
        rgb = _ramp(np.log10(flow) / np.log10(flow.max()), [(235,245,255),(120,180,230),(20,100,200),(8,48,107)]); a = np.where(flow < 3, 0, 190)
        label = "Flow accumulation (log scale)"
    elif name == "low_elevation":
        rgb = np.broadcast_to(np.array([214, 72, 95], float), (H, W, 3)); a = np.where(low > .5, 150, 0)
        label = "Low-elevation mask"
    else:
        return None
    img = Image.fromarray(np.dstack([rgb, a]).astype("uint8"), "RGBA").resize((W * 6, H * 6), Image.NEAREST)
    buf = io.BytesIO(); img.save(buf, "PNG"); return buf.getvalue()

RASTERS = [{"id": "slope", "label": "Slope (°)"}, {"id": "flow_accumulation", "label": "Flow accumulation"}, {"id": "low_elevation", "label": "Low-elevation mask"}]
