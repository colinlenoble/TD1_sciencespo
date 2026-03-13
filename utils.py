"""utils.py — reusable helpers for the UHI & Climate Justice notebook."""

import io
import base64

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import folium
from folium.plugins import DualMap

try:
    from PIL import Image as _PILImage
    _PIL_OK = True
except ImportError:
    _PIL_OK = False


# ── Heatwave detection ─────────────────────────────────────────────────────────

def get_heatwaves(df, temp_col="temperature", time_col="time",
                  q_pic=0.99, q_start=0.95, q_interrupt=0.90):
    """
    Quantile-based heatwave detection.

    A heatwave event is triggered when the daily mean temperature exceeds
    *Spic* (q_pic quantile).  The event is back-extended to contiguous days
    above *Sstart* (q_start) and ends when temperature drops below
    *Sinterup* (q_interrupt).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns *time_col* and *temp_col*.
    temp_col : str
    time_col : str
    q_pic : float
        Quantile threshold that triggers a new event (default 0.99).
    q_start : float
        Quantile used for back-extension and intensity normalisation (default 0.95).
    q_interrupt : float
        Quantile below which an ongoing event ends (default 0.90).

    Returns
    -------
    heatwaves : pd.DataFrame
        Columns: start_date, end_date, duration_days, max_temp, intensity, year.
    thresholds : dict
        {'Spic': float, 'Sstart': float, 'Sinterup': float}
    """
    d = df[[time_col, temp_col]].copy()
    d[time_col] = pd.to_datetime(d[time_col])
    d = d.sort_values(time_col).reset_index(drop=True)

    Spic     = d[temp_col].quantile(q_pic)
    Sstart   = d[temp_col].quantile(q_start)
    Sinterup = d[temp_col].quantile(q_interrupt)

    temps = d[temp_col].to_numpy(float)
    dates = d[time_col].to_numpy()
    n     = len(d)
    denom = Spic - Sstart if Spic != Sstart else np.nan

    events = []
    i = 0
    while i < n:
        if temps[i] > Spic:
            # Back-extend while above Sstart
            s = i
            while s > 0 and temps[s - 1] > Sstart:
                s -= 1

            # Forward-extend until below interruption threshold
            e = i
            while e + 1 < n and temps[e + 1] >= Sinterup:
                e += 1

            seg = temps[s : e + 1]
            intensity = (
                np.nan if (np.isnan(denom) or denom == 0)
                else float(np.sum(np.maximum(seg - Sstart, 0) / denom))
            )

            events.append({
                "start_date":    pd.Timestamp(dates[s]),
                "end_date":      pd.Timestamp(dates[e]),
                "duration_days": e - s + 1,
                "max_temp":      float(np.max(seg)),
                "intensity":     intensity,
            })
            i = e + 1
        else:
            i += 1

    hw = pd.DataFrame(events)
    if not hw.empty:
        hw["year"] = hw["start_date"].dt.year

    return hw, {"Spic": Spic, "Sstart": Sstart, "Sinterup": Sinterup}


# ── Raster → base64 PNG helpers ────────────────────────────────────────────────

def _to_png_url(rgba_uint8, flip_y=False):
    """Convert an (H, W, 4) uint8 RGBA array to a base64 data-URL PNG."""
    arr = rgba_uint8[::-1] if flip_y else rgba_uint8
    if _PIL_OK:
        img = _PILImage.fromarray(arr, "RGBA")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
    else:
        fig = plt.figure(figsize=(arr.shape[1] / 100, arr.shape[0] / 100), dpi=100)
        ax  = fig.add_axes([0, 0, 1, 1])
        ax.imshow(arr.astype(float) / 255, interpolation="nearest", aspect="auto")
        ax.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", transparent=True,
                    bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


def _mask_to_png(mask, color_hex, alpha=0.72, flip_y=False):
    """Render a boolean mask as a solid-colour PNG overlay."""
    from matplotlib.colors import to_rgba
    r, g, b, _ = to_rgba(color_hex)
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[mask.astype(bool)] = [int(r * 255), int(g * 255), int(b * 255), int(alpha * 255)]
    return _to_png_url(rgba, flip_y=flip_y)


def _arr_to_png(arr, cmap_name, alpha=0.72, flip_y=False):
    """Render a float array as a colourmap PNG overlay (NaNs are transparent)."""
    valid = np.isfinite(arr)
    vmin, vmax = np.nanmin(arr), np.nanmax(arr)
    norm    = (arr - vmin) / (vmax - vmin + 1e-12)
    cmap_fn = plt.get_cmap(cmap_name)
    rgba_f  = cmap_fn(np.where(valid, norm, 0.0))
    rgba    = (rgba_f * 255).astype(np.uint8)
    rgba[~valid, 3] = 0
    rgba[valid,  3] = int(alpha * 255)
    return _to_png_url(rgba, flip_y=flip_y)


# ── Spatial Explorer DualMap ───────────────────────────────────────────────────

def make_explorer_dualmap(left_specs, right_specs, bounds, center,
                          flip_y=False, zoom_start=12, basemap="CartoDB positron"):
    """
    Build a side-by-side Folium DualMap where each panel has its own
    independently toggled layer catalogue.

    Parameters
    ----------
    left_specs : list of (array_like, cmap_name: str, title: str)
        Layers for the **left** panel.  The first entry is shown by default.
    right_specs : list of (array_like, cmap_name: str, title: str)
        Layers for the **right** panel.  The first entry is shown by default.
    bounds : [[lat_min, lon_min], [lat_max, lon_max]]
        Geographic extent of the raster overlays.
    center : [lat, lon]
        Initial map centre.
    flip_y : bool
        Set True when the array row 0 corresponds to the southern edge
        (ascending y-axis).  For EPSG:4326 reprojections via rioxarray,
        this is usually False.
    zoom_start : int
        Initial zoom level (default 12).
    basemap : str
        Folium tile layer name (default "CartoDB positron").

    Returns
    -------
    folium.plugins.DualMap
    """
    # Pass tiles directly to DualMap — avoids duplicate tile layers that hide the background
    m = DualMap(location=center, zoom_start=zoom_start, tiles=basemap)

    _cache: dict = {}

    def _url(arr, cmap):
        key = (id(arr), cmap)
        if key not in _cache:
            _cache[key] = _arr_to_png(np.asarray(arr, dtype=float), cmap,
                                       flip_y=flip_y)
        return _cache[key]

    for i, (da, cmap, title) in enumerate(left_specs):
        folium.raster_layers.ImageOverlay(
            image=_url(da, cmap), bounds=bounds, name=title,
            opacity=1.0, show=(i == 0),
        ).add_to(m.m1)

    for i, (da, cmap, title) in enumerate(right_specs):
        folium.raster_layers.ImageOverlay(
            image=_url(da, cmap), bounds=bounds, name=title,
            opacity=1.0, show=(i == 0),
        ).add_to(m.m2)

    # Add LayerControls after all layers are populated on both panels
    folium.LayerControl(collapsed=False).add_to(m.m1)
    folium.LayerControl(collapsed=False).add_to(m.m2)

    return m
