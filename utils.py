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

from scipy.ndimage import convolve as ndimage_convolve



# ── Local Moran's I on a 2D raster (queen contiguity) ─────────────────────────
def local_morans_i(arr, weight_matrix, n_perms=999, alpha=0.05, seed=42):
    """
    Compute Local Moran's I for each cell of a 2D raster.

    Parameters
    ----------
    arr      : 2D float array (NaN = missing)
    weight_matrix : 3x3 array of spatial weights
    n_perms  : number of permutations for the pseudo p-value
    alpha    : significance level

    Returns
    -------
    li       : Local Moran's I values (same shape as arr)
    cluster  : integer map   1=HH  2=LL  3=HL  4=LH  0=ns  NaN=missing
    p_val    : pseudo p-values
    """
    valid = np.isfinite(arr)
    flat  = arr[valid]
    mu, sigma = flat.mean(), flat.std()
    z     = np.where(valid, (arr - mu) / sigma, np.nan)
    z0    = np.where(valid, z, 0.0)          # NaN → 0 for convolution

    #then normalize the weights so that they sum to 1 (row-standardization)
    W = weight_matrix / weight_matrix.sum()

    # you can try to change this to see if that changes the results
    n_neigh   = ndimage_convolve(valid.astype(float), W, mode="constant", cval=0)
    lag_sum   = ndimage_convolve(z0, W, mode="constant", cval=0)
    spatial_lag = np.where(n_neigh > 0, lag_sum / n_neigh, np.nan)
    li = np.where(valid, z * spatial_lag, np.nan)

    # ── Conditional permutation test ─────────────────────────────────────────
    # For each permutation, shuffle z values globally and recompute spatial lag.
    # p-value = fraction of |Li_perm| >= |Li|  (two-sided)
    rng     = np.random.default_rng(seed)
    flat_z  = z[valid]
    p_count = np.zeros(arr.shape, dtype=float)

    for _ in range(n_perms):
        z_perm       = np.full(arr.shape, np.nan)
        z_perm[valid] = rng.permutation(flat_z)
        z0_perm       = np.where(valid, z_perm, 0.0)
        lag_perm      = np.where(
            n_neigh > 0,
            ndimage_convolve(z0_perm, W, mode="constant", cval=0) / n_neigh,
            np.nan,
        )
        li_perm = np.where(valid, z * lag_perm, np.nan)
        # accumulate where |li_perm| >= |li|
        with np.errstate(invalid="ignore"):
            p_count += np.where(
                valid & np.isfinite(li_perm),
                (np.abs(li_perm) >= np.abs(li)).astype(float),
                0.0,
            )

    p_val = np.where(valid, p_count / n_perms, np.nan)

    # ── Classify ─────────────────────────────────────────────────────────────
    cluster = np.full(arr.shape, np.nan)
    sig = valid & (p_val <= alpha)

    cluster[sig & (z > 0) & (spatial_lag > 0)] = 1   # HH – hotspot
    cluster[sig & (z < 0) & (spatial_lag < 0)] = 2   # LL – coldspot
    cluster[sig & (z > 0) & (spatial_lag < 0)] = 3   # HL – outlier (high, low nbrs)
    cluster[sig & (z < 0) & (spatial_lag > 0)] = 4   # LH – outlier (low, high nbrs)
    cluster[valid & ~sig]                        = 0  # not significant

    return li, cluster, p_val


def local_lee_l(arr_x, arr_y, weight_matrix, n_perms=999, alpha=0.05, seed=42):
    """
    Local Lee's L bivariate spatial autocorrelation (Lee 2001).

    L_i = 0.5 * (z_x,i * lag(z_y)_i  +  z_y,i * lag(z_x)_i)

    Quadrant classification (significant cells only):
        1 = HH  z_x > 0, lag(z_y) > 0  → both co-cluster high
        2 = LL  z_x < 0, lag(z_y) < 0  → both co-cluster low  (vulnerability hotspot)
        3 = HL  z_x > 0, lag(z_y) < 0  → spatial mismatch
        4 = LH  z_x < 0, lag(z_y) > 0  → spatial mismatch
        0 = not significant
    """
    valid = np.isfinite(arr_x) & np.isfinite(arr_y)

    def _standardise(arr):
        flat = arr[valid]
        return np.where(valid, (arr - flat.mean()) / flat.std(), np.nan)

    zx = _standardise(arr_x)
    zy = _standardise(arr_y)

    W = weight_matrix / weight_matrix.sum()
    valid_f = valid.astype(float)
    n_neigh = ndimage_convolve(valid_f, W, mode="constant", cval=0)

    def _lag(z):
        z0 = np.where(valid, z, 0.0)
        lag_sum = ndimage_convolve(z0, W, mode="constant", cval=0)
        return np.where(n_neigh > 0, lag_sum / n_neigh, np.nan)

    lag_zx = _lag(zx)
    lag_zy = _lag(zy)

    L = np.where(valid, 0.5 * (zx * lag_zy + zy * lag_zx), np.nan)

    # ── Permutation test: fix z_x, shuffle z_y ───────────────────────────────
    rng    = np.random.default_rng(seed)
    flat_zy = zy[valid]
    p_count = np.zeros(arr_x.shape, dtype=float)

    for _ in range(n_perms):
        zy_perm        = np.full(arr_x.shape, np.nan)
        zy_perm[valid] = rng.permutation(flat_zy)
        lag_zy_perm    = _lag(zy_perm)
        zy_perm0       = np.where(valid, zy_perm, 0.0)
        L_perm = np.where(valid,
                          0.5 * (zx * lag_zy_perm + zy_perm * lag_zx), np.nan)
        with np.errstate(invalid="ignore"):
            p_count += np.where(
                valid & np.isfinite(L_perm),
                (np.abs(L_perm) >= np.abs(L)).astype(float),
                0.0,
            )

    p_val = np.where(valid, p_count / n_perms, np.nan)

    # ── Classify by sign of z_x and lag(z_y) ─────────────────────────────────
    cluster = np.full(arr_x.shape, np.nan)
    sig = valid & (p_val <= alpha)

    cluster[sig & (zx > 0) & (lag_zy > 0)] = 1   # HH
    cluster[sig & (zx < 0) & (lag_zy < 0)] = 2   # LL  ← double burden
    cluster[sig & (zx > 0) & (lag_zy < 0)] = 3   # HL
    cluster[sig & (zx < 0) & (lag_zy > 0)] = 4   # LH
    cluster[valid & ~sig]                   = 0   # not significant

    return L, cluster, p_val, lag_zy