# -*- coding: utf-8 -*-
"""
Gulf Airports dust_mass vs Time — Colored by Airport (Scatter)

- Reads MERRA-2 CSVs with columns: time, latitude, longitude, dust_mass
- For each airport, picks the nearest grid cell per timestamp (within MAX_KM)
- Plots ALL airports together on one figure as colored scatter (no heatmap, no contours)

Matplotlib only. One figure. Robust datetime parsing (day-first).
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import ScalarFormatter
from tqdm import tqdm

# ===================== USER SETTINGS =====================
DATA_DIR = r"/Volumes/My Passport/MODIS Data"   # <-- CHANGE THIS
FILE_PREFIX = "MERRA2_400"                # set "" to include all .csv
MAX_KM = 60.0                             # km; set None to accept nearest cell regardless of distance

# Airports to include (ICAO: (lat, lon))
AIRPORTS = {
    "OMAA": (24.4338, 54.6481),  # Abu Dhabi Intl
    "OMDB": (25.2532, 55.3657),  # Dubai Intl
    "OERK": (24.9578, 46.6988),  # Riyadh King Khalid
    "OEJN": (21.6796, 39.1565),  # Jeddah King Abdulaziz
}

# Scatter appearance
MARKERS = ["o", "^", "s", "D", "P", "X"]  # cycles if more airports
POINT_SIZE = 8
POINT_ALPHA = 0.35
Y_PAD_FRAC = 0.05  # 5% headroom on y-axis
# ========================================================


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in km."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def parse_datetime_series(time_series: pd.Series) -> pd.Series:
    """Fast explicit parse, fallback to dayfirst=True."""
    try:
        dt = pd.to_datetime(time_series, format="%d/%m/%Y %H:%M", utc=True, errors="coerce")
        if dt.notna().mean() > 0.8:
            return dt
    except Exception:
        pass
    return pd.to_datetime(time_series, utc=True, errors="coerce", dayfirst=True)


def _scan_files(data_dir: str):
    pattern = os.path.join(data_dir, "*.csv")
    return sorted(
        f for f in glob.glob(pattern)
        if os.path.basename(f).startswith(FILE_PREFIX) or FILE_PREFIX == ""
    )


def _collect_for_airport(files, icao: str, alat: float, alon: float) -> pd.DataFrame:
    """
    For one airport: across all files, pick nearest grid cell per timestamp.
    Returns columns: [datetime, dust_mass, dist_km, airport]
    """
    rows = []
    for f in tqdm(files, desc=f"{icao} CSVs"):
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"[{icao} skip] {os.path.basename(f)}: {e}")
            continue

        required = {"time", "latitude", "longitude", "dust_mass"}
        if not required.issubset(df.columns):
            continue

        df = df.dropna(subset=["time", "latitude", "longitude", "dust_mass"])
        if df.empty:
            continue

        df = df.copy()
        df["datetime"] = parse_datetime_series(df["time"])
        df = df.dropna(subset=["datetime"])
        if df.empty:
            continue

        df["dist_km"] = haversine_km(
            df["latitude"].values, df["longitude"].values, alat, alon
        )

        nearest = df.loc[df.groupby("datetime")["dist_km"].idxmin()]
        if MAX_KM is not None:
            nearest = nearest[nearest["dist_km"] <= MAX_KM]

        if not nearest.empty:
            tmp = nearest[["datetime", "dust_mass", "dist_km"]].copy()
            tmp["airport"] = icao
            rows.append(tmp)

    if not rows:
        return pd.DataFrame(columns=["datetime", "dust_mass", "dist_km", "airport"])

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values("datetime").drop_duplicates(subset=["datetime", "airport"])
    return out


def collect_all_airports(data_dir: str) -> pd.DataFrame:
    """Collect and concatenate data for all airports."""
    files = _scan_files(data_dir)
    if not files:
        print("No CSV files found. Check DATA_DIR or FILE_PREFIX.")
        return pd.DataFrame(columns=["datetime", "dust_mass", "dist_km", "airport"])

    all_air = []
    for icao, (lat, lon) in AIRPORTS.items():
        df_air = _collect_for_airport(files, icao, lat, lon)
        if not df_air.empty:
            all_air.append(df_air)

    if not all_air:
        return pd.DataFrame(columns=["datetime", "dust_mass", "dist_km", "airport"])

    return pd.concat(all_air, ignore_index=True)


def _datetime_series_for_mpl(dt_series: pd.Series) -> np.ndarray:
    """UTC → naive → numpy datetime for matplotlib (future-proof)."""
    s = pd.to_datetime(dt_series, utc=True, errors="coerce")
    s = s.dt.tz_convert(None)  # drop timezone for matplotlib
    # Wrap to avoid FutureWarning on pandas .to_pydatetime()
    return mdates.date2num(np.array(s.dt.to_pydatetime()))


def plot_airports_scatter_colored(df_all: pd.DataFrame):
    """
    Scatter plot colored by airport (one legend). No heatmap, no contours.
    """
    if df_all.empty:
        print("Nothing to plot (no points found near the specified airports).")
        return

    # Prepare y-limits to capture everything with a small margin
    y_all = (pd.to_numeric(df_all["dust_mass"], errors="coerce").to_numpy())*1e9
    y_all = y_all[np.isfinite(y_all)]
    if y_all.size == 0:
        print("All dust_mass values are non-finite.")
        return
    y_min = float(np.nanmin(y_all))
    y_max = float(np.nanmax(y_all))
    pad = (y_max - y_min) * Y_PAD_FRAC if y_max > y_min else (0.05 * (y_max if y_max > 0 else 1.0))
    y_lo = max(0.0, y_min - pad)  # dust_mass should be ≥ 0
    y_hi = y_max + pad

    fig, ax = plt.subplots(figsize=(14, 5), dpi=140)

    # Colors from default cycle; markers per-airport for clarity
    color_cycle = plt.rcParams.get("axes.prop_cycle").by_key()["color"]

    legend_entries = []
    for i, (icao, _coords) in enumerate(AIRPORTS.items()):
        dfi = df_all[df_all["airport"] == icao]
        if dfi.empty:
            continue

        xi = _datetime_series_for_mpl(dfi["datetime"])
        yi = pd.to_numeric(dfi["dust_mass"], errors="coerce").to_numpy()
        ok = np.isfinite(xi) & np.isfinite(yi)
        xi, yi = xi[ok], yi[ok]
        if xi.size == 0:
            continue

        col = color_cycle[i % len(color_cycle)]
        mark = MARKERS[i % len(MARKERS)]
        ax.scatter(
            xi, yi,
            s=POINT_SIZE, alpha=POINT_ALPHA,
            marker=mark, color=col, edgecolors="none",
            label=icao
        )

    # Axes/labels/legend
    airport_list = ", ".join(AIRPORTS.keys())
    ax.set_title(f"{airport_list} dust_mass vs Time (Colored by Airport)")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("dust_mass (\micro m$^{-3}$)")

    ax.set_ylim(y_lo, y_hi)

    ax.xaxis_date()
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    sf = ScalarFormatter(useMathText=True)
    sf.set_powerlimits((-2, 2))
    ax.yaxis.set_major_formatter(sf)

    ax.grid(True, which="both", alpha=0.25)
    ax.legend(title="Airport", loc="upper right", frameon=False)

    fig.tight_layout()
    plt.show()


def main():
    print(f"Scanning: {DATA_DIR}")
    df_all = collect_all_airports(DATA_DIR)
    if not df_all.empty:
        print("Samples per airport:")
        print(df_all.groupby("airport")["dust_mass"].count())
        print(df_all[["dust_mass", "dist_km"]].describe())
    plot_airports_scatter_colored(df_all)


if __name__ == "__main__":
    main()
