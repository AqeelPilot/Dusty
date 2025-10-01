
"""
Gulf Airports dust_mass vs Time — Colored by Airport (Scatter, in µg/m³)

- Reads MERRA-2 CSVs with columns: time, latitude, longitude, dust_mass (kg/m³)
- For each airport, picks the nearest grid cell per timestamp (within MAX_KM)
- Plots ALL airports together on one figure as colored scatter
- Y-axis in micrograms per cubic meter (µg/m³), no scientific notation, full range

Matplotlib only. One figure. Robust datetime parsing (day-first).
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm

# ===================== USER SETTINGS =====================
DATA_DIR = r"E:\MODIS Data"   # <-- CHANGE THIS
FILE_PREFIX = "MERRA2_400"                # set "" to include all .csv
MAX_KM = 60.0                             # km; set None to accept nearest cell regardless of distance

# Airports to include (ICAO: (lat, lon))
AIRPORTS = {
    "OMAA": (24.4338, 54.6481),
    "OMDB": (25.2532, 55.3657),
    "OERK": (24.9578, 46.6988),
    "OEJN": (21.6796, 39.1565),
}

# Scatter appearance
MARKERS = ["o", "^", "s", "D", "P", "X"]  # cycles if more airports
POINT_SIZE = 14
POINT_ALPHA = 0.55
Y_PAD_FRAC = 0.05  # 5% headroom on y-axis

# Units
UG_PER_KG = 1_000_000_000  # 1e9
Y_UNIT_FACTOR = UG_PER_KG
Y_UNIT_LABEL = r"$\mu$g m$^{-3}$"
# ========================================================


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def parse_datetime_series(time_series: pd.Series) -> pd.Series:
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

        df["dist_km"] = haversine_km(df["latitude"].values, df["longitude"].values, alat, alon)

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
    s = pd.to_datetime(dt_series, utc=True, errors="coerce")
    s = s.dt.tz_convert(None)                    # drop timezone for matplotlib
    return mdates.date2num(np.array(s.dt.to_pydatetime()))  # future-proof


def _nice_ylim(y_min: float, y_max: float, pad_frac: float):
    if not np.isfinite(y_min) or not np.isfinite(y_max):
        return 0.0, 1.0
    if y_max <= y_min:
        y_max = y_min + 1.0
    span = y_max - y_min
    lo = max(0.0, y_min - span * pad_frac)
    hi = y_max + span * pad_frac
    # round up hi to a nice step (1/2/5 * 10^k)
    steps = np.array([1, 2, 5])
    k = np.floor(np.log10(max(hi, 1e-9))) - 1
    base = 10 ** k
    best = steps[np.argmin(np.abs(hi / (steps * base) - np.round(hi / (steps * base))))]
    step = best * base
    hi = step * np.ceil(hi / step)
    return lo, hi


def plot_airports_scatter_colored(df_all: pd.DataFrame):
    if df_all.empty:
        print("Nothing to plot (no points found near the specified airports).")
        return

    # Prepare y-limits in µg/m³ (full range, small headroom)
    y_all = pd.to_numeric(df_all["dust_mass"], errors="coerce").to_numpy() * Y_UNIT_FACTOR
    y_all = y_all[np.isfinite(y_all)]
    if y_all.size == 0:
        print("All dust_mass values are non-finite.")
        return
    y_lo, y_hi = _nice_ylim(float(np.nanmin(y_all)), float(np.nanmax(y_all)), Y_PAD_FRAC)

    fig, ax = plt.subplots(figsize=(14, 5), dpi=140)
    color_cycle = plt.rcParams.get("axes.prop_cycle").by_key()["color"]

    for i, (icao, _coords) in enumerate(AIRPORTS.items()):
        dfi = df_all[df_all["airport"] == icao]
        if dfi.empty:
            continue

        xi = _datetime_series_for_mpl(dfi["datetime"])
        yi = pd.to_numeric(dfi["dust_mass"], errors="coerce").to_numpy() * Y_UNIT_FACTOR
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

    airport_list = ", ".join(AIRPORTS.keys())
    ax.set_title(f"{airport_list} dust_mass vs Time (Colored by Airport)")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel(f"dust_mass ({Y_UNIT_LABEL})")

    ax.set_ylim(y_lo, y_hi)

    # X axis formatting
    ax.xaxis_date()
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    # Y axis: plain numbers (no ×10^3)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))

    ax.grid(True, which="both", alpha=0.25)
    ax.legend(title="Airport", loc="upper right", frameon=False)

    fig.tight_layout()
    plt.show()


def compute_and_print_yearly_means(df_all: pd.DataFrame):
    """
    Print yearly dust_mass statistics per airport in µg/m³.
    Returns a DataFrame with columns: airport, year, mean_ugm3, median_ugm3, samples.
    """
    if df_all.empty:
        print("No data to summarize.")
        return pd.DataFrame(columns=["airport","year","mean_ugm3","median_ugm3","samples"])

    # Parse datetime (UTC), convert units to µg/m³, and extract year
    dt = pd.to_datetime(df_all["datetime"], utc=True, errors="coerce")
    df = df_all.assign(
        datetime=dt,
        year=dt.dt.year,
        dust_mass_ugm3=pd.to_numeric(df_all["dust_mass"], errors="coerce") * Y_UNIT_FACTOR
    ).dropna(subset=["datetime","dust_mass_ugm3","airport"])

    if df.empty:
        print("All rows became NaN after datetime/unit conversion.")
        return pd.DataFrame(columns=["airport","year","mean_ugm3","median_ugm3","samples"])

    # Group and summarize
    g = (
        df.groupby(["airport","year"])
          .agg(mean_ugm3=("dust_mass_ugm3","mean"),
               median_ugm3=("dust_mass_ugm3","median"),
               samples=("dust_mass_ugm3","size"))
          .reset_index()
          .sort_values(["airport","year"])
    )

    print("\nYearly dust_mass statistics by airport (µg/m³):")
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(
            g.to_string(
                index=False,
                formatters={
                    "mean_ugm3":   lambda v: f"{v:,.1f}",
                    "median_ugm3": lambda v: f"{v:,.1f}",
                    "samples":     lambda v: f"{int(v)}",
                },
            )
        )
    return g


def main():
    print(f"Scanning: {DATA_DIR}")
    df_all = collect_all_airports(DATA_DIR)
    if not df_all.empty:
        print("Samples per airport:")
        print(df_all.groupby("airport")["dust_mass"].count())
        # Optional brief stats in µg/m³
        stats = df_all.assign(dust_mass_ugm3=pd.to_numeric(df_all["dust_mass"], errors="coerce") * Y_UNIT_FACTOR)
        print(stats[["dust_mass_ugm3", "dist_km"]].describe())

        # >>> NEW: yearly means per airport (µg/m³)
        compute_and_print_yearly_means(df_all)

    plot_airports_scatter_colored(df_all)


if __name__ == "__main__":
    main()
