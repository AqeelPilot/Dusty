#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Monthly Means & Anomaly Analysis for Major Middle East Airports (Simple)
------------------------------------------------------------------------
This standalone script:
  1) Uses your MERRA2AODProcessor to download + convert MERRA-2 daily files
     for a bounding box that covers the selected Middle East airports.
  2) Extracts a time series for each airport (nearest grid point).
  3) Computes monthly means per year (µg/m³) for each airport.
  4) Builds a multi-year monthly "climatology" (per-month average across years).
  5) Computes anomalies for each (year, month) vs climatology:
        - pct_anom = 100 * (value - clim_mean) / clim_mean
        - zscore   = (value - clim_mean) / clim_std
  6) Saves:
        - Per-year plots with all 5 airports as lines (monthly_means_<YEAR>.png)
        - Per-airport plots with all years as lines (monthly_means_<ICAO>_by_year.png)
        - Per-airport anomaly heatmaps (anomalies_<ICAO>_pct.png)
        - A summary CSV highlighting the peak month per airport per year.

Edit the USER INPUTS below and run:
    python monthly_means_airports_analysis.py
"""

# ================== USER INPUTS (EDIT THESE) ==================
START_YEAR = 2020
END_YEAR   = 2023

# Output directories
OUTPUT_DATA_DIR = r"C:\data\merra2_me"     # where NetCDF/CSVs will be cached
OUTPUT_FIG_DIR  = r"C:\plots\merra2_me"    # where PNGs/CSVs will be saved

# Airports to track (ICAO: (lat, lon))
AIRPORTS = {
    "OMAA": (24.433, 54.651),  # Abu Dhabi
    "OMDB": (25.253, 55.365),  # Dubai
    "OMSJ": (25.329, 55.517),  # Sharjah
    "OERK": (24.959, 46.705),  # Riyadh
    "OEJN": (21.679, 39.156),  # Jeddah
}

# Padding (deg) around airports to form a minimal bounding box for downloads
BBOX_PAD_DEG = 1.0

# Save an additional summary CSV?
SUMMARY_CSV = r"C:\plots\merra2_me\airport_monthly_summary.csv"  # set to None to skip
# =============================================================

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Import your downloader/converter class ---
try:
    from MERRA2AODProcessor import MERRA2AODProcessor
except Exception:
    try:
        from Dust_Storm_Modules import MERRA2AODProcessor
    except Exception as e:
        raise ImportError(
            "Could not import MERRA2AODProcessor. Place the class in "
            "'MERRA2AODProcessor.py' or 'Dust_Storm_Modules.py' next to this script."
        ) from e


def compute_bbox_from_airports(airports: dict[str, tuple[float, float]], pad_deg: float = 1.0):
    lats = [v[0] for v in airports.values()]
    lons = [v[1] for v in airports.values()]
    lat_min = min(lats) - pad_deg
    lat_max = max(lats) + pad_deg
    lon_min = min(lons) - pad_deg
    lon_max = max(lons) + pad_deg
    return lat_min, lat_max, lon_min, lon_max


def ensure_download_and_csv(start_year, end_year, bbox, output_dir):
    """Use your class to fetch and convert files (it will skip existing ones)."""
    lat_min, lat_max, lon_min, lon_max = bbox
    start_date = f"{start_year}-01-01"
    end_date   = f"{end_year}-12-31"
    proc = MERRA2AODProcessor(
        start_date=start_date,
        end_date=end_date,
        region_bounds=(lat_min, lat_max, lon_min, lon_max),
        output_dir=output_dir
    )
    proc.download_files()
    proc.convert_to_csv()


def nearest_value(arr: np.ndarray, target: float) -> float:
    """Return the array element nearest to 'target'."""
    idx = np.argmin(np.abs(arr - target))
    return float(arr[idx])


def collect_airport_timeseries(output_dir: str, airports: dict[str, tuple[float, float]],
                               start_year: int, end_year: int, var_col: str = "dust_mass"):
    """
    Read all CSVs and build a time series per airport by selecting the nearest
    (lat, lon) grid point for each timestamp.
    Returns dict: { ICAO: DataFrame(time, value[var_col]) }.
    """
    out_path = Path(output_dir)
    csv_files = sorted(out_path.glob("MERRA2_400.tavg1_2d_aer_Nx.*.csv"))

    if not csv_files:
        print("No CSV files found in OUTPUT_DATA_DIR. Check paths/permissions.")
        return {k: pd.DataFrame(columns=["time", var_col]) for k in airports.keys()}

    nearest_grid = None  # { ICAO: (lat_grid, lon_grid) }
    series_store = {k: [] for k in airports.keys()}

    for f in tqdm(csv_files, desc="Reading daily CSVs for airport extraction"):
        yyyy = None
        try:
            ymd = f.stem.split(".")[-1]   # 'YYYYMMDD'
            yyyy = int(ymd[:4])
        except Exception:
            pass
        if (yyyy is not None) and not (start_year <= yyyy <= end_year):
            continue

        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"  Skipping unreadable CSV: {f.name} ({e})")
            continue

        if df.empty or not {"time", "latitude", "longitude", var_col}.issubset(df.columns):
            continue

        df["time"] = pd.to_datetime(df["time"], format="%d/%m/%Y %H:%M", errors="coerce")
        df = df.dropna(subset=["time"])
        if df.empty:
            continue

        # Determine nearest grid points once (grid is constant)
        if nearest_grid is None:
            lat_grid = np.sort(df["latitude"].unique())
            lon_grid = np.sort(df["longitude"].unique())
            nearest_grid = {}
            for icao, (alat, alon) in airports.items():
                lat_g = nearest_value(lat_grid, alat)
                lon_g = nearest_value(lon_grid, alon)
                nearest_grid[icao] = (lat_g, lon_g)

        for icao, (lat_g, lon_g) in nearest_grid.items():
            sub = df[(df["latitude"] == lat_g) & (df["longitude"] == lon_g)]
            if sub.empty:
                continue
            g = sub.groupby("time", as_index=False)[var_col].mean()
            series_store[icao].append(g)

    result = {}
    for icao, parts in series_store.items():
        if parts:
            ts = pd.concat(parts, ignore_index=True).sort_values("time").reset_index(drop=True)
            result[icao] = ts
        else:
            result[icao] = pd.DataFrame(columns=["time", var_col])

    return result


def monthly_means_per_year(airport_ts: dict[str, pd.DataFrame], var_col="dust_mass"):
    """
    Convert each airport's time series (kg/m³) into monthly means per year in µg/m³.
    Returns a dict { ICAO: DataFrame(Year, Month, mean_ugm3) }.
    """
    monthly = {}
    for icao, df in airport_ts.items():
        if df.empty:
            monthly[icao] = pd.DataFrame(columns=["Year", "Month", "mean_ugm3"])
            continue
        df = df.copy()
        df["ugm3"] = df[var_col] * 1e9
        df["Year"]  = df["time"].dt.year
        df["Month"] = df["time"].dt.month
        m = (df.groupby(["Year", "Month"], as_index=False)["ugm3"]
               .mean()
               .rename(columns={"ugm3": "mean_ugm3"}))
        monthly[icao] = m.sort_values(["Year", "Month"])
    return monthly


def compute_climatology(monthly_by_airport: dict[str, pd.DataFrame]):
    """
    For each airport, compute monthly climatology:
      - clim_mean_ugm3 = mean over all years for that month
      - clim_std_ugm3  = std  over all years for that month
    Returns dict { ICAO: DataFrame(Month, clim_mean_ugm3, clim_std_ugm3) }.
    """
    climatology = {}
    for icao, m in monthly_by_airport.items():
        if m.empty:
            climatology[icao] = pd.DataFrame(columns=["Month", "clim_mean_ugm3", "clim_std_ugm3"])
            continue
        c = (m.groupby("Month", as_index=False)["mean_ugm3"]
               .agg(clim_mean_ugm3="mean", clim_std_ugm3="std"))
        climatology[icao] = c.sort_values("Month")
    return climatology


def compute_anomalies(monthly_by_airport: dict[str, pd.DataFrame],
                      climatology: dict[str, pd.DataFrame]):
    """
    Merge each airport's monthly means with its climatology and compute:
      - pct_anom = 100 * (mean_ugm3 - clim_mean) / clim_mean
      - zscore   = (mean_ugm3 - clim_mean) / clim_std
    Returns dict { ICAO: DataFrame(Year, Month, mean_ugm3, clim_mean_ugm3, pct_anom, zscore) }
    """
    anomalies = {}
    for icao, m in monthly_by_airport.items():
        if m.empty or icao not in climatology or climatology[icao].empty:
            anomalies[icao] = pd.DataFrame(columns=["Year", "Month", "mean_ugm3", "clim_mean_ugm3", "pct_anom", "zscore"])
            continue
        c = climatology[icao]
        merged = m.merge(c, on="Month", how="left")
        # Avoid divide-by-zero
        merged["pct_anom"] = np.where(
            merged["clim_mean_ugm3"] != 0.0,
            100.0 * (merged["mean_ugm3"] - merged["clim_mean_ugm3"]) / merged["clim_mean_ugm3"],
            np.nan
        )
        merged["zscore"] = (merged["mean_ugm3"] - merged["clim_mean_ugm3"]) / merged["clim_std_ugm3"].replace(0.0, np.nan)
        anomalies[icao] = merged.sort_values(["Year", "Month"])
    return anomalies


def summarize_peaks(anomalies_by_airport: dict[str, pd.DataFrame]):
    """
    For each airport and year, find the peak month by mean_ugm3 and report
    its value and anomaly metrics.
    Returns a DataFrame with columns:
      [airport, year, peak_month, peak_mean_ugm3, peak_pct_anom, peak_zscore]
    """
    rows = []
    for icao, df in anomalies_by_airport.items():
        if df.empty:
            continue
        for y, grp in df.groupby("Year"):
            idx = grp["mean_ugm3"].idxmax()
            rec = grp.loc[idx]
            rows.append({
                "airport": icao,
                "year": int(y),
                "peak_month": int(rec["Month"]),
                "peak_mean_ugm3": float(rec["mean_ugm3"]),
                "peak_pct_anom": float(rec["pct_anom"]) if pd.notna(rec["pct_anom"]) else np.nan,
                "peak_zscore": float(rec["zscore"]) if pd.notna(rec["zscore"]) else np.nan,
            })
    if rows:
        return pd.DataFrame(rows).sort_values(["airport", "year"])
    return pd.DataFrame(columns=["airport","year","peak_month","peak_mean_ugm3","peak_pct_anom","peak_zscore"])


def plot_per_year_all_airports(monthly_by_airport: dict[str, pd.DataFrame],
                               start_year: int, end_year: int, out_dir: str):
    """
    For each year in the range, make a single line graph with 5 airport lines.
    Saves each figure as 'monthly_means_<YEAR>.png' in out_dir.
    """
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    years = range(start_year, end_year + 1)
    for y in years:
        plt.figure(figsize=(10, 6))
        any_data = False
        for icao, df in monthly_by_airport.items():
            sub = df[df["Year"] == y].sort_values("Month")
            if sub.empty:
                continue
            any_data = True
            plt.plot(sub["Month"], sub["mean_ugm3"], marker="o", label=icao)
        plt.xticks(range(1, 13), ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
        plt.xlabel("Month")
        plt.ylabel("Dust Concentration (µg/m³)")
        plt.title(f"Monthly Mean Dust (µg/m³) — {y} — OMAA/OMDB/OMSJ/OERK/OEJN")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(title="Airport", ncol=3)
        plt.tight_layout()

        if any_data:
            fig_path = outp / f"monthly_means_{y}.png"
            plt.savefig(fig_path, dpi=300)
            print(f"Saved: {fig_path}")
        else:
            print(f"No data available to plot for {y}.")
        plt.close()


def plot_per_airport_all_years(monthly_by_airport: dict[str, pd.DataFrame], out_dir: str):
    """
    For each airport, make a line graph with one line per year (12 months on x-axis).
    Saves 'monthly_means_<ICAO>_by_year.png' in out_dir.
    """
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    for icao, df in monthly_by_airport.items():
        if df.empty:
            print(f"{icao}: no monthly data to plot.")
            continue
        plt.figure(figsize=(10, 6))
        for y in sorted(df["Year"].unique()):
            sub = df[df["Year"] == y].sort_values("Month")
            if sub.empty:
                continue
            plt.plot(sub["Month"], sub["mean_ugm3"], marker="o", label=str(y))
        plt.xticks(range(1, 13), ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
        plt.xlabel("Month")
        plt.ylabel("Dust Concentration (µg/m³)")
        plt.title(f"Monthly Mean Dust (µg/m³) — {icao} — All Years")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(title="Year", ncol=2)
        plt.tight_layout()
        fig_path = outp / f"monthly_means_{icao}_by_year.png"
        plt.savefig(fig_path, dpi=300)
        print(f"Saved: {fig_path}")
        plt.close()


def plot_anomaly_heatmaps(anomalies_by_airport: dict[str, pd.DataFrame], out_dir: str):
    """
    For each airport, create a (Month x Year) heatmap of percentage anomalies.
    Saves 'anomalies_<ICAO>_pct.png' in out_dir.
    """
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    for icao, df in anomalies_by_airport.items():
        if df.empty:
            print(f"{icao}: no anomaly data to plot.")
            continue
        years = sorted(df["Year"].unique())
        months = list(range(1, 13))
        mat = np.full((len(months), len(years)), np.nan)
        for i, m in enumerate(months):
            for j, y in enumerate(years):
                row = df[(df["Year"] == y) & (df["Month"] == m)]
                if not row.empty:
                    mat[i, j] = row["pct_anom"].iloc[0]

        plt.figure(figsize=(10, 6))
        plt.imshow(mat, aspect="auto", origin="lower")
        plt.colorbar(label="Anomaly (%)")
        plt.yticks(range(len(months)), ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
        plt.xticks(range(len(years)), [str(y) for y in years], rotation=45)
        plt.title(f"Monthly Anomaly vs Climatology (%) — {icao}")
        plt.tight_layout()
        fig_path = outp / f"anomalies_{icao}_pct.png"
        plt.savefig(fig_path, dpi=300)
        print(f"Saved: {fig_path}")
        plt.close()


def main():
    # 0) Bounding box from airports
    bbox = compute_bbox_from_airports(AIRPORTS, BBOX_PAD_DEG)
    print(f"Using bounding box (lat_min, lat_max, lon_min, lon_max): {bbox}")

    # 1) Ensure data exist (download/convert; class skips existing)
    ensure_download_and_csv(
        START_YEAR, END_YEAR,
        bbox,
        OUTPUT_DATA_DIR
    )

    # 2) Nearest-grid time series per airport
    airport_ts = collect_airport_timeseries(
        output_dir=OUTPUT_DATA_DIR,
        airports=AIRPORTS,
        start_year=START_YEAR,
        end_year=END_YEAR,
        var_col="dust_mass"
    )

    # 3) Monthly means per year (µg/m³)
    monthly_by_ap = monthly_means_per_year(airport_ts, var_col="dust_mass")

    # 4) Climatology + anomalies
    clim = compute_climatology(monthly_by_ap)
    anomalies = compute_anomalies(monthly_by_ap, clim)

    # 5) Save/plot outputs
    plot_per_year_all_airports(monthly_by_ap, START_YEAR, END_YEAR, OUTPUT_FIG_DIR)
    plot_per_airport_all_years(monthly_by_ap, OUTPUT_FIG_DIR)
    plot_anomaly_heatmaps(anomalies, OUTPUT_FIG_DIR)

    # 6) Summary CSV (peak month per airport per year)
    summary_df = summarize_peaks(anomalies)
    if SUMMARY_CSV:
        Path(SUMMARY_CSV).parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(SUMMARY_CSV, index=False)
        print(f"Saved summary CSV -> {SUMMARY_CSV}")
    else:
        print("\nPeak-month summary (preview):")
        print(summary_df)

    # Also print climatology table for quick inspection
    for icao, c in clim.items():
        if c.empty:
            print(f"\n{icao} climatology: no data.")
            continue
        print(f"\n{icao} monthly climatology (µg/m³):")
        print(c.rename(columns={"clim_mean_ugm3":"mean","clim_std_ugm3":"std"}).round(3))

if __name__ == "__main__":
    main()
