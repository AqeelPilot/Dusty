#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple Monthly Means (uses your MERRA2AODProcessor)
1) Downloads + converts via your class (skips files that already exist)
2) Computes monthly means per year (µg/m³)
3) Plots all years on one chart and optionally writes a CSV

Edit the USER INPUTS below and run:
    python monthly_means_simple.py
"""

# ============== USER INPUTS (EDIT THESE) ==============
START_YEAR = 2022
END_YEAR   = 2022

# Region bounds (lat_min, lat_max, lon_min, lon_max)
LAT_MIN, LAT_MAX = 30.0, 75.0
LON_MIN, LON_MAX = -15.0, 60.0   # negatives (W) are fine; your class converts to 0–360

# Output directory (NetCDFs + CSVs will live here)
OUTPUT_DIR = r"C:\data\merra2"

# Optional outputs
SAVE_PLOT   = r"C:\plots\monthly_means_2020_2023.png"   # set to None to just show
MONTHLY_CSV = r"C:\plots\monthly_means_2020_2023.csv"   # set to None to skip CSV
# ======================================================

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Import your downloader/converter class ---
# It can live either in 'MERRA2AODProcessor.py' or inside 'Dust_Storm_Modules.py'
try:
    from Dust_Storm_Modules import MERRA2AODProcessor
except Exception:
    try:
        from Dust_Storm_Modules import MERRA2AODProcessor
    except Exception as e:
        raise ImportError(
            "Could not import MERRA2AODProcessor. Place the class in "
            "'MERRA2AODProcessor.py' or 'Dust_Storm_Modules.py' next to this script."
        ) from e


def ensure_download_and_csv(start_year, end_year, lat_min, lat_max, lon_min, lon_max, output_dir):
    """Use your class to fetch and convert files (it will skip existing ones)."""
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


def compute_monthly_means(output_dir, start_year, end_year, var_col="dust_mass"):
    """
    Reads all daily CSVs produced by your class and computes monthly mean per year.
    - Spatial mean per timestamp
    - Convert kg/m³ -> µg/m³ (×1e9)
    Returns tidy df: [Year, Month, mean_ugm3]
    """
    out_path = Path(output_dir)
    # Your class writes: MERRA2_400.tavg1_2d_aer_Nx.YYYYMMDD.csv
    csv_files = sorted(out_path.glob("MERRA2_400.tavg1_2d_aer_Nx.*.csv"))

    if not csv_files:
        print("No CSV files found in OUTPUT_DIR. Check paths/permissions.")
        return pd.DataFrame(columns=["Year", "Month", "mean_ugm3"])

    rows = []
    for f in tqdm(csv_files, desc="Reading daily CSVs (spatial→time means)"):
        # Filter by year via filename if possible
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

        if df.empty or "time" not in df.columns or var_col not in df.columns:
            continue

        # Times are saved like "%d/%m/%Y %H:%M"
        ts = pd.to_datetime(df["time"], format="%d/%m/%Y %H:%M", errors="coerce")
        df = df.assign(time=ts).dropna(subset=["time"])
        if df.empty:
            continue

        # Spatial mean for each timestamp
        g = df.groupby("time", as_index=False)[var_col].mean()
        rows.append(g)

    if not rows:
        return pd.DataFrame(columns=["Year", "Month", "mean_ugm3"])

    all_time = pd.concat(rows, ignore_index=True)
    all_time["ugm3"] = all_time[var_col] * 1e9
    all_time["Year"]  = all_time["time"].dt.year
    all_time["Month"] = all_time["time"].dt.month

    monthly = (
        all_time.groupby(["Year", "Month"], as_index=False)["ugm3"]
        .mean()
        .rename(columns={"ugm3": "mean_ugm3"})
    )
    monthly = monthly[(monthly["Year"] >= start_year) & (monthly["Year"] <= end_year)]
    return monthly.sort_values(["Year", "Month"])


def plot_monthly_lines(monthly_df, title, save_path=None):
    """Plot each year as a separate line across the 12 months."""
    import calendar
    if monthly_df.empty:
        print("Monthly dataframe is empty; nothing to plot.")
        return

    plt.figure(figsize=(10, 6))
    for year in sorted(monthly_df["Year"].unique()):
        sub = monthly_df[monthly_df["Year"] == year].sort_values("Month")
        plt.plot(sub["Month"], sub["mean_ugm3"], marker="o", label=str(year))

    plt.xticks(range(1, 13), [calendar.month_abbr[m] for m in range(1, 13)])
    plt.xlabel("Month")
    plt.ylabel("Dust Concentration (µg/m³)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Year", ncol=2)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Saved plot -> {save_path}")
    else:
        plt.show()


def main():
    # 1) Ensure data exist (download/convert; your class skips existing files)
    ensure_download_and_csv(
        START_YEAR, END_YEAR,
        LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
        OUTPUT_DIR
    )

    # 2) Monthly means
    monthly = compute_monthly_means(
        output_dir=OUTPUT_DIR,
        start_year=START_YEAR,
        end_year=END_YEAR,
        var_col="dust_mass"
    )

    if monthly.empty:
        print("No data aggregated; aborting plot.")
        return

    # Optional: save a pivot CSV for quick scanning
    if MONTHLY_CSV:
        pivot = monthly.pivot(index="Month", columns="Year", values="mean_ugm3").sort_index()
        Path(MONTHLY_CSV).parent.mkdir(parents=True, exist_ok=True)
        pivot.to_csv(MONTHLY_CSV, float_format="%.6f")
        print(f"Saved monthly means CSV -> {MONTHLY_CSV}")

    # 3) Plot
    title = f"Monthly Mean Dust Concentration (µg/m³): {START_YEAR}–{END_YEAR}"
    plot_monthly_lines(monthly, title, SAVE_PLOT)


if __name__ == "__main__":
    main()
