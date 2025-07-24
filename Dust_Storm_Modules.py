"""
Dust_Storm_Modules
------------------
This module includes processors for MERRA-2 dust and AODANA data,
storm detection, fusion storm tracking, and monthly analysis.

Classes:
- MERRA2AODProcessor
- MERRA2AODANAProcessor
- MonthlyDustAnalyzer
- StormDetector
- FusionStormDetector
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tqdm import tqdm
import requests
from tqdm.auto import tqdm as auto_tqdm
from geopy.distance import geodesic
import os
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
from scipy.ndimage import label
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import label, center_of_mass
from skimage.measure import regionprops


class MERRA2AODProcessor:
    def __init__(self, start_date, end_date, region_bounds, output_dir="data/merra2"):
        self.start_date = start_date
        self.end_date = end_date
        self.lat_min, self.lat_max, self.lon_min, self.lon_max = region_bounds

        if self.lon_min < 0:
            self.lon_min += 360
        if self.lon_max < 0:
            self.lon_max += 360

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.environ["NETRC"] = r"C:\\Users\\aqeel\\login_netrc"

    def download_files(self):
        print(" Downloading MERRA-2 NetCDF files...")
        for date in tqdm(
            pd.date_range(self.start_date, self.end_date), desc="Downloading"
        ):
            year, month, day = date.year, date.month, date.day
            base_url = (
                "https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2T1NXAER.5.12.4"
            )
            filename = f"MERRA2_400.tavg1_2d_aer_Nx.{year}{month:02d}{day:02d}.nc4"
            url = f"{base_url}/{year}/{month:02d}/{filename}"
            file_path = os.path.join(self.output_dir, filename)

            if os.path.exists(file_path):
                print(f" File already exists: {filename}")
                continue

            try:
                with requests.Session() as session:
                    session.trust_env = True
                    r = session.get(url, stream=True, timeout=60)
                    r.raise_for_status()

                    total_size = int(r.headers.get("content-length", 0))
                    with open(file_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
            except Exception as e:
                print(f" Download failed for {filename}: {e}")

    def convert_to_csv(self):
        print(" Converting NetCDF files to CSV...")
        files = sorted([f for f in os.listdir(self.output_dir) if f.endswith(".nc4")])

        #  Skip if the CSV already exists

        for filename in tqdm(files, desc="Converting"):
            try:
                output_csv = os.path.join(
                    self.output_dir, f"{os.path.splitext(filename)[0]}.csv"
                )
                if os.path.exists(output_csv):
                    print(f"CSV already exists for {filename}, skipping.")
                    continue
                file_path = os.path.join(self.output_dir, filename)
                ds = xr.open_dataset(file_path, engine="netcdf4")

                aod = ds["DUSMASS"][:, :, :]  # shape: (time, lat, lon)
                aod_region = aod.sel(
                    lat=slice(self.lat_min, self.lat_max),
                    lon=slice(self.lon_min, self.lon_max),
                )

                lat_vals = aod_region["lat"].values
                lon_vals = aod_region["lon"].values
                time_vals = aod_region["time"].values
                data = aod_region.values  # shape: (time, lat, lon)

                rows = []
                for t_idx, t in enumerate(time_vals):
                    for i, lat in enumerate(lat_vals):
                        for j, lon in enumerate(lon_vals):
                            val = data[t_idx, i, j]
                            if np.isfinite(val):
                                rows.append(
                                    [
                                        pd.to_datetime(str(t)).strftime(
                                            "%d/%m/%Y %H:%M"
                                        ),
                                        lat,
                                        lon,
                                        val,
                                    ]
                                )

                df = pd.DataFrame(
                    rows,
                    columns=["time", "latitude", "longitude", "dust_mass"],
                )
                out_csv = os.path.join(
                    self.output_dir, f"{os.path.splitext(filename)[0]}.csv"
                )
                df.to_csv(out_csv, index=False)
            except Exception as e:
                print(f" Error processing {filename}: {e}")


class MERRA2AODANAProcessor:
    def __init__(self, start_date, end_date, region_bounds, output_dir):
        self.start_date = start_date
        self.end_date = end_date
        self.lat_min, self.lat_max, self.lon_min, self.lon_max = region_bounds

        if self.lon_min < 0:
            self.lon_min += 360
        if self.lon_max < 0:
            self.lon_max += 360

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.environ["NETRC"] = r"C:\\Users\\aqeel\\login_netrc"

    def download_files(self):
        print(" Downloading MERRA-2 AODANA data (M2I3NXGAS)...")
        for date in tqdm(
            pd.date_range(self.start_date, self.end_date), desc="Downloading AODANA"
        ):
            year, month, day = date.year, date.month, date.day
            base_url = (
                "https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2I3NXGAS.5.12.4"
            )
            filename = f"MERRA2_400.inst3_2d_gas_Nx.{year}{month:02d}{day:02d}.nc4"
            url = f"{base_url}/{year}/{month:02d}/{filename}"
            file_path = os.path.join(self.output_dir, filename)

            if os.path.exists(file_path):
                continue

            try:
                with requests.Session() as session:
                    session.trust_env = True
                    response = session.get(url, stream=True, timeout=60)
                    response.raise_for_status()

                    with open(file_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
            except Exception as e:
                print(f" Failed to download {filename}: {e}")

    def process_files(self):
        print(" Processing AODANA data into CSV...")
        all_files = sorted(
            [f for f in os.listdir(self.output_dir) if f.endswith(".nc4")]
        )
        for filename in tqdm(all_files, desc="Processing AOD Files"):
            output_csv = os.path.join(
                self.output_dir, f"{os.path.splitext(filename)[0]}.csv"
            )
            if os.path.exists(output_csv):
                print(f"CSV already exists for {filename}, skipping conversion.")
                continue
            try:
                file_path = os.path.join(self.output_dir, filename)
                ds = xr.open_dataset(file_path)

                aod = ds["AODANA"][:, :, :]  # time, lat, lon
                aod_region = aod.sel(
                    lat=slice(self.lat_min, self.lat_max),
                    lon=slice(self.lon_min, self.lon_max),
                )

                time_vals = aod_region["time"].values
                lat_vals = aod_region["lat"].values
                lon_vals = aod_region["lon"].values
                data = aod_region.values  # shape: (time, lat, lon)

                rows = []
                for t_idx, t in enumerate(time_vals):
                    for i, lat in enumerate(lat_vals):
                        for j, lon in enumerate(lon_vals):
                            val = data[t_idx, i, j]
                            if pd.notna(val):
                                rows.append(
                                    [
                                        pd.to_datetime(str(t)).strftime(
                                            "%d/%m/%Y %H:%M"
                                        ),
                                        lat,
                                        lon,
                                        val,
                                    ]
                                )

                df = pd.DataFrame(
                    rows, columns=["time", "latitude", "longitude", "AODANA"]
                )
                csv_name = os.path.splitext(filename)[0] + ".csv"

                output_csv = os.path.join(
                    self.output_dir, f"{os.path.splitext(filename)[0]}_AODANA.csv"
                )
                if os.path.exists(output_csv):
                    print(f"CSV already exists for {filename}, skipping conversion.")
                    continue
                df.to_csv(os.path.join(self.output_dir, csv_name), index=False)
            except Exception as e:
                print(f" Error processing {filename}: {e}")

    def run(self):
        self.download_files()
        self.process_files()


class MonthlyDustAnalyzer:
    def __init__(self, output_dir, project_name):
        self.output_dir = output_dir
        self.project_name = project_name
        self.all_data = []

    def analyze(self, summary_csv_path):
        print("Analyzing dust data with time resolution...")

        for file in tqdm(sorted(os.listdir(self.output_dir))):
            if not file.endswith(".csv") or not file.startswith("MERRA2_400"):
                continue

            try:
                file_path = os.path.join(self.output_dir, file)
                df = pd.read_csv(file_path)

                if {"time", "latitude", "longitude", "dust_mass"}.issubset(df.columns):
                    df["time"] = pd.to_datetime(df["time"], format="%d/%m/%Y %H:%M")
                    self.all_data.append(df)
            except Exception as e:
                print(f"Error processing {file}: {e}")

        if not self.all_data:
            print("No valid CSVs found.")
            return

        # Combine all time-resolved entries
        full_df = pd.concat(self.all_data, ignore_index=True)

        # Extract hour + minute to preserve time-level patterns
        full_df["hour_minute"] = full_df["time"].dt.strftime("%H:%M")

        # Group by lat, lon, and time of day to calculate monthly mean and std
        stats_df = (
            full_df.groupby(["latitude", "longitude", "hour_minute"])["dust_mass"]
            .agg(monthly_mean="mean", std_dev="std")
            .reset_index()
        )

        # Merge back to get comparison at the same (lat, lon, hour:minute)
        full_df["hour_minute"] = full_df["time"].dt.strftime("%H:%M")
        merged = full_df.merge(stats_df, on=["latitude", "longitude", "hour_minute"])

        # Apply 2× std threshold
        merged["threshold"] = merged["monthly_mean"] + 2 * merged["std_dev"]
        merged["above_monthly_avg"] = merged["dust_mass"] > merged["threshold"]

        # Save results
        output_filename = f"{self.project_name}_monthly_dust_summary.csv"
        output_path = os.path.join(os.path.dirname(summary_csv_path), output_filename)
        merged.to_csv(output_path, index=False)
        print(f" Time-resolved monthly summary saved to: {output_path}")


class StormDetector:
    def __init__(self, input_dir, project_name, airport_csv_path, csv_output_path):
        self.input_dir = input_dir
        self.project_name = project_name
        self.results = []
        self.airport_csv_path = airport_csv_path
        self.csv_output_path = csv_output_path

        self.airports_df = pd.read_csv(airport_csv_path)
        self.airports_df = self.airports_df[self.airports_df["type"] == "large_airport"]

    def detect_storms(self, threshold_factor=1):
        print("Detecting storm blobs and tracking across days...")
        storm_id_counter = 1
        previous_day_blobs = []

        for file in tqdm(sorted(os.listdir(self.input_dir))):
            if not file.endswith(".csv") or not file.startswith("MERRA2_400"):
                continue

            file_path = os.path.join(self.input_dir, file)
            df = pd.read_csv(file_path)
            if {"latitude", "longitude", "dust_mass"}.issubset(df.columns) is False:
                continue

            df["time"] = pd.to_datetime(df["time"], format="%d/%m/%Y %H:%M")
            grouped = (
                df.groupby(["latitude", "longitude"])["dust_mass"].mean().reset_index()
            )

            lat_vals = sorted(df["latitude"].unique())
            lon_vals = sorted(df["longitude"].unique())

            data_array = np.full((len(lat_vals), len(lon_vals)), np.nan)
            for _, row in grouped.iterrows():
                i = lat_vals.index(row["latitude"])
                j = lon_vals.index(row["longitude"])
                data_array[i, j] = row["dust_mass"]

            mean = np.nanmean(data_array)
            std = np.nanstd(data_array)
            threshold = mean + threshold_factor * std
            mask = (data_array > threshold).astype(int)
            labeled, num_features = label(mask)

            day_results = []
            for blob_id in range(1, num_features + 1):
                indices = np.argwhere(labeled == blob_id)
                if len(indices) == 0:
                    continue

                lats = [lat_vals[i] for i, _ in indices]
                lons = [lon_vals[j] for _, j in indices]
                vals = [data_array[i, j] for i, j in indices]

                avg_mass = np.mean(vals)
                max_mass = np.max(vals)

                center_i, center_j = center_of_mass(mask, labeled, blob_id)
                center_lat = float(np.interp(center_i, range(len(lat_vals)), lat_vals))
                center_lon = float(np.interp(center_j, range(len(lon_vals)), lon_vals))

                airport_info = self._find_nearest_large_airport(center_lat, center_lon)
                if airport_info is None:
                    continue

                icao, distance_km = airport_info
                if not (0.01 <= distance_km <= 80):
                    continue

                best_match = None
                best_score = float("inf")

                for prev_blob in previous_day_blobs:
                    prev_lat = prev_blob["center_lat"]
                    prev_lon = prev_blob["center_lon"]
                    prev_size = prev_blob["blob_size"]

                    dist_km = geodesic(
                        (prev_lat, prev_lon), (center_lat, center_lon)
                    ).km
                    size_diff = abs(len(indices) - prev_size)

                    if dist_km < 50 and size_diff < 0.6 * prev_size:
                        score = dist_km + 0.1 * size_diff
                        if score < best_score:
                            best_score = score
                            best_match = prev_blob

                if best_match:
                    matched_id = best_match["storm_id"]
                else:
                    matched_id = storm_id_counter
                    storm_id_counter += 1

                blob_info = {
                    "file": file,
                    "storm_id": matched_id,
                    "blob_size": len(indices),
                    "avg_dust_mass": avg_mass,
                    "max_dust_mass": max_mass,
                    "center_lat": center_lat,
                    "center_lon": center_lon,
                    "nearest_airport_icao": icao,
                    "distance_to_airport_km": distance_km,
                }
                day_results.append(blob_info)

            previous_day_blobs = day_results
            self.results.extend(day_results)

    def _find_nearest_large_airport(self, lat, lon):
        min_dist = float("inf")
        nearest_airport = None

        for _, row in self.airports_df.iterrows():
            airport_coords = (row["latitude_deg"], row["longitude_deg"])
            dist = geodesic((lat, lon), airport_coords).km
            if dist < min_dist:
                min_dist = dist
                nearest_airport = row

        if nearest_airport is None:
            return None
        return nearest_airport["icao_code"], min_dist

    def save_results(self):
        if not self.results:
            print("No storms tracked.")
            return

        df = pd.DataFrame(self.results)
        output_path = os.path.join(
            self.csv_output_path, f"{self.project_name}_DUSMASS_storm_lifecycle.csv"
        )
        df.to_csv(output_path, index=False)
        print(f"Storm tracking results saved to {output_path}")


class FusionStormDetector:
    def __init__(self, aodana_dir, airport_csv_path, csv_output_path, project_name):
        self.aodana_dir = aodana_dir
        self.csv_output_path = csv_output_path
        self.project_name = project_name
        self.results = []
        self.daily_records = []
        self.storm_id_counter = 1
        self.previous_day_blobs = []

        self.airports_df = pd.read_csv(airport_csv_path)
        self.airports_df = self.airports_df[self.airports_df["type"] == "large_airport"]

    def detect_storms(self):
        aod_files = sorted(
            [f for f in os.listdir(self.aodana_dir) if f.endswith(".csv")]
        )

        for file in tqdm(aod_files, desc="AODANA Storm Tracking"):
            file_path = os.path.join(self.aodana_dir, file)
            aod_df = pd.read_csv(file_path)
            aod_df["time"] = pd.to_datetime(aod_df["time"], format="%d/%m/%Y %H:%M")
            self.daily_records.append(aod_df)

            filename_core = os.path.splitext(file)[0]
            parts = filename_core.split(".")
            date_str = parts[-1] if parts[-1].isdigit() else filename_core

            grouped = (
                aod_df.groupby(["latitude", "longitude"])["AODANA"].mean().reset_index()
            )
            lat_vals = sorted(grouped["latitude"].unique())
            lon_vals = sorted(grouped["longitude"].unique())

            aod_arr = np.full((len(lat_vals), len(lon_vals)), np.nan)
            for _, row in grouped.iterrows():
                i = lat_vals.index(row["latitude"])
                j = lon_vals.index(row["longitude"])
                aod_arr[i, j] = row["AODANA"]

            if np.all(np.isnan(aod_arr)):
                print(f"{file}: Empty AOD array. Skipping.")
                continue

            aod_thresh = np.nanmean(aod_arr) + 1 * np.nanstd(aod_arr)
            aod_mask = aod_arr > aod_thresh
            labeled, num_features = label(aod_mask)
            regions = regionprops(labeled)

            day_results = []
            for region in regions:
                coords = region.coords
                avg_aod = np.mean([aod_arr[i, j] for i, j in coords])
                max_aod = np.max([aod_arr[i, j] for i, j in coords])

                center_i, center_j = center_of_mass(aod_mask, labeled, region.label)
                center_lat = float(
                    np.interp(center_i, np.arange(len(lat_vals)), lat_vals)
                )
                center_lon = float(
                    np.interp(center_j, np.arange(len(lon_vals)), lon_vals)
                )

                airport_info = self._find_nearest_large_airport(center_lat, center_lon)
                if airport_info is None:
                    continue

                icao, distance_km = airport_info
                if not (0.01 <= distance_km <= 80):
                    continue

                best_match = None
                best_score = float("inf")

                for prev_blob in self.previous_day_blobs:
                    prev_lat = prev_blob["center_lat"]
                    prev_lon = prev_blob["center_lon"]
                    prev_size = prev_blob["blob_size"]

                    dist_km = geodesic(
                        (prev_lat, prev_lon), (center_lat, center_lon)
                    ).km
                    size_diff = abs(region.area - prev_size)

                    if dist_km < 50 and size_diff < 0.6 * prev_size:
                        score = dist_km + 0.1 * size_diff  # weighted score
                        if score < best_score:
                            best_score = score
                            best_match = prev_blob

                if best_match:
                    matched_id = best_match["storm_id"]
                else:
                    matched_id = self.storm_id_counter
                    self.storm_id_counter += 1

                blob_info = {
                    "date": date_str,
                    "storm_id": matched_id,
                    "blob_size": region.area,
                    "avg_AODANA": avg_aod,
                    "max_AODANA": max_aod,
                    "center_lat": center_lat,
                    "center_lon": center_lon,
                    "nearest_airport_icao": icao,
                    "distance_to_airport_km": distance_km,
                }
                day_results.append(blob_info)

            self.previous_day_blobs = day_results
            self.results.extend(day_results)

        self.save_results()
        self.generate_monthly_summary()

    def _find_nearest_large_airport(self, lat, lon):
        min_dist = float("inf")
        nearest_airport = None
        for _, row in self.airports_df.iterrows():
            airport_coords = (row["latitude_deg"], row["longitude_deg"])
            dist = geodesic((lat, lon), airport_coords).km
            if dist < min_dist:
                min_dist = dist
                nearest_airport = row
        if nearest_airport is None:
            return None
        return nearest_airport["icao_code"], min_dist

    def save_results(self):
        if not self.results:
            print("No storms detected.")
            return
        df = pd.DataFrame(self.results)
        output_path = os.path.join(
            self.csv_output_path, f"{self.project_name}_AOD_storm_lifecycle.csv"
        )
        df.to_csv(output_path, index=False)
        print(f"AOD-based storm tracking results saved to {output_path}")

    def generate_monthly_summary(self):
        print("Generating monthly AODANA summary...")
        if not self.daily_records:
            print("No fusion records to analyze.")
            return
        full_df = pd.concat(self.daily_records, ignore_index=True)
        full_df["hour_minute"] = full_df["time"].dt.strftime("%H:%M")
        stats_df = (
            full_df.groupby("hour_minute")
            .agg(
                regional_mean_aod=("AODANA", "mean"),
                regional_std_aod=("AODANA", "std"),
            )
            .reset_index()
        )
        merged = full_df.merge(stats_df, on="hour_minute")
        merged["threshold_aod"] = (
            merged["regional_mean_aod"] + 2 * merged["regional_std_aod"]
        )
        merged["above_monthly_avg_aod"] = merged["AODANA"] > merged["threshold_aod"]
        output_filename = f"{self.project_name}_monthly_AODANA_summary.csv"
        output_path = os.path.join(self.csv_output_path, output_filename)
        merged.to_csv(output_path, index=False)
        print(f"Monthly AODANA summary saved to: {output_path}")
