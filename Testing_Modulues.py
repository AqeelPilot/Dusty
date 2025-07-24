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
            filename = f"MERRA2_400.inst3_3d_gas_Nv.{year}{month:02d}{day:02d}.nc4"
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
            try:
                output_csv = os.path.join(
                    self.output_dir, f"{os.path.splitext(filename)[0]}.csv"
                )
                if os.path.exists(output_csv):
                    print(f"CSV already exists for {filename}, skipping.")
                    continue

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
                    rows, columns=["time", "latitude", "longitude", "aodana"]
                )
                csv_name = os.path.splitext(filename)[0] + ".csv"
                df.to_csv(os.path.join(self.output_dir, csv_name), index=False)
            except Exception as e:
                print(f" Error processing {filename}: {e}")

    def run(self):
        self.download_files()
        self.process_files()


class MonthlyDustAnalyzer:
    def __init__(self, dust_output_dir, aodana_output_dir, project_name):
        self.dust_output_dir = dust_output_dir
        self.aodana_output_dir = aodana_output_dir
        self.daily_data = []
        self.project_name = project_name

    def analyze(self, summary_csv_path):
        print("Analyzing dust data and AODANA for monthly spatial statistics...")

        # Step 1: Process dust mass data
        dust_data = []
        for file in tqdm(sorted(os.listdir(self.dust_output_dir))):
            if not file.endswith(".csv") or not file.startswith("MERRA2_400"):
                continue

            try:
                file_path = os.path.join(self.dust_output_dir, file)
                df = pd.read_csv(file_path)

                if {"time", "latitude", "longitude", "dust_mass"}.issubset(df.columns):
                    df["time"] = pd.to_datetime(df["time"], format="%d/%m/%Y %H:%M")
                    df["date"] = df["time"].dt.date
                    dust_data.append(df[["date", "latitude", "longitude", "dust_mass"]])
            except Exception as e:
                print(f"Error processing dust file {file}: {e}")

        # Step 2: Process AODANA data
        aodana_data = []
        for file in tqdm(sorted(os.listdir(self.aodana_output_dir))):
            if not file.endswith(".csv") or not file.startswith("MERRA2_400"):
                continue

            try:
                file_path = os.path.join(self.aodana_output_dir, file)
                df = pd.read_csv(file_path)

                if {"time", "latitude", "longitude", "aodana"}.issubset(df.columns):
                    df["time"] = pd.to_datetime(df["time"], format="%d/%m/%Y %H:%M")
                    df["date"] = df["time"].dt.date
                    aodana_data.append(df[["date", "latitude", "longitude", "aodana"]])
            except Exception as e:
                print(f"Error processing AODANA file {file}: {e}")

        if not dust_data and not aodana_data:
            print("No valid CSVs found.")
            return

        # Step 3: Combine and merge data
        combined_data = []

        if dust_data:
            dust_combined = pd.concat(dust_data, ignore_index=True)
            dust_daily = (
                dust_combined.groupby(["date", "latitude", "longitude"])["dust_mass"]
                .mean()
                .reset_index()
            )
            combined_data.append(dust_daily)

        if aodana_data:
            aodana_combined = pd.concat(aodana_data, ignore_index=True)
            aodana_daily = (
                aodana_combined.groupby(["date", "latitude", "longitude"])["aodana"]
                .mean()
                .reset_index()
            )
            combined_data.append(aodana_daily)

        # Step 4: Merge dust and AODANA data
        if len(combined_data) == 2:
            merged_daily = pd.merge(
                combined_data[0],
                combined_data[1],
                on=["date", "latitude", "longitude"],
                how="outer",
            )
        else:
            merged_daily = combined_data[0] if combined_data else pd.DataFrame()

        if merged_daily.empty:
            print("No data to analyze.")
            return

        # Step 5: Compute monthly statistics
        group_cols = ["latitude", "longitude"]
        stats_list = []

        if "dust_mass" in merged_daily.columns:
            dust_stats = (
                merged_daily.groupby(group_cols)["dust_mass"]
                .agg(dust_monthly_mean="mean", dust_std_dev="std")
                .reset_index()
            )
            stats_list.append(dust_stats)

        if "aodana" in merged_daily.columns:
            aodana_stats = (
                merged_daily.groupby(group_cols)["aodana"]
                .agg(aodana_monthly_mean="mean", aodana_std_dev="std")
                .reset_index()
            )
            stats_list.append(aodana_stats)

        # Merge all statistics
        final_stats = stats_list[0]
        for stats_df in stats_list[1:]:
            final_stats = pd.merge(final_stats, stats_df, on=group_cols, how="outer")

        # Step 6: Merge back with daily data and compute thresholds
        final_merged = pd.merge(merged_daily, final_stats, on=group_cols, how="left")

        # Compute thresholds and flags
        if "dust_mass" in final_merged.columns:
            final_merged["dust_threshold"] = (
                final_merged["dust_monthly_mean"] + 2 * final_merged["dust_std_dev"]
            )
            final_merged["dust_above_threshold"] = (
                final_merged["dust_mass"] > final_merged["dust_threshold"]
            )

        if "aodana" in final_merged.columns:
            final_merged["aodana_threshold"] = (
                final_merged["aodana_monthly_mean"] + 2 * final_merged["aodana_std_dev"]
            )
            final_merged["aodana_above_threshold"] = (
                final_merged["aodana"] > final_merged["aodana_threshold"]
            )

        # Create combined threshold flag
        threshold_cols = [
            col for col in final_merged.columns if col.endswith("_above_threshold")
        ]
        if threshold_cols:
            final_merged["any_above_threshold"] = final_merged[threshold_cols].any(
                axis=1
            )

        # Step 7: Save to CSV
        filename = f"{self.project_name}_monthly_dust_aodana_summary.csv"
        full_path = os.path.join(os.path.dirname(summary_csv_path), filename)

        # Rename columns for clarity
        if "dust_mass" in final_merged.columns:
            final_merged = final_merged.rename(columns={"dust_mass": "daily_mean_dust"})
        if "aodana" in final_merged.columns:
            final_merged = final_merged.rename(columns={"aodana": "daily_mean_aodana"})

        final_merged.to_csv(full_path, index=False)
        print(f"Monthly dust and AODANA summary saved to: {full_path}")


class StormDetector:
    def __init__(
        self,
        dust_input_dir,
        aodana_input_dir,
        project_name,
        airport_csv_path,
        csv_output_path,
    ):
        self.dust_input_dir = dust_input_dir
        self.aodana_input_dir = aodana_input_dir
        self.project_name = project_name
        self.results = []
        self.airport_csv_path = airport_csv_path
        self.csv_output_path = csv_output_path

        self.airports_df = pd.read_csv(airport_csv_path)
        self.airports_df = self.airports_df[self.airports_df["type"] == "large_airport"]

    def detect_storms(self, threshold_factor=2, use_combined_metrics=True):
        print("Detecting storm blobs using dust mass and AODANA data...")
        storm_id_counter = 1
        previous_day_blobs = []

        # Get all unique dates from both directories
        all_dates = set()

        for file in os.listdir(self.dust_input_dir):
            if file.endswith(".csv") and file.startswith("MERRA2_400"):
                date_str = file.split(".")[2]  # Extract date from filename
                all_dates.add(date_str)

        for file in os.listdir(self.aodana_input_dir):
            if file.endswith(".csv") and file.startswith("MERRA2_400"):
                date_str = file.split(".")[2]  # Extract date from filename
                all_dates.add(date_str)

        for date_str in tqdm(sorted(all_dates), desc="Processing dates"):
            # Load dust data for this date
            dust_file = f"MERRA2_400.tavg1_2d_aer_Nx.{date_str}.csv"
            dust_path = os.path.join(self.dust_input_dir, dust_file)
            dust_df = None
            if os.path.exists(dust_path):
                try:
                    dust_df = pd.read_csv(dust_path)
                    if {"latitude", "longitude", "dust_mass"}.issubset(dust_df.columns):
                        dust_df["time"] = pd.to_datetime(
                            dust_df["time"], format="%d/%m/%Y %H:%M"
                        )
                        dust_df = (
                            dust_df.groupby(["latitude", "longitude"])["dust_mass"]
                            .mean()
                            .reset_index()
                        )
                except Exception as e:
                    print(f"Error loading dust data for {date_str}: {e}")
                    dust_df = None

            # Load AODANA data for this date
            aodana_file = f"MERRA2_400.inst3_3d_gas_Nv.{date_str}.csv"
            aodana_path = os.path.join(self.aodana_input_dir, aodana_file)
            aodana_df = None
            if os.path.exists(aodana_path):
                try:
                    aodana_df = pd.read_csv(aodana_path)
                    if {"latitude", "longitude", "aodana"}.issubset(aodana_df.columns):
                        aodana_df["time"] = pd.to_datetime(
                            aodana_df["time"], format="%d/%m/%Y %H:%M"
                        )
                        aodana_df = (
                            aodana_df.groupby(["latitude", "longitude"])["aodana"]
                            .mean()
                            .reset_index()
                        )
                except Exception as e:
                    print(f"Error loading AODANA data for {date_str}: {e}")
                    aodana_df = None

            # Skip if no data available
            if dust_df is None and aodana_df is None:
                continue

            # Merge data if both available
            if dust_df is not None and aodana_df is not None:
                combined_df = pd.merge(
                    dust_df, aodana_df, on=["latitude", "longitude"], how="outer"
                )
            elif dust_df is not None:
                combined_df = dust_df.copy()
            else:
                combined_df = aodana_df.copy()

            # Create grid for analysis
            lat_vals = sorted(combined_df["latitude"].unique())
            lon_vals = sorted(combined_df["longitude"].unique())

            if (
                use_combined_metrics
                and "dust_mass" in combined_df.columns
                and "aodana" in combined_df.columns
            ):
                # Normalize both metrics and create combined score
                dust_norm = (
                    combined_df["dust_mass"] - combined_df["dust_mass"].min()
                ) / (combined_df["dust_mass"].max() - combined_df["dust_mass"].min())
                aodana_norm = (combined_df["aodana"] - combined_df["aodana"].min()) / (
                    combined_df["aodana"].max() - combined_df["aodana"].min()
                )
                combined_df["combined_metric"] = (dust_norm + aodana_norm) / 2
                metric_col = "combined_metric"
            elif "dust_mass" in combined_df.columns:
                metric_col = "dust_mass"
            else:
                metric_col = "aodana"

            # Create data array
            data_array = np.full((len(lat_vals), len(lon_vals)), np.nan)
            for _, row in combined_df.iterrows():
                if pd.notna(row[metric_col]):
                    i = lat_vals.index(row["latitude"])
                    j = lon_vals.index(row["longitude"])
                    data_array[i, j] = row[metric_col]

            # Calculate threshold
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

                avg_metric = np.mean(vals)
                max_metric = np.max(vals)

                center_i, center_j = center_of_mass(mask, labeled, blob_id)
                center_lat = float(np.interp(center_i, range(len(lat_vals)), lat_vals))
                center_lon = float(np.interp(center_j, range(len(lon_vals)), lon_vals))

                airport_info = self._find_nearest_large_airport(center_lat, center_lon)
                if airport_info is None:
                    continue

                icao, distance_km = airport_info
                if not (1 <= distance_km <= 300):
                    continue

                # Try to match this blob to previous day's blobs
                matched_id = None
                for prev_blob in previous_day_blobs:
                    prev_lat, prev_lon = (
                        prev_blob["center_lat"],
                        prev_blob["center_lon"],
                    )
                    dist = geodesic((prev_lat, prev_lon), (center_lat, center_lon)).km
                    if dist < 30:
                        matched_id = prev_blob["storm_id"]
                        break

                if matched_id is None:
                    matched_id = storm_id_counter
                    storm_id_counter += 1

                # Get individual metrics for this location
                dust_val = None
                aodana_val = None
                matching_row = combined_df[
                    (abs(combined_df["latitude"] - center_lat) < 0.1)
                    & (abs(combined_df["longitude"] - center_lon) < 0.1)
                ]
                if not matching_row.empty:
                    if "dust_mass" in matching_row.columns:
                        dust_val = matching_row["dust_mass"].iloc[0]
                    if "aodana" in matching_row.columns:
                        aodana_val = matching_row["aodana"].iloc[0]

                blob_info = {
                    "date": date_str,
                    "storm_id": matched_id,
                    "blob_size": len(indices),
                    "avg_combined_metric": avg_metric,
                    "max_combined_metric": max_metric,
                    "avg_dust_mass": dust_val,
                    "avg_aodana": aodana_val,
                    "center_lat": center_lat,
                    "center_lon": center_lon,
                    "nearest_airport_icao": icao,
                    "distance_to_airport_km": distance_km,
                    "metric_used": metric_col,
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
            self.csv_output_path, f"{self.project_name}_combined_storm_lifecycle.csv"
        )
        df.to_csv(output_path, index=False)
        print(f"Combined storm tracking results saved to {output_path}")


# Example usage function to demonstrate the enhanced workflow
def run_enhanced_analysis(
    start_date, end_date, region_bounds, project_name, airport_csv_path
):
    """
    Enhanced workflow that processes both dust mass and AODANA data
    """
    # Initialize processors
    dust_processor = MERRA2AODProcessor(
        start_date, end_date, region_bounds, "data/merra2_dust"
    )
    aodana_processor = MERRA2AODANAProcessor(
        start_date, end_date, region_bounds, "data/merra2_aodana"
    )

    # Download and process data
    print("=== Processing Dust Mass Data ===")
    dust_processor.download_files()
    dust_processor.convert_to_csv()

    print("=== Processing AODANA Data ===")
    aodana_processor.run()

    # Analyze monthly patterns
    print("=== Analyzing Monthly Patterns ===")
    analyzer = MonthlyDustAnalyzer(
        "data/merra2_dust", "data/merra2_aodana", project_name
    )
    analyzer.analyze("data/summary.csv")

    # Detect and track storms
    print("=== Detecting and Tracking Storms ===")
    detector = StormDetector(
        "data/merra2_dust",
        "data/merra2_aodana",
        project_name,
        airport_csv_path,
        "data/results",
    )
    detector.detect_storms(threshold_factor=2, use_combined_metrics=True)
    detector.save_results()

    print("Enhanced analysis complete!")
