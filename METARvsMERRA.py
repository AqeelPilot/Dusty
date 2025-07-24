import os
import pandas as pd
import numpy as np
from datetime import datetime
import json
from tqdm import tqdm

from Dust_Storm_Modules import MERRA2AODProcessor


class AirportDustFlagger:
    def __init__(self, metar_csv_path, config_path, airport_csv_path):
        self.metar_csv_path = metar_csv_path
        self.config_path = config_path
        self.airport_csv_path = airport_csv_path
        self._load_config()

    def _load_config(self):
        with open(self.config_path, "r") as f:
            config = json.load(f)

        self.project_name = config["project_name"]
        self.region_bounds = tuple(config["region_bounds"])
        self.start_date = config["start_date"]
        self.end_date = config["end_date"]
        self.output_dir = config["modis_output_dir"]
        self.csv_output_path = config["csv_output_path"]

    def _vectorized_haversine(self, lat1_array, lon1_array, lat2, lon2):
        lat1 = np.radians(lat1_array)
        lon1 = np.radians(lon1_array)
        lat2 = np.radians(lat2)
        lon2 = np.radians(lon2)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371
        return r * c

    def run(self):
        print("\n[1] Checking and processing MERRA-2 data...")
        processor = MERRA2AODProcessor(
            self.start_date, self.end_date, self.region_bounds, self.output_dir
        )
        processor.download_files()
        processor.convert_to_csv()

        print("\n[2] Loading METAR and airport data...")
        metar_df = pd.read_csv(self.metar_csv_path)
        metar_df["time"] = pd.to_datetime(metar_df["time"], errors="coerce")
        metar_df.dropna(subset=["time"], inplace=True)

        airport_df = pd.read_csv(self.airport_csv_path)
        airport_df = airport_df[airport_df["type"] == "large_airport"]
        unique_stations = metar_df["station"].unique()
        airport_df = airport_df[airport_df["icao_code"].isin(unique_stations)]

        results = []
        debug_logs = []

        print("\n[3] Matching MERRA-2 spikes with METAR reports...")
        merra_files = sorted(
            [
                f
                for f in os.listdir(self.output_dir)
                if f.endswith(".csv") and f.startswith("MERRA2_400")
            ]
        )
        all_merra_df = []
        for f in merra_files:
            df = pd.read_csv(os.path.join(self.output_dir, f))
            df["time"] = pd.to_datetime(df["time"], format="%d/%m/%Y %H:%M")
            df["hour"] = df["time"].dt.hour
            all_merra_df.append(df)
        if not all_merra_df:
            print("No MERRA CSVs found. Aborting.")
            return
        merra_full = pd.concat(all_merra_df, ignore_index=True)

        metar_times = set(metar_df["time"])
        merra_by_hour = {hour: df for hour, df in merra_full.groupby("hour")}

        for hour, hour_df in tqdm(merra_by_hour.items(), desc="[MERRA-led analysis]"):
            for _, airport in airport_df.iterrows():
                airport_lat = airport["latitude_deg"]
                airport_lon = airport["longitude_deg"]
                icao_code = airport["icao_code"]
                airport_name = airport["name"]

                hour_df = hour_df.copy()
                hour_df["dist"] = self._vectorized_haversine(
                    hour_df["latitude"].values,
                    hour_df["longitude"].values,
                    airport_lat,
                    airport_lon,
                )

                candidate_df = hour_df[hour_df["dist"] <= 50]
                threshold_df = hour_df[hour_df["dist"] <= 100]
                if threshold_df.empty:
                    continue

                monthly_mean = threshold_df["dust_mass"].mean()
                std_dev = threshold_df["dust_mass"].std()
                threshold_val = monthly_mean + 2 * std_dev

                filtered_df = candidate_df[candidate_df["dust_mass"] > threshold_val]

                for _, row in filtered_df.iterrows():
                    timestamp = row["time"]
                    metar_agrees = any(
                        abs((timestamp - mt).total_seconds()) <= 1800
                        for mt in metar_times
                    )
                    source = 3 if metar_agrees else 2

                    result = {
                        "datetime": timestamp,
                        "airport_name": airport_name,
                        "icao_code": icao_code,
                        "latitude": row["latitude"],
                        "longitude": row["longitude"],
                        "distance_km": row["dist"],
                        "dust_mass": row["dust_mass"],
                        "monthly_mean": monthly_mean,
                        "threshold": threshold_val,
                        "above_monthly_avg": True,
                        "Source": source,
                    }
                    results.append(result)

        for _, metar_row in tqdm(
            metar_df.iterrows(), total=len(metar_df), desc="[METAR-led analysis]"
        ):
            metar_time = metar_row["time"]
            station = metar_row["station"]
            airport_info = airport_df[airport_df["icao_code"] == station]
            if airport_info.empty:
                continue

            airport = airport_info.iloc[0]
            airport_lat = airport["latitude_deg"]
            airport_lon = airport["longitude_deg"]
            airport_name = airport["name"]

            time_window_start = metar_time - pd.Timedelta(minutes=30)
            time_window_end = metar_time + pd.Timedelta(minutes=30)

            merra_candidates = merra_full[
                (merra_full["time"] >= time_window_start)
                & (merra_full["time"] <= time_window_end)
            ].copy()

            merra_candidates["dist"] = self._vectorized_haversine(
                merra_candidates["latitude"].values,
                merra_candidates["longitude"].values,
                airport_lat,
                airport_lon,
            )

            candidate_df = merra_candidates[merra_candidates["dist"] <= 50]
            threshold_df = merra_candidates[merra_candidates["dist"] <= 100]

            if threshold_df.empty:
                continue

            monthly_mean = threshold_df["dust_mass"].mean()
            std_dev = threshold_df["dust_mass"].std()
            threshold_val = monthly_mean + 2 * std_dev

            match = candidate_df[candidate_df["dust_mass"] > threshold_val]
            source_code = 3 if not match.empty else 1

            for _, match_row in match.iterrows():
                result = {
                    "datetime": metar_time,
                    "airport_name": airport_name,
                    "icao_code": station,
                    "latitude": match_row["latitude"],
                    "longitude": match_row["longitude"],
                    "distance_km": match_row["dist"],
                    "dust_mass": match_row["dust_mass"],
                    "monthly_mean": monthly_mean,
                    "threshold": threshold_val,
                    "above_monthly_avg": True,
                    "Source": source_code,
                }
                results.append(result)

        results_df = pd.DataFrame(results)

        output_file = os.path.join(
            self.csv_output_path, f"{self.project_name}_metar_dust_crosscheck.csv"
        )
        results_df.to_csv(output_file, index=False)

        agreement_summary = results_df.copy()
        agreement_summary["agreement"] = agreement_summary["Source"].isin([1, 2, 3])
        summary = agreement_summary.groupby(
            ["airport_name", agreement_summary["datetime"].dt.hour]
        )["agreement"].agg(["count", "sum"])
        summary["percentage_agreement"] = 100 * summary["sum"] / summary["count"]

        print("\nSummary of METAR agreement by airport and hour:")
        print(summary[["percentage_agreement"]].round(1))

        print(f"\n Hourly crosscheck complete. Results saved to: {output_file}")
        return results_df


from Dust_Storm_Modules import MERRA2AODANAProcessor


class AirportAODFlagger:
    def __init__(self, metar_csv_path, config_path, airport_csv_path):
        self.metar_csv_path = metar_csv_path
        self.config_path = config_path
        self.airport_csv_path = airport_csv_path
        self._load_config()

    def _load_config(self):
        with open(self.config_path, "r") as f:
            config = json.load(f)

        self.project_name = config["project_name"]
        self.region_bounds = tuple(config["region_bounds"])
        self.start_date = config["start_date"]
        self.end_date = config["end_date"]
        self.output_dir = config["AOD_output_dir"]
        self.csv_output_path = config["csv_output_path"]

    def _vectorized_haversine(self, lat1_array, lon1_array, lat2, lon2):
        lat1 = np.radians(lat1_array)
        lon1 = np.radians(lon1_array)
        lat2 = np.radians(lat2)
        lon2 = np.radians(lon2)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371
        return r * c

    def run(self):
        print("\n[1] Checking and processing MERRA-2 AODANA data...")
        processor = MERRA2AODANAProcessor(
            self.start_date, self.end_date, self.region_bounds, self.output_dir
        )
        processor.download_files()
        processor.process_files()

        print("\n[2] Loading METAR and airport data...")
        metar_df = pd.read_csv(self.metar_csv_path)
        metar_df["time"] = pd.to_datetime(metar_df["time"], errors="coerce")
        metar_df.dropna(subset=["time"], inplace=True)

        airport_df = pd.read_csv(self.airport_csv_path)
        airport_df = airport_df[airport_df["type"] == "large_airport"]
        unique_stations = metar_df["station"].unique()
        airport_df = airport_df[airport_df["icao_code"].isin(unique_stations)]

        print("\n[3] Matching MERRA-2 AODANA spikes with METAR reports...")
        aod_files = sorted(
            [
                f
                for f in os.listdir(self.output_dir)
                if f.endswith(".csv") and f.startswith("MERRA2_400")
            ]
        )
        all_aod_df = []
        for f in aod_files:
            df = pd.read_csv(os.path.join(self.output_dir, f))
            df["time"] = pd.to_datetime(df["time"], format="%d/%m/%Y %H:%M")
            df["hour"] = df["time"].dt.hour
            all_aod_df.append(df)
        print(f"Found {len(all_aod_df)} AODANA files.")
        if not all_aod_df:
            print("No AODANA CSVs found. Aborting.")
            return
        aod_full = pd.concat(all_aod_df, ignore_index=True)

        metar_times = set(metar_df["time"])
        aod_by_hour = {hour: df for hour, df in aod_full.groupby("hour")}

        results = []
        for hour, hour_df in tqdm(aod_by_hour.items(), desc="[AODANA-led analysis]"):
            for _, airport in airport_df.iterrows():
                airport_lat = airport["latitude_deg"]
                airport_lon = airport["longitude_deg"]
                icao_code = airport["icao_code"]
                airport_name = airport["name"]

                hour_df = hour_df.copy()
                hour_df["dist"] = self._vectorized_haversine(
                    hour_df["latitude"].values,
                    hour_df["longitude"].values,
                    airport_lat,
                    airport_lon,
                )

                candidate_df = hour_df[hour_df["dist"] <= 50]
                threshold_df = hour_df[hour_df["dist"] <= 100]
                if threshold_df.empty:
                    continue

                monthly_mean = threshold_df["AODANA"].mean()
                std_dev = threshold_df["AODANA"].std()
                threshold_val = monthly_mean + 2 * std_dev

                filtered_df = candidate_df[candidate_df["AODANA"] > threshold_val]

                for _, row in filtered_df.iterrows():
                    timestamp = row["time"]
                    metar_agrees = any(
                        abs((timestamp - mt).total_seconds()) <= 1800
                        for mt in metar_times
                    )

                    result = {
                        "datetime": timestamp,
                        "airport_name": airport_name,
                        "icao_code": icao_code,
                        "latitude": row["latitude"],
                        "longitude": row["longitude"],
                        "distance_km": row["dist"],
                        "AODANA": row["AODANA"],
                        "monthly_mean_AOD": monthly_mean,
                        "threshold_AOD": threshold_val,
                        "above_monthly_avg_AOD": True,
                        "METAR_agrees": metar_agrees,
                    }
                    results.append(result)
        print(airport_df["icao_code"].unique())

        results_df = pd.DataFrame(results)
        output_file = os.path.join(
            self.csv_output_path, f"{self.project_name}_metar_AOD_crosscheck.csv"
        )
        results_df.to_csv(output_file, index=False)

        print(f"\n✅ AODANA crosscheck complete. Results saved to: {output_file}")
        return results_df
