import os
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
from tqdm.auto import tqdm
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import json

from Dust_Storm_Modules import (
    MERRA2AODProcessor,
    MonthlyDustAnalyzer,
    StormDetector,
    MERRA2AODANAProcessor,
    FusionStormDetector,
)


# ========================= WELCOME MESSAGE ========================= #
print("\033[95m" + "=" * 65)
print("\033[94m" + " " * 26 + "DUSTY")
print("\033[96m" + " " * 14 + "Developed by Muhammad Aqeel Abdulla")
print("\n\033[0m" + "DUSTY is designed to track dust storms over a range of days")
print("within the given timeframe and area of interest.")
print("It produces two CSV files in the path defined by `csv_output_path`")
print("in the configuration `.json` file:\n")
print(" - Monthly summary of detected dust storms")
print(" - Storm-tracking file for identifying storm evolution over time")
print("\033[95m" + "=" * 65 + "\033[0m\n")

# ================================================================ #


if __name__ == "__main__":
    Tk().withdraw()
    config_path = askopenfilename(
        title="Select Config JSON", filetypes=[("JSON files", "*.json")]
    )
    if not config_path:
        raise FileNotFoundError("No configuration file selected.")

    with open(config_path, "r") as f:
        config = json.load(f)

    region_bounds = tuple(config["region_bounds"])
    start_date = datetime.strptime(config["start_date"], "%Y-%m-%d")
    end_date = datetime.strptime(config["end_date"], "%Y-%m-%d")
    output_dir = config["modis_output_dir"]
    AOD_Directory = config["AOD_output_dir"]
    csv_output_path = config["csv_output_path"]
    project_name = config["project_name"]
    airport_csv_path = r"C:\Users\y46144ma\The University of Manchester\UOM-RG-FSE-DUST - Aqeel_Shared_Files\ENPROT_PowerBI\Flight Data\Airports Database\airports.csv"
    use_fusion = int(
        input("Would you like to use Fusion Storm Detector Module \n 1 or 0\n")
    )  # Add this to your JSON if needed

    processor = MERRA2AODProcessor(start_date, end_date, region_bounds, output_dir)
    processor.download_files()
    processor.convert_to_csv()
    AOD = MERRA2AODANAProcessor(start_date, end_date, region_bounds, AOD_Directory)
    #AOD.run()

    analyzer = MonthlyDustAnalyzer(output_dir, project_name)
    analyzer.analyze(csv_output_path)

    if use_fusion == 1:
        print("Running FusionStormDetector...")
        storm_detector = FusionStormDetector(
            AOD_Directory,
            airport_csv_path,
            csv_output_path,
            project_name,
        )
    else:
        print("Running StormDetector...")
        storm_detector = StormDetector(
            output_dir, project_name, airport_csv_path, csv_output_path
        )

    storm_detector.detect_storms()
    storm_detector.save_results()
