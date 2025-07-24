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
print("\033[94m" + " " * 23 + "DUSTY")
print("\033[96m" + " " * 10 + "Developed by Muhammad Aqeel Abdulla")
print(
    "DUSTY is designed to track dust storms over a range of days within the given timeframe and area of interest \nproduces 2 csv files in the csv_output_path defined in the .json file \n the first file produces a summary of the dust storm every month and the other file allows for the dust storm to be tracked"
)

print("\033[95m" + "=" * 65 + "\033[0m")
print("\n")
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
    airport_csv_path = r"C:\Users\aqeel\The University of Manchester\UOM-RG-FSE-DUST - Aqeel_Shared_Files\ENPROT_PowerBI\Flight Data\Airports Database\airports.csv"
    use_fusion = int(
        input("Would you like to use Fusion Storm Detector Module \n 1 or 0\n")
    )  # Add this to your JSON if needed

    processor = MERRA2AODProcessor(start_date, end_date, region_bounds, output_dir)
    processor.download_files()
    processor.convert_to_csv()
    AOD = MERRA2AODANAProcessor(start_date, end_date, region_bounds, AOD_Directory)
    AOD.run()

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
