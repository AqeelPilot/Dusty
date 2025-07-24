import os
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
from tqdm.auto import tqdm
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import json

from Testing_Modulues import (
    MERRA2AODProcessor,
    MonthlyDustAnalyzer,
    StormDetector,
    MERRA2AODANAProcessor,
)


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
    csv_output_path = config["csv_output_path"]
    project_name = config["project_name"]
    airport_csv_path = r"C:\Users\aqeel\The University of Manchester\UOM-RG-FSE-DUST - Aqeel_Shared_Files\ENPROT_PowerBI\Flight Data\Airports Database\airports.csv"

    processor = MERRA2AODProcessor(start_date, end_date, region_bounds, output_dir)
    processor.download_files()
    processor.convert_to_csv()
    AOD = MERRA2AODANAProcessor(start_date, end_date, region_bounds, output_dir)
    AOD_Data = AOD.run()

    analyzer = MonthlyDustAnalyzer(output_dir, project_name)
    analyzer.analyze(csv_output_path)
    storm_detector = StormDetector(
        output_dir, project_name, airport_csv_path, csv_output_path
    )
    storm_detector.detect_storms()
    storm_detector.save_results()
