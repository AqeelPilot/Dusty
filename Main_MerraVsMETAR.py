import os
from datetime import datetime
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import json

from METARvsMERRA import AirportDustFlagger, AirportAODFlagger

# ========================= WELCOME MESSAGE ========================= #
print("=" * 65)
print(" " * 23 + "METAR DUST CHECK")
print(" " * 10 + "Developed by Muhammad Aqeel Abdulla")
print("=" * 65)
print("\n")
# =================================================================== #

if __name__ == "__main__":
    # Hide GUI window and ask for config path
    Tk().withdraw()
    config_path = askopenfilename(
        title="Select Config JSON", filetypes=[("JSON files", "*.json")]
    )
    if not config_path:
        raise FileNotFoundError("No configuration file selected.")

    # Load config to extract METAR and airport CSV locations
    with open(config_path, "r") as f:
        config = json.load(f)

    # You can also make these static if you want to hardcode paths
    metar_csv_path = r"C:\Users\aqeel\Downloads\ME_Airports_METAR.csv"
    airport_csv_path = r"C:\Users\aqeel\The University of Manchester\UOM-RG-FSE-DUST - Aqeel_Shared_Files\ENPROT_PowerBI\Flight Data\Airports Database\airports.csv"

    # flagger = AirportDustFlagger(
    #     metar_csv_path=metar_csv_path,
    #     config_path=config_path,
    #     airport_csv_path=airport_csv_path,
    # )

    # flagger.run()

    flagger = AirportAODFlagger(
        metar_csv_path=metar_csv_path,
        config_path=config_path,
        airport_csv_path=airport_csv_path,
    )
    flagger.run()
