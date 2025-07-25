import os
from datetime import datetime
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import json

from METARvsMERRA import AirportDustFlagger, AirportAODFlagger

# ========================= WELCOME MESSAGE ========================= #
print("\033[95m" + "=" * 65)
print("\033[94m" + " " * 23 + "METAR DUST CHECK")
print("\033[96m" + " " * 10 + "Developed by Muhammad Aqeel Abdulla")
print("\033[95m" + "=" * 65 + "\033[0m")
print("\n\033[0m" + "METAR Dust Check is a verification module that cross-references")
print("airport METAR reports with satellite-based dust mass concentration data.")
print("It provides a dual-source validation of dust storm events by comparing:")
print("\n - Reported visibility and dust conditions in METARs")
print(" - Dust concentration anomalies from MERRA-2 datasets")
print("\nThe system flags dust storm detections using two passes:")
print(
    " 1. MERRA-led: Detects dust spikes in satellite data and checks for METAR agreement."
)
print(" 2. METAR-led: Flags METAR dust codes and confirms with satellite data.")
print("\nFinal outputs include:")
print(" - A CSV log of agreement statistics by airport and hour")
print(" - Source-coded detection outputs (1 = MERRA, 2 = METAR, 3 = Both)")
print("\033[95m" + "=" * 65 + "\033[0m\n")

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
