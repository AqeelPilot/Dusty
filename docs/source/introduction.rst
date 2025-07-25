Introduction to DUSTY
=====================

Overview
--------

**DUSTY** is a modular, Python-based atmospheric data analytics tool designed by **Muhammad Aqeel Abdulla** to track, quantify, and analyze dust storm events using satellite-derived datasets. The system integrates **MERRA-2** reanalysis data for both **dust mass surface concentration** (`DUSMASS`) and **aerosol optical depth** (`AODANA`), allowing for both classical and fusion-based storm detection workflows.

The tool is engineered to automate:
- Dust storm file acquisition and processing from NASA's GES DISC portal
- Conversion of raw `.nc4` NetCDF data into time-resolved `.csv` files
- Identification of storm "blobs" using thresholding and blob tracking
- Association of detected dust events with the nearest large airport
- Generation of both storm lifecycle and monthly anomaly summaries

DUSTY is configured via a `.json` configuration file and can be executed via a user-friendly command-line script, `DustyMain.py`.

Key Features
------------

- Fully automated NetCDF to CSV conversion using `xarray` and `pandas`
- Time-resolved detection of storm "blobs" using statistical thresholds
- Dual-mode operation: `DUSMASS`-only or `Fusion` with both `DUSMASS` and `AODANA`
- Proximity-based association of dust events with ICAO airport codes
- Multi-day storm tracking across geospatial and temporal domains
- Monthly summary reports of anomalies based on historical standard deviation

Main Components
---------------

The DUSTY system is built around the following core classes, defined in `Dust_Storm_Modules.py`:

**1. MERRA2AODProcessor**
   - Downloads and processes `DUSMASS` data from the MERRA-2 repository.
   - Clips data to a specified bounding box and converts to time-tagged CSV.

**2. MERRA2AODANAProcessor**
   - Handles acquisition and processing of `AODANA` aerosol optical depth data.
   - Outputs values in parallel with `DUSMASS` for fusion-based storm detection.

**3. MonthlyDustAnalyzer**
   - Performs a full statistical analysis of dust values by time of day.
   - Produces a CSV report highlighting dust spikes >2× standard deviation above monthly means.

**4. StormDetector**
   - Applies dynamic thresholding and connected-component labeling on `DUSMASS` CSVs.
   - Matches blobs to nearby large airports and tracks them across days.

**5. FusionStormDetector**
   - A parallel method to `StormDetector`, but uses `AODANA` data for detection.
   - Supports blob continuity across days and generates monthly summary reports.

Execution Flow
--------------

The system is run using `DustyMain.py`, which performs the following sequence:

1. Asks the user to load a JSON configuration file (via GUI).
2. Extracts region, date range, output directories, and project metadata.
3. Downloads and converts `DUSMASS` and `AODANA` NetCDF files to CSV.
4. Performs statistical baseline generation using `MonthlyDustAnalyzer`.
5. Based on user selection (`Fusion` or `Standard` mode), activates:
   - `StormDetector` for DUSMASS-only detection
   - `FusionStormDetector` for AODANA-powered storm tracking
6. Outputs:
   - `*_storm_lifecycle.csv`: a record of all tracked dust blobs
   - `*_monthly_dust_summary.csv` and `*_monthly_AODANA_summary.csv`: anomaly records

Configuration
-------------

The `.json` config file must define:

- `region_bounds`: `[lat_min, lat_max, lon_min, lon_max]`
- `start_date`, `end_date`: e.g. `"2023-07-01"`
- `modis_output_dir`, `AOD_output_dir`, `csv_output_path`
- `project_name`

An example prompt in the script asks:

.. code-block:: text

   Would you like to use Fusion Storm Detector Module 
   1 or 0

Selecting `1` will activate `FusionStormDetector`.

Outputs
-------

Each execution of DUSTY produces:

- `*_storm_lifecycle.csv`: Lifecycle of each detected storm with ID, size, mass, center, airport.
- `*_monthly_dust_summary.csv`: Statistical deviations for each lat/lon/time combination.
- `*_monthly_AODANA_summary.csv`: (Fusion mode) Same as above but based on `AODANA`.

Credits
-------

Developed by **Muhammad Aqeel Abdulla**, this tool forms the analytical foundation for integrating atmospheric datasets with air traffic and airport-centric impact modeling.

