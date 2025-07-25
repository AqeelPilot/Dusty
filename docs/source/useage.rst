Usage
=====

This guide walks you through how to run the DUSTY system from start to finish, using your own `.json` configuration file and selecting between classical and fusion-based dust storm detection modes.

Step-by-Step Execution
----------------------

To run the DUSTY pipeline:

1. Open a terminal in your project directory.

2. Run the main script:

   .. code-block:: bash

      python DustyMain.py

3. A file dialog will prompt you to select a configuration `.json` file.  
   Choose the one you created (see :doc:`installation` for structure).

4. You will be prompted:

   .. code-block:: text

      Would you like to use Fusion Storm Detector Module 
      1 or 0

   - Enter `1` for **FusionStormDetector** (uses both DUSMASS + AODANA)
   - Enter `0` for **StormDetector** (uses DUSMASS only)

5. The program will begin downloading and processing data.
   A progress bar will appear for each stage of the pipeline.

Modules Automatically Run
-------------------------

Based on your inputs, the following modules are executed:

- **MERRA2AODProcessor** – Downloads and processes dust mass concentration files (`DUSMASS`)
- **MERRA2AODANAProcessor** – Downloads and processes AODANA files if fusion mode is selected
- **MonthlyDustAnalyzer** – Computes time-resolved monthly dust statistics
- **StormDetector** or **FusionStormDetector** – Detects and tracks storm blobs day-by-day

Expected Outputs
----------------

All output files are saved in the `csv_output_path` defined in your `.json` config.

Key output files:

- `<project_name>_storm_lifecycle.csv`  
  A daily record of detected storm blobs with storm ID, size, dust mass, and airport info.

- `<project_name>_monthly_dust_summary.csv`  
  A statistical summary of dust activity across the region, highlighting anomalies.

- `<project_name>_monthly_AODANA_summary.csv`  
  (Fusion mode only) Similar summary based on aerosol optical depth.

- Multiple intermediate CSVs for each day’s DUSMASS and AODANA data

Troubleshooting
---------------

- **Download errors?** Make sure your `.netrc` file is correctly set up and your NASA Earthdata login is valid.
- **Empty output?** Ensure the date and region bounds match days and locations with known dust activity.
- **Slow performance?** Try reducing the date range or geographic area in your config.

Best Practices
--------------

- Keep each run in a unique project folder to avoid overwriting CSVs
- Always check that the output files are being written to the correct directory
- Use the same time format (`%d/%m/%Y %H:%M`) in your CSVs and code when debugging

