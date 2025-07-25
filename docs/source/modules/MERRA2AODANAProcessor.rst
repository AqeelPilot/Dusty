MERRA2AODANAProcessor
=====================

Overview
--------

The `MERRA2AODANAProcessor` module is designed to extract and convert aerosol optical depth data (`AODANA`) from NASA’s MERRA-2 atmospheric reanalysis dataset. Specifically, it focuses on transforming `.nc4` files (NetCDF format) into structured `.csv` outputs for downstream dust storm fusion analysis and visualization.

Unlike `DUSMASS`, which measures surface-level dust mass concentration, `AODANA` represents the **total column aerosol optical depth**—a key variable for estimating aerosol load in the atmosphere. Including this variable provides a second dimension to storm confidence assessments, especially in dusty but optically active regions.

Functionality
-------------

This module performs the following tasks:

- Iterates through a directory of `.nc4` files containing daily MERRA-2 `AODANA` data.
- Extracts metadata such as:
  - Time (hourly UTC timestamps)
  - Latitude and longitude grid
  - `AODANA` values for each grid point
- Filters out invalid or missing values (e.g., fill values)
- Writes the processed data into `.csv` files, one per day, preserving temporal and spatial resolution.

Scientific Motivation
---------------------

`AODANA` is a critical metric in remote sensing and atmospheric science as it:

- Provides insight into how optically thick the atmosphere is due to aerosols
- Helps distinguish between fine particulate pollution, sea salt, and dust
- Complements `DUSMASS` by adding a **vertical optical measurement** over surface-based mass estimation

This dual perspective is particularly useful when fusing datasets in modules like `FusionStormDetector`, which require strong evidence from both mass and optical profiles to confirm a dust storm.

Integration with Other Modules
------------------------------

- Used in conjunction with `MERRA2DUSMASSProcessor` for fusion-based blob detection in `FusionStormDetector`.
- May also be compared with satellite instruments or METARs for cross-validation.
- CSVs generated here are expected in the same format as DUSMASS files to allow seamless pairing and temporal matching.

File Structure
--------------

Each generated `.csv` file contains the following fields:

- `latitude`: Grid latitude in degrees
- `longitude`: Grid longitude in degrees
- `datetime`: UTC time (with hourly resolution)
- `AODANA`: Aerosol Optical Depth at 550 nm

The filename format is standardized as:YYYYMMDD.csv


where `YYYYMMDD` corresponds to the date of the original `.nc4` file.

Usage Considerations
--------------------

- This processor assumes a consistent grid and variable name (`AODANA`) in the MERRA-2 files.
- Missing data (e.g., due to quality filtering or oceanic regions) are excluded from the CSV output.
- The output folder is automatically created if it doesn't exist.

Dependencies
------------

- `netCDF4`
- `pandas`
- `numpy`
- `os` and `glob` for file handling
- `tqdm` for progress bars

Author
------

Developed by Muhammad Aqeel Abdulla as part of the Dusty pipeline for enhanced storm tracking and verification.
