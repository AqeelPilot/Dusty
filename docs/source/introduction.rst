Welcome to DUSTY
================

Overview
--------

**DUSTY** is a modular atmospheric analytics engine developed by **Muhammad Aqeel Abdulla** for tracking and characterizing dust storm events using satellite-based reanalysis datasets. The system is engineered to automate the detection, analysis, and lifecycle tracking of dust storms across both time and space using high-resolution NASA MERRA-2 data.

It supports both classical and fusion-based storm detection, correlating satellite observations of dust mass and aerosol optical depth with large airport locations to assess potential operational impact. DUSTY is optimized for environmental data scientists, atmospheric researchers, and aerospace engineers working on climate-aware aviation systems.

Capabilities
------------

- Downloads and converts MERRA-2 NetCDF files for both `DUSMASS` and `AODANA`
- Identifies and tracks dust storm “blobs” using dynamic spatial thresholds
- Associates detected storm centers with the nearest large airport (ICAO coded)
- Computes monthly statistical baselines and identifies anomalies
- Supports both standalone and fusion-based storm detection modes
- Outputs CSV summaries of storm evolution and statistical deviations

Use Cases
---------

DUSTY was developed as part of a broader research effort into the **interaction between dust storms and aviation operations**, particularly the ingestion risk posed to commercial aircraft engines. It has been applied to:

- Quantifying dust activity around airports
- Analyzing storm frequency and severity trends over months
- Supporting data fusion pipelines for multi-source dust monitoring
- Ground-truthing atmospheric model outputs with reanalysis data

Project Layout
--------------

The project is organized into modular components:

- :doc:`installation`: Set up DUSTY and its dependencies
- :doc:`useage`: Learn how to run and configure DUSTY
- :doc:`modules/MERRA2AODProcessor`: Convert and process DUSMASS files
- :doc:`modules/MonthlyDustAnalyzer`: Statistical monthly summaries
- :doc:`modules/StormDetector`: Classical storm detection
- :doc:`modules/FusionStormDetector`: Fusion-based storm detection


Author & License
----------------

Developed by **Muhammad Aqeel Abdulla**, this project was part of a research initiative at the University of Manchester. All rights reserved unless otherwise stated.

