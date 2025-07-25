FusionStormDetector Module
==========================

Overview
--------

The ``FusionStormDetector`` module is a high-confidence storm detection system developed as part of the **Dusty Pipeline**. It extends the core functionality of the ``StormDetector`` by integrating data from both **DUSMASS** (Dust Mass Concentration) and **AODANA** (Aerosol Optical Depth Analysis) variables, enabling **multi-source fusion detection** of dust storms.

By independently identifying dust storm blobs in both datasets and analyzing their spatial overlap, the module ensures only robust and physically consistent storms are flagged, thereby reducing false positives and improving validation accuracy.

Functionality
-------------

1. **Independent Blob Detection**:
   - Processes DUSMASS and AODANA files **separately** using dynamic thresholding and blob detection.
   - Detects candidate blobs from each source and stores them as individual feature sets.

2. **Blob Fusion Logic**:
   - Compares DUSMASS and AODANA blobs from the **same date**.
   - Determines spatial overlap using bounding box or pixel-level intersection methods.
   - Only blobs with **sufficient overlap** (defined by a configurable `%match threshold`) are retained as valid dust storms.

3. **Airport Proximity Filtering** *(Optional)*:
   - As in ``StormDetector``, filters blobs that are within 10–15 km of large airports using ICAO lookup.
   - Appends nearest airport ICAO and distance for situational awareness and downstream integration.

4. **Storm Lifecycle Tracking** *(Advanced)*:
   - If enabled, it tracks fused blobs across multiple dates using spatial continuity and mass similarity.
   - Outputs grouped events representing the evolution of a single storm across time.

Integration in Dusty Pipeline
-----------------------------

The ``FusionStormDetector`` replaces ``StormDetector`` when **fusion mode** is selected in the pipeline configuration (e.g., via the `mode_selector` flag in `DustyMain.py`). Its results are passed downstream for monthly aggregation, METAR cross-verification, or operational alerting.

Parameters & Customization
--------------------------

- ``overlap_threshold``: Minimum % overlap required to fuse DUSMASS and AODANA blobs.
- ``min_blob_area``: Minimum number of pixels for blob detection.
- ``apply_airport_filtering``: Boolean toggle to restrict detection near airports.
- ``enable_tracking``: Boolean to enable multi-day storm lifecycle tracking.

Dependencies
------------

- `NumPy`
- `Pandas`
- `Shapely` or `OpenCV` (for blob intersection logic)
- `Geopy` (for airport filtering)
- `Datetime` and `os` for tracking and file management

CSV Output Format
-----------------

Each output row contains:

- ``date``
- ``storm_id``
- ``center_lat``
- ``center_lon``
- ``avg_dust_mass``
- ``avg_aodana``
- ``overlap_ratio``
- ``nearest_airport_icao`` (if applicable)
- ``distance_to_airport_km`` (if applicable)

Advantages of Fusion Detection
------------------------------

- **Greater Confidence**: Confirms events using two independent physical variables.
- **Lower False Positives**: Reduces detection of transient anomalies that only appear in one dataset.
- **Enhanced Scientific Utility**: Supports climatology studies and airport-level hazard analytics.

Use Cases
---------

- High-confidence hazard detection near airports
- Scientific research on dust-aerosol coupling
- Validation of METAR-reported dust events with satellite fusion

``FusionStormDetector`` brings scientific rigor and spatial coherence to dust storm detection and is a cornerstone module in advanced deployments of the Dusty Pipeline.
