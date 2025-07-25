StormDetector Module
====================

Overview
--------

The ``StormDetector`` module is a core component of the **Dusty Pipeline**, responsible for identifying and characterizing dust storms from satellite-derived dust mass concentration data (DUSMASS). It uses advanced image processing techniques, including dynamic thresholding and blob detection, to detect coherent regions of high dust concentration that may represent dust storm events.

Functionality
-------------

This module processes gridded MERRA-2 DUSMASS data files (typically daily) and performs the following operations:

1. **Dynamic Thresholding**:
   - Calculates a baseline threshold per month or dynamically from the dataset.
   - Flags pixels exceeding this threshold as potential dust storm indicators.

2. **Blob Detection**:
   - Groups adjacent high-value pixels into blobs using contour-based or connected component labeling techniques.
   - Filters blobs based on spatial extent and mass thresholds.

3. **Geospatial Filtering (Optional)**:
   - Optionally filters detected blobs based on their proximity to large airports.
   - Airports within 10–15 km of a blob can be flagged using ICAO codes and distance calculations.

4. **Output Generation**:
   - Produces CSV summaries for each day, including:
     - Blob center coordinates (lat, lon)
     - Average and maximum DUSMASS within each blob
     - Airport proximity details (if applicable)

5. **Configurable Parameters**:
   - Threshold multiplier
   - Minimum blob size
   - Airport filtering toggle
   - Region of interest (ROI) masking

Integration in Dusty Pipeline
-----------------------------

``StormDetector`` is typically called from the ``DustyMain.py`` script during standard (non-fusion) runs. It feeds its results into the monthly aggregation module or into verification modules like ``AirportDustFlagger``.

Dependencies
------------

- `NumPy`
- `Pandas`
- `OpenCV` or `scikit-image` (for image processing)
- `Geopy` (for airport distance calculations)

CSV Output Format
-----------------

Each output file includes the following columns:

- ``date``
- ``storm_id``
- ``center_lat``
- ``center_lon``
- ``avg_dust_mass``
- ``max_dust_mass``
- ``nearest_airport_icao`` (if enabled)
- ``distance_to_airport_km`` (if enabled)

Use Cases
---------

- Real-time or retrospective dust storm tracking
- Data-driven airport disruption analysis
- Feeding storm blobs into machine learning or forecasting models
