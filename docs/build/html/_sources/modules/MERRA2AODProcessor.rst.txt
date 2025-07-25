MERRA2AODProcessor
==================

.. automodule:: MERRA2AODProcessor
   :members:
   :undoc-members:
   :show-inheritance:


The `MERRA2AODProcessor` class is responsible for downloading, extracting, and converting daily dust surface mass concentration (`DUSMASS`) values from the NASA MERRA-2 dataset. The converted data is stored as `.csv` files for downstream analysis and storm detection.

This module enables researchers to extract data from a user-defined region and time window, preparing it in a standardized format for further processing by other DUSTY modules.

Overview
--------

**Class:** `MERRA2AODProcessor`  
**Location:** `Dust_Storm_Modules.py`

Constructor
-----------

.. code-block:: python

   MERRA2AODProcessor(start_date, end_date, region_bounds, output_dir="data/merra2")

- `start_date`: `datetime` – Beginning of the data range
- `end_date`: `datetime` – End of the data range
- `region_bounds`: Tuple of `[lat_min, lat_max, lon_min, lon_max]`
- `output_dir`: Path to save the downloaded and processed files

Methods
-------

.. autoclass:: Dust_Storm_Modules.MERRA2AODProcessor
   :members:
   :undoc-members:
   :show-inheritance:

Workflow
--------

1. **`download_files()`**  
   Downloads daily MERRA-2 `DUSMASS` `.nc4` files from NASA's GES DISC for the specified region and date range.

2. **`convert_to_csv()`**  
   Converts `.nc4` files into flat `.csv` format:
   - Time-resolved
   - Includes `latitude`, `longitude`, and `dust_mass` columns
   - Filters only valid (finite) values

Notes
-----

- The processor automatically adjusts longitudes < 0 to 0–360 format, as required by MERRA-2.
- The NetCDF files are opened using `xarray`, and the output CSVs are formatted using `pandas`.

Example Usage
-------------

.. code-block:: python

   from Dust_Storm_Modules import MERRA2AODProcessor
   from datetime import datetime

   processor = MERRA2AODProcessor(
       start_date=datetime(2023, 6, 1),
       end_date=datetime(2023, 6, 7),
       region_bounds=(15, 35, 40, 65),
       output_dir="data/merra2"
   )
   processor.download_files()
   processor.convert_to_csv()

