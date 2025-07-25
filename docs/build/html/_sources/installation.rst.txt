Installation
============

This page outlines the steps required to set up and run the DUSTY system for tracking and analyzing dust storms using MERRA-2 datasets.

Prerequisites
-------------

- Python 3.9 or higher
- Git (for cloning the repository)
- A Unix-like or Windows OS with internet access
- A NASA Earthdata account (required to download MERRA-2 files)

Required Python Packages
------------------------

To install the necessary dependencies, run:

.. code-block:: bash

   pip install -r requirements.txt

Your `requirements.txt` file should include (or manually install):

- numpy
- pandas
- xarray
- matplotlib
- tqdm
- requests
- netCDF4
- scipy
- geopy
- scikit-image

Clone the Repository
--------------------

Use the following command to clone the project:

.. code-block:: bash

   git clone https://github.com/AqeelPilot/DustFlight.git
   cd DustFlight

Set Up Authentication (Earthdata)
----------------------------------

MERRA-2 data requires NASA Earthdata login credentials. You need to:

1. Create an Earthdata account: https://urs.earthdata.nasa.gov
2. Create a `.netrc` file in your home directory with the following format:

.. code-block:: text

   machine urs.earthdata.nasa.gov
   login your_username
   password your_password

3. On Windows, ensure the path to `.netrc` is set using:

.. code-block:: python

   os.environ["NETRC"] = r"C:\\Users\\yourname\\login_netrc"

Or, edit this path in `MERRA2AODProcessor` and `MERRA2AODANAProcessor` constructors.

Configuration File
------------------

DUSTY requires a configuration `.json` file. This file specifies:

- `region_bounds`: `[lat_min, lat_max, lon_min, lon_max]`
- `start_date`, `end_date`: `"YYYY-MM-DD"`
- `modis_output_dir`, `AOD_output_dir`, `csv_output_path`: output folder paths
- `project_name`: project identifier used in output file names

Example:

.. code-block:: json

   {
     "region_bounds": [15, 35, 40, 65],
     "start_date": "2023-06-01",
     "end_date": "2023-06-30",
     "modis_output_dir": "data/merra2",
     "AOD_output_dir": "data/aodana",
     "csv_output_path": "output",
     "project_name": "DustyGulf"
   }

Running DUSTY
-------------

Once installed and configured, run the main script:

.. code-block:: bash

   python DustyMain.py

You will be prompted to select the configuration file and choose whether to use the Fusion Storm Detector.

