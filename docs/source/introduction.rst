Introduction
===============


This page takes you through the basics of using Dusty, including how to set up your environment and run your first simulation.



Example usage
=============

.. code-block:: python

   from Dust_Storm_Modules import MERRA2AODProcessor

   processor = MERRA2AODProcessor("20220101", "20220105", (10, 30, 30, 50))
   processor.download_files()
I am going to do this Radio Check