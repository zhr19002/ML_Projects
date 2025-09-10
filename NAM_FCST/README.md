## Script Descriptions

- **`grib_downloader.py`**  
  Downloads NAM GRIB2 forecast files from NOAA's archive for a specified date range and forecast hours, ensuring logging, file existence checks, and data integrity.

- **`grib2nc.py`**  
  Converts GRIB2 forecast files into NetCDF format by extracting selected variables (`u10`, `v10`), applying a geographic subset, and saving the results with logging and error handling.

- **`nc2csv.py`**  
  Processes NetCDF forecast files to compute the mean 10-meter wind components (`u10`, `v10`), converts forecast timestamps from UTC to EST, and saves the results into a CSV file.
