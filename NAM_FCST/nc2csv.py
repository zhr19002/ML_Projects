import os
import glob
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta

# --- Path Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NC_DIR = os.path.join(BASE_DIR, "netcdf_files")
OUTPUT_CSV = os.path.join(BASE_DIR, "Sikorsky_fcst_00.csv")

# --- Loop through NetCDF files ---
records = []
nc_files = sorted(glob.glob(os.path.join(NC_DIR, "*.nc")))
for nc_file in nc_files:
    try:
        ds = xr.open_dataset(nc_file)
        u10_mean = ds.u10.mean().item() if "u10" in ds else None
        v10_mean = ds.v10.mean().item() if "v10" in ds else None

        # Extract timestamp and convert into EST
        fname = os.path.basename(nc_file)
        name_no_ext = os.path.splitext(fname)[0]

        parts = name_no_ext.split("_")
        date_str = parts[1]
        hour_str = parts[2].replace("f", "")

        timestamp_utc = datetime.strptime(date_str, "%Y%m%d") + timedelta(hours=int(hour_str))
        timestamp_est = timestamp_utc - timedelta(hours=5)
        timestamp = timestamp_est.strftime("%Y-%m-%d %H:%M:%S")

        records.append({"TmStamp": timestamp, "u10": u10_mean, "v10": v10_mean})
        print(f"Processed: {fname}")

    except Exception as e:
        print(f"Error processing {nc_file}: {e}")

# --- Save to CSV ---
df = pd.DataFrame(records)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved summary to: {OUTPUT_CSV}")