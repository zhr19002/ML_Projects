import os
import glob
import logging
import xarray as xr

# --- Path Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GRIB_DIR = os.path.join(BASE_DIR, "grib_files")
NC_DIR = os.path.join(BASE_DIR, "netcdf_files")
LOG_FILE = os.path.join(BASE_DIR, "grib2nc.log")

# --- Geographic subset ---
LON_MIN = -73.3
LON_MAX = -73.0
LAT_MIN = 41.1
LAT_MAX = 41.3

# --- Variables to extract with cfgrib filter keys ---
level_filters = {
    "u10": {"shortName": "10u"},
    "v10": {"shortName": "10v"},
}

# --- Setup output directory and logging ---
os.makedirs(NC_DIR, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    filemode='a',
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

def convert_grib_to_netcdf(grib_path):
    fname = os.path.basename(grib_path)
    name_no_ext = os.path.splitext(fname)[0]
    nc_path = os.path.join(NC_DIR, f"{name_no_ext}.nc")

    if os.path.exists(nc_path):
        logging.info(f"{nc_path} already exists. Skipping {fname}.")
        return

    datasets = []
    for var, filt in level_filters.items():
        try:
            logging.info(f"{fname} → extracting '{var}'")
            ds = xr.open_dataset(grib_path, engine="cfgrib", filter_by_keys=filt, decode_timedelta=False)
            if 'longitude' in ds.coords:
                ds = ds.assign_coords(longitude=xr.where(ds.longitude > 180, ds.longitude - 360, ds.longitude))
            datasets.append(ds)
        except Exception as e:
            logging.error(f"Failed to extract {var} from {fname}: {e}")

    if not datasets:
        logging.warning(f"No valid variables found in {fname}")
        return

    try:
        ds_merged = xr.merge(datasets, compat="no_conflicts")

        # Apply geographic subset
        if all(v is not None for v in [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX]):
            if 'latitude' in ds_merged.coords and 'longitude' in ds_merged.coords:
                ds_merged = ds_merged.where(
                    (ds_merged.longitude >= LON_MIN) & (ds_merged.longitude <= LON_MAX) &
                    (ds_merged.latitude >= LAT_MIN) & (ds_merged.latitude <= LAT_MAX),
                    drop=True
                )

        ds_merged.to_netcdf(nc_path)
        logging.info(f"Saved NetCDF: {nc_path}")
        print(f"Converted: {fname}")

    except Exception as e:
        logging.exception(f"Error writing NetCDF for {fname}: {e}")
        print(f"Error processing {fname}. See log for details.")


def main():
    print("Starting GRIB2 to NetCDF conversion...")

    grib_files = sorted(glob.glob(os.path.join(GRIB_DIR, "nam_*.grb2")))
    if not grib_files:
        print(f"No GRIB2 files found in {GRIB_DIR}")
        return

    for grib_path in grib_files:
        convert_grib_to_netcdf(grib_path)

    print("Conversion completed.")


if __name__ == "__main__":
    main()