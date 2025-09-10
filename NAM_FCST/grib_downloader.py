import os
import requests
from datetime import datetime, timedelta
import logging

# --- Configuration ---
START_DATE = "20250601"
END_DATE = "20250630"
MAX_FORECAST_HOUR = 24

# --- Directory Setup ---
SCRIPT_DIR = os.path.dirname(__file__)
GRIB_DIR = os.path.join(SCRIPT_DIR, "grib_files")
LOG_FILE = os.path.join(SCRIPT_DIR, "grib_downloader.log")

os.makedirs(GRIB_DIR, exist_ok=True)

# --- Logging Setup ---
logging.basicConfig(
    filename=LOG_FILE,
    filemode='a',
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

# --- Download Function ---
def download_file(url, dest_path):
    try:
        logging.info(f"Downloading: {url}")
        response = requests.get(url, timeout=60)
        if response.status_code == 200 and response.content.strip():
            with open(dest_path, 'wb') as f:
                f.write(response.content)
            logging.info(f"Saved: {dest_path}")
            return True
        else:
            logging.warning(f"Failed (empty or bad response): {url}")
            return False
    except Exception as e:
        logging.error(f"Error downloading {url}: {e}")
        return False

# --- Date Generator ---
def daterange(start_date, end_date):
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    delta = timedelta(days=1)
    while start <= end:
        yield start.strftime("%Y%m%d")
        start += delta

# --- Main Logic ---
def main():
    print("Starting NAM GRIB2 file download...")
    logging.info("=== NAM GRIB2 Download Started ===")
    
    try:
        for current_date in daterange(START_DATE, END_DATE):
            logging.info(f"Processing date: {current_date}")
            year = current_date[:4]
            month = current_date[4:6]

            base_url = f"https://www.ncei.noaa.gov/data/north-american-mesoscale-model/access/forecast/{year}{month}/{current_date}"

            for fh in range(1, MAX_FORECAST_HOUR + 1):
                fh_str = f"{fh:03d}"
                filename = f"nam_{current_date}_f{fh_str}.grb2"
                file_path = os.path.join(GRIB_DIR, filename)
                file_url = f"{base_url}/nam_218_{current_date}_0000_{fh_str}.grb2"

                if not (os.path.exists(file_path) and os.path.getsize(file_path) > 0):
                    success = download_file(file_url, file_path)
                    if not success and os.path.exists(file_path):
                        os.remove(file_path)
                        logging.info(f"Removed incomplete file: {file_path}")
                else:
                    logging.info(f"Already exists: {file_path}")

    except Exception as e:
        logging.exception("Unexpected error occurred during download.")
        print("An error occurred. Check the log file for details.")
    else:
        logging.info("=== NAM GRIB2 Download Completed ===")
        print("Download completed successfully.")

if __name__ == "__main__":
    main()