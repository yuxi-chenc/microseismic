import requests
import os
import csv
from datetime import datetime, timedelta
import argparse
from tqdm import tqdm

def setup_logging(log_file_path):
    """Initializes the log file and writes the header."""
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    with open(log_file_path, "w", newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['year', 'month', 'day', 'download_status'])

def log_status(log_file_path, date, status):
    """Writes the download status for a single day to the log file."""
    with open(log_file_path, "a", newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([date.year, f"{date.month:02}", f"{date.day:02}", status])

def download_daily_data(session, date_to_download, params):
    """
    Downloads data for a single specified day.
    
    Args:
        session (requests.Session): The session object for making requests.
        date_to_download (datetime): The date for which to download data.
        params (dict): A dictionary containing network, station, etc.
        
    Returns:
        bool: True if download is successful, False otherwise.
    """
    start_time = date_to_download.strftime('%Y-%m-%dT00:00:00')
    end_time = (date_to_download + timedelta(days=1)).strftime('%Y-%m-%dT00:00:00')
    
    url = (
        f"{params['base_url']}?net={params['network']}&sta={params['station']}"
        f"&loc={params['location']}&channel={params['channel_pattern']}"
        f"&starttime={start_time}&endtime={end_time}&quality=M"
    )
    
    file_name = f"{params['station']}_{date_to_download.strftime('%Y%m%d')}.mseed"
    output_file_path = os.path.join(params['output_dir'], file_name)

    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()

        if response.content:
            with open(output_file_path, "wb") as f:
                f.write(response.content)
            return True
        else:
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"Error downloading for date {date_to_download.strftime('%Y-%m-%d')}: {e}")
        return False

def main(args):
    """Main execution function."""
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    log_file = os.path.join(args.output_dir, "download_log.csv")
    setup_logging(log_file)
    
    total_days = (end_date - start_date).days + 1
    
    download_params = {
        "base_url": "https://ws.resif.fr/fdsnws/dataselect/1/query",
        "network": args.network,
        "station": args.station,
        "location": args.location,
        "channel_pattern": args.channel + "*",
        "output_dir": args.output_dir
    }
    
    with requests.Session() as session:
        for i in tqdm(range(total_days), desc="Downloading Data"):
            current_date = start_date + timedelta(days=i)
            success = download_daily_data(session, current_date, download_params)
            log_status(log_file, current_date, 'T' if success else 'F')
            
    print("\nDownload process finished.")
    print(f"Log file saved to: {log_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Downloads seismic waveform data from the RESIF FDSN web service.")
    
    parser.add_argument('--start_date', type=str, default='2013-01-09', help='Start date in YYYY-MM-DD format.')
    parser.add_argument('--end_date', type=str, default='2019-12-31', help='End date in YYYY-MM-DD format.')
    parser.add_argument('--output_dir', type=str, default='./data_downed', help='Directory to save mseed files and the log.')
    parser.add_argument('--network', type=str, default='MT', help='Network code (Net).')
    parser.add_argument('--station', type=str, default='THE', help='Station code (Sta).')
    parser.add_argument('--location', type=str, default='06', help='Location code (Loc).')
    parser.add_argument('--channel', type=str, default='CH', help='Channel code (Cha), a wildcard * will be appended.')
    
    args = parser.parse_args()
    main(args)