import tensorflow as tf
import numpy as np
import sys
import os
import argparse
from obspy import read
import pandas as pd
import copy
import csv
import fnmatch
from scipy.signal import butter, filtfilt
from datetime import datetime
import scipy.signal as signal
from models import base_model 
import function_tools

def bandpass_filter(data, fs, lowcut, highcut, order=4):
    """Applies a Butterworth bandpass filter."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

def max_min_mean(data):
    """Applies min-max normalization and removes the mean from 1D data."""
    max_val, min_val = np.max(data), np.min(data)
    if (max_val - min_val) > 1e-9:
        data = (data - min_val) / (max_val - min_val)
    data = data - np.mean(data)
    return data
    
def regulation(data):
    """Applies normalization to the entire dataset of chunks."""
    print("  Applying regulation (normalization)...")
    for i in range(data.shape[0]):
        for j in range(data.shape[3]):
            data[i, :, 0, j] = max_min_mean(data[i, :, 0, j])
    return data

def find_stream_by_pattern(stream, pattern):
    """Finds a stream component by wildcard pattern."""
    for st in stream:
        if fnmatch.fnmatch(st.stats.channel, pattern):
            return st
    return None

def evnt_time_get(iso_time_str):
    """Parses time from a UTCDateTime string."""
    iso_time_str = str(iso_time_str)
    if '.' in iso_time_str:
        date_time_str, microseconds_str = iso_time_str.rsplit('.', 1)
        microseconds = int(microseconds_str.rstrip('Z')) / 1e6
    else:
        date_time_str, microseconds = iso_time_str.rstrip('Z'), 0
    dt_object = datetime.strptime(date_time_str, '%Y-%m-%dT%H:%M:%S')
    return dt_object.year, dt_object.month, dt_object.day, dt_object.hour, dt_object.minute, dt_object.second, microseconds

def main(args):
    """Main universal prediction function."""
    
    print(f"Setting GPU ID to: {args.gpu_id}")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    checkpoint_path = os.path.join(args.model_base_path, args.model_name, 'model_save', args.ckpt_name)
    checkpoint_path = os.path.abspath(checkpoint_path.replace("\\", "/"))

    model = base_model()
    print(f"Loading model weights from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path + '.index'):
        print(f"Error: Model checkpoint not found at the specified path: {checkpoint_path}")
        sys.exit(1)
    model.load_weights(checkpoint_path).expect_partial()
    print("Model loaded successfully.")

    output_csv_path = os.path.abspath(args.output_csv.replace("\\", "/"))
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, "w", newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['num', 'year', 'month', 'day', 'hour', 'min', 'sec', 'T', 'type', 'pro'])
    print(f"Output will be saved to: {output_csv_path}")
    
    input_dir = os.path.abspath(args.input_dir.replace("\\", "/"))
    filenameList = [f for f in sorted(os.listdir(input_dir)) if fnmatch.fnmatch(f, args.file_pattern)]
    
    if not filenameList:
        print(f"Warning: No files found in '{input_dir}' matching pattern '{args.file_pattern}'.")
        return

    print(f"Found {len(filenameList)} files to process.")
    total_pick_num = 0

    for i_file, filename in enumerate(filenameList):
        print(f"\n--- Processing file {i_file+1}/{len(filenameList)}: {filename} ---")
        filepath = os.path.join(input_dir, filename)
        
        try:
            stream = read(filepath)
        except Exception as e:
            print(f"  Could not read file. Error: {e}. Skipping.")
            continue
        
        if len(stream) < 3:
            print("  Stream contains less than 3 traces. Skipping.")
            continue

        st_E = find_stream_by_pattern(stream, '*E')
        st_N = find_stream_by_pattern(stream, '*N')
        st_Z = find_stream_by_pattern(stream, '*Z')

        if not all([st_E, st_N, st_Z]):
            print("  Could not find all E, N, Z components. Skipping.")
            continue
        
        npts = st_E.stats.npts
        if npts < 6000:
            print("  Trace length is less than 6000 samples. Skipping.")
            continue

        data_E = bandpass_filter(st_E.data, st_E.stats.sampling_rate, 1, 49)
        data_N = bandpass_filter(st_N.data, st_N.stats.sampling_rate, 1, 49)
        data_Z = bandpass_filter(st_Z.data, st_Z.stats.sampling_rate, 1, 49)
        
        data_out_list = function_tools.segment_data(data_E, data_N, data_Z)
        data_out_np = np.array(data_out_list)
        data_out_np = regulation(data_out_np)
        
        print(f"  Predicting on {data_out_np.shape[0]} data chunks...")
        result = model.predict(data_out_np, batch_size=args.batch_size)
        
        print("  Post-processing results...")
        _, result_RF, _ = function_tools.reconstruct_data(result, npts)
        RF_start, RF_end, pro_r = function_tools.pick_continuous_events(result_RF, args.threshold)
        
        print(f"  Found {len(RF_start)} events. Appending to CSV...")
        with open(output_csv_path, "a", newline='') as f:
            csv_writer = csv.writer(f)
            for i in range(len(RF_start)):
                pick_time = st_E.stats.starttime + (RF_start[i] * 0.01)
                T_time = (RF_end[i] - RF_start[i]) * 0.01
                year,month,day,hour,minute,sec,mirosec = evnt_time_get(pick_time)
                csv_writer.writerow([total_pick_num,year,month,day,hour,minute,f"{sec+mirosec:.2f}",T_time,'R',pro_r[i]])
                total_pick_num += 1

    print(f"\nAll tasks finished! A total of {total_pick_num} events were written to {output_csv_path}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Universal Seismic Event Prediction Script", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    core_group = parser.add_argument_group('Core Arguments')
    core_group.add_argument('--input_dir', type=str, required=True, help='Directory containing the input .mseed files.')
    core_group.add_argument('--file_pattern', type=str, required=True, help='Wildcard pattern to match .mseed files (e.g., "*.mseed" or "STA4.*.mseed").')
    core_group.add_argument('--output_csv', type=str, required=True, help='Full path for the output CSV results file.')
    
    model_group = parser.add_argument_group('Model Loading Arguments')
    model_group.add_argument('--model_base_path', type=str, default="./attention_model/model/", help='Base directory containing different model folders.')
    model_group.add_argument('--model_name', type=str, required=True, help='Name of the trained model subdirectory (e.g., "GRUV3").')
    model_group.add_argument('--ckpt_name', type=str, default="model_cyx.ckpt", help='Name of the checkpoint file itself.')

    pred_group = parser.add_argument_group('General Prediction Arguments')
    pred_group.add_argument('--gpu_id', type=str, default="0", help='GPU device ID to use.')
    pred_group.add_argument('--threshold', type=float, default=0.7, help='Probability threshold for event picking.')
    pred_group.add_argument('--batch_size', type=int, default=128, help='Batch size for model prediction.')

    args = parser.parse_args()
    main(args)