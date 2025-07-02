# -*- coding: utf-8 -*-
#!/usr/bin/env python

import numpy as np
import os
import sys
import pandas as pd
import fnmatch
import argparse
from obspy import read
from scipy.signal import butter, filtfilt, windows
from datetime import datetime

def gauss_label(npts, std_dev_factor=4):
    """
    Generates a Gaussian-shaped label.
    """
    if npts <= 0:
        return np.array([])
    std = (npts - 1) / (2 * std_dev_factor)
    g = windows.gaussian(npts, std=std)
    return g

def rangle_label(t1, t2):
    """
    Generates a rectangular-shaped label (array of ones).
    """
    length = t2 - t1
    if length <= 0:
        return np.array([])
    return np.ones(length)

def data_cut(comp1_data, comp2_data, comp3_data, npts, output_list):
    """
    Cuts long time-series data into fixed-size chunks of 6000 points.
    """
    chunk_size = 6000
    num_chunks = npts // chunk_size
    
    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size
        
        chunk1 = comp1_data[start:end]
        chunk2 = comp2_data[start:end]
        chunk3 = comp3_data[start:end]
        
        stacked_chunk = np.stack([chunk1, chunk2, chunk3], axis=-1)
        stacked_chunk = stacked_chunk.reshape((chunk_size, 1, 3))
        
        output_list.append(stacked_chunk)
        
    return output_list

def bandpass_filter(data, fs, lowcut, highcut, order=4):
    """Applies a Butterworth bandpass filter."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def max_min_mean(data):
    data = (data - min(data))/(max(data)-min(data))
    data = data - np.mean(data)
    return data
def regulation(data):
    """Normalizes and de-means the data in chunks."""
    nums = np.shape(data)
    long = nums[0]
    for i in range(long):
        data[i,0:6000,0,0] =  max_min_mean(data[i,0:6000,0,0])
        data[i,0:6000,0,1] =  max_min_mean(data[i,0:6000,0,1])
        data[i,0:6000,0,2] =  max_min_mean(data[i,0:6000,0,2])
    return data


def find_stream_by_pattern(stream_list, pattern):
    """
    Finds the first stream in a list that matches a channel wildcard pattern.
    """
    for st in stream_list:
        if fnmatch.fnmatch(st.stats.channel, pattern):
            return st
    return None

def evnt_time_get(iso_time_str):
    """Parses year, month, day, hour, minute, second from a UTCDateTime string."""
    iso_time_str = str(iso_time_str).rstrip('Z')
    if '.' in iso_time_str:
        date_time_str, microsec_str = iso_time_str.rsplit('.', 1)
        microseconds = int(microsec_str) / 1e6
    else:
        date_time_str, microseconds = iso_time_str, 0
    dt_object = datetime.strptime(date_time_str, '%Y-%m-%dT%H:%M:%S')
    return dt_object.year, dt_object.month, dt_object.day, dt_object.hour, dt_object.minute, dt_object.second, microseconds

def find_events_by_date(csv_file, year, month, day):
    """Finds events for a specific date in the catalog CSV file."""
    df = pd.read_csv(csv_file)
    mask = (df['year'] == year) & (df['month'] == month) & (df['day'] == day)
    filtered_df = df.loc[mask]
    
    event_ids = filtered_df['num'].tolist()
    return event_ids, df["hour"], df["min"], df["sec"], df["T"], df["type"]

def KSG_get(data, window_size, data_length):
    """Calculates the normalized rolling standard deviation (KSG)."""
    s = pd.Series(data)
    G_return = s.rolling(window=window_size, center=True, min_periods=1).std().to_numpy()
    G_return[np.isnan(G_return)] = 0
    
    min_g, max_g = np.min(G_return), np.max(G_return)
    if (max_g - min_g) < 1e-9:
        return np.zeros_like(G_return)
    return (G_return - min_g) / (max_g - min_g)

def generate_event_labels(args):
    """
    Main logic for generating waveform data (X) and event labels (Y)
    from mseed files and a catalog.
    """
    print("--- Starting Event Label Generation ---")
    
    EQ_label_out = gauss_label(201) 
    EQ_label_out /= np.max(EQ_label_out)

    data_out, label_out, filenameList = [], [], []

    print(f"Searching for files in '{args.data_dir}' with pattern '{args.file_pattern}'...")
    for iFile in sorted(os.listdir(args.data_dir)):
        if fnmatch.fnmatch(iFile, args.file_pattern):
            filenameList.append(iFile)
    
    if not filenameList:
        print("Error: No files found matching the pattern. Please check --data_dir and --file_pattern.")
        return

    print(f"Found {len(filenameList)} files to process.")

    for i_set, filename in enumerate(filenameList):
        print(f"Processing file {i_set + 1}/{len(filenameList)}: {filename}")
        try:
            data = read(os.path.join(args.data_dir, filename))
        except Exception as e:
            print(f"  Warning: Could not read file {filename}. Error: {e}. Skipping.")
            continue

        year, month, day, _, _, _, _ = evnt_time_get(data[0].stats.starttime)
        numbers, hours, mins, sec, T, ty = find_events_by_date(args.catalog_file, year, month, day)
        
        if not numbers:
            print(f" No catalog events found for {year}-{month}-{day}. Skipping label generation for this file.")

        if len(data) % 3 != 0:
            print("  Warning: Number of traces is not a multiple of 3. Skipping.")
            continue
            
        st_num = len(data) // 3
        for i in range(st_num):
            streams_to_check = [data[i], data[i + st_num], data[i + 2 * st_num]]

            st1, st2, st3 = streams_to_check[0], streams_to_check[1], streams_to_check[2]
            if st1.stats.npts < 6000 or st1.stats.npts != st2.stats.npts or st1.stats.npts != st3.stats.npts:
                continue
            
            # --- MODIFIED SECTION ---
            st_E = find_stream_by_pattern(streams_to_check, args.east_pattern)
            st_N = find_stream_by_pattern(streams_to_check, args.north_pattern)
            st_Z = find_stream_by_pattern(streams_to_check, args.vert_pattern)

            if not all([st_E, st_N, st_Z]):
                print(f"  Warning: Could not find all E,N,Z components using patterns for traces starting with {st1.id}. Skipping.")
                continue
            # --- END OF MODIFIED SECTION ---

            data_E = bandpass_filter(st_E.data, args.fs, args.low_cut, args.high_cut)
            data_N = bandpass_filter(st_N.data, args.fs, args.low_cut, args.high_cut)
            data_Z = bandpass_filter(st_Z.data, args.fs, args.low_cut, args.high_cut)

            s_year, s_month, s_day, s_hour, s_min, s_sec, s_micro = evnt_time_get(st1.stats.starttime)
            npts = st1.stats.npts
            label_noise = np.ones(npts)
            label_RF = np.zeros(npts)
            label_EQ = np.zeros(npts)

            for event_idx in range(len(numbers)):
                en = numbers[event_idx] - 1
                s_time_samples = int(((hours[en]-s_hour)*3600 + (mins[en]-s_min)*60 + sec[en]-s_sec - s_micro) * args.fs)
                
                if not (0 <= s_time_samples < npts): continue

                t1 = s_time_samples
                t2 = t1 + int(T[en] * args.fs)
                
                try:
                    if ty[en] == "R":
                        label_RF[t1:t2] = rangle_label(t1, t2)
                    elif ty[en] == "S":
                        label_EQ[t1 - 100 : t1 + 100] = EQ_label_out[:-1]
                except (ValueError, IndexError):
                    pass

            label_noise -= (label_RF + label_EQ)
            label_noise[label_noise < 0] = 0

            data_out = data_cut(data_E, data_N, data_Z, npts, data_out)
            label_out = data_cut(label_noise, label_RF, label_EQ, npts, label_out)
    
    if not data_out:
        print("Error: No data was processed. Halting.")
        return

    print("Converting lists to numpy arrays...")
    data_out_np = np.array(data_out)
    label_out_np = np.array(label_out)

    print("Applying regulation (normalization)...")
    data_out_np = regulation(data_out_np)
    
    print(f"Saving data array to {args.output_x}")
    np.save(args.output_x, data_out_np)
    print(f"Saving labels array to {args.output_y}")
    np.save(args.output_y, label_out_np)
    print("--- Event Label Generation Finished ---")

def generate_attribute_labels(args):
    """
    Main logic for generating attribute labels (SSD) from pre-existing
    waveform data (X).
    """
    print("--- Starting Attribute Label Generation ---")
    
    if not os.path.exists(args.input_x):
        print(f"Error: Input file not found at '{args.input_x}'. Please check the path.")
        return

    print(f"Loading data from {args.input_x}...")
    x_data = np.load(args.input_x)
    nums = np.shape(x_data)
    G_label = np.zeros(nums)
    
    print(f"Calculating SSD labels for {nums[0]} samples...")
    for i in range(nums[0]):
        if (i + 1) % 1000 == 0 or (i + 1) == nums[0]:
            print(f"  Progress: {((i+1)/nums[0])*100:.2f}%")
        
        sample = x_data[i]
        G_label[i, :, 0, 0] = KSG_get(sample[:, 0, 0], args.window_size, 6000)
        G_label[i, :, 0, 1] = KSG_get(sample[:, 0, 1], args.window_size, 6000)
        G_label[i, :, 0, 2] = KSG_get(sample[:, 0, 2], args.window_size, 6000)
        
    print(f"Saving attribute labels to {args.output_y}...")
    np.save(args.output_y, G_label)
    print("--- Attribute Label Generation Finished ---")


def main(args):
    """Main dispatcher function."""
    output_paths = [args.output_x, args.output_y]
    if args.label_type == 'attribute':
        output_paths = [args.output_y]

    for path in output_paths:
        out_dir = os.path.dirname(path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)
            print(f"Created output directory: {out_dir}")

    if args.label_type == 'event':
        generate_event_labels(args)
    elif args.label_type == 'attribute':
        generate_attribute_labels(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate event or attribute labels for seismic data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--label_type', 
        type=str, 
        required=True, 
        choices=['event', 'attribute'],
        help="The type of labels to generate: 'event' for waveform/labels, 'attribute' for SSD."
    )
    parser.add_argument(
        '--output_y', 
        type=str, 
        default='./labels/y_data.npy',
        help="Path to save the output labels (Y for events, or SSD for attributes)."
    )

    event_group = parser.add_argument_group('Arguments for --label_type=event')
    event_group.add_argument('--data_dir', type=str, default='./data/', help="Directory containing mseed files.")
    event_group.add_argument('--catalog_file', type=str, default='./datalog.csv', help="Path to the data catalog CSV file.")
    event_group.add_argument('--file_pattern', type=str, default='*.mseed', help="Pattern to match data files (e.g., '*2013*.mseed').")
    event_group.add_argument('--output_x', type=str, default='./x_data.npy', help="Path to save the output waveform data (X).")
    event_group.add_argument('--fs', type=float, default=100.0, help="Sampling rate in Hz.")
    event_group.add_argument('--low_cut', type=float, default=1.0, help="Bandpass filter low-cut frequency.")
    event_group.add_argument('--high_cut', type=float, default=49.0, help="Bandpass filter high-cut frequency.")
    
    event_group.add_argument('--east_pattern', type=str, default='*E', help="Wildcard pattern for the East component channel.")
    event_group.add_argument('--north_pattern', type=str, default='*N', help="Wildcard pattern for the North component channel.")
    event_group.add_argument('--vert_pattern', type=str, default='*Z', help="Wildcard pattern for the Vertical component channel.")
    
    attr_group = parser.add_argument_group('Arguments for --label_type=attribute')
    attr_group.add_argument('--input_x', type=str, default='./x_data.npy', help="Path to the input waveform data (X) for attribute calculation.")
    attr_group.add_argument('--window_size', type=int, default=100, help="Window size for rolling standard deviation.")
    
    args = parser.parse_args()
    main(args)