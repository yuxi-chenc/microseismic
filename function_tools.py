# -*- coding: utf-8 -*-
"""
Signal Processing and Data Handling Utilities for Seismology

This module provides a collection of functions for normalizing, processing,
and labeling seismic data. It includes tools for event picking, data windowing,
and generating characteristic function labels.

Author: cyx
Date: 2023.04.11
"""

import os
import shutil
import copy
import numpy as np
import matplotlib.pyplot as plt

# --- Constants for Configuration ---
WINDOW_SIZE = 6000  # The size of each data window (e.g., 6000 samples for 1 minute at 100Hz).
PICK_BUFFER = 50    # Buffer zone around a picked event to prevent re-picking.
CAR_RF_THRESHOLD = 0.02 # Threshold to determine the start/end of a car or rockfall event.

# --- Data Normalization ---

def _min_max_normalize_and_center(data_segment: np.ndarray) -> np.ndarray:
    """
    Applies Min-Max normalization to a data segment and then centers it by
    subtracting the mean.

    Args:
        data_segment (np.ndarray): A 1D array of seismic data.

    Returns:
        np.ndarray: The normalized and centered data segment.
    """
    # Avoid division by zero if the signal is flat
    if (max_val := np.max(data_segment)) == (min_val := np.min(data_segment)):
        return data_segment - np.mean(data_segment)
    
    normalized_data = (data_segment - min_val) / (max_val - min_val)
    return normalized_data - np.mean(normalized_data)

def normalize_by_window(data: np.ndarray) -> np.ndarray:
    """
    Applies window-based normalization to 3-component seismic data.

    Each component of each sample is normalized independently over the defined WINDOW_SIZE.

    Args:
        data (np.ndarray): A 4D numpy array with shape
                           (num_samples, time_steps, 1, num_components).

    Returns:
        np.ndarray: The data array with each component normalized.
    """
    num_samples = data.shape[0]
    for i in range(num_samples):
        for component in range(data.shape[3]): # Iterate through E, N, Z components
            data[i, :WINDOW_SIZE, 0, component] = _min_max_normalize_and_center(
                data[i, :WINDOW_SIZE, 0, component]
            )
    return data

# --- Event Picking Algorithms ---

def pick_earthquake_events(characteristic_function: np.ndarray, threshold: float) -> tuple[list[int], list[float]]:
    """
    Picks distinct earthquake events from a characteristic function.

    This function iteratively finds the maximum value in the function. If it exceeds
    the threshold, it's marked as a pick, and a buffer zone around it is
    zeroed out to prevent multiple picks for the same event.

    Args:
        characteristic_function (np.ndarray): The 1D array representing the
                                              likelihood of an earthquake.
        threshold (float): The minimum probability value to consider as a pick.

    Returns:
        tuple[list[int], list[float]]: A tuple containing:
            - A list of integer indices representing the P-wave arrival times.
            - A list of the corresponding probability values for each pick.
    """
    picks = []
    probabilities = []
    # Work on a copy to avoid modifying the original array
    temp_function = characteristic_function.copy()
    
    while True:
        max_value = np.max(temp_function)
        if max_value <= threshold:
            break
            
        max_index = np.argmax(temp_function)
        
        picks.append(max_index)
        probabilities.append(max_value)
        
        # Zero out a window around the pick to avoid re-picking the same event
        start_buffer = max(0, max_index - PICK_BUFFER)
        end_buffer = min(len(temp_function), max_index + PICK_BUFFER)
        temp_function[start_buffer:end_buffer] = 0
        
    return picks, probabilities

def pick_continuous_events(characteristic_function: np.ndarray, threshold: float) -> tuple[list[int], list[int], list[float]]:
    """
    Picks continuous events like cars or rockfalls from a characteristic function.

    This function identifies events by locating peaks above a threshold and then
    searching backward and forward to find the start and end points where the
    signal drops below a secondary, lower threshold.

    Args:
        characteristic_function (np.ndarray): The 1D array representing the
                                              likelihood of a car or rockfall.
        threshold (float): The minimum peak value to consider as an event.

    Returns:
        tuple[list[int], list[int], list[float]]: A tuple containing:
            - A list of event start indices.
            - A list of event end indices.
            - A list of the peak probability values for each event.
    """
    pick_starts, pick_ends, probabilities = [], [], []
    temp_function = characteristic_function.copy()
    
    while True:
        max_value = np.max(temp_function)
        if max_value <= threshold:
            break
            
        peak_index = np.argmax(temp_function)
        probabilities.append(max_value)
        
        # Search backwards for the start time
        start_time = peak_index
        for i in range(peak_index, 2, -1):
            if all(temp_function[i-3:i] <= CAR_RF_THRESHOLD):
                start_time = i
                break
        else: # If loop completes without break
             start_time = 0
        pick_starts.append(start_time)

        # Search forwards for the end time
        end_time = peak_index
        for i in range(peak_index, len(temp_function) - 3):
             if all(temp_function[i:i+3] <= CAR_RF_THRESHOLD):
                 end_time = i
                 break
        else: # If loop completes without break
            end_time = len(temp_function) - 1
        pick_ends.append(end_time)
        
        # Zero out the detected event to find the next one
        temp_function[start_time:end_time + 1] = 0

    return pick_starts, pick_ends, probabilities

# --- Label Generation ---

def generate_gaussian_label(mu: int = 0, sigma: int = 20) -> np.ndarray:
    """
    Generates a 1D Gaussian distribution label, typically for P-wave arrivals.

    Args:
        mu (int, optional): The mean of the distribution (center of the pick). Defaults to 0.
        sigma (int, optional): The standard deviation of the distribution. Defaults to 20.

    Returns:
        np.ndarray: An array containing the Gaussian label values.
    """
    x = np.arange(-100, 101, 1)  # Create a 201-point window for the label
    gaussian = np.exp(-(x - mu)**2 / (2 * sigma**2))
    return gaussian / (np.sqrt(2 * np.pi) * sigma)


def generate_trapezoid_label(start: int, end: int) -> np.ndarray:
    """
    Generates a trapezoidal label for continuous events like rockfalls or cars.

    The label ramps up at the beginning and ramps down at the end.

    Args:
        start (int): The starting index of the event.
        end (int): The ending index of the event.

    Returns:
        np.ndarray: An array containing the trapezoidal label values.
    """
    duration = end - start
    if duration <= 0:
        return np.array([])
        
    label = np.ones(duration)
    
    # Create sinusoidal ramps for smooth transitions
    ramp_duration = int(duration / 10)
    if ramp_duration > 0:
        x_ramp = np.arange(ramp_duration)
        ramp_up = np.sin(np.pi / (2 * ramp_duration) * x_ramp)
        ramp_down = np.cos(np.pi / (2 * ramp_duration) * x_ramp)
        
        label[:ramp_duration] = ramp_up
        label[duration - ramp_duration:] = ramp_down
        
    return label

# --- Data Segmentation and Reconstruction ---

def segment_data(data_e: np.ndarray, data_n: np.ndarray, data_z: np.ndarray) -> list[np.ndarray]:
    """
    Segments 3-component continuous data into overlapping windows.

    Args:
        data_e (np.ndarray): The East component data stream.
        data_n (np.ndarray): The North component data stream.
        data_z (np.ndarray): The Z (vertical) component data stream.

    Returns:
        list[np.ndarray]: A list of data segments, each with shape
                          (WINDOW_SIZE, 1, 3).
    """
    npts = len(data_e)
    num_segments = (npts - 1) // WINDOW_SIZE
    segmented_data = []

    for i in range(num_segments):
        start, end = i * WINDOW_SIZE, (i + 1) * WINDOW_SIZE
        segment = np.zeros((WINDOW_SIZE, 1, 3), dtype=float)
        segment[:, 0, 0] = data_e[start:end]
        segment[:, 0, 1] = data_n[start:end]
        segment[:, 0, 2] = data_z[start:end]
        segmented_data.append(segment)

    # Add the final overlapping window from the end of the data
    final_segment = np.zeros((WINDOW_SIZE, 1, 3), dtype=float)
    final_segment[:, 0, 0] = data_e[-WINDOW_SIZE:]
    final_segment[:, 0, 1] = data_n[-WINDOW_SIZE:]
    final_segment[:, 0, 2] = data_z[-WINDOW_SIZE:]
    segmented_data.append(final_segment)

    return segmented_data

def reconstruct_data(segmented_results: list[np.ndarray], npts: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstructs continuous data streams from segmented model predictions.

    Args:
        segmented_results (list[np.ndarray]): A list of predicted segments from the model.
        npts (int): The total number of points in the original continuous data.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the three
            reconstructed continuous characteristic functions (e.g., car, rockfall, earthquake).
    """
    num_segments = len(segmented_results)
    # Initialize empty arrays for the reconstructed time series
    car_stream = np.zeros(npts)
    rf_stream = np.zeros(npts)
    eq_stream = np.zeros(npts)

    for i in range(num_segments - 1):
        start, end = i * WINDOW_SIZE, (i + 1) * WINDOW_SIZE
        segment = segmented_results[i]
        car_stream[start:end] = segment[:, 0, 0]
        rf_stream[start:end] = segment[:, 0, 1]
        eq_stream[start:end] = segment[:, 0, 2]

    # Handle the final overlapping segment
    final_segment = segmented_results[-1]
    car_stream[-WINDOW_SIZE:] = final_segment[:, 0, 0]
    rf_stream[-WINDOW_SIZE:] = final_segment[:, 0, 1]
    eq_stream[-WINDOW_SIZE:] = final_segment[:, 0, 2]

    return car_stream, rf_stream, eq_stream

# --- File System Utility ---

def copy_directory(src_folder: str, dst_folder: str):
    """
    Recursively copies a directory and its contents.

    Args:
        src_folder (str): The path to the source folder.
        dst_folder (str): The path to the destination folder.
    """
    if os.path.exists(dst_folder):
        shutil.rmtree(dst_folder) # Remove existing to avoid merge conflicts
    shutil.copytree(src_folder, dst_folder)