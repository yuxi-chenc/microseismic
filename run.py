# -*- coding: utf-8 -*-
"""
Created on Mon May 26 11:23:06 2025

@author: Yuxi Chen
"""

"""
Seismic Data Analysis Pipeline: Label Generation, Model Training, and Prediction

This script provides a unified interface for:
1. Generating event/attribute labels from seismic data
2. Training machine learning models for seismic event classification
3. Making predictions on new seismic data using trained models

Here are some running examples provided.
"""
import os

# --- SELECT THE TASK TO RUN ---
run_case = "train"  
'''
Options for run_case:
eve : Generates event labels (Noise, Event Type) from raw seismic data.
att : Generates attribute labels (e.g., SSD) from existing waveform data (requires 'eve' stage first).
pre_train: Pre-trains a model on attribute labels.
train: Trains or fine-tunes the main model on event labels.
pred: Makes predictions on new data using a trained model.
'''

if run_case == "eve":
    command_event = (
        "python label_make.py "
        "--label_type event "             # Specifies the 'event' mode for the label maker script.
        "--data_dir ./test/data "         # Directory containing raw .mseed data.
        "--catalog_file ./datalog.csv "   # Path to the event catalog file.
        "--file_pattern \"*.mseed\" "     # Wildcard pattern to match data files.
        "--output_x ./test/label/x_data.npy " # Output path for the processed waveform data (X).
        "--output_y ./test/label/y_data.npy"  # Output path for the generated event labels (Y).
    )
    os.system(command_event)

elif run_case == "att":
    command_attribute = (
        "python label_make.py " 
        "--label_type attribute "       # Specifies the 'attribute' mode for the label maker script.
        "--input_x ./test/label/x_data.npy "  # Path to the input waveform data from the 'eve' stage.
        "--output_y ./test/label/ssd_data.npy " # Output path for the attribute labels (e.g., SSD).
        "--window_size 100"             # Window size for calculating the attribute.
    )
    os.system(command_attribute)

elif run_case == "pre_train":
    command_pretrain = (
        "python pre_train.py "
        "--x_train_path ./test/label/x_data.npy "    # Path to the training data (X).
        "--y_train_path ./test/label/ssd_data.npy "  # Path to the attribute labels (Y) for pre-training.
        "--model_path ./models/model_f/test "       # Base directory to save the pre-trained model.
        "--epochs 10 "                              # Number of training epochs.
        "--batch_size 10 "                          # Number of samples per batch.
        "--gpu_id 0 "                               # GPU device ID to use.
        "--seed 116"                                # Random seed for reproducibility.
        "--ckpt_name model.ckpt"                    # Name for the checkpoint file.
    )
    os.system(command_pretrain)

elif run_case == "train":
    command_train = (
        "python train.py "
        "--x_train_path ./test/label/x_data.npy "     # Path to the training data (X).
        "--y_train_path ./test/label/y_data.npy "     # Path to the event labels (Y) for training.
        "--model_path ./models/test1 "                # Base directory to save the final model.
        "--initial_model_path ./model_save "          # Path to pre-trained weights for fine-tuning.
        "--ckpt_name model.ckpt "                     # Name for the final checkpoint file.
        "--pre_train yes "                            # Enable fine-tuning mode from pre-trained weights.
        "--epochs 50 "                                # Number of training epochs.
        "--batch_size 32 "                            # Number of samples per batch.
    )
    os.system(command_train)

elif run_case == "pred":
    command_pred = (
        "python predict.py "
        "--model_base_path ./attention_model/model/ "   # Base directory where different trained models are stored.
        "--model_name GRUV3 "                           # Specific name of the model folder to load.
        "--ckpt_name model_cyx.ckpt "                   # Name of the checkpoint file to load.
        "--input_dir ./data "                           # Directory containing .mseed files for prediction.
        "--file_pattern \"*201401*.mseed\" "            # Pattern to select which files to predict on.
        "--output_csv ./predictions/2014_Jan_picks.csv " # Path for the output CSV file with results.
        "--threshold 0.75 "                             # Probability threshold for event picking.
        "--batch_size 128 "                             # Batch size for prediction to manage memory.
        "--gpu_id 0 "                                   # GPU device ID to use.
    )
    os.system(command_pred)