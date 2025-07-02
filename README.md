# Intelligent Recognition Network for Microseismic Signals based on Waveform Attributes (SSD)

This repository provides a complete deep learning pipeline for seismic data processing, including label generation, model training, and event prediction, as presented in our corresponding research paper.

## Features
- **Label Generation**: Scripts to process raw seismic data and generate various types of labels (e.g., event-based and attribute-based)
- **Flexible Model Training**: Training script supporting building models from scratch or fine-tuning pre-trained weights
- **Universal Event Prediction**: Prediction script to apply trained models on new, continuous seismic data
- **Modular Workflow**: Entire pipeline controlled through central runner script (`run.py`)

## Requirements
To run this project, first install necessary Python libraries (recommended in a virtual environment):
```bash
pip install tensorflow numpy obspy pandas scipy matplotlib pywavelets
'''exit
## How to Run
All functionalities are integrated into run.py. Modify the run_case variable at the top of the script:
# --- SELECT THE TASK TO RUN ---
run_case = "train"  
'''
Options for run_case:
eve      : Generates event labels (Noise, Event Type) from raw seismic data
att      : Generates attribute labels (e.g., SSD) from waveform data
pre_train: Pre-trains a model on attribute labels
train    : Trains/fine-tunes main model on event labels
pred     : Makes predictions on new data

