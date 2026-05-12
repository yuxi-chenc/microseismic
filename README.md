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
```

## How to Run
All functionalities are integrated into run.py. Modify the run_case variable at the top of the script:
```bash
run_case = "train"  
Options for run_case:
eve      : Generates event labels (Noise, Event Type) from raw seismic data
att      : Generates attribute labels (e.g., SSD) from waveform data
pre_train: Pre-trains a model on attribute labels
train    : Trains/fine-tunes main model on event labels
pred     : Makes predictions on new data
```
## Execution Steps
- Choose Task: Set run_case in run.py to desired task (e.g., run_case = "pred")
- Configure Parameters: Modify command-line parameters in corresponding if/elif block
- Run Script:
```bash
python run.py
```
## Model
-Architectures: Defined in models.py and models_f.py
-Pre-trained Weights: Provided in model/ directory

## make_sample_labels_general.py

This script is used to generate three-component waveform samples and point-wise labels for rockfall-event recognition.

The script does not require the input data to be in SAC format. The only requirement is that each record can provide three-component waveform data and, for rockfall samples, the corresponding rockfall-event time window.

### 1. Purpose

The script converts raw three-component waveform records into training samples and labels.

For each input record, the script:

1. reads the three-component waveform data;
2. reads the rockfall-event start and end times if the record is a rockfall sample;
3. normalizes the waveform data;
4. adds Gaussian white noise;
5. generates point-wise rockfall and non-rockfall labels;
6. cuts the waveform into fixed-length samples;
7. normalizes each sample locally;
8. balances rockfall and non-rockfall samples;
9. saves the generated samples and labels as `.npy` files.

### 2. Input requirements

The input data format is flexible. The data do not have to be SAC files.

Each record only needs to provide the following information:

```text
data_e   # E-component waveform
data_n   # N-component waveform
data_z   # Z-component waveform
t1_sec   # rockfall-event start time, in seconds
t2_sec   # rockfall-event end time, in seconds

## Dataset
All data and code are openly available.
- Curated datasets from paper (download and place in appropriate directory):
Download Link: https://doi.org/YOUR_DOI_HERE

-Continuous Raw Data Sources
Séchilienne Rockslide	OMIV/RESIF	Use day_datadown.py script	https://doi.com/10.15778/RESIF.MT
Illgraben Rockslide	GFZ Data Services	Manual download	https://doi.com/10.5880/GFZ.2.4/2016.001.

## Reference:
Bianchi, M., Evans, P. L., Heinloo, A., & Quinteros, J. (2015). Webdc3 web interface. GFZ Data Services. doi: 10.5880/GFZ.2.4/2016.001388
Helmstetter, A., & Garambois, S. (2010). Seismic monitoring of S´echilienne rockslide (French Alps): Analysis of seismic signals and their correlation with rainfalls. Journal of Geophysical Research: Earth Surface, 115 (F3). doi: 10.1029/2009JF001532409
French Landslide Observatory – Seismological Datacenter / RESIF. (2006). Observatoire Multi-disciplinaire des Instabilit´es de Versants (OMIV) [Data set]. RESIF - R´eseau French Landslide Observatory – Seismological Datacenter / RESIF. (2006). Observatoire Multi-disciplinaire des Instabilit´es de Versants (OMIV) [Data set]. RESIF - R´eseau



