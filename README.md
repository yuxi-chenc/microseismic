# Intelligent Recognition Network for Microseismic Signals Based on Waveform Attributes (SSD)

This repository provides a complete deep learning pipeline for microseismic signal processing and rockfall-event recognition, including sample-label generation, seismic-attribute feature construction, model training, and event prediction.

This code is associated with the paper:

**Seismic Insights into the Role of Rockfall in Rockslide Destruction Processes**

The workflow is designed for three-component seismic or microseismic waveform data. Users can either directly use the released sample dataset and trained model provided by this project, or prepare their own waveform data following the sample-label generation and prediction schemes described below.

---

## Features

- **Sample and Label Generation**  
  Generate three-component waveform samples and point-wise rockfall/non-rockfall labels for model training.

- **SSD-Based Waveform Attributes**  
  Construct sum-of-squared deviation (SSD) and other waveform-attribute features from seismic waveform samples.

- **Attribute-Guided Model Training**  
  Train the model through waveform-attribute pre-training and rockfall-event fine-tuning.

- **Prediction with a Trained Model**  
  Apply the trained model to continuous three-component seismic data and output per-station prediction CSV files.

- **Flexible Data Input**  
  Support both released sample datasets and user-defined three-component waveform data.

---

## Requirements

To run this project, first install the necessary Python libraries. A virtual environment is recommended.

```bash
pip install tensorflow numpy obspy pandas scipy matplotlib pywavelets
```

The main dependencies include:

```text
tensorflow
numpy
scipy
pandas
matplotlib
pywavelets
obspy
```

`obspy` is mainly used for reading and processing SAC-format seismic waveform data. If your data are stored in other formats, the input-reading part can be modified accordingly.

---

## Repository Structure

A recommended repository structure is:

```text
.
├── make_sample_labels.py           # Generate waveform samples and point-wise labels
├── feature_set.py                  # Generate seismic-attribute feature datasets
├── physical_features.py            # Seismic-attribute feature functions
├── composetrain.py                 # Main script for pre-training and fine-tuning
├── func_train_f.py                 # Seismic-attribute pre-training function
├── func_train.py                   # Rockfall-event fine-tuning function
├── model_with_GCAM_f.py            # Source model for seismic-attribute pre-training
├── model_with_GCAM.py              # Target model for rockfall-event recognition
├── function_tools.py               # Utility functions for label generation and data cutting
│
├── predict/                        # Prediction-only workflow
│   ├── main_predict.py             # Prediction entry script
│   ├── predictor.py                # Per-station prediction and CSV output
│   ├── config.py                   # Prediction configuration
│   └── model_with_GCAM.py          # Recognition model architecture
│
├── data/                           # Example data for prediction
│   ├── station_id.csv              # Station table with Station_ID column
│   └── sample/                     # Example SAC waveform data
│
├── modelarbs/                      # Trained model weights for prediction
├── labels/                         # Generated waveform samples and labels
├── labels_feature/                 # Generated seismic-attribute features
├── output/                         # Prediction outputs
└── README.md
```

---

## Overall Workflow

The complete training workflow is:

```text
Raw three-component seismic waveform data
        ↓
make_sample_labels.py
        ↓
x_data.npy / y_data.npy
        ↓
feature_set.py
        ↓
seismic-attribute feature data
        ↓
composetrain.py
        ↓
seismic-attribute pre-training + rockfall-event fine-tuning
```

The prediction workflow is:

```text
Continuous three-component SAC data
        ↓
predict/main_predict.py
        ↓
load trained model
        ↓
predictor.py
        ↓
output/pred_csv/station_<station_id>.csv
```

---

# Dataset

All data and code are openly available.

The curated sample dataset used in the corresponding paper can be downloaded from:

```text
Download Link: https://doi.org/YOUR_DOI_HERE
```

Users can directly use the released sample dataset for model training. Users can also prepare their own three-component waveform data and generate training samples using `make_sample_labels.py`.

### Continuous Raw Data Sources

| Dataset | Data Center | Access Method | Link |
|---|---|---|---|
| Séchilienne Rockslide | OMIV / RESIF | Use `day_datadown.py` script | https://doi.org/10.15778/RESIF.MT |
| Illgraben Rockslide | GFZ Data Services | Manual download | https://doi.org/10.5880/GFZ.2.4/2016.001 |

---

# How to Train

This repository supports two training modes:

1. **Training with the released sample dataset**
2. **Training with your own three-component seismic waveform data**

---

## 1. Training with the Released Sample Dataset

The released sample dataset can be downloaded from:

```text
DOI: [https://doi.org/YOUR_DOI_HERE](https://doi.org/10.5281/zenodo.20130810)
```

The released dataset contains waveform samples, event labels, and seismic-attribute feature files required for training. Therefore, users can directly train the model without regenerating labels or seismic attributes.

After downloading the dataset, place the files in the expected directory. For example:

```text
labels_feature/
└── 75percent/
    ├── x_run_data_reduced.npy
    ├── y_run_data_reduced.npy
    ├── reduced_y_physics_get_sum_sq_diff.npy
    ├── reduced_y_physics_get_raw_kurtosis.npy
    └── ...
```

The training scripts use fixed data paths by default. Please check the paths in:

```text
func_train_f.py
func_train.py
composetrain.py
```

In `func_train_f.py`, the seismic-attribute pre-training paths are usually defined as:

```python
x_path = "./labels_feature/75percent/x_run_data_reduced.npy"
y_path = f"./labels_feature/75percent/reduced_y_physics_{phy_name}.npy"
```

In `func_train.py`, the rockfall-event fine-tuning paths are usually defined as:

```python
x_path = "./labels_feature/75percent/x_run_data_reduced.npy"
y_path = "./labels_feature/75percent/y_run_data_reduced.npy"
```

The seismic attributes used for training are specified in `composetrain.py`. For example:

```python
target_features = [
    "get_sum_sq_diff",
    "get_raw_kurtosis",
]
```

Each attribute in `target_features` will be used for one complete training process, including seismic-attribute pre-training and rockfall-event fine-tuning.

To start training, run:

```bash
python composetrain.py
```

The script performs:

```text
1. Load waveform samples.
2. Load selected seismic-attribute feature labels.
3. Pre-train the source model using seismic-attribute labels.
4. Save the pre-trained model weights.
5. Load the pre-trained weights into the target model.
6. Fine-tune the target model using rockfall/non-rockfall labels.
7. Save the final model weights and training curves.
```

If the released dataset already contains the required seismic-attribute feature files, it is not necessary to run `feature_set.py` again.

---

## 2. Training with Your Own Three-Component Seismic Waveform Data

Users can also train the model using their own data.

Each waveform record should provide three-component waveform data:

```text
data_e   # E-component waveform
data_n   # N-component waveform
data_z   # Z-component waveform
```

For rockfall-event samples, the corresponding event time window should also be provided:

```text
t1_sec   # rockfall-event start time, in seconds
t2_sec   # rockfall-event end time, in seconds
```

For non-rockfall samples, `t1_sec` and `t2_sec` can be set to `None`.

The input data do not have to be SAC files. Any data format can be used as long as the data-reading part in `make_sample_labels.py` can finally return:

```python
data_e, data_n, data_z, t1_sec, t2_sec
```

Then run:

```bash
python make_sample_labels.py
```

This step generates waveform samples and point-wise labels:

```text
x_data.npy
y_data.npy
```

The expected data shapes are:

```text
X: (number of samples, sample length, 1, 3)
Y: (number of samples, sample length, 1, 2)
```

where:

```text
X[:, :, :, 0] = E-component waveform
X[:, :, :, 1] = N-component waveform
X[:, :, :, 2] = Z-component waveform

Y[:, :, :, 0] = non-rockfall label
Y[:, :, :, 1] = rockfall label
```

After generating `x_data.npy`, use `feature_set.py` to generate seismic-attribute features:

```bash
python feature_set.py
```

Before running `feature_set.py`, set the input waveform path:

```python
INPUT_X_PATH = "./labels_physics/x_run_data.npy"
```

Then select the seismic attributes to be calculated:

```python
TARGET_FEATURES = [
    "get_sum_sq_diff",
    "get_raw_kurtosis",
]
```

Each selected seismic attribute will generate one feature file:

```text
y_physics_<feature_name>.npy
```

For example:

```text
y_physics_get_sum_sq_diff.npy
y_physics_get_raw_kurtosis.npy
```

The seismic-attribute feature generation process may take a long time, especially when the dataset is large or multiple attributes are selected.

After the waveform samples, event labels, and seismic-attribute features are prepared, run:

```bash
python composetrain.py
```

The model will be trained in two stages:

```text
1. Seismic-attribute pre-training
2. Rockfall-event fine-tuning
```

---

# make_sample_labels.py

This script is used to generate three-component waveform samples and point-wise labels for rockfall-event recognition.

The script does not require the input data to be in SAC format. The only requirement is that each record can provide three-component waveform data and, for rockfall samples, the corresponding rockfall-event time window.

## Purpose

For each input record, the script:

```text
1. reads the three-component waveform data;
2. reads the rockfall-event start and end times if the record is a rockfall sample;
3. normalizes the waveform data;
4. adds Gaussian white noise;
5. generates point-wise rockfall and non-rockfall labels;
6. cuts the waveform into fixed-length samples;
7. normalizes each sample locally;
8. balances rockfall and non-rockfall samples;
9. saves the generated samples and labels as .npy files.
```

## Input Requirements

Each record only needs to provide:

```text
data_e   # E-component waveform
data_n   # N-component waveform
data_z   # Z-component waveform
t1_sec   # rockfall-event start time, in seconds
t2_sec   # rockfall-event end time, in seconds
```

For non-rockfall samples, `t1_sec` and `t2_sec` can be set to `None`.

The three waveform components must have the same length:

```text
len(data_e) = len(data_n) = len(data_z)
```

The event time window should be relative to the beginning of the current waveform record.

A recommended data organization is:

```text
label_data/
├── RF_multiple_sta_label/
│   ├── event_001/
│   ├── event_002/
│   └── ...
│
├── RF_single_sta_label/
│   ├── event_001/
│   ├── event_002/
│   └── ...
│
└── Non_RF_label/
    ├── sample_001/
    ├── sample_002/
    └── ...
```

The rockfall-event categories can be adjusted in `make_sample_labels.py` according to the actual folder names. Other categories can be treated as non-rockfall samples.

---

# Seismic-Attribute Feature Generation

After generating `x_data.npy`, seismic-attribute features should be produced using `feature_set.py`.

The script first loads the waveform sample file, for example:

```python
INPUT_X_PATH = "./labels_physics/x_run_data.npy"
```

The output directory is defined as:

```python
OUTPUT_DIR = "./labels_feature/"
```

Run:

```bash
python feature_set.py
```

For each selected seismic attribute, the script saves one feature file:

```text
y_physics_<feature_name>.npy
```

The expected output shape is:

```text
(N, 6000, 1, 3)
```

This means that the seismic-attribute curve is calculated separately for the E, N, and Z components of each waveform sample.

---

# Available Seismic Attributes

The file `physical_features.py` provides multiple seismic-attribute feature functions. Users can flexibly select one or more attributes according to the research purpose.

Available attributes include:

```text
get_sum_sq_diff
get_rms
get_std
get_energy
get_zcr
get_env_mean_max_ratio
get_env_median_max_ratio
get_rise_fall_ratio
get_raw_kurtosis
get_env_kurtosis
get_raw_skewness
get_env_skewness
get_crest_factor
get_autocorr_peaks
get_autocorr_energy_ratio
get_linear_decay_error
get_spec_mean
get_dom_freq
get_quartile_freq
get_spec_peaks_count
get_nyquist_band_energy
get_spec_centroid
get_gyration_radius
get_spec_bandwidth
```

To select the seismic attributes used for feature generation, modify `TARGET_FEATURES` in `feature_set.py`.

Example:

```python
TARGET_FEATURES = [
    "get_sum_sq_diff",
    "get_raw_kurtosis",
]
```

---

# Prediction

The prediction workflow is provided in the `predict/` folder. It is designed for applying a trained model to continuous three-component seismic data and generating per-station prediction CSV files.

Only prediction is performed in this workflow. The later steps used in other workflows, such as consensus filtering, waveform slicing, impact extraction, and `.npy` dataset saving, are not included.

The prediction workflow is:

```text
Continuous three-component SAC data
        ↓
predict/main_predict.py
        ↓
load trained recognition model
        ↓
predictor.py
        ↓
output/pred_csv/station_<station_id>.csv
```

---

## Prediction Files

The prediction folder contains:

```text
predict/
├── main_predict.py          # Main entry for prediction only
├── predictor.py             # Per-station prediction and CSV writing
├── config.py                # Prediction configuration
└── model_with_GCAM.py       # Recognition model architecture
```

A trained model is provided in:

```text
modelarbs/model_get_sum_sq_diff_finetune/model_save/
```

The default checkpoint path is defined in `predict/config.py`:

```python
MODEL_CKPT = "./modelarbs/model_get_sum_sq_diff_finetune/model_save/model_get_sum_sq_diff.ckpt"
```

Do not include the `.index` suffix in `MODEL_CKPT`.

---

## Prediction Example

We provide an example dataset in the `data/` folder for testing the prediction workflow.

A typical example structure is:

```text
data/
├── station_id.csv
└── sample/
    └── 20250703_195655/
        ├── 0188.HHE.sac
        ├── 0188.HHN.sac
        ├── 0188.HHZ.sac
        └── ...
```

The station table should contain a column named:

```text
Station_ID
```

The SAC filename pattern expected by the default prediction code is:

```text
{4-digit station ID}.HH{E|N|Z}.sac
```

For example:

```text
0188.HHE.sac
0188.HHN.sac
0188.HHZ.sac
```

The default paths in `predict/config.py` are:

```python
STATION_CSV = "./data/station_id.csv"
DATA_DIR = "./data/sample/20250703_195655"
MODEL_CKPT = "./modelarbs/model_get_sum_sq_diff_finetune/model_save/model_get_sum_sq_diff.ckpt"
OUTPUT_DIR = "./output"
PRED_CSV_DIR = os.path.join(OUTPUT_DIR, "pred_csv")
```

To run prediction with the provided example data:

```bash
cd predict
python main_predict.py
```

Or run from the repository root if imports and paths are set accordingly:

```bash
python predict/main_predict.py
```

The prediction results will be saved to:

```text
output/pred_csv/
```

Each station will generate one CSV file:

```text
station_<station_id>.csv
```

For example:

```text
output/pred_csv/station_188.csv
```

---

## Prediction Output Format

Each prediction CSV contains detected rockfall-event picks for one station.

The output columns are:

```text
num, station_id, year, month, day, hour, min, sec, duration_s, type, prob
```

where:

| Column | Description |
|---|---|
| `num` | Pick index for the current station |
| `station_id` | Station identifier |
| `year` | Pick year |
| `month` | Pick month |
| `day` | Pick day |
| `hour` | Pick hour |
| `min` | Pick minute |
| `sec` | Pick second |
| `duration_s` | Estimated event duration in seconds |
| `type` | Event type, where `R` indicates rockfall |
| `prob` | Maximum predicted rockfall probability |

---

## Prediction Configuration

The prediction parameters are set in `predict/config.py`.

```python
STATION_CSV = "./data/station_id.csv"  # Station table with Station_ID column

DATA_DIR = "./data/sample/20250703_195655"  # Input SAC data directory

MODEL_CKPT = "./modelarbs/model_get_sum_sq_diff_finetune/model_save/model_get_sum_sq_diff.ckpt"  # Fine-tuned model checkpoint

OUTPUT_DIR = "./output"  # Output directory

PRED_CSV_DIR = os.path.join(OUTPUT_DIR, "pred_csv")  # Prediction CSV output directory

FS = 100  # Sampling rate in Hz

SEG_LEN = 6000  # Model input segment length; 6000 samples at 100 Hz = 60 s

P_THRESHOLD = 0.9  # Rockfall probability threshold

PICK_RATIO = 0.1  # Event boundary search ratio

PICK_MIN_CONSEC = 3  # Stop boundary search after this number of consecutive low-probability samples

PICK_MAX_GAP = 5  # Maximum allowed gap during boundary search

BANDPASS_LOW = 1.0  # Low cutoff frequency for bandpass filtering

BANDPASS_HIGH = 50.0  # High cutoff frequency for bandpass filtering

BLACKLIST_STATIONS = []  # Stations excluded from prediction

CUDA_DEVICE = "0"  # GPU device ID; use "" for CPU
```

The most commonly modified parameters are:

```python
STATION_CSV
DATA_DIR
MODEL_CKPT
P_THRESHOLD
BLACKLIST_STATIONS
CUDA_DEVICE
```

---

## Changing the Prediction Input Format

The default prediction code assumes SAC files and uses ObsPy to read them. If users want to use another data format, they only need to modify the data-loading part in:

```text
predict/predictor.py
```

The key function is:

```python
process_stream(st)
```

and the core requirement is that the prediction pipeline finally obtains:

```python
data_E, data_N, data_Z, npts, starttime
```

where:

```text
data_E    # E-component waveform
data_N    # N-component waveform
data_Z    # Z-component waveform
npts      # number of samples
starttime # waveform start time
```

If users do not want to use a station CSV, they can also modify the station-file mapping function in:

```text
predict/main_predict.py
```

The default function is:

```python
build_station_file_map()
```

By default, it reads `Station_ID` from `station_id.csv` and searches for files named like:

```text
0188.HH*.sac
```

If the file names already contain station IDs, this function can be modified to extract station IDs directly from the file names. The `station_id` does not have to be a real field station number, but it should uniquely identify each station or input channel group.

---

# Model

The model architecture is defined in:

```text
model_with_GCAM.py
model_with_GCAM_f.py
```

- **`model_with_GCAM_f.py`**: source model for seismic-attribute pre-training.
- **`model_with_GCAM.py`**: target model for rockfall-event recognition.

The recognition model contains convolutional downsampling and upsampling blocks, GRU-based temporal modeling, attention modules, and output layers for point-wise prediction.

The final recognition model outputs two channels:

```text
channel 0: non-rockfall probability
channel 1: rockfall probability
```

The trained model used for prediction is provided in:

```text
modelarbs/model_get_sum_sq_diff_finetune/model_save/
```

---

# Output Files

During training, model weights and training curves are saved automatically.

For seismic-attribute pre-training, the outputs are saved under:

```text
modelpercent/model_<feature_name>_pretrain/
├── model_save/
└── result/
```

For rockfall-event fine-tuning, the outputs are saved under:

```text
modelpercent/model_<feature_name>_finetune/
├── model_save/
└── result/
```

Prediction outputs are saved under:

```text
output/pred_csv/
```

---

# Notes

- The training input data format is flexible and does not have to be SAC.
- The prediction example uses SAC files and ObsPy for reading waveform data.
- For training, each record only needs to provide three-component waveform data and, for rockfall samples, the event start and end times.
- For prediction, each station should provide E, N, and Z components.
- Station ID is used to distinguish different stations and write separate prediction CSV files. It does not have to be a real field station number, but it should uniquely identify each station.
- If the prediction input format changes, modify `predict/predictor.py`.
- If the station-file organization changes, modify `predict/main_predict.py`.
- Seismic-attribute feature generation may take a long time for large datasets.
- Multiple seismic attributes are provided in `physical_features.py` and can be selected flexibly.
- The prediction workflow only outputs `pred_csv`; it does not perform consensus filtering, slicing, or impact-event extraction.

---

# Reference

Bianchi, M., Evans, P. L., Heinloo, A., & Quinteros, J. (2015). WebDC3 web interface. GFZ Data Services. https://doi.org/10.5880/GFZ.2.4/2016.001

Helmstetter, A., & Garambois, S. (2010). Seismic monitoring of Séchilienne rockslide (French Alps): Analysis of seismic signals and their correlation with rainfalls. *Journal of Geophysical Research: Earth Surface*, 115(F3). https://doi.org/10.1029/2009JF001532

French Landslide Observatory – Seismological Datacenter / RESIF. (2006). Observatoire Multi-disciplinaire des Instabilités de Versants (OMIV) [Data set]. RESIF - Réseau Sismologique et Géodésique Français. https://doi.org/10.15778/RESIF.MT

---

# Citation

If you use this code or dataset in your research, please cite the associated paper and dataset.

```text
Dataset DOI:https://doi.org/10.5281/zenodo.20130810
Paper citation: Seismic Insights into the Role of Rockfall in Rockslide Destruction Processes.
```
