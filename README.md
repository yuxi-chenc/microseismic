🌊 An Intelligent Recognition Network for Microseismic Signals based on Waveform Attributes (SSD)
This repository provides a complete deep learning pipeline for seismic data processing, including label generation, model training, and event prediction, as presented in our corresponding research paper.

✨ Features
Label Generation: Scripts to process raw seismic data and generate various types of labels (e.g., event-based and attribute-based).

Flexible Model Training: A training script that supports building models from scratch or fine-tuning from pre-trained weights.

Universal Event Prediction: A prediction script to apply trained models on new, continuous seismic data to detect and classify events.

Modular Workflow: The entire pipeline is controlled through a central runner script (run.py), making it easy to execute different stages.

🔧 Requirements
To run this project, first install the necessary Python libraries. It is recommended to use a virtual environment.

pip install tensorflow numpy obspy pandas scipy matplotlib pywavelets

🚀 How to Run
All functionalities of this pipeline are integrated into the run.py script, which provides a simple interface to execute different tasks.

To run a specific task, simply open the run.py file and modify the run_case variable at the top of the script.

Configuration
In run.py, find this section:

# --- SELECT THE TASK TO RUN ---
run_case = "train"  
'''
Options for run_case:
eve : Generates event labels (e.g., Noise, Event Type) from raw seismic data.
att : Generates attribute labels (e.g., SSD) from existing waveform data.
pre_train: Pre-trains a model on attribute labels.
train: Trains or fine-tunes the main model on event labels.
pred: Makes predictions on new data using a trained model.
'''

Execution Steps
Choose a Task: Change the value of the run_case variable in run.py to your desired task (e.g., run_case = "pred").

Configure Parameters: Within the corresponding if/elif block for your chosen run_case, modify the command-line parameters (e.g., file paths, epochs, batch size) to match your needs.

Run the Script: Open your terminal, navigate to the project's root directory, and execute the script:

python run.py

The runner script will then automatically call the appropriate sub-script (label_make.py, train.py, or predict.py) with the parameters you have configured.

🧠 Models
This repository provides the model architectures used in the paper.

Model definitions can be found in models.py and models_f.py.

The trained model parameters used in the paper are provided in the model/ directory.

💾 Dataset
All data and code supporting this study are openly available.

Training & Testing Sets
The curated datasets used for training and testing in the paper can be downloaded from the following link. Please place the data in the appropriate directory as referenced in your configuration.

Download Link: https://doi.org/YOUR_DOI_HERE

Continuous Raw Data
The continuous seismic data sources are listed below.

Séchilienne Rockslide: Data are provided by the Observatoire Multi-disciplinaire des Instabilités de Versants (OMIV) and are openly available through the RESIF - Réseau Sismologique et géodésique Français data center. A script, day_datadown.py, is provided in this repository to facilitate downloading this data.

DOI: https://doi.org/10.15778/RESIF.MT

Reference: Helmstetter, A., et al. (2010), Seismic Monitoring of the Séchilienne Landslide.

Illgraben Rockslide: Data are available at the GFZ Data Services repository.

DOI: https://doi.org/10.5880/GFZ.4.1.2016.001
