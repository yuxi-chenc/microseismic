# An intelligent recognition network for microseismic signals based on waveform attributes (SSD)

A machine learning model for identifying rockfall signals from microseismic data using waveform guidance.

This repository provides a complete deep learning pipeline for seismic data processing, including label generation, model training, and event prediction. 

Label Generation: Scripts to process raw seismic data and generate various types of labels (e.g., event labels, attribute labels).

Model Training: A flexible training script that supports training from scratch or fine-tuning from pre-trained weights.

Event Prediction: A universal prediction script to apply trained models on new, continuous seismic data to detect and classify events.

Modular Design: The entire workflow is controlled through a central runner script (run.py), making it easy to execute different stages of the pipeline.
