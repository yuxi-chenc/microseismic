# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import sys
import os
import argparse
from models_f import base_model 

def main(args):
    """Main training logic function."""
    
    print(f"Setting GPU ID to: {args.gpu_id}")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    model_save_path = os.path.join(args.model_path, "model_save")
    result_path = os.path.join(args.model_path, "result")
    

    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(result_path, exist_ok=True)
    print(f"Model will be saved to: {model_save_path}")
    print(f"Training results will be saved to: {result_path}")

    print(f"Loading training data X from: {args.x_train_path}")
    x_train = np.load(args.x_train_path)
    print(f"Loading training labels Y from: {args.y_train_path}")
    y_train = np.load(args.y_train_path)
    print(f"Data loaded. X shape: {x_train.shape}, Y shape: {y_train.shape}")
    
    print(f"Shuffling data using random seed: {args.seed}")
    np.random.seed(args.seed)
    np.random.shuffle(x_train)
    np.random.seed(args.seed)
    np.random.shuffle(y_train)

    model = base_model()
    
    print(f"Compiling model with optimizer '{args.optimizer}' and mean squared error loss.")
    model.compile(optimizer=args.optimizer,
                  loss=tf.keras.losses.mean_squared_error)

    checkpoint_save_path = os.path.join(model_save_path, args.ckpt_name)
    print(f"Checkpoint path set to: {checkpoint_save_path}")
    
    if os.path.exists(checkpoint_save_path + '.index'):
        print('-------------Model found, loading weights-----------------')
        model.load_weights(checkpoint_save_path)
    else:
        print('-------------No model found, training from scratch-----------------')

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                     save_weights_only=True,
                                                     save_best_only=True)

    print("---------------- Starting Model Training ----------------")
    history = model.fit(x_train, y_train, 
                        batch_size=args.batch_size, 
                        epochs=args.epochs,
                        validation_split=args.validation_split, 
                        validation_freq=args.validation_freq,
                        callbacks=[cp_callback])
    
    model.summary()

    print("Training finished. Saving loss and val_loss...")
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    np.save(os.path.join(result_path, "loss.npy"), loss)
    np.save(os.path.join(result_path, "val_loss.npy"), val_loss)
    print("All tasks completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Model Training Script")

    path_group = parser.add_argument_group('File and Path Arguments')
    path_group.add_argument('--x_train_path', type=str, default="./test/label/x_data.npy", help='Path to the training data (X) .npy file.')
    path_group.add_argument('--y_train_path', type=str, default="./test/label/y_data.npy", help='Path to the training labels (Y) .npy file.')
    path_group.add_argument('--model_path', type=str, default="./models/test", help='Base directory to save all files for this training run.')
    path_group.add_argument('--ckpt_name', type=str, default="model.ckpt", help='Name for the checkpoint file.')

    train_group = parser.add_argument_group('Training Hyperparameters')
    train_group.add_argument('--gpu_id', type=str, default="0", help='GPU device ID to use.')
    train_group.add_argument('--seed', type=int, default=116, help='Random seed for shuffling data.')
    train_group.add_argument('--optimizer', type=str, default='adam', help='Optimizer to use during training.')
    train_group.add_argument('--batch_size', type=int, default=50, help='Number of samples per batch.')
    train_group.add_argument('--epochs', type=int, default=50, help='Total number of training epochs.')
    train_group.add_argument('--validation_split', type=float, default=0.1, help='Fraction of the training data to be used as validation data.')
    train_group.add_argument('--validation_freq', type=int, default=2, help='Frequency (in epochs) at which to perform validation.')

    args = parser.parse_args()
    main(args)