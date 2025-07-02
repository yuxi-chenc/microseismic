import tensorflow as tf
import numpy as np
import shutil
import sys
import os
import argparse
from models import base_model
from tensorflow.keras.optimizers import Adam


def copy_files(src_folder, dst_folder):
    """Recursively copies folder contents."""
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    for item in os.listdir(src_folder):
        src_item = os.path.join(src_folder, item)
        dst_item = os.path.join(dst_folder, item)
        if os.path.isfile(src_item):
            shutil.copy2(src_item, dst_item)
        elif os.path.isdir(src_item):
            copy_files(src_item, dst_item)

def main(args):
    """Main training logic function."""
    
    args.model_path = os.path.abspath(args.model_path.replace("\\", "/"))
    args.x_train_path = os.path.abspath(args.x_train_path.replace("\\", "/"))
    args.y_train_path = os.path.abspath(args.y_train_path.replace("\\", "/"))
    if args.initial_model_path:
        args.initial_model_path = os.path.abspath(args.initial_model_path.replace("\\", "/"))

    print(f"Setting GPU ID to: {args.gpu_id}")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    model_save_path = os.path.join(args.model_path, "model_save")
    result_path = os.path.join(args.model_path, "result")
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(result_path, exist_ok=True)
    print(f"Model will be saved to: {model_save_path}")
    print(f"Results will be saved to: {result_path}")

    print(f"Loading training data X from: {args.x_train_path}")
    x_train = np.load(args.x_train_path)
    print(f"Loading training labels Y from: {args.y_train_path}")
    y_train = np.load(args.y_train_path)
    print(f"Data loaded. X shape: {x_train.shape}, Y shape: {y_train.shape}")
    
    print(f"Shuffling data with random seed: {args.seed}")
    np.random.seed(args.seed)
    np.random.shuffle(x_train)
    np.random.seed(args.seed)
    np.random.shuffle(y_train)

    model = base_model()
    
    if args.pre_train.lower() == 'yes':
        print(f"Pre-training enabled. Copying weights from {args.initial_model_path}...")
        copy_files(args.initial_model_path, model_save_path)
        
        checkpoint_load_path = os.path.join(model_save_path, args.ckpt_name)
        if os.path.exists(checkpoint_load_path + '.index'):
             print(f"Loading weights from {checkpoint_load_path} for fine-tuning...")
             model.load_weights(checkpoint_load_path)
        else:
             print(f"Warning: Pre-train enabled, but no checkpoint found at {checkpoint_load_path}")
        
        if args.freeze_layers > 0:
            print(f"Freezing the first {args.freeze_layers} layers of the model...")
            for layer in model.layers[0:args.freeze_layers]:
                layer.trainable = False

    print(f"Compiling model with optimizer '{args.optimizer}' and loss '{args.loss_function}'")
    model.compile(optimizer=args.optimizer,
                  loss=args.loss_function,
                  metrics=['categorical_accuracy'])

    checkpoint_save_path = os.path.join(model_save_path, args.ckpt_name)
    if os.path.exists(checkpoint_save_path + '.index') and not args.pre_train.lower() == 'yes':
        print('-------------Found existing model, loading weights-----------------')
        model.load_weights(checkpoint_save_path)
    else:
        print('-------------No model found or in pre-train mode, training from scratch/loaded weights-----------------')

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                     save_weights_only=True,
                                                     save_best_only=True,
                                                     monitor='val_categorical_accuracy',
                                                     mode='max',
                                                     verbose=1)

    print("---------------- Starting Model Training ----------------")
    history = model.fit(x_train, y_train, 
                        batch_size=args.batch_size, 
                        epochs=args.epochs,
                        validation_split=args.validation_split, 
                        validation_freq=args.validation_freq,
                        callbacks=[cp_callback])
    
    model.summary()

    print("Training finished. Saving history...")
    acc = history.history['categorical_accuracy']
    val_acc = history.history['val_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    np.save(os.path.join(result_path, "acc.npy"), acc)
    np.save(os.path.join(result_path, "val_acc.npy"), val_acc)
    np.save(os.path.join(result_path, "loss.npy"), loss)
    np.save(os.path.join(result_path, "val_loss.npy"), val_loss)
    print("All tasks completed!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Classification Model Training Script")

    path_group = parser.add_argument_group('File and Path Arguments')
    path_group.add_argument('--x_train_path', type=str, default="./test/label/x_data.npy", help='Path to the training data (X) .npy file.')
    path_group.add_argument('--y_train_path', type=str, default="./test/label/y_data.npy", help='Path to the training labels (Y) .npy file.')
    path_group.add_argument('--model_path', type=str, default="./models/test1", help='Base directory to save all files for this training run.')
    path_group.add_argument('--initial_model_path', type=str, default="./model_1_hour_filter/model_GRUv1_f/model_save", help='Source directory for pre-trained weights (used only if pre_train=yes).')
    path_group.add_argument('--ckpt_name', type=str, default="model.ckpt", help='Name for the checkpoint file.')

    model_group = parser.add_argument_group('Preprocessing and Model Arguments')
    model_group.add_argument('--pre_train', type=str, choices=['yes', 'no'], default='no', help='Whether to load pre-trained weights for fine-tuning (yes/no).')
    model_group.add_argument('--freeze_layers', type=int, default=8, help='Number of layers to freeze from the beginning of the model during fine-tuning.')

    train_group = parser.add_argument_group('Training Hyperparameters')
    train_group.add_argument('--gpu_id', type=str, default="1", help='GPU device ID to use.')
    train_group.add_argument('--seed', type=int, default=116, help='Random seed for shuffling data.')
    train_group.add_argument('--optimizer', type=str, default='adam', help='Optimizer to use during training.')
    train_group.add_argument('--loss_function', type=str, default='categorical_crossentropy', help='Loss function to use.')
    train_group.add_argument('--batch_size', type=int, default=50, help='Number of samples per batch.')
    train_group.add_argument('--epochs', type=int, default=20, help='Total number of training epochs.')
    train_group.add_argument('--validation_split', type=float, default=0.1, help='Fraction of the training data to be used as validation data.')
    train_group.add_argument('--validation_freq', type=int, default=2, help='Frequency (in epochs) at which to perform validation.')

    args = parser.parse_args()
    main(args)