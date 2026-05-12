import tensorflow as tf
import numpy as np
import sys
import os
from model_with_GCAM import base_model as TargetModel
from model_with_GCAM_f import base_model as SourceModel

def normalization_per_channel(data):
    # 1. 基础 Min-Max 归一化：将数据缩放到 [0, 1]
    # 假设数据维度为 (N, 6000, 1, 3)，在 axis=1 (时间轴) 上求极值
    min_val = np.min(data, axis=1, keepdims=True)
    max_val = np.max(data, axis=1, keepdims=True)
    normalized_data = (data - min_val) / (max_val - min_val + 1e-10)
    
    # 2. 去均值处理 (Zero-centering)
    # 归一化后减去自身的均值，使数据分布在 0 附近（范围约 [-0.5, 0.5]）
    channel_mean = np.mean(normalized_data, axis=1, keepdims=True)
    final_data = normalized_data - channel_mean
    
    return final_data

def train_finetune(phy_name, pretrain_weights_path):
    x_path = f"./labels_feature/75percent/x_run_data_reduced.npy"
    y_path = f"./labels_feature/75percent/y_run_data_reduced.npy"

        
    x_train = np.load(x_path)
    y_train = np.load(y_path)

    x_train = normalization_per_channel(x_train)

    np.random.seed(116)
    np.random.shuffle(x_train)
    np.random.seed(116)
    np.random.shuffle(y_train)

    source_model = SourceModel()
    dummy_input = tf.zeros((1, x_train.shape[1], x_train.shape[2], x_train.shape[3]))
    source_model(dummy_input) 
    print(f"Attempting to load weights from: {pretrain_weights_path}")
    try:
        # 记录加载前的权重均值（可选，用于双重保险）
        initial_weight_mean = np.mean(source_model.layers[0].get_weights()[0])
        
        # 加载权重
        load_status = source_model.load_weights(pretrain_weights_path)
        
        # 1. 基础判断：load_status 包含了加载结果的断言信息
        # 2. 数值判断：对比加载后权重是否发生了变化
        current_weight_mean = np.mean(source_model.layers[0].get_weights()[0])
        
        if initial_weight_mean != current_weight_mean:
            print("Successfully loaded pre-trained weights. (Weights have been updated)")
        else:
            # 如果预训练权重恰好全 0 或者和初始化一模一样，会进这里，通常对于地震信号模型来说概率极低
            print("Weights loaded, but no numerical change detected. Please check checkpoint file.")
            
    except Exception as e:
        print(f"Error: Failed to load weights! Reason: {e}")
        # 如果加载失败，通常不建议继续微调，可以直接退出或报错
        return None
    target_model = TargetModel()
    target_model(dummy_input)

    for i in range(len(target_model.layers) - 3):
        target_model.layers[i].set_weights(source_model.layers[i].get_weights())

    for layer in target_model.layers[0:8]:
        layer.trainable = False

    target_model.compile(optimizer="adam",
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['binary_accuracy']
                  )

    base_dir = f"./modelpercent/model_{phy_name}_finetune"
    save_dir = os.path.join(base_dir, "model_save")
    result_dir = os.path.join(base_dir, "result")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    checkpoint_save_path = os.path.join(save_dir, f"model_{phy_name}.ckpt")

    if os.path.exists(checkpoint_save_path + '.index'):
        target_model.load_weights(checkpoint_save_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                     save_weights_only=True,
                                                     save_best_only=True)

    history = target_model.fit(x_train, y_train, batch_size=100, epochs=50,
                        validation_split=0.1, validation_freq=2,
                        callbacks=[cp_callback],verbose = 1)
    
    target_model.summary()

    acc = history.history['binary_accuracy']
    val_acc = history.history['val_binary_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    np.save(os.path.join(result_dir, "acc.npy"), acc)
    np.save(os.path.join(result_dir, "val_acc.npy"), val_acc)
    np.save(os.path.join(result_dir, "loss.npy"), loss)
    np.save(os.path.join(result_dir, "val_loss.npy"), val_loss)