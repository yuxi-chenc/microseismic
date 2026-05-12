import tensorflow as tf
import numpy as np
import sys
import os
from model_with_GCAM_f import base_model

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

def train_pretrain(phy_name):
    # 路径配置
    x_path = f"./labels_feature/75percent/x_run_data_reduced.npy"
    y_path = f"./labels_feature/75percent/reduced_y_physics_{phy_name}.npy" 
    
    
    #y_path = f"./labels_feature/y_physics_{phy_name}.npy" 
    #print(x_path)
    #print(y_path)
    
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        print(f"Data not found: {x_path} or {y_path}")
        return None

    # 加载数据
    x_train = np.load(x_path)
    #print("load finishing")
    y_train = np.load(y_path)

    # 归一化
    x_train = normalization_per_channel(x_train)

    # 打乱数据
    np.random.seed(116)
    np.random.shuffle(x_train)
    np.random.seed(116)
    np.random.shuffle(y_train)

    # 实例化模型
    model = base_model()

    # 编译模型
    # 修改说明: 回归任务不适合用 'accuracy'。
    # Loss 已经是 mse，这里添加 'mae' (平均绝对误差) 作为额外的监控指标，比 mse 更直观。
    model.compile(optimizer="adam",
                  loss="mse",
                  metrics=['mae']
                  )

    # 目录配置
    base_dir = f"./modelpercent/model_{phy_name}_pretrain"
    save_dir = os.path.join(base_dir, "model_save")
    result_dir = os.path.join(base_dir, "result") # 新增结果保存目录
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    checkpoint_save_path = os.path.join(save_dir, "model_pretrain.ckpt")

    if os.path.exists(checkpoint_save_path + '.index'):
        print('-------------load the model-----------------')
        model.load_weights(checkpoint_save_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                     save_weights_only=True,
                                                     save_best_only=True)

    # 训练模型并获取 history
    history = model.fit(x_train, y_train, batch_size=100, epochs=50,
                        validation_split=0.1, validation_freq=2,
                        callbacks=[cp_callback],
                        verbose = 1)
    
    # 保存训练过程中的 Loss 和 Metrics
    # history.history 字典中包含: 'loss', 'mae', 'val_loss', 'val_mae'
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    mae = history.history['mae']
    val_mae = history.history['val_mae']

    np.save(os.path.join(result_dir, "loss.npy"), loss)
    np.save(os.path.join(result_dir, "val_loss.npy"), val_loss)
    np.save(os.path.join(result_dir, "mae.npy"), mae)
    np.save(os.path.join(result_dir, "val_mae.npy"), val_mae)
    
    print(f"Training finished for {phy_name}. Loss saved to {result_dir}")

    return checkpoint_save_path