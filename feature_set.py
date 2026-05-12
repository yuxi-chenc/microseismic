import numpy as np
import os
import sys
import time
import importlib

# 尝试导入您的物理特征库
# 确保 physical_features.py 在当前目录下
try:
    import physical_features as pf
except ImportError:
    print("❌ 错误: 未找到 physical_features.py，请确保它在当前目录中。")
    sys.exit()

# ================= 配置区域 =================
# 输入数据路径 (请修改为您实际的 X 数据路径)
INPUT_X_PATH = "./labels_physics/x_run_data.npy"   # 例如 x_test_data.npy 或 x_combined.npy

# 输出保存目录
OUTPUT_DIR = "./labels_feature/"

# 滑动窗口大小
WINDOW_SIZE = 100
DATA_LENGTH = 6000

# 要生成的特征函数列表


TARGET_FEATURES = [
    #'get_sum_sq_diff',
    # --- 1. 基础能量与统计特征 (Basic Stats) ---
    #'get_rms',                  # 均方根 (能量)
    #'get_energy',               # 绝对振幅和
    #'get_zcr',                  # 过零率

    # --- 2. 波形形态特征 (Waveform Morphology) ---
    #'get_env_mean_max_ratio',   # 包络均值比
    #'get_env_median_max_ratio', # 包络中值比
    #'get_rise_fall_ratio',      # 上升下降时间比
    'get_raw_kurtosis',         # 原始信号峰度
    'get_env_kurtosis',         # 包络峰度
    'get_raw_skewness',         # 原始信号偏度
    'get_env_skewness',         # 包络偏度
    'get_crest_factor',         # 波峰因数 (Peak/RMS)

    # --- 3. 自相关与衰减特征 (Autocorr & Decay) ---
    'get_autocorr_peaks',       # 自相关峰值数
    'get_autocorr_energy_ratio',# 自相关能量比
    'get_linear_decay_error',   # 线性衰减拟合误差

    # --- 4. 频谱特征 (Spectral Features) ---
    'get_spec_mean',            # 频谱均值
    'get_dom_freq',             # 主频
    'get_quartile_freq',        # 频率中位数
    'get_spec_peaks_count',     # 显著频谱峰值数
    'get_nyquist_band_energy',  # 低频段能量
    'get_spec_centroid',        # 频谱质心
    'get_gyration_radius',      # 回转半径
    'get_spec_bandwidth',       # 频谱带宽
]



# ===========================================

def normalize_trace(data):
    """
    [修正顺序] 
    1. 先归一化到 [0, 1]
    2. 后执行去均值 (Zero-centering)
    """
    d_min = np.min(data)
    d_max = np.max(data)
    den = d_max - d_min
    if den == 0:
        return np.zeros_like(data)
    
    norm_data = (data - d_min) / den
    final_data = norm_data - np.mean(norm_data)
    return final_data

def generate_labels():
    # 1. 检查并加载数据
    if not os.path.exists(INPUT_X_PATH):
        print(f"❌ 错误: 输入文件不存在 -> {INPUT_X_PATH}")
        return

    print(f"📦 正在加载输入数据: {INPUT_X_PATH} ...")
    X_data = np.load(INPUT_X_PATH)
    
    # 检查形状 (N, 6000, 1, 3)
    if X_data.ndim != 4 or X_data.shape[1] != 6000 or X_data.shape[3] != 3:
        print(f"⚠️ 警告: 输入数据形状 {X_data.shape} 可能不符合 (N, 6000, 1, 3) 的预期。")
        print("   脚本将尝试按最后一维作为通道(Channel)处理。")

    N = X_data.shape[0]
    print(f"✅ 数据加载成功，共 {N} 个样本。")

    # 创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 2. 遍历特征进行计算
    for func_name in TARGET_FEATURES:
        # 检查函数是否存在
        if not hasattr(pf, func_name):
            print(f"⚠️ 跳过: physical_features 中找不到函数 '{func_name}'")
            continue
        
        print(f"\n🚀 正在计算特征: [{func_name}] ...")
        feature_func = getattr(pf, func_name)
        
        # 结果容器
        y_physics_out = []
        
        start_time = time.time()
        
        # 遍历所有样本
        for i in range(N):
            # 打印进度条 (每10个样本刷新一次)
            if i % 10 == 0:
                progress = (i / N) * 100
                elapsed = time.time() - start_time
                print(f"\r   进度: {i}/{N} ({progress:.1f}%) - 耗时: {elapsed:.1f}s", end="")

            # 容器：存放当前样本 3 个通道的特征曲线
            sample_features = []
            
            # 遍历 3 个通道 (E, N, Z)
            for ch in range(3):
                # 提取单条波形数据 (6000,)
                trace_data = X_data[i, :, 0, ch]
                
                # === [关键修改] ===
                # 在计算物理特征前，先对该分量进行 [0, 1] 归一化
                # 这一步至关重要，确保 RMS/能量等特征在同一尺度下计算
                trace_data = normalize_trace(trace_data)
                # =================
                
                # 调用物理特征函数计算
                feat_curve = feature_func(trace_data, WINDOW_SIZE, DATA_LENGTH)
                
                sample_features.append(feat_curve)
            
            # 堆叠通道: [Arr1, Arr2, Arr3] -> (3, 6000) -> 转置为 (6000, 3)
            sample_features = np.array(sample_features).T 
            
            # 增加维度以匹配输入格式: (6000, 3) -> (6000, 1, 3)
            sample_features = np.expand_dims(sample_features, axis=1)
            
            y_physics_out.append(sample_features)
        
        print(f"\r   进度: {N}/{N} (100.0%) - 完成!                  ")
        
        # 转换为 numpy 数组: (N, 6000, 1, 3)
        y_physics_out = np.array(y_physics_out)
        
        # 3. 保存结果
        save_name = f"y_physics_{func_name}.npy"
        save_path = os.path.join(OUTPUT_DIR, save_name)
        np.save(save_path, y_physics_out)
        
        print(f"💾 已保存: {save_path}")
        print(f"   输出形状: {y_physics_out.shape}")

if __name__ == "__main__":
    generate_labels()