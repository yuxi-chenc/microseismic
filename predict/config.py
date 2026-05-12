import os

# =========================
# Path settings
# =========================

STATION_CSV = "./data/station_id.csv"  # 台站坐标表，必须包含 Station_ID 列

DATA_DIR = "./data/sample/20250703_195655"  # 原始 SAC 数据目录

MODEL_CKPT = "./modelarbs/model_get_sum_sq_diff_finetune/model_save/model_get_sum_sq_diff.ckpt"  # 微调后的识别模型权重路径，不要加 .index 后缀

OUTPUT_DIR = "./output"  # 总输出目录

PRED_CSV_DIR = os.path.join(OUTPUT_DIR, "pred_csv")  # 单台站预测结果 CSV 保存目录


# =========================
# Prediction parameters
# =========================

FS = 100  # 采样率，单位 Hz；如果原始数据不是 100 Hz，程序会重采样到该采样率

SEG_LEN = 6000  # 模型输入长度；6000 点在 100 Hz 下对应 60 s

P_THRESHOLD = 0.9  # 落石事件预测概率阈值；越高越严格，误检少但可能漏检更多

PICK_RATIO = 0.1  # 事件边界搜索比例；边界阈值 = 峰值概率 * PICK_RATIO

PICK_MIN_CONSEC = 3  # 边界搜索时，连续低于边界阈值多少个点后停止

PICK_MAX_GAP = 5  # 边界搜索允许的最大间断点数

BANDPASS_LOW = 1.0  # 预测前带通滤波低截止频率，单位 Hz

BANDPASS_HIGH = 50.0  # 预测前带通滤波高截止频率，单位 Hz


# =========================
# Station filtering
# =========================

BLACKLIST_STATIONS = []  # 不参与预测的台站编号；如果不排除任何台站，设为 []


# =========================
# Runtime settings
# =========================

CUDA_DEVICE = "0"  # 使用的 GPU 编号；如果用 CPU，可以设为 ""