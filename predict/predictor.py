# ============================================================
# predictor.py  —  单台站预测模块
# 逻辑来源：predarbs.py
# 输入：某台站三分量 SAC/mseed 文件
# 输出：该台站当天的落石拾取 CSV
# ============================================================

import os
import csv
import numpy as np
from obspy import read
from datetime import datetime

import config


# ── 数据预处理 ────────────────────────────────────────────────

def normalize_slice(seg: np.ndarray) -> np.ndarray:
    """Min-max 归一化后去均值"""
    min_val = np.min(seg, axis=0, keepdims=True)
    max_val = np.max(seg, axis=0, keepdims=True)
    seg = (seg - min_val) / (max_val - min_val + 1e-10)
    seg = seg - np.mean(seg, axis=0, keepdims=True)
    return seg


def data_cut_and_norm(data_E, data_N, data_Z, npts: int) -> np.ndarray:
    """将三分量数据切成 SEG_LEN 长度的段并归一化"""
    seg_len = config.SEG_LEN
    num_segments = int(np.ceil(npts / seg_len))
    data_out = []

    for i in range(num_segments):
        start = i * seg_len
        end   = start + seg_len
        seg   = np.zeros((seg_len, 1, 3))

        if end <= npts:
            seg[:, 0, 0] = data_E[start:end]
            seg[:, 0, 1] = data_N[start:end]
            seg[:, 0, 2] = data_Z[start:end]
        else:
            # 最后一段不足时取末尾 seg_len 个样本
            seg[:, 0, 0] = data_E[npts - seg_len:npts]
            seg[:, 0, 1] = data_N[npts - seg_len:npts]
            seg[:, 0, 2] = data_Z[npts - seg_len:npts]

        data_out.append(normalize_slice(seg))

    return np.array(data_out)


def stitch_predictions(preds: np.ndarray, npts: int) -> np.ndarray:
    """将分段预测结果拼接回原始时间轴长度"""
    seg_len  = config.SEG_LEN
    channels = preds.shape[-1]
    full_pred = np.zeros((npts, channels))

    for i in range(preds.shape[0]):
        start   = i * seg_len
        end     = start + seg_len
        current = preds[i, :, 0, :]

        if end <= npts:
            full_pred[start:end] = current
        else:
            full_pred[npts - seg_len:npts] = current

    return full_pred


# ── 拾取算法 ─────────────────────────────────────────────────

def car_RF_pick(result_car, p_threshold: float,
                ratio=None, min_consecutive=None, max_gap=None):
    """
    在概率曲线上逐个拾取落石事件。
    返回 (pick_start_list, pick_end_list, prob_list)，单位：样本索引
    """
    ratio           = ratio           or config.PICK_RATIO
    min_consecutive = min_consecutive or config.PICK_MIN_CONSEC
    max_gap         = max_gap         or config.PICK_MAX_GAP

    pick_start, pick_end, RF_p = [], [], []
    temp = np.asarray(result_car, dtype=float).copy()
    n    = len(temp)

    if n == 0:
        return pick_start, pick_end, RF_p

    while True:
        b = float(np.max(temp))
        if b <= p_threshold:
            break

        RF_p.append(b)
        a       = int(np.argmax(temp))
        low_thr = b * ratio

        # 向左搜索起点
        i, below_run, gap, last_good = a, 0, 0, a
        while i >= 0:
            if temp[i] >= low_thr:
                last_good, below_run, gap = i, 0, 0
            else:
                below_run += 1
                gap       += 1
                if below_run >= min_consecutive or gap > max_gap:
                    break
            i -= 1
        start_time = max(0, last_good)

        # 向右搜索终点
        i, below_run, gap, last_good = a, 0, 0, a
        while i <= n - 1:
            if temp[i] >= low_thr:
                last_good, below_run, gap = i, 0, 0
            else:
                below_run += 1
                gap       += 1
                if below_run >= min_consecutive or gap > max_gap:
                    break
            i += 1
        end_time = min(n - 1, last_good)

        pick_start.append(int(start_time))
        pick_end.append(int(end_time))
        temp[int(start_time):int(end_time) + 1] = 0.0

    return pick_start, pick_end, RF_p


# ── 流数据预处理 ─────────────────────────────────────────────

def process_stream(st):
    """
    读取 ObsPy Stream，重采样 + 滤波，返回三分量数组。
    返回 (data_E, data_N, data_Z, npts, starttime) 或全 None
    """
    try:
        st.merge(fill_value=0)

        current_sr = st[0].stats.sampling_rate
        if abs(current_sr - config.FS) > 1e-4:
            st.resample(float(config.FS))

        st.filter('bandpass',
                  freqmin=config.BANDPASS_LOW,
                  freqmax=config.BANDPASS_HIGH,
                  corners=4, zerophase=True)
        st.sort()

        try:
            tr_Z = st.select(component="Z")[0]
            tr_N = st.select(component="N")[0]
            tr_E = st.select(component="E")[0]
        except IndexError:
            return None, None, None, 0, None

        npts   = min(tr_Z.stats.npts, tr_N.stats.npts, tr_E.stats.npts)
        data_Z = tr_Z.data[:npts]
        data_N = tr_N.data[:npts]
        data_E = tr_E.data[:npts]

        return data_E, data_N, data_Z, npts, tr_Z.stats.starttime

    except Exception:
        return None, None, None, 0, None


# ── 时间解析 ─────────────────────────────────────────────────

def _parse_pick_time(pick_dt):
    """将 ObsPy UTCDateTime 转为 (datetime, float_seconds) 便于写 CSV"""
    iso_str = str(pick_dt)
    if '.' in iso_str:
        date_str, frac = iso_str.rsplit('.', 1)
        frac = frac.rstrip('Z')[:6].ljust(6, '0')
        micros = float("0." + frac)
    else:
        date_str = iso_str.rstrip('Z')
        micros   = 0.0

    dt_obj     = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S')
    final_sec  = dt_obj.second + micros + (pick_dt.microsecond / 1e6)
    return dt_obj, final_sec


# ── CSV 写入 ─────────────────────────────────────────────────

def _init_csv(path: str):
    with open(path, "w", newline='') as f:
        csv.writer(f).writerow(
            ['num', 'station_id', 'year', 'month', 'day',
             'hour', 'min', 'sec', 'duration_s', 'type', 'prob']
        )


def _append_csv(path: str, row: list):
    with open(path, "a", newline='') as f:
        csv.writer(f).writerow(row)


# ── 主接口 ────────────────────────────────────────────────────

def predict_station(model, station_id: int, sac_files: list[str],
                    out_csv: str) -> list[dict]:
    _init_csv(out_csv)
    events   = []
    num_pick = 0

    from obspy import Stream
    st = Stream()
    for fpath in sac_files:
        try:
            st += read(fpath)
        except Exception:
            continue
            
    if not st:
        return events

    data_E, data_N, data_Z, npts, starttime = process_stream(st)

    if npts is None or npts < config.SEG_LEN:
        return events

    input_batch = data_cut_and_norm(data_E, data_N, data_Z, npts)
    preds       = model.predict(input_batch, verbose=0)
    full_pred   = stitch_predictions(preds, npts)
    result_target = full_pred[:, 1]  

    rf_starts, rf_ends, rf_probs = car_RF_pick(result_target, config.P_THRESHOLD)

    for k in range(len(rf_starts)):
        pick_dt  = starttime + (rf_starts[k] / config.FS)
        duration = (rf_ends[k] - rf_starts[k]) / config.FS

        dt_obj, final_sec = _parse_pick_time(pick_dt)

        _append_csv(out_csv, [
            num_pick,
            station_id,
            dt_obj.year,
            dt_obj.month,
            dt_obj.day,
            dt_obj.hour,
            dt_obj.minute,
            f"{final_sec:.4f}",
            f"{duration:.4f}",
            'R',
            f"{rf_probs[k]:.4f}"
        ])

        events.append({
            "station_id"  : station_id,
            "abs_time_s"  : float(pick_dt),
            "start_sample": rf_starts[k],
            "end_sample"  : rf_ends[k],
            "duration_s"  : duration,
            "prob"        : rf_probs[k],
            "file"        : sac_files[0], 
            "starttime"   : starttime,
        })
        num_pick += 1

    print(f"  [predictor] Station {station_id}: {num_pick} picks → {out_csv}")
    return events


def run_prediction_all_stations(model, station_file_map: dict) -> dict:
    """
    对所有台站批量预测。

    Parameters
    ----------
    model            : 已加载权重的 Keras 模型
    station_file_map : {station_id: [sac_file_path, ...]}

    Returns
    -------
    all_events : {station_id: [event_dict, ...]}
    """
    os.makedirs(config.PRED_CSV_DIR, exist_ok=True)
    all_events = {}

    for station_id, files in station_file_map.items():
        out_csv = os.path.join(config.PRED_CSV_DIR, f"station_{station_id}.csv")
        events  = predict_station(model, station_id, files, out_csv)
        all_events[station_id] = events

    return all_events
