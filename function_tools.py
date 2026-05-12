import numpy as np
import copy

def gauss_label(mu, sigma):
    if sigma <= 0:
        sigma = 1e-6
    x = np.arange(-100, 101, 1)
    left = 1 / (np.sqrt(2 * np.pi) * np.sqrt(sigma))
    right = np.exp(-(x - mu)**2 / (2 * sigma))
    return left * right

def rangle_label(start, end):
    long_num = int(end - start)
    if long_num <= 0:
        return np.zeros(1)
    data = np.ones(long_num)
    t = int(long_num / 10)
    if t < 1: 
        return data
    x = np.arange(0, t, 1)
    y = np.sin(np.pi / (2 * t) * x)
    y1 = np.sin(np.pi / (2 * t) * (x + t))
    data[0:t] = y
    data[long_num - t : long_num] = y1
    return data

def data_cut(data_E, data_N, data_Z, npts, data_out=None):
    if data_out is None:
        data_out = []
    window_len = 6000
    if npts < window_len:
        return data_out 
    data_num = int((npts - 1) // window_len)
    data_save_template = np.zeros((window_len, 1, 2), dtype=float)
    for i in range(data_num):
        temp_chunk = data_save_template.copy()
        idx_start = i * window_len
        idx_end = (i + 1) * window_len
        temp_chunk[:, 0, 0] = data_E[idx_start:idx_end]
        temp_chunk[:, 0, 1] = data_N[idx_start:idx_end]
        data_out.append(temp_chunk)
    if npts >= window_len:
        temp_chunk_last = data_save_template.copy()
        temp_chunk_last[:, 0, 0] = data_E[-window_len:]
        temp_chunk_last[:, 0, 1] = data_N[-window_len:]
        data_out.append(temp_chunk_last)
    return data_out
def data_cut_back_v1(data_in, npts):
    if isinstance(data_in, list):
        num_chunks = len(data_in)
    else:
        num_chunks = data_in.shape[0]
    if num_chunks == 0:
        return np.zeros(npts), np.zeros(npts), np.zeros(npts)
    window_len = 6000
    data_car_save = np.zeros((npts))
    data_RF_save = np.zeros((npts))
    for i in range(num_chunks - 1):
        data_save_in = data_in[i]
        idx_start = i * window_len
        idx_end = (i + 1) * window_len
        if idx_end > npts:
            break
        data_car_save[idx_start:idx_end] = data_save_in[:, 0, 0] 
        data_RF_save[idx_start:idx_end]  = data_save_in[:, 0, 1] 
       
    if num_chunks > 0:
        data_save_in = data_in[num_chunks - 1]
        start_pos = npts - window_len
        if start_pos < 0: 
            start_pos = 0
            valid_len = npts
            data_car_save[0:npts] = data_save_in[-valid_len:, 0, 0]
            data_RF_save[0:npts]  = data_save_in[-valid_len:, 0, 1]
            
        else:
            data_car_save[start_pos:npts] = data_save_in[:, 0, 0] 
            data_RF_save[start_pos:npts]  = data_save_in[:, 0, 1] 
           
    return data_car_save, data_RF_save




def data_cut_back(data_in, npts):
    if isinstance(data_in, list):
        num_chunks = len(data_in)
    else:
        num_chunks = data_in.shape[0]
    if num_chunks == 0:
        return np.zeros(npts), np.zeros(npts), np.zeros(npts)
    window_len = 6000
    data_car_save = np.zeros((npts))
    data_EQ_save = np.zeros((npts))
    data_RF_save = np.zeros((npts))
    for i in range(num_chunks - 1):
        data_save_in = data_in[i]
        idx_start = i * window_len
        idx_end = (i + 1) * window_len
        if idx_end > npts:
            break
        data_car_save[idx_start:idx_end] = data_save_in[:, 0, 0] 
        data_RF_save[idx_start:idx_end]  = data_save_in[:, 0, 1] 
        data_EQ_save[idx_start:idx_end]  = data_save_in[:, 0, 2]
    if num_chunks > 0:
        data_save_in = data_in[num_chunks - 1]
        start_pos = npts - window_len
        if start_pos < 0: 
            start_pos = 0
            valid_len = npts
            data_car_save[0:npts] = data_save_in[-valid_len:, 0, 0]
            data_RF_save[0:npts]  = data_save_in[-valid_len:, 0, 1]
            data_EQ_save[0:npts]  = data_save_in[-valid_len:, 0, 2]
        else:
            data_car_save[start_pos:npts] = data_save_in[:, 0, 0] 
            data_RF_save[start_pos:npts]  = data_save_in[:, 0, 1] 
            data_EQ_save[start_pos:npts]  = data_save_in[:, 0, 2]
    return data_car_save, data_RF_save, data_EQ_save

def regularization(data_E, data_N, data_Z):
    def _normalize_single(arr):
        val_min = np.min(arr)
        val_max = np.max(arr)
        val_range = val_max - val_min
        if val_range == 0:
            return np.zeros_like(arr)
        arr_nor = (arr - val_min) / val_range
        return arr_nor - np.mean(arr_nor)
    st_E_data = _normalize_single(data_E)
    st_N_data = _normalize_single(data_N)
    st_Z_data = _normalize_single(data_Z)
    return st_E_data, st_N_data, st_Z_data