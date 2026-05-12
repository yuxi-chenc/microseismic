"""
Generate three-component waveform samples and point-wise rockfall labels.

This version is not limited to SAC files. The only required information is:
1. Three-component waveform data for each record: E, N, Z.
2. The rockfall time interval in the waveform, if the record is a rockfall event.

The code keeps the original processing logic:
read data -> add noise -> remove mean -> generate labels -> cut samples -> local normalization -> balance RF/Non-RF -> save npy files.

You only need to adjust the configuration section and, if necessary, the read_one_record() function
according to your own data format.
"""

import os
import sys
import random
import numpy as np

sys.path.append("./")
import function_tools


# ============================================================
# 1. Configuration
# ============================================================

# Root folder of the prepared dataset.
# Each subfolder is treated as one category.
DATA_ROOT = "./label_data"

# Output files.
X_SAVE_PATH = "./labels/x_ones_data.npy"
Y_SAVE_PATH = "./labels/y_ones_data.npy"

# Category names treated as rockfall-event data.
# All other categories under DATA_ROOT are treated as non-rockfall data.
RF_CATEGORIES = [
    "RF_multiple_sta_label",
    "RF_single_sta_label",
]

# Sampling rate of the waveform data, used to convert event time in seconds to sample index.
SAMPLE_RATE = 100

# Random SNR range used for data augmentation.
SNR_MIN = 10
SNR_MAX = 20

# Supported default input formats in this script:
#   "npz" : each record folder contains one npz file with keys E, N, Z, t1, t2
#   "npy" : each record folder contains E.npy, N.npy, Z.npy and optional label.txt
# You can also modify read_one_record() for your own format.
INPUT_FORMAT = "npz"


# ============================================================
# 2. Utility functions
# ============================================================

def wgn(x, snr):
    """Add Gaussian white noise according to the given SNR."""
    long = len(x)
    if np.sum(np.abs(x)) == 0:
        return x
    ps = np.sum(np.abs(x) ** 2) / long
    pn = ps / (10 ** (snr / 10))
    noise = np.random.normal(0, 1, long) * np.sqrt(pn)
    return x + noise


def regularization_global(data_e, data_n, data_z):
    """Global min-max normalization before adding noise."""
    def norm(d):
        d = np.asarray(d, dtype=np.float32)
        den = np.max(d) - np.min(d)
        if den == 0:
            return np.zeros_like(d, dtype=np.float32)
        return (d - np.min(d)) / den

    return norm(data_e), norm(data_n), norm(data_z)


def normalize_batch(data_list):
    """Normalize each cut sample independently to [0, 1]."""
    normalized_out = []
    for seg in data_list:
        mins = np.min(seg, axis=0, keepdims=True)
        maxs = np.max(seg, axis=0, keepdims=True)
        den = maxs - mins
        den[den == 0] = 1.0
        normalized_out.append((seg - mins) / den)
    return normalized_out


def read_label_txt(label_path):
    """
    Read rockfall start/end time from label.txt.

    Recommended label.txt format:
        t1 12.35
        t2 16.80

    or simply:
        12.35 16.80
    """
    if not os.path.exists(label_path):
        return None, None

    with open(label_path, "r", encoding="utf-8") as f:
        text = f.read().strip().replace(",", " ")

    if not text:
        return None, None

    parts = text.split()

    # Format 1: t1 12.35 t2 16.80
    if "t1" in parts and "t2" in parts:
        t1 = float(parts[parts.index("t1") + 1])
        t2 = float(parts[parts.index("t2") + 1])
        return t1, t2

    # Format 2: 12.35 16.80
    nums = []
    for p in parts:
        try:
            nums.append(float(p))
        except ValueError:
            pass

    if len(nums) >= 2:
        return nums[0], nums[1]

    return None, None


def read_one_record(record_path, input_format="npz"):
    """
    Read one record and return:
        data_e, data_n, data_z, t1_sec, t2_sec

    t1_sec and t2_sec are only required for rockfall records.
    For non-rockfall records, they can be None.

    You can modify this function if your new data are stored in another format.
    """

    if input_format == "npz":
        # Recommended simple format:
        # one file in each record folder, for example data.npz
        # required keys: E, N, Z
        # optional keys for rockfall samples: t1, t2
        npz_files = [f for f in os.listdir(record_path) if f.endswith(".npz")]
        if len(npz_files) == 0:
            raise FileNotFoundError(f"No .npz file found in {record_path}")

        data = np.load(os.path.join(record_path, npz_files[0]))
        data_e = data["E"]
        data_n = data["N"]
        data_z = data["Z"]
        t1_sec = float(data["t1"]) if "t1" in data.files else None
        t2_sec = float(data["t2"]) if "t2" in data.files else None
        return data_e, data_n, data_z, t1_sec, t2_sec

    if input_format == "npy":
        # Alternative format:
        # each record folder contains:
        #   E.npy
        #   N.npy
        #   Z.npy
        #   label.txt  optional for non-rockfall, required for rockfall
        data_e = np.load(os.path.join(record_path, "E.npy"))
        data_n = np.load(os.path.join(record_path, "N.npy"))
        data_z = np.load(os.path.join(record_path, "Z.npy"))
        t1_sec, t2_sec = read_label_txt(os.path.join(record_path, "label.txt"))
        return data_e, data_n, data_z, t1_sec, t2_sec

    raise ValueError(f"Unsupported INPUT_FORMAT: {input_format}")


def check_three_components(data_e, data_n, data_z, record_path):
    """Basic checks for three-component waveform data."""
    data_e = np.asarray(data_e, dtype=np.float32).reshape(-1)
    data_n = np.asarray(data_n, dtype=np.float32).reshape(-1)
    data_z = np.asarray(data_z, dtype=np.float32).reshape(-1)

    if not (len(data_e) == len(data_n) == len(data_z)):
        raise ValueError(f"Three components have different lengths in {record_path}")

    if len(data_e) == 0:
        raise ValueError(f"Empty waveform in {record_path}")

    return data_e, data_n, data_z


# ============================================================
# 3. Main process
# ============================================================

def main():
    os.makedirs(os.path.dirname(X_SAVE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(Y_SAVE_PATH), exist_ok=True)

    buffer_rf_x = []
    buffer_rf_y = []
    buffer_nonrf_dict = {}

    category_names = sorted(os.listdir(DATA_ROOT))

    for category_name in category_names:
        category_path = os.path.join(DATA_ROOT, category_name)
        if not os.path.isdir(category_path):
            continue

        is_rf_category = category_name in RF_CATEGORIES
        label_out_choose = 1 if is_rf_category else 0

        if not is_rf_category and category_name not in buffer_nonrf_dict:
            buffer_nonrf_dict[category_name] = {"X": [], "Y": []}

        print(f"Processing category: {category_name} -> {'[RF]' if is_rf_category else '[Non-RF]'}")

        record_names = sorted(os.listdir(category_path))

        for i, record_name in enumerate(record_names):
            if i % 20 == 0:
                print(f"  Progress: {i}/{len(record_names)}")

            record_path = os.path.join(category_path, record_name)
            if not os.path.isdir(record_path):
                continue

            try:
                data_e, data_n, data_z, t1_sec, t2_sec = read_one_record(record_path, INPUT_FORMAT)
                data_e, data_n, data_z = check_three_components(data_e, data_n, data_z, record_path)
            except Exception as e:
                print(f"  Skip {record_path}: {e}")
                continue

            npts = len(data_e)
            snr = random.randint(SNR_MIN, SNR_MAX)

            # 1. Global preprocessing for noise augmentation.
            data_e, data_n, data_z = regularization_global(data_e, data_n, data_z)

            # 2. Add noise.
            input_data_e = wgn(data_e, snr)
            input_data_n = wgn(data_n, snr)
            input_data_z = wgn(data_z, snr)

            # 3. Remove mean.
            input_data_e -= np.mean(input_data_e)
            input_data_n -= np.mean(input_data_n)
            input_data_z -= np.mean(input_data_z)

            # 4. Generate point-wise labels.
            label_non_rf = np.ones(npts, dtype=np.float32)
            label_rf = np.zeros(npts, dtype=np.float32)
            label_placeholder = np.zeros(npts, dtype=np.float32)

            if label_out_choose == 1:
                if t1_sec is None or t2_sec is None:
                    print(f"  Skip RF record without t1/t2: {record_path}")
                    continue

                t1 = int(float(t1_sec) * SAMPLE_RATE)
                t2 = int(float(t2_sec) * SAMPLE_RATE)

                # Keep indices inside waveform range.
                t1 = max(0, min(t1, npts))
                t2 = max(0, min(t2, npts))

                if t2 <= t1:
                    print(f"  Skip invalid RF interval: {record_path}, t1={t1_sec}, t2={t2_sec}")
                    continue

                rf_label_out = function_tools.rangle_label(t1, t2)
                valid_len = min(len(rf_label_out), t2 - t1)
                label_rf[t1:t1 + valid_len] = rf_label_out[:valid_len]
                label_non_rf = label_non_rf - label_rf

            # 5. Cut waveform and labels into samples.
            temp_data_out = function_tools.data_cut(
                input_data_e, input_data_n, input_data_z, npts, []
            )
            temp_label_out = function_tools.data_cut(
                label_non_rf, label_rf, label_placeholder, npts, []
            )

            # 6. Local normalization after cutting.
            temp_data_out = normalize_batch(temp_data_out)

            # 7. Store samples.
            if is_rf_category:
                buffer_rf_x.extend(temp_data_out)
                buffer_rf_y.extend(temp_label_out)
            else:
                buffer_nonrf_dict[category_name]["X"].extend(temp_data_out)
                buffer_nonrf_dict[category_name]["Y"].extend(temp_label_out)

    # ========================================================
    # 4. Stratified sampling for Non-RF samples
    # ========================================================
    print("-" * 40)
    count_rf = len(buffer_rf_x)
    print(f"RF samples: {count_rf}")

    if count_rf == 0:
        print("Error: no RF samples found.")
        sys.exit()

    non_rf_categories = list(buffer_nonrf_dict.keys())
    num_categories = len(non_rf_categories)
    print(f"Non-RF categories: {num_categories}, {non_rf_categories}")

    if num_categories == 0:
        final_x = buffer_rf_x
        final_y = buffer_rf_y
    else:
        quota_per_category = int(count_rf / num_categories)
        print(f"Sampling strategy: RF={count_rf}, quota for each Non-RF category={quota_per_category}")

        final_x = []
        final_y = []
        final_x.extend(buffer_rf_x)
        final_y.extend(buffer_rf_y)

        for cat in non_rf_categories:
            cat_samples_x = buffer_nonrf_dict[cat]["X"]
            cat_samples_y = buffer_nonrf_dict[cat]["Y"]
            cat_count = len(cat_samples_x)

            print(f"  Non-RF category {cat}: available={cat_count}, target={quota_per_category}")

            if cat_count > quota_per_category:
                indices = random.sample(range(cat_count), quota_per_category)
                for idx in indices:
                    final_x.append(cat_samples_x[idx])
                    final_y.append(cat_samples_y[idx])
            else:
                final_x.extend(cat_samples_x)
                final_y.extend(cat_samples_y)

    # ========================================================
    # 5. Shuffle, format, and save
    # ========================================================
    print("-" * 40)
    print("Shuffling and saving...")

    combined = list(zip(final_x, final_y))
    random.shuffle(combined)

    if len(combined) == 0:
        print("Error: no samples to save.")
        sys.exit()

    final_x, final_y = zip(*combined)

    x_final = np.array(final_x, dtype=np.float32)
    y_temp = np.array(final_y, dtype=np.float32)

    # Keep two label channels: Non-RF and RF.
    y_final = y_temp[:, :, :, :2]

    print("Final data shapes:")
    print(f"X: {x_final.shape}")
    print(f"Y: {y_final.shape}")

    np.save(X_SAVE_PATH, x_final)
    np.save(Y_SAVE_PATH, y_final)

    print(f"Saved X to: {X_SAVE_PATH}")
    print(f"Saved Y to: {Y_SAVE_PATH}")


if __name__ == "__main__":
    main()
