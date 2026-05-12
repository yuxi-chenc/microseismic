import os
import glob
import argparse
import pandas as pd

import config
from predictor import run_prediction_all_stations

os.environ["CUDA_VISIBLE_DEVICES"] = getattr(config, "CUDA_DEVICE", "0")


# ── 模型加载 ─────────────────────────────────────────────────

def load_model():
    """加载训练好的落石识别模型权重。"""
    from model_with_GCAM import base_model

    model = base_model()
    ckpt = config.MODEL_CKPT

    if os.path.exists(ckpt + ".index"):
        model.load_weights(ckpt).expect_partial()
        print(f"[main] Model loaded from: {ckpt}")
    else:
        print(f"[main] WARNING: checkpoint not found at: {ckpt}")
        print("[main] Please check config.MODEL_CKPT.")

    return model


# ── 构建台站文件映射 ─────────────────────────────────────────

def build_station_file_map(
    data_dir: str,
    station_csv: str,
    date_mask: str = "",
) -> dict[int, list[str]]:
    """
    扫描 data_dir，将 SAC 文件按台站 ID 分组。

    文件名约定：
        {4位台站ID}.HH{E|N|Z}.sac

    例如：
        0188.HHE.sac
        0188.HHN.sac
        0188.HHZ.sac

    Parameters
    ----------
    data_dir : str
        SAC 文件所在目录。
    station_csv : str
        台站表路径，需要包含 Station_ID 列。
    date_mask : str
        可选过滤字符串。为空表示不过滤。

    Returns
    -------
    station_file_map : dict[int, list[str]]
        {station_id: [sac_file_path, ...]}
    """
    df = pd.read_csv(station_csv)

    if "Station_ID" not in df.columns:
        raise ValueError(f"[main] Station CSV must contain 'Station_ID': {station_csv}")

    blacklist = getattr(config, "BLACKLIST_STATIONS", [])
    df = df[~df["Station_ID"].isin(blacklist)]

    station_file_map: dict[int, list[str]] = {}

    for _, row in df.iterrows():
        sid = int(row["Station_ID"])

        # 兼容长编号，只取末尾 4 位作为 SAC 文件名前缀
        str_id = str(sid)
        short_id = str_id[-4:].zfill(4) if len(str_id) >= 4 else str_id.zfill(4)

        pattern = os.path.join(data_dir, f"{short_id}.HH*.sac")
        files = sorted(glob.glob(pattern))

        if date_mask:
            files = [f for f in files if date_mask in f]

        if files:
            station_file_map[sid] = files

    return station_file_map


# ── 主预测流程 ───────────────────────────────────────────────

def main(
    data_dir: str,
    station_csv: str = "",
    date_mask: str = "",
):
    """
    单独预测主流程。

    只生成每个台站的预测 CSV：
        output/pred_csv/station_<station_id>.csv

    不执行：
        consensus filtering
        raw waveform slicing
        impact extraction
        npy dataset saving
    """
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.PRED_CSV_DIR, exist_ok=True)

    station_csv = station_csv or config.STATION_CSV

    # Step 1: 构建台站文件映射
    print("\n=== Step 1: Build station file map ===")
    station_file_map = build_station_file_map(
        data_dir=data_dir,
        station_csv=station_csv,
        date_mask=date_mask,
    )

    print(f"[main] Stations with data: {len(station_file_map)}")

    if not station_file_map:
        print("[main] No station data found. Exit.")
        return

    # Step 2: 加载模型
    print("\n=== Step 2: Load model ===")
    model = load_model()

    # Step 3: 单台站预测，并输出 pred_csv
    print("\n=== Step 3: Per-station prediction ===")
    all_events = run_prediction_all_stations(model, station_file_map)

    total_picks = sum(len(events) for events in all_events.values())

    print("\n=== Prediction finished ===")
    print(f"[main] Total picked events: {total_picks}")
    print(f"[main] Prediction CSV directory: {config.PRED_CSV_DIR}")

    if total_picks == 0:
        print("[main] No picks detected.")
    else:
        print("[main] Done. Only pred_csv files were generated.")


# ── CLI ──────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rockfall prediction only: generate per-station pred_csv files."
    )

    parser.add_argument(
        "--data_dir",
        default=config.DATA_DIR,
        help="Directory containing SAC files.",
    )
    parser.add_argument(
        "--station_csv",
        default=config.STATION_CSV,
        help="Station CSV file containing Station_ID column.",
    )
    parser.add_argument(
        "--date_mask",
        default="",
        help="Optional string filter for input files.",
    )

    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        station_csv=args.station_csv,
        date_mask=args.date_mask,
    )
