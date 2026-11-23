"""Hàm đọc dữ liệu dùng chung cho cả nhóm (chỉ cần numpy, pandas, Pillow).

Dùng trong notebook:
    import sys; sys.path.append("FinalDeepData/common")
    from data_utils import load_hourly, make_windows, load_task3, encode_labels, decode_tokens
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


# ----------------------------------------------------------------------------- Task 1-2

def load_hourly(path):
    """Đọc file hanoi_hourly_*.csv, trả về DataFrame có chỉ mục thời gian (giờ GMT+7)."""
    df = pd.read_csv(path, parse_dates=["time"]).set_index("time")
    step = df.index.to_series().diff().dropna().unique()
    if len(step) != 1 or step[0] != pd.Timedelta(hours=1):
        raise ValueError("Chuoi thoi gian khong lien mach theo tung gio")
    return df


def make_windows(df, feature_cols, target_col="pm2_5", lookback=48, horizons=(2,)):
    """Cắt cửa sổ trượt cho Task 1 và Task 2.

    Mẫu ở thời điểm hiện tại t:
        X = feature_cols của các giờ t-lookback+1 ... t   (lookback giờ, tính cả giờ t)
        y = target_col tại t+h với mỗi h trong horizons
    Task 1: horizons=(2,)            -> PM2.5 lúc 12:00 khi hiện tại là 10:00
    Task 2: horizons=(1, 2, 3, 4, 5) -> 5 giờ liên tiếp sau giờ hiện tại

    Mỗi mẫu thuộc tập (cột split) chứa các giờ cần dự báo. Mẫu nào có nhãn nằm ở hai tập
    (ngay chỗ giáp ranh) bị bỏ. Đầu vào được phép lấy số của tập trước vì đó là quá khứ.
    Lưu ý: chuẩn hoá (scaler) phải fit trên mẫu train rồi mới áp cho validation, test.

    Trả về dict: X (N, lookback, F) float32, y (N, len(horizons)) float32,
                 time (N,) thời điểm hiện tại t, split (N,) "train" / "validation" / "test".
    """
    horizons = tuple(int(h) for h in horizons)
    values = df[list(feature_cols)].to_numpy(np.float32)
    target = df[target_col].to_numpy(np.float32)
    split = df["split"].to_numpy()
    n = len(df)
    t = np.arange(lookback - 1, n - max(horizons))
    windows = np.lib.stride_tricks.sliding_window_view(values, (lookback, values.shape[1]))[:, 0]
    X = windows[t - lookback + 1]
    y = np.stack([target[t + h] for h in horizons], axis=1)
    first, last = split[t + min(horizons)], split[t + max(horizons)]
    keep = first == last
    return {
        "X": np.ascontiguousarray(X[keep]),
        "y": y[keep],
        "time": df.index.to_numpy()[t[keep]],
        "split": first[keep],
    }


# ----------------------------------------------------------------------------- Task 3

# Bộ token cho decoder: chữ số 0-9 là token 0-9
PAD, BOS, EOS = 10, 11, 12
VOCAB_SIZE = 13


def load_task3(dataset_dir, split, width=256, height=32):
    """Đọc một tập ảnh chuỗi số (mnist_sequences_v1 hoặc v2).

    Nhãn được đọc dạng chuỗi để giữ số 0 ở đầu. Ảnh được đệm nền đen bên phải cho đủ
    `width` pixel (không kéo giãn, để bề rộng ảnh không lộ ra độ dài chuỗi).
    Trả về X (N, height, width) uint8 và danh sách nhãn (str).
    """
    root = Path(dataset_dir)
    df = pd.read_csv(root / f"{split}_labels.csv", dtype={"label": str}, encoding="utf-8-sig")
    X = np.zeros((len(df), height, width), np.uint8)
    for i, rel in enumerate(df["image_path"]):
        a = np.asarray(Image.open(root / rel).convert("L"))
        h, w = a.shape
        if h != height or w > width:
            raise ValueError(f"{rel}: kich thuoc {a.shape} vuot qua {height}x{width}")
        X[i, :, :w] = a
    return X, df["label"].tolist()


def encode_labels(labels, max_len=8):
    """Đổi nhãn chuỗi thành token cho teacher forcing.

    decoder_in  = <bos> d1 ... dn <pad> ...     (đầu vào của decoder)
    decoder_out = d1 ... dn <eos> <pad> ...     (đích cần dự đoán; bỏ qua <pad> khi tính loss)
    Hai mảng có cùng độ dài max_len + 1.
    """
    n = len(labels)
    dec_in = np.full((n, max_len + 1), PAD, np.int32)
    dec_out = np.full((n, max_len + 1), PAD, np.int32)
    for i, s in enumerate(labels):
        digits = [int(c) for c in s]
        if len(digits) > max_len:
            raise ValueError(f"Nhan '{s}' dai hon max_len={max_len}")
        dec_in[i, 0] = BOS
        dec_in[i, 1:len(digits) + 1] = digits
        dec_out[i, :len(digits)] = digits
        dec_out[i, len(digits)] = EOS
    return dec_in, dec_out


def decode_tokens(tokens):
    """Đổi dãy token dự đoán thành chuỗi chữ số, dừng ở <eos>."""
    out = []
    for tok in tokens:
        tok = int(tok)
        if tok == EOS:
            break
        if 0 <= tok <= 9:
            out.append(str(tok))
    return "".join(out)


def augment(X, rng, noise_max=12.0, brightness=(0.8, 1.1)):
    """Tăng cường ảnh khi huấn luyện: đổi độ sáng và thêm nhiễu Gauss (không đổi nhãn).

    X: (N, H, W) uint8; rng: np.random.default_rng(...). Trả về mảng uint8 mới.
    """
    Xf = X.astype(np.float32)
    scale = rng.uniform(*brightness, size=(len(X), 1, 1)).astype(np.float32)
    sigma = rng.uniform(0, noise_max, size=(len(X), 1, 1)).astype(np.float32)
    Xf = Xf * scale + rng.normal(0, 1, size=X.shape).astype(np.float32) * sigma
    return np.clip(Xf, 0, 255).astype(np.uint8)
