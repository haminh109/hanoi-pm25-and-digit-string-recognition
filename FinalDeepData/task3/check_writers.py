"""Kiểm tra mỗi ảnh chuỗi số được ghép từ chữ số của bao nhiêu người viết.

Mã người viết của từng ảnh MNIST lấy từ nhãn mở rộng của QMNIST (cột 2), do generate_data_v3.py tải về
data/raw/. Với mỗi bộ dữ liệu:
- v3 (có cột `source` trong file nhãn): đọc thẳng chỉ số các ảnh MNIST đã ghép, rồi kiểm tra
  (1) mọi chữ số của một ảnh cùng một người viết và đúng bằng cột `writer`,
  (2) ảnh MNIST nguồn đúng chữ số của nhãn và không dùng lặp lại trong cùng một ảnh,
  (3) không người viết nào xuất hiện ở hai tập.
- v1, v2 (file nhãn không lưu ảnh nguồn): chạy lại đúng script sinh với cùng seed để biết mỗi chữ số lấy
  từ ảnh MNIST nào. Nhãn chạy lại phải trùng file nhãn; số ảnh trùng từng pixel chỉ để tham khảo, vì
  ảnh v2 chỉ trùng từng pixel khi dùng cùng phiên bản Pillow với lúc sinh.

Chạy (trong thư mục task3):  python check_writers.py
"""
import gzip
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def qmnist_writers(split, n):
    raw = gzip.open(DATA / "raw" / f"qmnist-{split}-labels-idx2-int.gz").read()
    _, rows, cols = np.frombuffer(raw[:12], ">i4")
    return np.frombuffer(raw[12:], ">i4").reshape(rows, cols)[:n, 2].astype(np.int64)


with np.load(DATA / "raw" / "mnist.npz") as d:
    X_TRAIN, Y_TRAIN, X_TEST, Y_TEST = d["x_train"], d["y_train"], d["x_test"], d["y_test"]
WRITER = {"train": qmnist_writers("train", len(Y_TRAIN)), "test": qmnist_writers("test", len(Y_TEST))}


def check_v3(folder):
    rows, writers_of = [], {}
    for split in ["train", "validation", "test"]:
        df = pd.read_csv(folder / f"{split}_labels.csv", dtype={"label": str, "source": str}, encoding="utf-8-sig")
        mnist = "test" if split == "test" else "train"
        labels_mnist = Y_TEST if mnist == "test" else Y_TRAIN
        same, label_ok, no_repeat = 0, 0, 0
        for label, w, src in zip(df["label"], df["writer"], df["source"]):
            idx = np.array(src.split(), dtype=np.int64)
            same += bool(np.all(WRITER[mnist][idx] == w))
            label_ok += "".join(map(str, labels_mnist[idx])) == label
            no_repeat += len(set(idx)) == len(idx)
        writers_of[split] = set(df["writer"])
        rows.append({"bộ": folder.name, "tập": split, "số ảnh": len(df), "số người viết": df["writer"].nunique(),
                     "mọi chữ số cùng một người viết": same, "ảnh nguồn đúng nhãn": label_ok,
                     "không dùng lặp ảnh nguồn": no_repeat})
    overlap = {f"{a}-{b}": len(writers_of[a] & writers_of[b])
               for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]}
    return rows, overlap


def replay_v1(name, images, labels, count, seed, folder):
    g1 = load_module("generate_data")
    rng = np.random.default_rng(seed)
    pools = [np.flatnonzero(labels == k) for k in range(10)]
    df = pd.read_csv(folder / f"{name}_labels.csv", dtype={"label": str}, encoding="utf-8-sig")
    sources, n_same_pixels = [], 0
    for i in range(count):
        length = int(rng.integers(g1.MIN_LENGTH, g1.MAX_LENGTH + 1))
        digits = rng.integers(0, 10, size=length)
        assert "".join(str(int(d)) for d in digits) == df["label"][i], "chay lai khong khop nhan"
        idx, pieces = [], []
        for digit in digits:                    # đúng thứ tự gọi rng của generate_data.py
            k = int(rng.choice(pools[int(digit)]))
            im = Image.fromarray(images[k]).rotate(float(rng.uniform(-6, 6)), resample=Image.Resampling.BICUBIC,
                                                   expand=True, fillcolor=0)
            size = int(rng.integers(24, 29))
            im.thumbnail((size, size), Image.Resampling.LANCZOS)
            idx.append(k)
            pieces.append(im)
        gaps = rng.integers(0, 6, size=length - 1)
        canvas = Image.new("L", (8 + sum(im.width for im in pieces) + int(gaps.sum()), g1.HEIGHT), color=0)
        x = 4
        for j, im in enumerate(pieces):
            y = int(rng.integers(2, g1.HEIGHT - im.height - 1))
            canvas.paste(im, (x, y))
            x += im.width + (int(gaps[j]) if j < length - 1 else 0)
        sources.append(idx)
        n_same_pixels += np.array_equal(np.asarray(canvas), np.asarray(Image.open(folder / df["image_path"][i])))
    return sources, n_same_pixels


def replay_v2(name, images, labels, count, seed, folder):
    g2 = load_module("generate_data_v2")
    P = g2.PARAMS
    rng = np.random.default_rng(seed)
    pools = [np.flatnonzero(labels == k) for k in range(10)]
    df = pd.read_csv(folder / f"{name}_labels.csv", dtype={"label": str}, encoding="utf-8-sig")
    sources, n_same_pixels = [], 0
    for i in range(count):
        length = int(rng.integers(g2.MIN_LENGTH, g2.MAX_LENGTH + 1))
        digits = rng.integers(0, 10, size=length)
        assert "".join(str(int(d)) for d in digits) == df["label"][i], "chay lai khong khop nhan"
        # Từ đây chép đúng make_image của generate_data_v2.py, chỉ thêm việc ghi lại chỉ số ảnh nguồn
        idx, pieces = [], []
        for d in digits:
            while True:
                k = int(rng.choice(pools[int(d)]))
                p = g2.make_digit(images[k], rng)
                if p is not None and int(p.sum()) > 255 * 15:
                    idx.append(k)
                    pieces.append(p)
                    break
        gaps = []
        for j in range(len(pieces) - 1):
            lo = -min(3, pieces[j].shape[1] // 3, pieces[j + 1].shape[1] // 3)
            gaps.append(int(rng.integers(lo, P["gap_px"][1] + 1)))
        canvas = np.zeros((g2.HEIGHT, 8 + sum(p.shape[1] for p in pieces) + sum(gaps)), np.uint8)
        x = 4
        for j, p in enumerate(pieces):
            h, w = p.shape
            y = (g2.HEIGHT - h) // 2 + int(rng.integers(-P["vertical_jitter_px"], P["vertical_jitter_px"] + 1))
            y = min(max(y, 0), g2.HEIGHT - h)
            region = canvas[y:y + h, x:x + w]
            np.maximum(region, p, out=region)
            x += w + (gaps[j] if j < len(gaps) else 0)
        img = Image.fromarray(canvas)
        if rng.random() < P["blur_prob"]:
            img = img.filter(ImageFilter.GaussianBlur(float(rng.uniform(*P["blur_radius"]))))
        peak = float(rng.uniform(*P["ink_peak"]))
        a = np.round(np.asarray(img, dtype=np.float32) * peak).astype(np.uint8)
        sources.append(idx)
        n_same_pixels += a.shape[1] <= g2.MAX_WIDTH and np.array_equal(a, np.asarray(Image.open(folder / df["image_path"][i])))
    return sources, n_same_pixels


def check_old(version):
    folder = DATA / f"mnist_sequences_{version}"
    split_file = DATA / "mnist_sequences_v1" / "source_split_indices.npz"   # v1 và v2 dùng chung cách chia
    with np.load(split_file) as s:
        ids = {"train": s["train_indices"], "validation": s["validation_indices"], "test": np.arange(len(Y_TEST))}
    replay, seeds = (replay_v1, {"train": 43, "validation": 44, "test": 45}) if version == "v1" \
        else (replay_v2, load_module("generate_data_v2").SEEDS)
    rows = []
    for split in ["train", "validation", "test"]:
        x, y, w = (X_TEST, Y_TEST, WRITER["test"]) if split == "test" else (X_TRAIN, Y_TRAIN, WRITER["train"])
        count = len(pd.read_csv(folder / f"{split}_labels.csv", encoding="utf-8-sig"))
        sources, same_pixels = replay(split, x[ids[split]], y[ids[split]], count, seeds[split], folder)
        n_writers = np.array([len(set(w[ids[split][np.array(src)]])) for src in sources])
        lengths = np.array([len(src) for src in sources])
        rows.append({"bộ": folder.name, "tập": split, "số ảnh": count,
                     "mọi chữ số cùng một người viết": int((n_writers == 1).sum()),
                     "số người viết trung bình mỗi ảnh": round(float(n_writers.mean()), 2),
                     "mỗi chữ số một người viết khác nhau": int((n_writers == lengths).sum()),
                     "ảnh chạy lại trùng từng pixel": int(same_pixels)})
    return rows


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    for version in ["v3", "v3_big"]:
        folder = DATA / f"mnist_sequences_{version}"
        if folder.exists():
            rows, overlap = check_v3(folder)
            print(pd.DataFrame(rows).to_string(index=False))
            print("Số người viết chung giữa hai tập:", overlap, "\n")
    old = [row for version in ["v1", "v2"] if (DATA / f"mnist_sequences_{version}").exists()
           for row in check_old(version)]
    if old:
        print(pd.DataFrame(old).to_string(index=False))
