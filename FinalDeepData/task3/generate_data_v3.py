"""Sinh bộ ảnh chuỗi chữ số viết tay v3 cho Task 3: mỗi chuỗi do MỘT người viết.

Vì sao có v3. Ở v1 và v2, mỗi chữ số trong chuỗi được lấy ngẫu nhiên từ cả kho MNIST, nên một chuỗi
5 chữ số thường là chữ của 5 người khác nhau (kiểm tra bằng check_writers.py: 0/24.000 ảnh của mỗi bản
có mọi chữ số cùng một người viết). Chuỗi số viết tay thật luôn do một người viết, nên v3 sửa lại:

- Mỗi ảnh chỉ ghép chữ số của cùng một người viết. MNIST không kèm mã người viết, nhưng bộ QMNIST
  (Yadav & Bottou, 2019) đã truy từng ảnh MNIST về tờ mẫu gốc trong NIST Special Database 19 và công bố
  mã người viết (writer ID). QMNIST-train khớp từng ảnh và đúng thứ tự với MNIST-train; 10.000 ảnh đầu
  của QMNIST-test khớp với MNIST-test. Script tự tải hai file nhãn mở rộng của QMNIST (khoảng 1 MB),
  kiểm tra mã MD5 và kiểm tra nhãn chữ số khớp MNIST trước khi dùng.
- Kiểu chữ (cỡ chữ, góc xoay, độ nghiêng, độ dày nét, khoảng cách) được chọn một lần cho cả chuỗi; từng
  chữ số chỉ dao động nhẹ quanh đó, như khi một người viết liền một dãy số bằng một cây bút.
- Chia train và validation theo người viết: 20% người viết của MNIST-train (chọn ngẫu nhiên trong từng
  nhóm tờ mẫu của NIST) chỉ dùng cho validation. MNIST vốn đã tách người viết giữa train và test.
  Vì vậy không người viết nào xuất hiện ở hai tập: validation và test đều đo khả năng đọc chữ của người lạ.

Cách sinh một ảnh:
1. Chọn độ dài 3-8 và các chữ số ngẫu nhiên đều, như v2 (nhãn không phụ thuộc người viết).
2. Chọn ngẫu nhiên một người viết trong số những người có đủ mẫu cho các chữ số đó. Chữ số lặp lại
   (ví dụ hai chữ 7) lấy hai mẫu khác nhau của cùng người đó.
3. Chọn kiểu chữ cho cả chuỗi, biến đổi từng chữ số, cắt sát nét rồi ghép thành hàng như v2.

Giữ nguyên như v2: ảnh cao 32 pixel, rộng tối đa 256; khe giữa hai chữ số từ -3 đến +6 pixel (có lúc
chạm hoặc chồng nhẹ); làm mờ nhẹ một nửa số ảnh; đổi độ đậm nét; nhãn lưu dạng chuỗi, khi đọc phải dùng
dtype={"label": str}. File nhãn có thêm hai cột:
- `writer`: mã người viết NIST của cả chuỗi (mã từ 2100 trở lên là học sinh trung học, nhỏ hơn là nhân
  viên Cục Điều tra Dân số Mỹ);
- `source`: chỉ số các ảnh MNIST đã ghép, từ trái sang phải (train và validation lấy từ MNIST-train,
  test lấy từ MNIST-test).

Chạy:  python generate_data_v3.py
       python generate_data_v3.py --out data/mnist_sequences_v3_big --train 60000
"""
import argparse
import csv
import gzip
import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
import PIL
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
MNIST_PATH = RAW_DIR / "mnist.npz"
MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
QMNIST_URL = "https://raw.githubusercontent.com/facebookresearch/qmnist/master/"
QMNIST_FILES = {"train": ("qmnist-train-labels-idx2-int.gz", "0058f8dd561b90ffdd0f734c6a30e5e4"),
                "test": ("qmnist-test-labels-idx2-int.gz", "5b5b05890a5e13444e108efe57b788aa")}

HEIGHT, MAX_WIDTH = 32, 256
MIN_LENGTH, MAX_LENGTH = 3, 8
SPLIT_SEED = 2030                    # chọn người viết cho validation
VALIDATION_WRITER_SHARE = 0.2
SEEDS = {"train": 2031, "validation": 2032, "test": 2033}

PARAMS = {
    "height": HEIGHT,
    "max_width": MAX_WIDTH,
    "length_range": [MIN_LENGTH, MAX_LENGTH],
    # Kiểu chữ chung của cả chuỗi
    "digit_height_px": [19, 27],
    "rotation_deg": 7,
    "shear": 0.3,
    "gap_px": [-2, 5],
    "thicker_stroke_prob": 0.25,
    "thinner_stroke_prob": 0.20,
    # Dao động nhẹ giữa các chữ số trong chuỗi
    "digit_height_jitter_px": 1,
    "rotation_jitter_deg": 3,
    "shear_jitter": 0.05,
    "gap_jitter_px": 2,
    "gap_limit_px": [-3, 6],
    "vertical_jitter_px": 2,
    # Cả ảnh
    "blur_prob": 0.5,
    "blur_radius": [0.3, 0.8],
    "ink_peak": [0.7, 1.0],
}


def download(url, path, md5=None):
    if not path.exists():
        print("Dang tai", url)
        temporary = path.with_name(path.name + ".part")
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(path)
    if md5 is not None and hashlib.md5(path.read_bytes()).hexdigest() != md5:
        raise ValueError(f"{path.name}: ma MD5 khong khop. Hay xoa file nay roi chay lai.")


def load_writers(y_train, y_test):
    """Mã người viết và nhóm tờ mẫu NIST (hsf) của từng ảnh MNIST, lấy từ nhãn mở rộng của QMNIST."""
    result = {}
    for split, y in [("train", y_train), ("test", y_test)]:
        name, md5 = QMNIST_FILES[split]
        download(QMNIST_URL + name, RAW_DIR / name, md5)
        raw = gzip.open(RAW_DIR / name).read()
        magic, rows, cols = np.frombuffer(raw[:12], ">i4")
        ext = np.frombuffer(raw[12:], ">i4").reshape(rows, cols)[:len(y)]
        # Cột 0: chữ số, cột 1: nhóm tờ mẫu hsf, cột 2: mã người viết. QMNIST khớp từng ảnh với MNIST.
        if magic != 3074 or not np.array_equal(ext[:, 0], y):
            raise ValueError(f"{name} khong khop voi MNIST-{split}")
        result[split] = (ext[:, 2].astype(np.int64), ext[:, 1].astype(np.int64))
    return result


def split_writers(writer, series, rng):
    """Chia người viết của MNIST-train: 20% cho validation (chọn riêng trong từng nhóm hsf), còn lại cho train."""
    validation = []
    for s in np.unique(series):
        ws = np.unique(writer[series == s])
        k = int(round(len(ws) * VALIDATION_WRITER_SHARE))
        validation += rng.permutation(ws)[:k].tolist()
    validation = np.sort(np.array(validation, dtype=np.int64))
    return np.setdiff1d(np.unique(writer), validation), validation


class WriterPool:
    """Chỉ số các ảnh MNIST của một tập, xếp theo (người viết, chữ số)."""

    def __init__(self, indices, writer, labels):
        self.writers = np.unique(writer[indices])
        groups = {(int(w), c): [] for w in self.writers for c in range(10)}
        for i in indices:
            groups[int(writer[i]), int(labels[i])].append(int(i))
        self.groups = {key: np.array(v, dtype=np.int64) for key, v in groups.items()}
        self.counts = np.array([[len(self.groups[int(w), c]) for c in range(10)] for w in self.writers])

    def sample(self, digits, rng):
        """Chọn một người viết có đủ mẫu cho các chữ số, rồi lấy mẫu không lặp lại trong chuỗi."""
        need = np.bincount(digits, minlength=10)
        eligible = np.flatnonzero((self.counts >= need).all(axis=1))
        w = int(self.writers[int(rng.choice(eligible))])
        picks = {c: rng.choice(self.groups[w, c], size=int(need[c]), replace=False).tolist()
                 for c in range(10) if need[c] > 0}
        return w, [picks[int(d)].pop() for d in digits]


def make_style(rng):
    """Kiểu chữ của một người viết, dùng chung cho cả chuỗi."""
    r = rng.random()
    if r < PARAMS["thicker_stroke_prob"]:
        stroke = "thicker"
    elif r < PARAMS["thicker_stroke_prob"] + PARAMS["thinner_stroke_prob"]:
        stroke = "thinner"
    else:
        stroke = "normal"
    return {
        "stroke": stroke,
        "thin_level": int(rng.integers(60, 121)),
        "height": int(rng.integers(PARAMS["digit_height_px"][0], PARAMS["digit_height_px"][1] + 1)),
        "rotation": float(rng.uniform(-PARAMS["rotation_deg"], PARAMS["rotation_deg"])),
        "shear": float(rng.uniform(-PARAMS["shear"], PARAMS["shear"])),
        "gap": int(rng.integers(PARAMS["gap_px"][0], PARAMS["gap_px"][1] + 1)),
    }


def apply_stroke(im, style):
    if style["stroke"] == "thicker":
        return im.filter(ImageFilter.MaxFilter(3))
    if style["stroke"] == "thinner":
        lo = style["thin_level"]
        return im.point([0 if p <= lo else int((p - lo) * 255 / (255 - lo)) for p in range(256)])
    return im


def make_digit(img28, style, rng):
    """Biến đổi một ảnh MNIST 28x28 theo kiểu chữ của chuỗi (cộng dao động nhẹ), rồi cắt sát nét chữ."""
    rotation = style["rotation"] + float(rng.uniform(-PARAMS["rotation_jitter_deg"], PARAMS["rotation_jitter_deg"]))
    s = style["shear"] + float(rng.uniform(-PARAMS["shear_jitter"], PARAMS["shear_jitter"]))
    h = style["height"] + int(rng.integers(-PARAMS["digit_height_jitter_px"], PARAMS["digit_height_jitter_px"] + 1))

    piece = None
    for keep_stroke_style in (True, False):    # nét làm mảnh mà mất gần hết thì giữ độ dày gốc của chữ số này
        im = Image.fromarray(img28)
        if keep_stroke_style:
            im = apply_stroke(im, style)
        # Đặt vào khung 56x56 để xoay và nghiêng không bị cắt mất nét
        canvas = Image.new("L", (56, 56), 0)
        canvas.paste(im, (14, 14))
        canvas = canvas.rotate(rotation, resample=Image.Resampling.BICUBIC, fillcolor=0)
        canvas = canvas.transform((56, 56), Image.Transform.AFFINE, (1, s, -s * 28, 0, 1, 0),
                                  resample=Image.Resampling.BICUBIC, fillcolor=0)
        a = np.asarray(canvas)
        ys, xs = np.nonzero(a > 40)
        if len(xs) == 0:
            continue
        a = a[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
        w = max(3, int(round(a.shape[1] * h / a.shape[0])))
        piece = np.asarray(Image.fromarray(a).resize((w, h), Image.Resampling.LANCZOS))
        if int(piece.sum()) > 255 * 15:        # đủ nét để nhìn thấy
            return piece
    return piece


def make_image(digit_images, rng):
    """Ghép các ảnh 28x28 của cùng một người viết thành một ảnh chuỗi số cao 32 pixel."""
    style = make_style(rng)
    pieces = [make_digit(img, style, rng) for img in digit_images]

    gaps = []
    lo_limit, hi_limit = PARAMS["gap_limit_px"]
    for j in range(len(pieces) - 1):
        # Khoảng cách quen tay của người viết, dao động nhẹ; không cho chồng quá 1/3 bề rộng chữ số hẹp hơn
        lo = max(lo_limit, -min(pieces[j].shape[1] // 3, pieces[j + 1].shape[1] // 3))
        g = style["gap"] + int(rng.integers(-PARAMS["gap_jitter_px"], PARAMS["gap_jitter_px"] + 1))
        gaps.append(min(max(g, lo), hi_limit))

    width = 4 + sum(p.shape[1] for p in pieces) + sum(gaps) + 4
    canvas = np.zeros((HEIGHT, width), np.uint8)
    x = 4
    for j, p in enumerate(pieces):
        h, w = p.shape
        y = (HEIGHT - h) // 2 + int(rng.integers(-PARAMS["vertical_jitter_px"], PARAMS["vertical_jitter_px"] + 1))
        y = min(max(y, 0), HEIGHT - h)
        region = canvas[y:y + h, x:x + w]
        np.maximum(region, p, out=region)
        x += w + (gaps[j] if j < len(gaps) else 0)

    img = Image.fromarray(canvas)
    if rng.random() < PARAMS["blur_prob"]:
        img = img.filter(ImageFilter.GaussianBlur(float(rng.uniform(*PARAMS["blur_radius"]))))
    peak = float(rng.uniform(*PARAMS["ink_peak"]))
    img = Image.fromarray(np.round(np.asarray(img, dtype=np.float32) * peak).astype(np.uint8))

    scaled = False
    if img.width > MAX_WIDTH:
        new_h = max(1, round(HEIGHT * MAX_WIDTH / img.width))
        small = img.resize((MAX_WIDTH, new_h), Image.Resampling.LANCZOS)
        img = Image.new("L", (MAX_WIDTH, HEIGHT), 0)
        img.paste(small, (0, (HEIGHT - new_h) // 2))
        scaled = True
    return img, scaled


def create_split(pool, images, name, count, seed, out):
    rng = np.random.default_rng(seed)
    (out / name).mkdir()
    n_scaled, used_writers = 0, set()
    with (out / f"{name}_labels.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label", "length", "writer", "source"])
        for i in range(count):
            length = int(rng.integers(MIN_LENGTH, MAX_LENGTH + 1))
            digits = rng.integers(0, 10, size=length)
            label = "".join(str(int(d)) for d in digits)
            w, source = pool.sample(digits, rng)
            img, scaled = make_image([images[k] for k in source], rng)
            n_scaled += scaled
            used_writers.add(w)
            rel = f"{name}/{i:06d}.png"
            img.save(out / rel)
            writer.writerow([rel, label, length, w, " ".join(str(k) for k in source)])
            if (i + 1) % 5000 == 0:
                print(f"{name}: {i + 1}/{count}")
    return {"images_shrunk_to_fit_256px": n_scaled, "writers_available": len(pool.writers),
            "writers_used": len(used_writers)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "mnist_sequences_v3"))
    ap.add_argument("--train", type=int, default=20000)
    ap.add_argument("--validation", type=int, default=2000)
    ap.add_argument("--test", type=int, default=2000)
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"Thu muc da ton tai: {out}. Hay doi --out neu muon tao bo moi.")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    download(MNIST_URL, MNIST_PATH)
    with np.load(MNIST_PATH, allow_pickle=False) as d:
        x_train, y_train, x_test, y_test = d["x_train"], d["y_train"], d["x_test"], d["y_test"]
    writers = load_writers(y_train, y_test)
    w_train, hsf_train = writers["train"]
    w_test, _ = writers["test"]

    train_writers, val_writers = split_writers(w_train, hsf_train, np.random.default_rng(SPLIT_SEED))
    train_ids = np.flatnonzero(np.isin(w_train, train_writers))
    val_ids = np.flatnonzero(np.isin(w_train, val_writers))
    test_writers = np.unique(w_test)
    # Ba nhóm người viết không giao nhau (MNIST đã tách người viết giữa train và test)
    assert not set(train_writers) & set(val_writers)
    assert not (set(train_writers) | set(val_writers)) & set(test_writers)

    out.mkdir(parents=True)
    np.savez(out / "source_split_indices.npz", train_indices=train_ids,
             validation_indices=val_ids, test_indices=np.arange(len(x_test)))
    (out / "writer_split.json").write_text(json.dumps(
        {"train": train_writers.tolist(), "validation": val_writers.tolist(), "test": test_writers.tolist()}),
        encoding="utf-8")

    counts = {"train": args.train, "validation": args.validation, "test": args.test}
    pools = {"train": (WriterPool(train_ids, w_train, y_train), x_train),
             "validation": (WriterPool(val_ids, w_train, y_train), x_train),
             "test": (WriterPool(np.arange(len(x_test)), w_test, y_test), x_test)}
    stats = {}
    for name in ["train", "validation", "test"]:
        pool, images = pools[name]
        stats[name] = create_split(pool, images, name, counts[name], SEEDS[name], out)
        stats[name]["mnist_digits_available"] = int(pool.counts.sum())

    config = {"version": "v3", "same_writer_per_image": True, "counts": counts,
              "seeds": SEEDS, "writer_split_seed": SPLIT_SEED, "validation_writer_share": VALIDATION_WRITER_SHARE,
              "params": PARAMS, "split_stats": stats,
              "writer_ids": "QMNIST extended labels (Yadav & Bottou, 2019), cot 2; " + QMNIST_URL,
              "library_versions": {"numpy": np.__version__, "pillow": PIL.__version__}}
    (out / "generation_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Hoan thanh. Du lieu luu tai:", out)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
