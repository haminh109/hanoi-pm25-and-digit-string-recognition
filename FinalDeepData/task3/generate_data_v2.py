"""Sinh bộ ảnh chuỗi chữ số viết tay v2 (khó hơn v1) cho Task 3.

Khác với v1 (generate_data.py):
- Cắt sát từng chữ số trước khi ghép, nên khoảng cách giữa các chữ số do "khe" quyết định.
  Khe từ -3 đến +6 pixel: có lúc các chữ số chạm hoặc chồng nhẹ lên nhau như chữ viết thật.
- Xoay tối đa 10 độ, nghiêng (shear) tối đa 0,35, đổi độ dày nét, chiều cao chữ số 18-28 pixel.
- Làm mờ nhẹ (một nửa số ảnh) và đổi độ sáng nét chữ.
- Ảnh cao 32 pixel, rộng tối đa 256 pixel; ảnh nào rộng hơn được thu nhỏ cho vừa.

Giữ nguyên như v1:
- Chữ trắng trên nền đen, độ dài chuỗi 3-8, mọi chữ số 0-9 (kể cả số 0 ở đầu).
- Nhãn lưu dạng chuỗi; khi đọc phải dùng dtype={"label": str}.
- Dùng lại đúng cách chia nguồn MNIST của v1 (data/mnist_sequences_v1/source_split_indices.npz):
  ảnh train chỉ dùng chữ số của 48.000 ảnh MNIST-train, validation dùng 12.000 ảnh còn lại,
  test dùng MNIST-test. Không chữ số nào xuất hiện ở hai tập.

Chạy:  python generate_data_v2.py
       python generate_data_v2.py --out data/mnist_sequences_v2_big --train 60000
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "mnist.npz"
SPLIT_FILE = ROOT / "data" / "mnist_sequences_v1" / "source_split_indices.npz"

HEIGHT, MAX_WIDTH = 32, 256
MIN_LENGTH, MAX_LENGTH = 3, 8
SEEDS = {"train": 2027, "validation": 2028, "test": 2029}

PARAMS = {
    "height": HEIGHT,
    "max_width": MAX_WIDTH,
    "length_range": [MIN_LENGTH, MAX_LENGTH],
    "digit_height_px": [18, 28],
    "rotation_deg": 10,
    "shear": 0.35,
    "gap_px": [-3, 6],
    "thicker_stroke_prob": 0.25,
    "thinner_stroke_prob": 0.20,
    "blur_prob": 0.5,
    "blur_radius": [0.3, 0.8],
    "ink_peak": [0.7, 1.0],
    "vertical_jitter_px": 2,
}


def make_digit(img28, rng):
    """Biến đổi một ảnh MNIST 28x28 rồi cắt sát nét chữ. Trả về mảng uint8 hoặc None."""
    im = Image.fromarray(img28)

    # Độ dày nét
    r = rng.random()
    if r < PARAMS["thicker_stroke_prob"]:
        im = im.filter(ImageFilter.MaxFilter(3))
    elif r < PARAMS["thicker_stroke_prob"] + PARAMS["thinner_stroke_prob"]:
        lo = int(rng.integers(60, 121))
        lut = [0 if p <= lo else int((p - lo) * 255 / (255 - lo)) for p in range(256)]
        im = im.point(lut)

    # Đặt vào khung 56x56 để xoay và nghiêng không bị cắt mất nét
    canvas = Image.new("L", (56, 56), 0)
    canvas.paste(im, (14, 14))
    canvas = canvas.rotate(
        float(rng.uniform(-PARAMS["rotation_deg"], PARAMS["rotation_deg"])),
        resample=Image.Resampling.BICUBIC,
        fillcolor=0,
    )
    s = float(rng.uniform(-PARAMS["shear"], PARAMS["shear"]))
    canvas = canvas.transform(
        (56, 56),
        Image.Transform.AFFINE,
        (1, s, -s * 28, 0, 1, 0),
        resample=Image.Resampling.BICUBIC,
        fillcolor=0,
    )

    a = np.asarray(canvas)
    ys, xs = np.nonzero(a > 40)
    if len(xs) == 0:
        return None
    a = a[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]

    h = int(rng.integers(PARAMS["digit_height_px"][0], PARAMS["digit_height_px"][1] + 1))
    w = max(3, int(round(a.shape[1] * h / a.shape[0])))
    piece = Image.fromarray(a).resize((w, h), Image.Resampling.LANCZOS)
    return np.asarray(piece)


def make_image(digits, pools, images, rng):
    pieces = []
    for d in digits:
        while True:
            p = make_digit(images[int(rng.choice(pools[int(d)]))], rng)
            if p is not None and int(p.sum()) > 255 * 15:  # đủ nét để nhìn thấy
                pieces.append(p)
                break

    gaps = []
    for j in range(len(pieces) - 1):
        # Không cho chồng quá 1/3 bề rộng chữ số hẹp hơn, để chữ số không bị che mất
        lo = -min(3, pieces[j].shape[1] // 3, pieces[j + 1].shape[1] // 3)
        gaps.append(int(rng.integers(lo, PARAMS["gap_px"][1] + 1)))

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


def create_split(images, labels, name, count, seed, out):
    rng = np.random.default_rng(seed)
    pools = [np.flatnonzero(labels == k) for k in range(10)]
    (out / name).mkdir()
    n_scaled = 0
    with (out / f"{name}_labels.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label", "length"])
        for i in range(count):
            length = int(rng.integers(MIN_LENGTH, MAX_LENGTH + 1))
            digits = rng.integers(0, 10, size=length)
            label = "".join(str(int(d)) for d in digits)
            img, scaled = make_image(digits, pools, images, rng)
            n_scaled += scaled
            rel = f"{name}/{i:06d}.png"
            img.save(out / rel)
            writer.writerow([rel, label, length])
            if (i + 1) % 5000 == 0:
                print(f"{name}: {i + 1}/{count}")
    return n_scaled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "mnist_sequences_v2"))
    ap.add_argument("--train", type=int, default=20000)
    ap.add_argument("--validation", type=int, default=2000)
    ap.add_argument("--test", type=int, default=2000)
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"Thu muc da ton tai: {out}. Hay doi --out neu muon tao bo moi.")

    with np.load(RAW, allow_pickle=False) as d:
        x_train, y_train, x_test, y_test = d["x_train"], d["y_train"], d["x_test"], d["y_test"]
    with np.load(SPLIT_FILE, allow_pickle=False) as s:
        train_ids, val_ids = s["train_indices"], s["validation_indices"]
    assert len(np.intersect1d(train_ids, val_ids)) == 0

    out.mkdir(parents=True)
    np.savez(out / "source_split_indices.npz", train_indices=train_ids,
             validation_indices=val_ids, test_indices=np.arange(len(x_test)))

    counts = {"train": args.train, "validation": args.validation, "test": args.test}
    sources = {
        "train": (x_train[train_ids], y_train[train_ids]),
        "validation": (x_train[val_ids], y_train[val_ids]),
        "test": (x_test, y_test),
    }
    scaled = {}
    for name in ["train", "validation", "test"]:
        imgs, labs = sources[name]
        scaled[name] = create_split(imgs, labs, name, counts[name], SEEDS[name], out)

    config = {"version": "v2", "counts": counts, "seeds": SEEDS, "params": PARAMS,
              "images_shrunk_to_fit_256px": scaled,
              "source_split": "giong v1: data/mnist_sequences_v1/source_split_indices.npz"}
    (out / "generation_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Hoan thanh. Du lieu luu tai:", out, "| anh phai thu nho:", scaled)


if __name__ == "__main__":
    main()
