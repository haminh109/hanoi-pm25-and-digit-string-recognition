from pathlib import Path
import csv
import urllib.request

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent

RAW = ROOT / "data" / "raw" / "mnist.npz"
OUT = ROOT / "data" / "mnist_sequences_v1"

NUM_IMAGES = {
    "train": 20000,
    "validation": 2000,
    "test": 2000
}

SEED = 42
MIN_LENGTH = 3
MAX_LENGTH = 8
HEIGHT = 32


def create_split(images, labels, name, count, seed):
    rng = np.random.default_rng(seed)

    # Danh sách chỉ số ảnh cho từng chữ số 0–9
    pools = [
        np.flatnonzero(labels == digit)
        for digit in range(10)
    ]

    image_dir = OUT / name
    image_dir.mkdir()

    csv_path = OUT / f"{name}_labels.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label", "length"])

        for i in range(count):
            # Chọn độ dài và các chữ số ngẫu nhiên
            length = int(
                rng.integers(MIN_LENGTH, MAX_LENGTH + 1)
            )

            digits = rng.integers(0, 10, size=length)

            # Giữ nhãn dưới dạng chuỗi, không chuyển sang số nguyên
            label = "".join(str(int(d)) for d in digits)

            pieces = []

            for digit in digits:
                # Lấy một ảnh gốc có nhãn tương ứng
                idx = int(rng.choice(pools[int(digit)]))
                im = Image.fromarray(images[idx])

                # Xoay nhẹ, mở rộng khung để tránh cắt nét
                im = im.rotate(
                    float(rng.uniform(-6, 6)),
                    resample=Image.Resampling.BICUBIC,
                    expand=True,
                    fillcolor=0
                )

                # Thay đổi kích thước nhẹ, giữ tỷ lệ ảnh
                size = int(rng.integers(24, 29))
                im.thumbnail(
                    (size, size),
                    Image.Resampling.LANCZOS
                )

                pieces.append(im)

            # Khoảng cách giữa các ảnh chữ số
            gaps = rng.integers(0, 6, size=length - 1)

            width = (
                8
                + sum(im.width for im in pieces)
                + int(gaps.sum())
            )

            canvas = Image.new(
                "L",
                (width, HEIGHT),
                color=0
            )

            x = 4

            for j, im in enumerate(pieces):
                # Dịch vị trí dọc một chút
                y = int(
                    rng.integers(
                        2,
                        HEIGHT - im.height - 1
                    )
                )

                canvas.paste(im, (x, y))
                x += im.width

                if j < length - 1:
                    x += int(gaps[j])

            relative_path = f"{name}/{i:06d}.png"

            canvas.save(OUT / relative_path)
            writer.writerow([relative_path, label, length])

            if (i + 1) % 1000 == 0:
                print(f"{name}: {i + 1}/{count}")


def main():
    RAW.parent.mkdir(parents=True, exist_ok=True)

    # Chỉ tải MNIST nếu chưa có file
    if not RAW.exists():
        print("Dang tai MNIST...")

        url = (
            "https://storage.googleapis.com/"
            "tensorflow/tf-keras-datasets/mnist.npz"
        )

        temporary = RAW.with_suffix(".part")

        urllib.request.urlretrieve(url, temporary)
        temporary.replace(RAW)

    with np.load(RAW, allow_pickle=False) as data:
        x_train = data["x_train"]
        y_train = data["y_train"]

        x_test = data["x_test"]
        y_test = data["y_test"]

    # Chia ảnh gốc TRƯỚC khi ghép thành chuỗi
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(x_train))

    train_ids = indices[:48000]
    val_ids = indices[48000:]

    # Tránh ghi đè bộ dữ liệu đã tạo
    if OUT.exists():
        raise FileExistsError(
            f"Thu muc da ton tai: {OUT}. "
            "Hay doi ten OUT neu muon tao bo moi."
        )

    OUT.mkdir(parents=True)

    # Lưu lại cách chia ảnh gốc để đối chiếu
    np.savez(
        OUT / "source_split_indices.npz",
        train_indices=train_ids,
        validation_indices=val_ids,
        test_indices=np.arange(len(x_test))
    )

    create_split(
        x_train[train_ids],
        y_train[train_ids],
        "train",
        NUM_IMAGES["train"],
        SEED + 1
    )

    create_split(
        x_train[val_ids],
        y_train[val_ids],
        "validation",
        NUM_IMAGES["validation"],
        SEED + 2
    )

    create_split(
        x_test,
        y_test,
        "test",
        NUM_IMAGES["test"],
        SEED + 3
    )

    print("Hoan thanh. Du lieu luu tai:", OUT)


if __name__ == "__main__":
    main()