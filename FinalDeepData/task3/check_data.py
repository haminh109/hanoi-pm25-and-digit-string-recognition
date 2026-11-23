from pathlib import Path
import pandas as pd
from PIL import Image

root = (
    Path(__file__).resolve().parent
    / "data"
    / "mnist_sequences_v1"
)

# Quan trọng: đọc label dưới dạng chuỗi để giữ số 0 đầu
labels = pd.read_csv(
    root / "train_labels.csv",
    dtype={"label": str}
)

sample = labels.iloc[0]

print("Nhan:", sample["label"])
print("Do dai:", sample["length"])

Image.open(root / sample["image_path"]).show()