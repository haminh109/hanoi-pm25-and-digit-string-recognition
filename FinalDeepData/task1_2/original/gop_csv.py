from pathlib import Path
import pandas as pd

# Thư mục chứa file Python này
folder = Path(__file__).resolve().parent

# Đọc hai CSV gốc
# Bỏ 3 dòng đầu vì đây là metadata; dòng thứ 4 mới là tên cột
weather = pd.read_csv(
    folder / "raw data/open-meteo-21.00N105.75E19m.csv",
    skiprows=3,
    encoding="utf-8-sig"
)

air_quality = pd.read_csv(
    folder / "raw data/open-meteo-21.00N105.90E19m.csv",
    skiprows=3,
    encoding="utf-8-sig"
)

# Ghép các cột có cùng thời điểm
merged = pd.merge(
    weather,
    air_quality,
    on="time",
    how="outer",
    validate="one_to_one"
)

# Lưu thành file mới
output = folder / "hanoi_weather_air_quality.csv"

merged.to_csv(
    output,
    index=False,
    encoding="utf-8-sig"
)

print("Kích thước sau khi gộp:", merged.shape)
print("File đã lưu tại:", output)
print(merged.head())