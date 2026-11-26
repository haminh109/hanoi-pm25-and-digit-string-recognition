"""Tải bộ dữ liệu mở rộng 08/2022–10/2026 bằng chính script download_openmeteo.py của nhóm.

Dùng cho mục 14 (nâng cấp N6) của notebook 02_task1_model.ipynb. Script này chỉ đổi khoảng thời gian
và tên file. Mọi tham số tải (hai ô lưới, mô hình ERA5, độ cao 19 m) lấy từ download_openmeteo.py,
nên bản mở rộng và bản v1 cùng một nguồn. Sau khi tải, script của nhóm tự so khớp các cột chung
với hanoi_hourly_2023_2025_v1.csv (đều phải "khop").

Chạy (cần Internet, không cần API key):
    python download_2022_2026.py
Kết quả:
    raw_2022_2026/weather_<năm>.csv, raw_2022_2026/air_<năm>.csv   số liệu gốc đúng như API trả về
    hanoi_hourly_2022_2026_v2.csv                                    file gộp, 37.272 giờ, 19 biến
    columns_units_2022_2026_v2.json                                  đơn vị của từng cột
Cột split trong file gộp do script của nhóm ghi theo mốc cũ; notebook tự chia lại tập.
"""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("download_openmeteo", HERE / "download_openmeteo.py")
dl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dl)
dl.START, dl.END = "2022-08-01", "2026-10-31"

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--raw-dir", str(HERE / "raw_2022_2026"),
                "--out", str(HERE / "hanoi_hourly_2022_2026_v2.csv"),
                "--compare", str(HERE / "hanoi_hourly_2023_2025_v1.csv")]
    dl.main()
    units = HERE / "hanoi_hourly_2022_2026_v2.json"
    if units.exists():
        units.replace(HERE / "columns_units_2022_2026_v2.json")
