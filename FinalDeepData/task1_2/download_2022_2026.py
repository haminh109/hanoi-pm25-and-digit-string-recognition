"""Tải bộ dữ liệu mở rộng 08/2022–10/2026 bằng chính script download_openmeteo.py của nhóm.

Dùng cho mục 14 (nâng cấp N6) của notebook 02_task1_model.ipynb. Script không sửa file của nhóm,
chỉ đổi khoảng thời gian và thêm hai tham số cho phần khí tượng để số liệu khớp với bộ v1:
  - models=era5: nếu thiếu, Open-Meteo trả về mô hình mặc định ở điểm lưới khác (20,98 B; 105,79 Đ),
    nhiệt độ, gió, mưa lệch hẳn so với bộ v1;
  - elevation=19: độ cao của bộ v1; nếu thiếu, API dùng 13 m nên nhiệt độ và áp suất lệch nhẹ.
Sau khi tải, script của nhóm tự so khớp các cột chung với hanoi_hourly_2023_2025_v1.csv (đều phải "khop").

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
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("download_openmeteo", HERE / "download_openmeteo.py")
dl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dl)
dl.START, dl.END = "2022-08-01", "2026-10-31"


def fetch(source, year):
    params = {"latitude": source["latitude"], "longitude": source["longitude"],
              "start_date": max(dl.START, f"{year}-01-01"), "end_date": min(dl.END, f"{year}-12-31"),
              "hourly": ",".join(source["hourly"]), "timezone": dl.TIMEZONE, "format": "csv"}
    if source is dl.WEATHER:
        params["models"] = "era5"
        params["elevation"] = 19
    url = source["url"] + "?" + urllib.parse.urlencode(params, safe=",")
    req = urllib.request.Request(url, headers={"User-Agent": "deep-learning-course-project"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                return resp.read().decode("utf-8-sig")
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            print(f"Loi ({e}), thu lai...")
            time.sleep(10 * (attempt + 1))
    sys.exit("Khong tai duoc sau 4 lan thu. Kiem tra ket noi Internet.")


dl.fetch = fetch
if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--raw-dir", str(HERE / "raw_2022_2026"),
                "--out", str(HERE / "hanoi_hourly_2022_2026_v2.csv"),
                "--compare", str(HERE / "hanoi_hourly_2023_2025_v1.csv")]
    dl.main()
    units = HERE / "hanoi_hourly_2022_2026_v2.json"
    if units.exists():
        units.replace(HERE / "columns_units_2022_2026_v2.json")
