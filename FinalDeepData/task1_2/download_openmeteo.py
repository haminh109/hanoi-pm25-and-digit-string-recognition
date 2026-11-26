"""Tải số liệu Hà Nội theo giờ từ Open-Meteo cho Task 1-2 và gộp thành một file.

Bản v2 thêm các biến còn thiếu so với bản v1 của nhóm, quan trọng nhất là hướng gió
(wind_direction_10m). Hai điểm lưới giữ đúng như v1 để số liệu khớp với bản cũ:
  - khí tượng:           21.00 N, 105.75 E  (Historical Weather API, mô hình ERA5, độ cao 19 m)
  - chất lượng không khí: 21.00 N, 105.90 E  (Air Quality API, dữ liệu CAMS)
Giờ trả về theo GMT+7 (Asia/Bangkok, cùng múi giờ Việt Nam), từ 2023-01-01 00:00 đến 2025-12-31 23:00.

Chạy trên Google Colab hoặc máy có Internet (không cần API key, chỉ cần pandas):
    python download_openmeteo.py
Kết quả:
    raw/weather_<năm>.csv, raw/air_<năm>.csv  số liệu gốc đúng như API trả về
    hanoi_hourly_2023_2025_v2.csv              file gộp, tên cột gọn, có cột split
    columns_units_v2.json                      đơn vị của từng cột

Dựng lại file gộp từ một thư mục số liệu gốc có sẵn (không cần mạng):
    python download_openmeteo.py --offline --raw-dir raw
"""
import argparse
import io
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
START, END = "2023-01-01", "2025-12-31"
TIMEZONE = "Asia/Bangkok"

WEATHER = {
    "url": "https://archive-api.open-meteo.com/v1/archive",
    "latitude": 21.0,
    "longitude": 105.75,
    # Cố định mô hình và độ cao như v1 (sửa ngày 26/11/2026):
    #   - thiếu models=era5, API dùng mô hình mặc định ở ô lưới khác (20,98 N; 105,79 E):
    #     nhiệt độ lệch tới 8 °C, gió lệch tới 26 km/h so với v1;
    #   - thiếu elevation=19, API tự lấy độ cao theo toạ độ yêu cầu (13 m ở toạ độ này):
    #     áp suất lệch khoảng 0,7 hPa, nhiệt độ lệch 0,1 °C ở một số giờ.
    "extra_params": {"models": "era5", "elevation": 19},
    "hourly": [
        "temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation",
        "surface_pressure", "cloud_cover", "wind_speed_10m", "wind_speed_100m",
        "wind_direction_10m", "wind_gusts_10m", "shortwave_radiation",
    ],
}
AIR = {
    "url": "https://air-quality-api.open-meteo.com/v1/air-quality",
    "latitude": 21.0,
    "longitude": 105.9,
    "hourly": [
        "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide",
        "ozone", "dust", "aerosol_optical_depth",
    ],
}
AIR_NAMES = set(AIR["hourly"])

# Chia tập theo thời gian (giờ địa phương)
SPLITS = [("train", "2025-01-01"), ("validation", "2025-07-01"), ("test", None)]


def split_of(times):
    out = pd.Series("test", index=times.index)
    out[times < pd.Timestamp(SPLITS[1][1])] = "validation"
    out[times < pd.Timestamp(SPLITS[0][1])] = "train"
    return out


def fetch(source, year):
    params = {
        "latitude": source["latitude"],
        "longitude": source["longitude"],
        "start_date": max(START, f"{year}-01-01"),
        "end_date": min(END, f"{year}-12-31"),
        "hourly": ",".join(source["hourly"]),
        "timezone": TIMEZONE,
        "format": "csv",
    }
    params.update(source.get("extra_params", {}))
    url = source["url"] + "?" + urllib.parse.urlencode(params, safe=",")
    req = urllib.request.Request(url, headers={"User-Agent": "deep-learning-course-project"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                return resp.read().decode("utf-8-sig")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            if e.code == 400:
                sys.exit(f"API tu choi yeu cau ({source['url']}): {body}")
            wait = 10 * (attempt + 1)
            print(f"HTTP {e.code}, thu lai sau {wait}s...")
        except urllib.error.URLError as e:
            wait = 10 * (attempt + 1)
            print(f"Loi mang ({e.reason}), thu lai sau {wait}s...")
        time.sleep(wait)
    sys.exit("Khong tai duoc sau 4 lan thu. Kiem tra ket noi Internet.")


def parse_openmeteo_csv(text):
    """Tách phần đầu (toạ độ, múi giờ) và bảng số liệu của một file CSV Open-Meteo."""
    lines = text.splitlines()
    meta = dict(zip(lines[0].split(","), lines[1].split(","))) if lines and lines[0].startswith("latitude") else {}
    start = next(i for i, ln in enumerate(lines) if ln.startswith("time,"))
    df = pd.read_csv(io.StringIO("\n".join(lines[start:])))
    units = {}
    rename = {}
    for col in df.columns:
        m = re.match(r"^(.*?)\s*\((.*)\)\s*$", col)
        if m:
            rename[col] = m.group(1)
            units[m.group(1)] = m.group(2)
    df = df.rename(columns=rename)
    df["time"] = pd.to_datetime(df["time"])
    return meta, df, units


def combine(frames):
    df = pd.concat(frames, ignore_index=True).sort_values("time")
    dup = df["time"].duplicated().sum()
    if dup:
        print(f"Bo {dup} dong trung thoi diem")
        df = df.drop_duplicates("time")
    return df.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="khong tai, chi dung file trong --raw-dir")
    ap.add_argument("--raw-dir", default=str(ROOT / "raw"))
    ap.add_argument("--out", default=str(ROOT / "hanoi_hourly_2023_2025_v2.csv"))
    ap.add_argument("--compare", default=str(ROOT / "hanoi_hourly_2023_2025_v1.csv"),
                    help="file cu de so khop cac cot trung nhau (bo qua neu khong co)")
    args = ap.parse_args()
    raw_dir = Path(args.raw_dir)

    if not args.offline:
        raw_dir.mkdir(parents=True, exist_ok=True)
        for name, source in [("weather", WEATHER), ("air", AIR)]:
            for year in range(int(START[:4]), int(END[:4]) + 1):
                print(f"Dang tai {name} {year}...")
                text = fetch(source, year)
                (raw_dir / f"{name}_{year}.csv").write_text(text, encoding="utf-8")

    weather, air, units, metas = [], [], {}, {}
    for path in sorted(raw_dir.glob("*.csv")):
        meta, df, u = parse_openmeteo_csv(path.read_text(encoding="utf-8-sig"))
        kind = "air" if AIR_NAMES & set(df.columns) else "weather"
        (air if kind == "air" else weather).append(df)
        units.update(u)
        metas[kind] = meta
    if not weather or not air:
        sys.exit(f"Thieu file khi tuong hoac chat luong khong khi trong {raw_dir}")

    merged = pd.merge(combine(weather), combine(air), on="time", how="outer", validate="one_to_one")
    merged = merged.sort_values("time").reset_index(drop=True)

    # Kiểm tra
    full = pd.date_range(merged["time"].min(), merged["time"].max(), freq="h")
    missing = len(full.difference(merged["time"]))
    print(f"So dong: {len(merged)} | tu {merged['time'].min()} den {merged['time'].max()} | "
          f"gio bi thieu: {missing}")
    for kind, meta in metas.items():
        print(f"Diem luoi {kind}: lat {meta.get('latitude')}, lon {meta.get('longitude')}, "
              f"mui gio {meta.get('timezone')} {meta.get('timezone_abbreviation')}")
    nan = merged.isna().sum()
    print("O trong theo cot:", {c: int(v) for c, v in nan.items() if v} or "khong co")

    merged["split"] = split_of(merged["time"])
    print("So dong moi tap:", merged["split"].value_counts().reindex(["train", "validation", "test"]).to_dict())

    compare = Path(args.compare)
    if compare.exists() and compare.resolve() != Path(args.out).resolve():
        old = pd.read_csv(compare, parse_dates=["time"])
        both = merged.merge(old, on="time", suffixes=("", "_old"))
        for col in [c for c in old.columns if c not in ("time", "split") and c in merged.columns]:
            diff = (both[col] - both[col + "_old"]).abs().max()
            status = "khop" if diff <= 0.051 else "KHAC - kiem tra lai diem luoi hoac nguon du lieu"
            print(f"So voi {compare.name}: {col} lech toi da {diff:.3f} -> {status}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False, date_format="%Y-%m-%dT%H:%M")
    units_path = out.with_name(out.stem.replace("hanoi_hourly_2023_2025", "columns_units") + ".json")
    units_path.write_text(json.dumps(units, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Da luu:", out, "va", units_path.name)


if __name__ == "__main__":
    main()
