# FinalDeepData: bộ dữ liệu hoàn chỉnh cho đề tài 4

Dự báo PM2.5 Hà Nội (Task 1–2) và đọc chuỗi chữ số viết tay (Task 3). Đóng gói ngày 22/11/2026.

Gói gồm bộ dữ liệu gốc của nhóm (giữ nguyên), bản đã làm sạch dùng ngay cho Task 1–2, bộ ảnh Task 3 v2 khó hơn, script tải số liệu đủ biến, và hai file code dùng chung (đọc dữ liệu, tính chỉ số).

## Cấu trúc thư mục

```
FinalDeepData/
├── README.md
├── common/
│   ├── data_utils.py                  đọc dữ liệu, cắt cửa sổ trượt, đọc ảnh, mã hoá nhãn
│   └── metrics.py                     R², NSE, MAE, RMSE theo slide chương 2.1 + chỉ số Task 3
├── task1_2/
│   ├── hanoi_hourly_2023_2025_v1.csv  DÙNG NGAY cho Task 1–2
│   ├── columns_units_v1.json          đơn vị từng cột
│   ├── download_openmeteo.py          tải bản v2 đủ biến (có hướng gió), xem mục "Việc còn lại"
│   └── original/                      nguyên trạng của nhóm
│       ├── gop_csv.py
│       ├── hanoi_weather_air_quality.csv
│       └── raw data/                  2 file Open-Meteo gốc
└── task3/
    ├── generate_data.py, check_data.py   nguyên trạng của nhóm (sinh bản v1)
    ├── generate_data_v2.py               sinh bản v2
    └── data/
        ├── raw/mnist.npz
        ├── mnist_sequences_v1/           20.000 / 2.000 / 2.000 ảnh, bản của nhóm, giữ nguyên
        └── mnist_sequences_v2/           20.000 / 2.000 / 2.000 ảnh, bản mới khó hơn
```

## Việc còn lại: tải bản v2 cho Task 1–2

Máy đóng gói không kết nối được tới Open-Meteo, nên gói chưa có file `hanoi_hourly_2023_2025_v2.csv`. Tải file zip lên Google Colab rồi chạy (không cần API key, mất khoảng một phút):

```
!unzip -q FinalDeepData_hoan_chinh.zip
!python FinalDeepData/task1_2/download_openmeteo.py
```

- Script tải 6 lượt (3 năm × 2 nguồn), lưu số liệu gốc vào `task1_2/raw/`, rồi tạo `hanoi_hourly_2023_2025_v2.csv` và `columns_units_v2.json`.
- Nó tự so các cột trùng với file v1 và in "khop" nếu lệch tối đa ≤ 0,05. Nếu in "KHAC", báo lại cả nhóm trước khi dùng.
- Biến thêm so với v1:
  - khí tượng: `dew_point_2m`, `cloud_cover`, `wind_direction_10m` (hướng gió), `wind_gusts_10m`, `shortwave_radiation`;
  - không khí: `carbon_monoxide`, `sulphur_dioxide`, `ozone`, `dust`, `aerosol_optical_depth`.
- Tên biến lấy theo trang tài liệu Open-Meteo, nhưng script chưa được chạy với API thật. Nếu API báo lỗi một tên biến, xoá biến đó khỏi danh sách `WEATHER` hoặc `AIR` ở đầu script rồi chạy lại.
- Phần gộp, kiểm tra và chia tập của script đã được thử trên số liệu gốc của nhóm: chính file v1 trong gói được tạo bằng lệnh `--offline` của script này.
- Có file v2 rồi thì chỉ cần đổi tên file trong đoạn code đọc dữ liệu bên dưới. Riêng hướng gió là góc 0–360°, nên đổi sang sin/cos trước khi đưa vào mô hình (359° và 1° thực ra rất gần nhau):

```python
import numpy as np
df = load_hourly("FinalDeepData/task1_2/hanoi_hourly_2023_2025_v2.csv")   # load_hourly: xem mục Task 1–2
rad = np.deg2rad(df.pop("wind_direction_10m"))
df["wind_dir_sin"], df["wind_dir_cos"] = np.sin(rad), np.cos(rad)
```

## Task 1–2: chuỗi thời gian Hà Nội

`task1_2/hanoi_hourly_2023_2025_v1.csv` có đúng số liệu của file `hanoi_weather_air_quality.csv` của nhóm. Chỉ khác hai điểm: tên cột bỏ phần đơn vị, và có thêm cột `split`.

- 26.304 dòng theo giờ, từ 01/01/2023 00:00 đến 31/12/2025 23:00, giờ GMT+7. Không thiếu giờ, không trùng, không ô trống.
- `pred_column` = `pm2_5`, bước thời gian 1 giờ.

| Cột | Đơn vị | Nguồn |
| --- | --- | --- |
| `time` | giờ GMT+7 | |
| `temperature_2m` | °C | Historical Weather API, điểm lưới 21,00°N 105,75°E |
| `relative_humidity_2m` | % | như trên |
| `surface_pressure` | hPa | như trên |
| `wind_speed_10m`, `wind_speed_100m` | km/h | như trên |
| `precipitation` | mm, tổng của giờ trước đó | như trên |
| `pm10`, `pm2_5`, `nitrogen_dioxide` | µg/m³ | Air Quality API (mô hình CAMS), điểm lưới 21,00°N 105,90°E |
| `split` | `train` / `validation` / `test` | chia theo thời gian |

**Chia tập:** train 2023–2024 (17.544 dòng), validation 01–06/2025 (4.344 dòng), test 07–12/2025 (4.416 dòng, có mùa đông cuối năm).

**Mốc cần thắng (persistence, lấy PM2.5 hiện tại làm dự báo) trên tập test:**

| | R² | NSE | MAE (µg/m³) | RMSE (µg/m³) |
| --- | --- | --- | --- | --- |
| Task 1, dự báo 2 giờ tới | 0,779 | 0,765 | 10,0 | 15,6 |
| Task 2, gộp chung 5 bước | 0,603 | 0,553 | 13,6 | 21,5 |

Với Task 2, NSE theo từng bước 1 → 5 giờ tới lần lượt là 0,925 / 0,765 / 0,568 / 0,357 / 0,148.

**Lưu ý khi viết báo cáo:**
- Hai điểm lưới cách nhau khoảng 15,6 km: mỗi API trả về ô lưới gần nhất của mô hình riêng.
- PM2.5 là số của mô hình khí quyển CAMS, không phải số đo tại trạm. Trung bình theo tháng chỉ dao động 39–55 µg/m³, mùa đông không cao hẳn như số đo trạm thường thấy.
- Số theo giờ có biến động thật: dựng lại PM2.5 từ các mốc 3 giờ (nội suy tuyến tính hoặc spline) sai 3,6–4,3 µg/m³, nên không phải số nội suy. Dùng bước 1 giờ là ổn.

**Đọc dữ liệu và cắt cửa sổ:**

```python
import sys; sys.path.append("FinalDeepData/common")
from data_utils import load_hourly, make_windows

df = load_hourly("FinalDeepData/task1_2/hanoi_hourly_2023_2025_v1.csv")
features = [c for c in df.columns if c != "split"]

task1 = make_windows(df, features, "pm2_5", lookback=48, horizons=(2,))          # 10:00 -> 12:00
task2 = make_windows(df, features, "pm2_5", lookback=48, horizons=(1, 2, 3, 4, 5))
X_train = task1["X"][task1["split"] == "train"]   # (17495, 48, 9)
y_train = task1["y"][task1["split"] == "train"]   # (17495, 1)
```

Mẫu có nhãn nằm vắt qua hai tập (ngay chỗ giáp ranh) bị bỏ. Đầu vào được phép lấy số của tập trước, vì đó là quá khứ. Chuẩn hoá phải fit trên tập train rồi mới áp cho validation và test.

## Task 3: ảnh chuỗi chữ số viết tay

| | v1 (của nhóm, giữ nguyên) | v2 (mới) |
| --- | --- | --- |
| Số ảnh train / validation / test | 20.000 / 2.000 / 2.000 | 20.000 / 2.000 / 2.000 |
| Kích thước | cao 32, rộng 80–253 pixel | cao 32, rộng 35–254 pixel |
| Khoảng cách chữ số | luôn tách rời (viền đen sẵn của MNIST + khe 0–5 pixel) | cắt sát, khe −3 đến +6 pixel: có lúc chạm hoặc chồng nhẹ |
| Biến dạng | xoay ±6°, cỡ 24–28 pixel | xoay ±10°, nghiêng ±0,35, đổi độ dày nét, cao 18–28 pixel, mờ nhẹ, đổi độ sáng |
| Bề rộng ảnh theo độ dài | gần như tách hẳn (3 chữ số: 80–102 pixel; 8 chữ số: 213–253 pixel) | chồng lấn nhiều (3 chữ số: 35–117; 8 chữ số: 107–254) |

Giống nhau ở cả hai bản:
- Chữ trắng trên nền đen, độ dài chuỗi 3–8, 10 chữ số xuất hiện đều nhau.
- Khoảng 10% nhãn bắt đầu bằng số 0.
- Chia nguồn MNIST giống hệt nhau: ảnh train chỉ dùng chữ số từ 48.000 ảnh MNIST-train, validation dùng 12.000 ảnh còn lại, test dùng MNIST-test. Không chữ số nào xuất hiện ở hai tập.

**Gợi ý dùng:** train trên v1 trước cho mô hình chạy được, rồi train trên v2 (hoặc gộp v1 + v2). Nếu nhóm định thử trên ảnh chữ viết tay thật thì nên dùng mô hình đã train trên v2, vì v2 giống chữ viết thật hơn.

**Ba điều bắt buộc khi đọc:**
1. Nhãn phải đọc dạng chuỗi: `pd.read_csv(..., dtype={"label": str})`. Đọc thiếu tham số này thì 2.038 nhãn train của v1 mất số 0 ở đầu.
2. Đệm ảnh nền đen bên phải cho đủ 32 × 256, không kéo giãn: kéo giãn làm chữ số béo gầy khác nhau tuỳ độ dài chuỗi.
3. Bộ token của decoder: 0–9 là chữ số, 10 = `<pad>`, 11 = `<bos>`, 12 = `<eos>`.

`load_task3` và `encode_labels` đã làm sẵn cả ba việc:

```python
from data_utils import load_task3, encode_labels, decode_tokens, augment
import numpy as np

X, labels = load_task3("FinalDeepData/task3/data/mnist_sequences_v2", "train")  # (20000, 32, 256) uint8
dec_in, dec_out = encode_labels(labels, max_len=8)   # teacher forcing: <bos> d1..dn / d1..dn <eos>
X_aug = augment(X[:64], np.random.default_rng(0))   # nhiễu + độ sáng, dùng khi huấn luyện
decode_tokens([2, 0, 3, 6, 9, 12])                   # -> "20369"
```

Sinh lại hoặc sinh bộ lớn hơn (chạy trong thư mục `task3`, khoảng 1 phút cho mỗi 24.000 ảnh): `python generate_data_v2.py --out data/mnist_sequences_v2_big --train 60000`. Seed và tham số nằm trong `generation_config.json`.

## Tính chỉ số

```python
import metrics
metrics.regression_report(y_true, y_pred)   # Task 1: {"R2", "NSE", "MAE", "RMSE"}
metrics.seq2seq_report(Y_true, Y_pred)      # Task 2, mảng (N, 5): theo từng bước + gộp chung
metrics.digit_level_report(true_strs, pred_strs)        # Task 3, cách B
metrics.numeric_report_by_length(true_strs, pred_strs)  # Task 3, cách A
metrics.sequence_accuracy(true_strs, pred_strs), metrics.cer(true_strs, pred_strs)
```

Không dùng `sklearn.metrics.r2_score` cho cột R²: hàm đó tính 1 − SSE/SST, tức là NSE. R² trong slide là bình phương hệ số tương quan, và `metrics.r2` tính đúng công thức đó.

## Đã kiểm tra những gì

- **Task 1–2:**
  - File gộp của nhóm khớp từng giá trị với hai file gốc.
  - Chuỗi liền mạch theo giờ, không trùng, không ô trống.
  - Số theo giờ không phải số nội suy (xem lưu ý ở trên).
- **Task 3 v1:**
  - Dựng lại 300 ảnh đầu của mỗi tập bằng script của nhóm với cùng seed: giống hệt cả nhãn lẫn từng pixel.
  - Nguồn train và validation không trùng nhau.
  - Xem trực tiếp ảnh mẫu, nhãn khớp hình.
- **Task 3 v2:** đủ 24.000 ảnh, cao 32, rộng không quá 256, không ảnh trống, nhãn khớp cột độ dài, xem trực tiếp ảnh mẫu.
- **`data_utils` và `metrics`:**
  - Căn chỉnh cửa sổ đúng trên 200 mẫu ngẫu nhiên; mã hoá rồi giải mã lại nhãn đúng; phần đệm ảnh toàn số 0.
  - `metrics.nse` cho cùng kết quả với `sklearn.metrics.r2_score`, còn `metrics.r2` thì khác, đúng như mong đợi.

## Ghi nguồn

- Open-Meteo, Historical Weather API và Air Quality API. Trang tài liệu yêu cầu ghi nguồn Open-Meteo và nhà cung cấp dữ liệu CAMS.
- MNIST: LeCun, Y., Bottou, L., Bengio, Y., Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11).
- Bộ v2 sinh bằng `task3/generate_data_v2.py` với seed 2027 (train), 2028 (validation), 2029 (test).
