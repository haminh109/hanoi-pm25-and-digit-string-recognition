# FinalDeepData: bộ dữ liệu hoàn chỉnh cho đề tài 4

Dự báo PM2.5 Hà Nội (Task 1–2) và đọc chuỗi chữ số viết tay (Task 3). Đóng gói ngày 22/11/2026; thêm bộ Task 3 v3 ngày 25/11/2026; thêm bộ Task 1–2 mở rộng ngày 26/11/2026.

Gói gồm bộ dữ liệu gốc của nhóm (giữ nguyên), bản đã làm sạch dùng ngay cho Task 1–2, bộ ảnh Task 3 v3 (mỗi chuỗi do một người viết, bộ chính), script tải số liệu đủ biến, và hai file code dùng chung (đọc dữ liệu, tính chỉ số).

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
│   ├── hanoi_hourly_2022_2026_v2.csv  bản mở rộng 08/2022–10/2026, 19 biến (có hướng gió)
│   ├── columns_units_2022_2026_v2.json
│   ├── download_2022_2026.py          tải lại bản mở rộng (gọi download_openmeteo.py, chỉ đổi khoảng thời gian)
│   ├── download_openmeteo.py          script tải của nhóm: ERA5, độ cao 19 m như v1 (sửa ngày 26/11/2026)
│   └── original/                      nguyên trạng của nhóm
│       ├── gop_csv.py
│       ├── hanoi_weather_air_quality.csv
│       └── raw data/                  2 file Open-Meteo gốc
└── task3/
    ├── generate_data.py, check_data.py   nguyên trạng của nhóm (sinh bản v1)
    ├── generate_data_v2.py               sinh bản v2
    ├── generate_data_v3.py               sinh bản v3: mỗi chuỗi do một người viết (BỘ CHÍNH)
    ├── check_writers.py                  đếm số người viết trong từng ảnh của v1, v2, v3
    └── data/
        ├── raw/                          mnist.npz + 2 file nhãn mở rộng của QMNIST (mã người viết)
        ├── mnist_sequences_v1/           20.000 / 2.000 / 2.000 ảnh, bản của nhóm, giữ nguyên
        ├── mnist_sequences_v2/           20.000 / 2.000 / 2.000 ảnh, bản khó hơn (chỉ để so sánh)
        └── mnist_sequences_v3/           20.000 / 2.000 / 2.000 ảnh, mỗi chuỗi do một người viết
```

## Bản mở rộng cho Task 1–2 (08/2022–10/2026)

`task1_2/hanoi_hourly_2022_2026_v2.csv` có 37.272 giờ, từ 01/08/2022 đến 31/10/2026, giờ Việt Nam (GMT+7), cùng hai điểm lưới như v1.
- **Số liệu ô nhiễm có từ 7 giờ ngày 4/8/2022.** 79 giờ đầu trống số liệu ô nhiễm, vì CAMS chưa có số; bỏ các giờ này trước khi dùng.
- **Biến thêm so với v1:**
  - khí tượng: `dew_point_2m`, `cloud_cover`, `wind_direction_10m` (hướng gió), `wind_gusts_10m`, `shortwave_radiation`;
  - không khí: `carbon_monoxide`, `sulphur_dioxide`, `ozone`, `dust`, `aerosol_optical_depth`.
- **Khớp với v1:** ở những giờ trùng nhau (2023–2025), 9 cột chung khớp hoàn toàn với `hanoi_hourly_2023_2025_v1.csv`.
- **Cột `split`** do script ghi theo mốc cũ: 2026 bị tính vào test. Notebook `02_task1_model.ipynb` tự chia lại tập, và tách riêng 01–10/2026 để kiểm tra thêm.

Tải lại (cần Internet, không cần API key):

```
python FinalDeepData/task1_2/download_2022_2026.py
```

**Cả nhóm dùng chung một bộ số liệu.** Ở những giờ trùng nhau, ba nguồn dưới đây giống nhau từng giá trị:
- `hanoi_hourly_2023_2025_v1.csv`;
- `hanoi_hourly_2022_2026_v2.csv`;
- số liệu do `download_openmeteo.py` tải về.

Nên đọc thẳng các file CSV này thay vì tự tải lại. Ai tự viết code tải riêng thì phải gửi đúng các tham số như `download_openmeteo.py`:
- khí tượng (Historical Weather API): ô lưới 21,00°N 105,75°E, `models=era5`, `elevation=19`;
- không khí (Air Quality API): ô lưới 21,00°N 105,90°E, dữ liệu CAMS Global;
- giờ GMT+7 (`timezone=Asia/Bangkok`). Nếu lấy giờ UTC thì cộng 7 giờ trước khi ghép với các file trên.

Tham số `elevation` quyết định cách API hiệu chỉnh áp suất và nhiệt độ theo độ cao. Nếu không ghi, API tự lấy độ cao theo toạ độ yêu cầu. Vì vậy đổi toạ độ là số liệu đổi theo, dù vẫn rơi vào cùng ô lưới.

Ví dụ, yêu cầu tại 21,03°N 105,85°E (trung tâm Hà Nội) vẫn rơi vào hai ô lưới trên, nhưng API dùng độ cao 24 m:
- áp suất thấp hơn khoảng 0,57 hPa ở mọi giờ;
- độ ẩm lệch 1% ở 0,4% số giờ.

Thêm `elevation=19` thì trùng hoàn toàn.

**`download_openmeteo.py` đã sửa ngày 26/11/2026.** Bản trước không gửi `models=era5` và `elevation=19` khi tải khí tượng:
- Thiếu `models=era5`, Open-Meteo trả về mô hình mặc định ở một ô lưới khác (20,98°N 105,79°E). Khi đó nhiệt độ lệch tới 8 °C và gió lệch tới 26 km/h so với v1.
- Thiếu `elevation=19`, API dùng độ cao 13 m. Khi đó áp suất lệch khoảng 0,7 hPa, và nhiệt độ lệch 0,1 °C ở một số giờ.

Hướng gió là góc 0–360°, nên đổi sang hai thành phần u, v (hoặc sin, cos) trước khi đưa vào mô hình, vì 359° và 1° thực ra rất gần nhau.

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

**Dùng bộ v3.** Ở v1 và v2, mỗi chữ số được lấy từ một ảnh MNIST bất kỳ, nên một chuỗi là chữ của nhiều người: `check_writers.py` chạy lại hai script cũ với đúng seed và cho thấy 0/24.000 ảnh của mỗi bản có mọi chữ số cùng một người viết (trung bình 5,5 người viết mỗi ảnh). Chuỗi số viết tay thật luôn do một người viết, nên theo góp ý của giảng viên nhóm sinh lại bộ v3.

| | v1 (của nhóm, giữ nguyên) | v2 | **v3 (bộ chính)** |
| --- | --- | --- | --- |
| Người viết trong một chuỗi | nhiều người (mỗi chữ số một ảnh MNIST bất kỳ) | nhiều người | **một người** (mã người viết theo QMNIST) |
| Số ảnh train / validation / test | 20.000 / 2.000 / 2.000 | 20.000 / 2.000 / 2.000 | 20.000 / 2.000 / 2.000 (thêm bộ 60.000 ảnh train) |
| Kích thước | cao 32, rộng 80–253 pixel | cao 32, rộng 35–254 pixel | cao 32, rộng 28–256 pixel |
| Khoảng cách chữ số | luôn tách rời (viền đen sẵn của MNIST + khe 0–5 pixel) | cắt sát, khe −3 đến +6 pixel: có lúc chạm hoặc chồng nhẹ | như v2, nhưng mỗi người viết có khoảng cách quen tay riêng |
| Biến dạng | xoay ±6°, cỡ 24–28 pixel | từng chữ số xoay ±10°, nghiêng ±0,35, đổi độ dày nét, cao 18–28 pixel; cả ảnh mờ nhẹ, đổi độ sáng | cùng các phép biến đổi và cùng giới hạn như v2, nhưng cỡ chữ, góc xoay, độ nghiêng, độ dày nét chọn **một lần cho cả chuỗi**, từng chữ số chỉ dao động nhẹ |
| Chia train / validation | theo ảnh MNIST: 48.000 / 12.000 ảnh MNIST-train | như v1 | **theo người viết**: 431 / 108 người viết của MNIST-train |
| Bề rộng ảnh theo độ dài | gần như tách hẳn (3 chữ số: 80–102 pixel; 8 chữ số: 213–253 pixel) | chồng lấn nhiều (3 chữ số: 35–117; 8 chữ số: 107–254) | chồng lấn nhiều hơn (3 chữ số: 28–147; 8 chữ số: 79–256) |

**Mã người viết.** MNIST không cho biết ai viết chữ số nào. Bộ QMNIST (Yadav & Bottou, 2019) đã truy từng ảnh MNIST về tờ mẫu gốc trong NIST Special Database 19 và công bố mã người viết trong file nhãn mở rộng (cột 2). Ảnh QMNIST-train khớp từng ảnh, đúng thứ tự với MNIST-train; 10.000 ảnh đầu của QMNIST-test khớp với MNIST-test. `generate_data_v3.py` tự tải hai file nhãn này vào `data/raw/` (khoảng 1 MB), kiểm tra mã MD5 và kiểm tra nhãn chữ số khớp MNIST trước khi dùng. Mã từ 2100 trở lên là học sinh trung học, nhỏ hơn 2100 là nhân viên Cục Điều tra Dân số Mỹ.

**Cách v3 sinh một ảnh:**
1. Chọn độ dài 3–8 và các chữ số ngẫu nhiên đều như v1, v2, nên nhãn không phụ thuộc người viết.
2. Chọn ngẫu nhiên một người viết trong số những người có đủ mẫu cho các chữ số đó. Chữ số lặp lại (ví dụ hai chữ 7) lấy hai mẫu khác nhau của cùng người.
3. Chọn kiểu chữ cho cả chuỗi, biến đổi từng chữ số, cắt sát nét rồi ghép thành hàng như v2. Chuỗi nào rộng quá 256 pixel thì thu nhỏ cả ảnh cho vừa (56/20.000 ảnh train).

**Chia tập của v3 theo người viết.** MNIST vốn đã tách người viết giữa MNIST-train và MNIST-test. v3 chia tiếp người viết của MNIST-train: 20% số người viết (chọn ngẫu nhiên trong từng nhóm tờ mẫu của NIST) chỉ dùng cho validation.

| Tập | Số ảnh | Người viết | Ảnh MNIST nguồn |
| --- | --- | --- | --- |
| Train | 20.000 | 431 người viết của MNIST-train | 47.790 ảnh |
| Validation | 2.000 | 108 người viết khác của MNIST-train | 12.210 ảnh |
| Test | 2.000 | 276 trong 297 người viết của MNIST-test | 10.000 ảnh |

Không người viết nào xuất hiện ở hai tập, nên validation và test đều là chữ của người mà mô hình chưa gặp khi huấn luyện. Mỗi học sinh trong MNIST-test chỉ viết khoảng 20 chữ số, còn mỗi nhân viên khoảng 100, nên chuỗi test dài thường lấy từ nhân viên hơn (tỷ lệ ảnh của học sinh: 81% ở chuỗi 3 chữ số, 53% ở chuỗi 8 chữ số).

File nhãn của v3 có thêm hai cột: `writer` (mã người viết của cả chuỗi) và `source` (chỉ số các ảnh MNIST đã ghép, từ trái sang phải; train và validation lấy từ MNIST-train, test lấy từ MNIST-test).

Giống nhau ở cả ba bản:
- Chữ trắng trên nền đen, độ dài chuỗi 3–8, 10 chữ số xuất hiện đều nhau.
- Khoảng 10% nhãn bắt đầu bằng số 0.
- Không ảnh MNIST nào được dùng ở hai tập.

**Gợi ý dùng:** huấn luyện và đánh giá trên v3. Bộ v2 chỉ dùng để so sánh với cách ghép cũ.

**Ba điều bắt buộc khi đọc:**
1. Nhãn phải đọc dạng chuỗi: `pd.read_csv(..., dtype={"label": str})`. Đọc thiếu tham số này thì 2.038 nhãn train của v1 mất số 0 ở đầu.
2. Đệm ảnh nền đen bên phải cho đủ 32 × 256, không kéo giãn: kéo giãn làm chữ số béo gầy khác nhau tuỳ độ dài chuỗi.
3. Bộ token của decoder: 0–9 là chữ số, 10 = `<pad>`, 11 = `<bos>`, 12 = `<eos>`.

`load_task3` và `encode_labels` đã làm sẵn cả ba việc:

```python
from data_utils import load_task3, encode_labels, decode_tokens, augment
import numpy as np

X, labels = load_task3("FinalDeepData/task3/data/mnist_sequences_v3", "train")  # (20000, 32, 256) uint8
dec_in, dec_out = encode_labels(labels, max_len=8)   # teacher forcing: <bos> d1..dn / d1..dn <eos>
X_aug = augment(X[:64], np.random.default_rng(0))   # nhiễu + độ sáng, dùng khi huấn luyện
decode_tokens([2, 0, 3, 6, 9, 12])                   # -> "20369"
```

Sinh lại hoặc sinh bộ lớn hơn (chạy trong thư mục `task3`, khoảng 20 giây cho mỗi 24.000 ảnh): `python generate_data_v3.py` (bộ chính) và `python generate_data_v3.py --out data/mnist_sequences_v3_big --train 60000` (60.000 ảnh train; 20.000 ảnh train đầu, validation và test trùng v3). Seed, tham số, số người viết và phiên bản thư viện nằm trong `generation_config.json`. Ảnh chỉ trùng từng pixel khi sinh bằng cùng phiên bản Pillow; bộ trong gói được sinh bằng NumPy 2.5 và Pillow 13.

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
  - Script tải đã sửa (26/11/2026):
    - `download_openmeteo.py` chạy mặc định cho 2023–2025 khớp từng giá trị với v1 ở 9 cột chung, và với bản mở rộng ở cả 19 biến;
    - `download_2022_2026.py` tải lại ra đúng từng byte file mở rộng và 10 file gốc.
- **Task 3 v1:**
  - Dựng lại 300 ảnh đầu của mỗi tập bằng script của nhóm với cùng seed: giống hệt cả nhãn lẫn từng pixel.
  - Nguồn train và validation không trùng nhau.
  - Xem trực tiếp ảnh mẫu, nhãn khớp hình.
- **Task 3 v2:** đủ 24.000 ảnh, cao 32, rộng không quá 256, không ảnh trống, nhãn khớp cột độ dài, xem trực tiếp ảnh mẫu.
- **Mã người viết:** nhãn chữ số của QMNIST trùng MNIST ở cả 60.000 ảnh train và 10.000 ảnh test; khoảng cách giữa ảnh MNIST và ảnh QMNIST cùng chỉ số rất nhỏ (L2 trung vị 8,7 trên thang 0–255, so với khoảng 2.200 giữa hai ảnh khác nhau cùng chữ số), đúng như README của QMNIST.
- **Task 3 v3** (`check_writers.py`):
  - cả 24.000 ảnh (và 64.000 ảnh của bộ lớn) có mọi chữ số cùng một người viết, đúng bằng cột `writer`;
  - ảnh MNIST nguồn đúng chữ số của nhãn, không ảnh nguồn nào dùng hai lần trong cùng một ảnh;
  - không người viết nào xuất hiện ở hai tập;
  - 20.000 ảnh train đầu, validation và test của bộ lớn trùng từng pixel với v3.
- **v1, v2 chạy lại** (`check_writers.py`): nhãn và từng pixel của cả 48.000 ảnh trùng bản gốc (với Pillow 10.4); 0 ảnh có mọi chữ số cùng một người viết.
- **`data_utils` và `metrics`:**
  - Căn chỉnh cửa sổ đúng trên 200 mẫu ngẫu nhiên; mã hoá rồi giải mã lại nhãn đúng; phần đệm ảnh toàn số 0.
  - `metrics.nse` cho cùng kết quả với `sklearn.metrics.r2_score`, còn `metrics.r2` thì khác, đúng như mong đợi.

## Ghi nguồn

- Open-Meteo, Historical Weather API và Air Quality API. Trang tài liệu yêu cầu ghi nguồn Open-Meteo và nhà cung cấp dữ liệu CAMS.
- MNIST: LeCun, Y., Bottou, L., Bengio, Y., Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11).
- QMNIST (mã người viết): Yadav, C., Bottou, L. (2019). Cold case: the lost MNIST digits. *Advances in Neural Information Processing Systems 32*. https://github.com/facebookresearch/qmnist (giấy phép BSD).
- Bộ v2 sinh bằng `task3/generate_data_v2.py` với seed 2027 (train), 2028 (validation), 2029 (test).
- Bộ v3 sinh bằng `task3/generate_data_v3.py` với seed 2030 (chia người viết), 2031 (train), 2032 (validation), 2033 (test).
