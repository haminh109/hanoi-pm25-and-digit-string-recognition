# Dự báo PM2.5 Hà Nội và đọc chuỗi chữ số viết tay

*Short-term PM2.5 forecasting for Hanoi and handwritten digit-string recognition with attention-based recurrent networks. Deep Learning final project.*

Đây là bài cuối kỳ môn Deep Learning, làm theo nhóm. Đề tài dùng hai bộ dữ liệu do nhóm tự xây dựng:

| Task | Bài toán | Dữ liệu |
|---|---|---|
| Task 1–2 | Dự báo nồng độ PM2.5 ở Hà Nội vài giờ tới. Task 1 dự báo một bước, Task 2 dự báo nhiều bước | Số liệu theo giờ 2023–2025 từ Open-Meteo (khí tượng và chất lượng không khí) |
| Task 3 | Đọc chuỗi 3–8 chữ số viết tay trong ảnh, ví dụ ra `"20369"` | Ảnh ghép từ chữ số MNIST, sinh bằng script của nhóm |

Repo hiện có:
- gói dữ liệu chung của nhóm cho cả ba task;
- toàn bộ phần Task 3 (notebook, script, kết quả).

## Task 3: mô hình CNN + BiLSTM + LSTM decoder + Attention

![Sơ đồ kiến trúc mô hình Task 3](task3_outputs/figures/model_architecture.png)

Mô hình hoạt động qua ba bước:
1. CNN kiểu VGG biến ảnh 32 × 256 thành một dãy 64 cột, xếp từ trái sang phải.
2. BiLSTM đọc dãy cột này theo cả hai chiều.
3. LSTM decoder sinh lần lượt từng chữ số, dừng ở `<eos>`. Ở mỗi bước, attention cộng (Bahdanau) chỉ cho decoder biết cần nhìn vào cột nào.

Trong hình, các đường xanh là trọng số attention thật của mô hình khi đọc ảnh ví dụ.

**Kết quả trên 2.000 ảnh test v2** (ảnh mô hình chưa từng thấy, ghép từ chữ số của MNIST-test):

| Mô hình | Số tham số | Đọc đúng cả chuỗi | CER |
|---|---:|---:|---:|
| CNN + nhiều đầu softmax (mốc so sánh) | 9.709.429 | 82,15% | 4,33% |
| CNN + LSTM decoder + Attention (bỏ BiLSTM) | 2.056.013 | 80,25% | 7,67% |
| CNN + BiLSTM + LSTM decoder (bỏ attention) | 2.252.877 | 91,70% | 1,95% |
| **CNN + BiLSTM + LSTM decoder + Attention** | 2.318.669 | **93,55%** | **1,20%** |
| Mô hình trên, huấn luyện với 60.000 ảnh thay vì 20.000 | 2.318.669 | **95,30%** | **0,90%** |

- **Độ ổn định:** chạy mô hình chính với ba seed được 93,55 ± 0,10%.
- **Quy đổi chuỗi ra số, theo từng chữ số:** R² = 0,970, NSE = 0,969, MAE = 0,051, RMSE = 0,502.
- **Đổi bộ dữ liệu:**
  - mô hình học trên v2 đọc đúng 90,05% ảnh v1;
  - mô hình học trên v1 chỉ đọc đúng 15,90% ảnh v2.
- **Beam search** (k = 3 và 5) không cải thiện so với greedy.

Notebook `05_task3_model.ipynb` có thêm phân tích lỗi và bản đồ attention. Các bảng kết quả đầy đủ nằm trong `task3_outputs/results/`.

## Cấu trúc thư mục

```
.
├── 04_task3_data.ipynb          Task 3: khám phá dữ liệu, đệm ảnh, mã hoá nhãn, tăng cường dữ liệu
├── 05_task3_model.ipynb         Task 3: mô hình, huấn luyện, greedy/beam search, đánh giá, thí nghiệm so sánh
├── task3_model.py               Task 3: riêng mô hình chính dạng script (huấn luyện + đánh giá trên tập test)
├── task3_outputs/
│   ├── figures/                 hình dùng cho báo cáo
│   ├── results/                 bảng kết quả (CSV) và task3_summary.json
│   └── models/                  lịch sử huấn luyện từng epoch (*_history.json); trọng số không đăng kèm
├── FinalDeepData/               gói dữ liệu chung của nhóm, xem FinalDeepData/README.md
│   ├── common/                  data_utils.py (đọc dữ liệu), metrics.py (R², NSE, MAE, RMSE, chỉ số Task 3)
│   ├── task1_2/                 số liệu PM2.5 và khí tượng Hà Nội theo giờ, 2023–2025
│   └── task3/                   script sinh ảnh chuỗi chữ số
└── Proposal_PM2.5_Ha_Noi_va_chuoi_so_viet_tay.pdf
```

## Dữ liệu

- **Task 1–2:** dùng ngay file `FinalDeepData/task1_2/hanoi_hourly_2023_2025_v1.csv`. File có 26.304 giờ, không thiếu giờ nào và có sẵn cột chia train/validation/test theo thời gian. `FinalDeepData/README.md` mô tả từng cột, cách cắt cửa sổ trượt và mốc dự báo persistence cần vượt qua.
- **Task 3:** bộ ảnh (48.000 ảnh, khoảng 190 MB) không có trong repo. Sinh lại bằng hai lệnh, mỗi lệnh khoảng một phút:

  ```bash
  cd FinalDeepData/task3
  python generate_data.py        # v1: tự tải MNIST (11 MB) rồi sinh 20.000 / 2.000 / 2.000 ảnh
  python generate_data_v2.py     # v2: bộ khó hơn, chữ số chạm hoặc chồng nhau (cần chạy v1 trước)
  cd ../..
  ```

  Seed được cố định, nên nhãn sinh ra luôn giống hệt bộ ảnh nhóm đã dùng để ra các kết quả ở trên. Về ảnh:
  - Bộ v1 giống hệt từng pixel.
  - Bộ v2 giống hệt từng pixel khi dùng Pillow 10.4 và NumPy 2.1, là môi trường nhóm đã dùng.
  - Với Pillow 13 và NumPy 2.5, khoảng 2% ảnh v2 lệch nhẹ: mỗi ảnh lệch vài chục pixel, độ xám lệch tối đa 24/255.

  Bộ 60.000 ảnh (`mnist_sequences_v2_big`) được notebook và `task3_model.py --data v2_big` tự sinh khi cần.

## Chạy code Task 3

Cần Python 3.10 trở lên:

```bash
pip install "keras>=3" tensorflow numpy pandas pillow matplotlib h5py
```

Code viết bằng Keras 3, nên cũng chạy được với PyTorch: cài `torch` rồi đặt `KERAS_BACKEND=torch`.

- **Notebook:** mở từ thư mục gốc của repo, sau khi đã sinh dữ liệu.
  - Repo không kèm trọng số, nên `05_task3_model.ipynb` huấn luyện 8 mô hình, mất khoảng 3 giờ trên GPU Apple M4 Pro.
  - Đặt `RUN_EXPERIMENTS = False` để chỉ huấn luyện mô hình chính, khoảng 25 phút.
- **Script** (kết quả lưu vào `checkpoints/`):

  ```bash
  python task3_model.py                        # 20.000 ảnh v2, seed 42, tối đa 30 epoch
  python task3_model.py --data v2_big          # 60.000 ảnh v2
  python task3_model.py --data v1 --seed 43    # dữ liệu v1, seed khác
  ```

- **Google Colab:**
  1. Mở notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haminh109/hanoi-pm25-and-digit-string-recognition/blob/main/05_task3_model.ipynb)
  2. Chọn *Runtime → Change runtime type → GPU*.
  3. Thêm một ô ở đầu notebook rồi chạy lần lượt:

  ```python
  !git clone -q https://github.com/haminh109/hanoi-pm25-and-digit-string-recognition.git
  %cd hanoi-pm25-and-digit-string-recognition
  !cd FinalDeepData/task3 && python generate_data.py && python generate_data_v2.py
  ```

- **Máy Mac dùng Anaconda:** nếu `import tensorflow` bị treo (lỗi `mutex.cc ... Lock blocking`, do xung đột với `pyarrow`), dùng một môi trường mới không có `pyarrow`, hoặc chạy trên Colab.

## Ghi nguồn dữ liệu

- Số liệu Task 1–2: [Open-Meteo](https://open-meteo.com/), gồm Historical Weather API và Air Quality API. Dữ liệu chất lượng không khí lấy từ mô hình CAMS (Copernicus Atmosphere Monitoring Service). Dữ liệu dùng theo giấy phép CC BY 4.0.
- MNIST: LeCun, Y., Bottou, L., Bengio, Y., Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11).
