"""Bốn chỉ số R², NSE, MAE, RMSE theo đúng công thức slide chương 2.1, và chỉ số bổ trợ cho Task 3.

Lưu ý: sklearn.metrics.r2_score tính 1 - SSE/SST, tức là NSE, KHÔNG phải R² trong slide.
R² trong slide là bình phương hệ số tương quan Pearson giữa thực tế và dự báo.
Luôn tính trên đơn vị gốc (µg/m³), sau khi đã đổi ngược chuẩn hoá và log.
"""
import numpy as np


def r2(obs, pred):
    obs, pred = np.asarray(obs, float).ravel(), np.asarray(pred, float).ravel()
    num = np.sum((obs - obs.mean()) * (pred - pred.mean()))
    den = np.sqrt(np.sum((obs - obs.mean()) ** 2) * np.sum((pred - pred.mean()) ** 2))
    return float((num / den) ** 2) if den > 0 else float("nan")


def nse(obs, pred):
    obs, pred = np.asarray(obs, float).ravel(), np.asarray(pred, float).ravel()
    sst = np.sum((obs - obs.mean()) ** 2)
    return float(1 - np.sum((obs - pred) ** 2) / sst) if sst > 0 else float("nan")


def mae(obs, pred):
    return float(np.mean(np.abs(np.asarray(obs, float) - np.asarray(pred, float))))


def rmse(obs, pred):
    return float(np.sqrt(np.mean((np.asarray(obs, float) - np.asarray(pred, float)) ** 2)))


def regression_report(obs, pred):
    return {"R2": r2(obs, pred), "NSE": nse(obs, pred), "MAE": mae(obs, pred), "RMSE": rmse(obs, pred)}


def seq2seq_report(obs, pred):
    """Task 2: obs, pred có dạng (N, 5).

    Trả về hai cách tính: theo từng bước (danh sách 5 dict) và gộp chung (trải phẳng N x 5 cặp).
    R² và NSE gộp chung không bằng trung bình của 5 giá trị theo bước, nên ghi rõ cách tính.
    """
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    per_step = [regression_report(obs[:, k], pred[:, k]) for k in range(obs.shape[1])]
    return {"per_step": per_step, "pooled": regression_report(obs.ravel(), pred.ravel())}


# ----------------------------------------------------------------------------- Task 3

def sequence_accuracy(true, pred):
    """Tỷ lệ chuỗi đọc đúng hoàn toàn."""
    return float(np.mean([t == p for t, p in zip(true, pred)]))


def _edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(true, pred):
    """Character error rate: tổng số thao tác thêm, xoá, thay / tổng số chữ số đúng."""
    return float(sum(_edit_distance(t, p) for t, p in zip(true, pred)) / sum(len(t) for t in true))


def digit_level_report(true, pred):
    """Cách B (mục 6.3 của proposal): căn phải hai chuỗi như đặt phép cộng, đệm 0 bên trái,
    mỗi cặp chữ số cùng hàng là một cặp (thực tế, dự báo), rồi tính bốn chỉ số."""
    o, p = [], []
    for t, q in zip(true, pred):
        n = max(len(t), len(q))
        t, q = t.rjust(n, "0"), (q or "0").rjust(n, "0")
        o.extend(int(c) for c in t)
        p.extend(int(c) for c in q)
    return regression_report(o, p)


def numeric_report_by_length(true, pred):
    """Cách A: coi chuỗi là số nguyên, tính bốn chỉ số riêng cho từng độ dài chuỗi thật."""
    out = {}
    lengths = np.array([len(t) for t in true])
    for n in sorted(set(lengths)):
        idx = np.flatnonzero(lengths == n)
        o = [float(true[i]) for i in idx]
        p = [float(pred[i]) if pred[i] else 0.0 for i in idx]
        out[int(n)] = regression_report(o, p)
    return out
