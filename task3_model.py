import argparse
import json
import subprocess
import sys
from pathlib import Path
import keras
import numpy as np
import pandas as pd
from keras import layers, ops

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="v2", choices=["v2", "v2_big", "v1"])
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--epochs", type=int, default=30)
args = parser.parse_args()
keras.utils.set_random_seed(args.seed)

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / "FinalDeepData" / "common"))  # data_utils.py, metrics.py dùng chung của nhóm
import metrics
from data_utils import BOS, PAD, VOCAB_SIZE, augment, decode_tokens, encode_labels, load_task3
data_dir = root / "FinalDeepData" / "task3" / "data"
train_dir = data_dir / f"mnist_sequences_{args.data}"
eval_dir = data_dir / f"mnist_sequences_{args.data.replace('_big', '')}"  # v2_big: validation, test của v2
save_path = root / "checkpoints" / f"task3_model_{args.data}_seed{args.seed}"

# 1. Load the dataset
if args.data == "v2_big" and not train_dir.exists():  # sinh 60.000 ảnh train bằng script của nhóm (vài phút)
    subprocess.run([sys.executable, str(data_dir.parent / "generate_data_v2.py"), "--out", str(train_dir),
                    "--train", "60000"], check=True)
X_train, y_train = load_task3(train_dir, "train")  # ảnh uint8 (N, 32, 256), nhãn là chuỗi chữ số
X_val, y_val = load_task3(eval_dir, "validation")
X_test, y_test = load_task3(eval_dir, "test")
dec_in_train, dec_out_train = encode_labels(y_train)  # <bos> d1..dn <pad>..  và  d1..dn <eos> <pad>..
dec_in_val, dec_out_val = encode_labels(y_val)
print(f"Train {X_train.shape}, validation {X_val.shape}, test {X_test.shape}")


def to_float(X):  # uint8 (N, 32, 256) -> float32 (N, 32, 256, 1) trong [0, 1]
    return (X.astype("float32") / 255.0)[..., None]


X_val, X_test = to_float(X_val), to_float(X_test)


# 2. Data Augmentation
class DigitSeqDataset(keras.utils.PyDataset):  # xáo trộn mỗi epoch, đổi độ sáng + thêm nhiễu Gauss
    def __init__(self, X, dec_in, dec_out, batch_size=64):
        super().__init__()
        self.X, self.dec_in, self.dec_out, self.batch_size = X, dec_in, dec_out, batch_size
        self.rng = np.random.default_rng(args.seed)
        self.order = np.arange(len(X))
        self.rng.shuffle(self.order)

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, i):
        idx = self.order[i * self.batch_size:(i + 1) * self.batch_size]
        x = to_float(augment(self.X[idx], self.rng))  # nhãn giữ nguyên
        return (x, self.dec_in[idx]), self.dec_out[idx]  # mô hình nhận (ảnh, đầu vào decoder)

    def on_epoch_end(self):
        self.rng.shuffle(self.order)


train_data = DigitSeqDataset(X_train, dec_in_train, dec_out_train)

# 3. Build CNN + BiLSTM + LSTM decoder + Attention Model
T_DEC = 9  # số bước của decoder: tối đa 8 chữ số + <eos>
conv_arch = ((1, 32, (2, 2)), (1, 64, (2, 2)), (2, 128, (2, 1)), (2, 256, (2, 1)))  # (số conv, số kênh, pool)
image = keras.Input(shape=(32, 256, 1), name="image")
x = image
for num_convs, num_channels, pool_size in conv_arch:
    for _ in range(num_convs):  # khối VGG (chương 5.2)
        x = layers.Conv2D(num_channels, (3, 3), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=pool_size)(x)  # sau 4 khối: (2, 64, 256)
x = layers.Permute((2, 1, 3))(x)  # (64, 2, 256): đưa chiều rộng ảnh lên trước
x = layers.Reshape((64, 2 * 256))(x)  # 64 cột đặc trưng từ trái sang phải
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.2)(x)
bilstm = layers.Bidirectional(layers.LSTM(128, return_sequences=True, return_state=True))  # BiLSTM (chương 7)
H, h_fw, c_fw, h_bw, c_bw = bilstm(x)
h0, c0 = layers.Concatenate()([h_fw, h_bw]), layers.Concatenate()([c_fw, c_bw])  # trạng thái đầu của decoder
encoder = keras.Model(image, [H, h0, c0], name="encoder")


class BahdanauAttention(layers.Layer):  # e_i = v^T tanh(W h_i + U s), alpha = softmax(e), c = sum alpha_i h_i
    def __init__(self, units):
        super().__init__()
        self.W = layers.Dense(units, use_bias=False)
        self.U = layers.Dense(units)
        self.v = layers.Dense(1, use_bias=False)

    def call(self, s, H, WH):
        e = self.v(ops.tanh(WH + ops.expand_dims(self.U(s), 1)))  # (B, 64, 1): điểm của từng cột
        return ops.sum(ops.softmax(e, axis=1) * H, axis=1)  # vector ngữ cảnh c_t: (B, 256)


class Img2Seq(keras.Model):  # LSTM decoder có attention (chương 8)
    def __init__(self):
        super().__init__()
        self.encoder = encoder
        self.embedding = layers.Embedding(VOCAB_SIZE, 64)
        self.attention = BahdanauAttention(128)
        self.cell = layers.LSTMCell(256)
        self.dropout = layers.Dropout(0.3)
        self.fc = layers.Dense(VOCAB_SIZE, activation="softmax")

    def call(self, inputs, training=False):
        # fit: inputs = (ảnh, dec_in) -> teacher forcing;  predict: chỉ có ảnh -> greedy search
        images, dec_in = inputs if isinstance(inputs, (tuple, list)) else (inputs, None)
        H, h0, c0 = self.encoder(images, training=training)
        WH, state = self.attention.W(H), [h0, c0]  # W h_i chỉ cần tính một lần cho mỗi ảnh
        y = ops.full((ops.shape(images)[0],), BOS, dtype="int32")  # bước đầu: <bos>
        outputs = []
        for t in range(T_DEC):
            if dec_in is not None:
                y = dec_in[:, t]  # teacher forcing: token đúng của bước trước
            context = self.attention(state[0], H, WH)
            x_t = ops.concatenate([self.embedding(y), context], axis=-1)  # token bước trước + c_t
            s, state = self.cell(x_t, state, training=training)
            probs = self.fc(self.dropout(ops.concatenate([s, context], axis=-1), training=training))
            outputs.append(probs)
            y = ops.cast(ops.argmax(probs, axis=-1), "int32")  # greedy: token có xác suất cao nhất
        return ops.stack(outputs, axis=1)  # (B, 9, 13): xác suất 13 token ở 9 bước


model = Img2Seq()
model((X_val[:2], dec_in_val[:2]))  # gọi thử trên 2 ảnh để Keras tạo trọng số trước khi train


# 4. Train
def masked_loss(y_true, y_pred):  # cross-entropy trên các token thật, bỏ qua <pad>
    y_true = ops.cast(y_true, "int32")
    loss = keras.losses.sparse_categorical_crossentropy(y_true, y_pred)  # (B, 9)
    mask = ops.cast(ops.not_equal(y_true, PAD), loss.dtype)
    return ops.sum(loss * mask) / ops.sum(mask)


def token_acc(y_true, y_pred):
    y_true, pred = ops.cast(y_true, "int32"), ops.cast(ops.argmax(y_pred, axis=-1), "int32")
    mask = ops.not_equal(y_true, PAD)
    correct = ops.logical_and(ops.equal(pred, y_true), mask)
    return ops.sum(ops.cast(correct, "float32")) / ops.sum(ops.cast(mask, "float32"))


def seq_acc(y_true, y_pred):  # 1 nếu đọc đúng cả chuỗi, 0 nếu sai; Keras lấy trung bình trên các ảnh
    y_true, pred = ops.cast(y_true, "int32"), ops.cast(ops.argmax(y_pred, axis=-1), "int32")
    ok = ops.logical_or(ops.equal(pred, y_true), ops.equal(y_true, PAD))
    return ops.cast(ops.all(ok, axis=1), "float32")


model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0), loss=masked_loss,
              metrics=[token_acc, seq_acc])
checkpoint = keras.callbacks.ModelCheckpoint(
    f"{save_path}.weights.h5", monitor="val_seq_acc", mode="max", save_best_only=True, save_weights_only=True)
early_stop = keras.callbacks.EarlyStopping(monitor="val_seq_acc", mode="max", patience=5)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5)
history = model.fit(train_data, validation_data=((X_val, dec_in_val), dec_out_val), epochs=args.epochs,
                    callbacks=[checkpoint, early_stop, reduce_lr], verbose=2).history
model.load_weights(f"{save_path}.weights.h5")

# 5. Evaluate test set
probs = model.predict(X_test, batch_size=250, verbose=0)  # greedy search: (N, 9, 13)
y_pred = [decode_tokens(tokens) for tokens in probs.argmax(axis=-1)]  # token -> chuỗi chữ số, dừng ở <eos>
lengths, correct = np.array([len(t) for t in y_test]), np.array([t == p for t, p in zip(y_test, y_pred)])
by_length = {n: {"n_images": int(np.sum(lengths == n)), "seq_acc": correct[lengths == n].mean(), **r}
             for n, r in metrics.numeric_report_by_length(y_test, y_pred).items()}  # cách A: coi chuỗi là số
report = pd.DataFrame.from_dict(by_length, orient="index").rename_axis("length")
print(report.to_string(float_format="{:,.4f}".format))
val_acc = history["val_seq_acc"]
results = {"data": args.data, "seed": args.seed, "best_val_seq_acc": max(val_acc),
           "best_epoch": int(np.argmax(val_acc)) + 1,
           "test_seq_acc": metrics.sequence_accuracy(y_test, y_pred), "test_cer": metrics.cer(y_test, y_pred),
           "test_len_acc": np.mean([len(p) for p in y_pred] == lengths),
           **{f"test_{k}_B": v for k, v in metrics.digit_level_report(y_test, y_pred).items()},  # cách B
           "test_by_length_A": by_length, "history": history}
Path(f"{save_path}.json").write_text(json.dumps(results, indent=1, default=float))
print("Test: đọc đúng cả chuỗi {test_seq_acc:.4f}, CER {test_cer:.4f} | cách B: R2 {test_R2_B:.4f}, "
      "NSE {test_NSE_B:.4f}, MAE {test_MAE_B:.4f}, RMSE {test_RMSE_B:.4f}".format(**results))
