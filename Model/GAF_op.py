import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.models as models
import cv2
from tqdm import tqdm
import os

# ======================
# 参数
# ======================
DATASETS = ["PEMS04", "PEMS08", "METR-LA", "BJMetro", "HZMetro", "XMBRT"]
ROOT = "/data/LiFeiFei/MM/datasets"
SAVE_ROOT = "./outputs"

HISTORY = 12
PRED = 12
IMG_SIZE = 64

BATCH_SIZE = 128
EPOCHS = 40
LR = 1e-3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ======================
# 数据加载
# ======================
def load_dataset(name):

    base = os.path.join(ROOT, name)

    if name == "PEMS04":
        data = np.load(os.path.join(base, "PEMS04.npz"))
        X = data[data.files[0]]

    elif name == "PEMS08":
        data = np.load(os.path.join(base, "pems08.npz"))
        X = data[data.files[0]]

    elif name == "METR-LA":
        df = pd.read_hdf(os.path.join(base, "metr-la.h5"))
        X = df.values

    elif name == "BJMetro":
        X = np.load(os.path.join(base, "ts_bjmetro_hs12_ts12.npy"))

    elif name == "HZMetro":
        X = np.load(os.path.join(base, "ts_hzmetro_hs12_ts12.npy"))

    elif name == "XMBRT":
        X = np.load(os.path.join(base, "ts_xmbrt_hs12_ts12.npy"))

    else:
        raise ValueError

    return X


# ======================
# 滑窗统一
# ======================
def build_windows(X):

    if X.ndim == 3:
        return X[:, :HISTORY, :], X[:, HISTORY:HISTORY+PRED, :]

    T, N = X.shape
    xs, ys = [], []

    for t in range(HISTORY, T-PRED):
        xs.append(X[t-HISTORY:t])
        ys.append(X[t:t+PRED])

    return np.array(xs), np.array(ys)


# ======================
# 图像函数（不变）
# ======================
def normalize_ts(ts):
    ts_min, ts_max = ts.min(), ts.max()
    return 2 * (ts - ts_min) / (ts_max - ts_min + 1e-6) - 1

def build_multiscale(seq):
    scales = [1, 3, 6, 12]
    return [np.convolve(seq, np.ones(w)/w, mode='same') for w in scales]

def build_gaf(ts):
    ts = normalize_ts(ts)
    phi = np.arccos(np.clip(ts, -1, 1))
    return np.cos(phi[:, None] + phi[None, :])

def build_image(seq):
    scales = build_multiscale(seq)
    channels = []
    for s in scales:
        s = normalize_ts(s)
        channels.extend([
            np.tile(s, (len(s), 1)),
            build_gaf(s),
            np.tile(np.diff(s, prepend=s[0]), (len(s), 1))
        ])
    img = np.stack(channels)
    img = np.array([cv2.resize(c, (IMG_SIZE, IMG_SIZE)) for c in img])
    return img.reshape(3, 4, IMG_SIZE, IMG_SIZE).mean(axis=1)


# ======================
# Dataset
# ======================
class TrafficDataset(Dataset):

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.N = X.shape[2]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        node = np.random.randint(0, self.N)

        seq = self.X[idx, :, node]
        img = build_image(seq)

        y = self.Y[idx, :, node]

        return torch.tensor(img, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# ======================
# 划分
# ======================
def split_dataset(dataset):

    total = len(dataset)
    idx = list(range(total))

    train_end = int(0.6 * total)
    val_end = int(0.8 * total)

    return (
        Subset(dataset, idx[:train_end]),
        Subset(dataset, idx[train_end:val_end]),
        Subset(dataset, idx[val_end:])
    )


# ======================
# 模型
# ======================
class ResNetPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(512, PRED)

    def forward(self, x):
        return self.model(x)


# ======================
# 指标
# ======================
def metric(pred, true, mean, std):

    pred = pred.cpu().numpy() * std + mean
    true = true.cpu().numpy() * std + mean

    mae = np.mean(np.abs(pred-true))
    rmse = np.sqrt(np.mean((pred-true)**2))
    mape = np.mean(np.abs((pred-true)/(true+1e-5))) * 100

    return mae, rmse, mape


# ======================
# 🔥 单数据集训练
# ======================
def run_dataset(dataset_name):

    print(f"\n🚀 Running: {dataset_name}")

    save_dir = os.path.join(SAVE_ROOT, dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    # ===== 数据 =====
    X_raw = load_dataset(dataset_name)
    X, Y = build_windows(X_raw)

    mean, std = X.mean(), X.std()
    X = (X - mean) / (std + 1e-6)

    dataset = TrafficDataset(X, Y)
    train_set, val_set, test_set = split_dataset(dataset)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    model = ResNetPredictor().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best_loss = 1e9

    # ===== 训练 =====
    for epoch in range(EPOCHS):

        model.train()
        total_loss = 0

        for x, y in tqdm(train_loader, desc=f"{dataset_name} Epoch {epoch+1}"):

            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # ===== 验证 =====
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                val_loss += loss_fn(model(x), y).item()

        val_loss /= len(val_loader)

        print(f"Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))

    # ===== 测试 =====
    loader = DataLoader(test_set, batch_size=128)

    MAE, RMSE, MAPE = [], [], []

    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pth")))
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)

            mae, rmse, mape = metric(pred, y, mean, std)
            MAE.append(mae)
            RMSE.append(rmse)
            MAPE.append(mape)

    result = f"MAE:{np.mean(MAE):.4f}, RMSE:{np.mean(RMSE):.4f}, MAPE:{np.mean(MAPE):.2f}%"
    print(result)

    with open(os.path.join(save_dir, "result.txt"), "w") as f:
        f.write(result)


# ======================
# MAIN
# ======================
if __name__ == "__main__":

    for d in DATASETS:
        run_dataset(d)