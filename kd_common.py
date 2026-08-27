"""6주차 지식 증류 실습 공용 모듈.

데이터: Fashion-MNIST (28x28x1, 10클래스)
학습 표본을 10,000장으로 제한한다 — 엣지 개발의 현실(데이터도 적다)에 맞추고,
학생 모델이 데이터를 다 소화하지 못하는 구간을 만들기 위해서다.
"""
import gzip, os, struct, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA = "/root/data"
DEV = torch.device("cpu")
torch.set_num_threads(2)


# ────────────────────────────── 데이터 ──────────────────────────────
def _idx(path):
    with gzip.open(path, "rb") as f:
        magic, = struct.unpack(">I", f.read(4))
        ndim = magic & 0xFF
        shape = struct.unpack(">" + "I" * ndim, f.read(4 * ndim))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


def _prep(x):
    return torch.from_numpy((x.astype(np.float32) / 255.0 - 0.2860) / 0.3530).unsqueeze(1)


def load(name="fmnist", n_train=10000, n_teacher=0, seed=0):
    """학생용 학습 집합과 (선택) 교사용 학습 집합을 **서로 겹치지 않게** 나눠 돌려준다.

    n_teacher > 0 이면 (xs, ys, xt_pool, yt_pool, xte, yte) 를 돌려준다.
    교사를 학생과 다른 데이터로 학습시켜야 교사의 예측이 학생에게
    '새로운 정보'가 된다 — 같은 데이터로 학습한 교사는 그 데이터를 외워 버려
    소프트 타깃이 사실상 원-핫이 되고, 증류가 아무 정보도 전달하지 못한다.
    """
    d = os.path.join(DATA, name)
    xtr = _idx(os.path.join(d, "train-images-idx3-ubyte.gz"))
    ytr = _idx(os.path.join(d, "train-labels-idx1-ubyte.gz"))
    xte = _idx(os.path.join(d, "t10k-images-idx3-ubyte.gz"))
    yte = _idx(os.path.join(d, "t10k-labels-idx1-ubyte.gz"))

    rng = np.random.RandomState(seed)
    per_s, per_t = n_train // 10, n_teacher // 10
    si, ti = [], []
    for c in range(10):
        pool = rng.permutation(np.where(ytr == c)[0])
        si.append(pool[:per_s])
        ti.append(pool[per_s:per_s + per_t])
    si = rng.permutation(np.concatenate(si))
    out = [_prep(xtr[si]), torch.from_numpy(ytr[si].astype(np.int64))]
    if n_teacher:
        ti = rng.permutation(np.concatenate(ti))
        out += [_prep(xtr[ti]), torch.from_numpy(ytr[ti].astype(np.int64))]
    out += [_prep(xte), torch.from_numpy(yte.astype(np.int64))]
    return tuple(out)


# ────────────────────────────── 모델 ──────────────────────────────
class Net(nn.Module):
    """폭(w) 하나로 크기를 조절하는 3-conv CNN.

    w=4 → 2,954 파라미터 (학생)      w=32 → 104,202
    w=8 → 8,778                      w=64 → 392,714
    """

    def __init__(self, w=16):
        super().__init__()
        self.c1 = nn.Conv2d(1, w, 3, padding=1)
        self.c2 = nn.Conv2d(w, 2 * w, 3, padding=1)
        self.c3 = nn.Conv2d(2 * w, 4 * w, 3, padding=1)
        self.fc = nn.Linear(4 * w * 3 * 3, 10)

    def forward(self, x, return_feat=False):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)   # 14
        x = F.max_pool2d(F.relu(self.c2(x)), 2)   # 7
        f = F.relu(self.c3(x))
        x = F.max_pool2d(f, 2)                    # 3
        out = self.fc(x.flatten(1))
        return (out, f) if return_feat else out


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# ────────────────────────────── 학습·평가 ──────────────────────────────
@torch.no_grad()
def evaluate(model, x, y, bs=1000):
    model.eval()
    correct = 0
    for i in range(0, len(x), bs):
        correct += (model(x[i:i + bs]).argmax(1) == y[i:i + bs]).sum().item()
    return 100.0 * correct / len(x)


@torch.no_grad()
def logits_of(model, x, bs=1000):
    model.eval()
    return torch.cat([model(x[i:i + bs]) for i in range(0, len(x), bs)])


def kd_loss(s_logits, t_logits, labels, T=4.0, alpha=0.7):
    """KD 손실 = alpha * T^2 * KL(교사 soft || 학생 soft) + (1-alpha) * CE(정답, 학생)"""
    soft = F.kl_div(F.log_softmax(s_logits / T, dim=1),
                    F.log_softmax(t_logits / T, dim=1),
                    reduction="batchmean", log_target=True) * (T * T)
    hard = F.cross_entropy(s_logits, labels)
    return alpha * soft + (1 - alpha) * hard


def train(model, xtr, ytr, xte, yte, epochs=30, bs=128, lr=1e-3, seed=0,
          teacher_logits=None, T=4.0, alpha=0.7, label_smooth=0.0, log=None,
          curve=False):
    """teacher_logits 가 주어지면 KD, 아니면 일반 CE 학습.

    curve=True 이면 (최종정확도, 최고정확도, 에폭별 정확도 리스트) 를 돌려준다.
    """
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = len(xtr)
    g = torch.Generator().manual_seed(seed + 12345)
    best, hist = 0.0, []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            out = model(xtr[b])
            if teacher_logits is None:
                loss = F.cross_entropy(out, ytr[b], label_smoothing=label_smooth)
            else:
                loss = kd_loss(out, teacher_logits[b], ytr[b], T=T, alpha=alpha)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        acc = evaluate(model, xte, yte)
        best = max(best, acc); hist.append(round(acc, 2))
        if log and (ep + 1) % log == 0:
            print(f"    ep{ep+1:3d}  test {acc:6.2f}", flush=True)
    final = evaluate(model, xte, yte)
    return (final, best, hist) if curve else (final, best)


def bench_latency(model, x, n=200, warmup=30):
    """배치 1 추론 지연(ms) 중앙값."""
    model.eval()
    with torch.no_grad():
        for i in range(warmup):
            model(x[i:i + 1])
        ts = []
        for i in range(n):
            t0 = time.perf_counter()
            model(x[i:i + 1])
            ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))


def save_and_size(model, path):
    torch.save(model.state_dict(), path)
    return os.path.getsize(path)
