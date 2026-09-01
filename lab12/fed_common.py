"""12주차 공통 모듈 — 연합 학습 · 통신 비용 · 기울기 누출 · 차등 프라이버시.

이번 주의 질문은 "데이터를 안 보내면 안전한가"이다.
답은 실험으로 낸다. 우리가 보내는 것(기울기)에서 무엇이 복원되는지 직접 꺼내 본다.
"""
import copy, math, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

torch.set_num_threads(2)
DEV = "cpu"
ROOT = "/root/lab12/data"


# ══════════════════════════════════════════════════════════════════
# 데이터
# ══════════════════════════════════════════════════════════════════
def load_mnist(n_train=12000, n_test=2000, seed=0):
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    tr = datasets.MNIST(ROOT, train=True, download=True, transform=tf)
    te = datasets.MNIST(ROOT, train=False, download=True, transform=tf)
    g = np.random.RandomState(seed)
    itr = g.permutation(len(tr))[:n_train]
    ite = g.permutation(len(te))[:n_test]
    # 텐서로 한 번에 올려 둔다 — 매 라운드 DataLoader 를 새로 만드는 비용을 없앤다
    X = torch.stack([tr[i][0] for i in itr])
    Y = torch.tensor([tr[i][1] for i in itr])
    Xt = torch.stack([te[i][0] for i in ite])
    Yt = torch.tensor([te[i][1] for i in ite])
    return X, Y, Xt, Yt


def dirichlet_split(Y, n_clients, alpha, seed=0, n_classes=10):
    """디리클레 분포로 클라이언트에 라벨을 쏠리게 나눈다.

    alpha 가 크면 IID 에 가깝고, 작으면 한 클라이언트가 몇 개 숫자만 갖는다.
    연합 학습 문헌의 표준 비-IID 시뮬레이션 방법이다.
    """
    g = np.random.RandomState(seed)
    Y = Y.numpy()
    idx_by_c = [np.where(Y == c)[0] for c in range(n_classes)]
    for a in idx_by_c:
        g.shuffle(a)
    parts = [[] for _ in range(n_clients)]
    for c in range(n_classes):
        p = g.dirichlet(np.repeat(alpha, n_clients))
        cuts = (np.cumsum(p) * len(idx_by_c[c])).astype(int)[:-1]
        for k, chunk in enumerate(np.split(idx_by_c[c], cuts)):
            parts[k].extend(chunk.tolist())
    out = []
    for p in parts:
        g.shuffle(p)
        out.append(np.array(p, dtype=np.int64))
    return out


def label_skew(parts, Y, n_classes=10):
    """각 클라이언트가 실제로 몇 개 클래스를 갖는지 — 비-IID 의 정도를 수치로."""
    Y = Y.numpy()
    cnt = []
    for p in parts:
        if len(p) == 0:
            cnt.append(0); continue
        cnt.append(int((np.bincount(Y[p], minlength=n_classes) > 0).sum()))
    return cnt


# ══════════════════════════════════════════════════════════════════
# 모델
# ══════════════════════════════════════════════════════════════════
class SmallCNN(nn.Module):
    """IoT 기기에 올릴 만한 크기 — 연합 학습 실험용."""
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 8, 5, padding=2)
        self.c2 = nn.Conv2d(8, 16, 5, padding=2)
        self.f1 = nn.Linear(16 * 7 * 7, 64)
        self.f2 = nn.Linear(64, 10)

    def forward(self, x):
        x = Fn.max_pool2d(Fn.relu(self.c1(x)), 2)
        x = Fn.max_pool2d(Fn.relu(self.c2(x)), 2)
        x = x.flatten(1)
        return self.f2(Fn.relu(self.f1(x)))


class MLP(nn.Module):
    """첫 층이 원본 입력을 그대로 받는다 — 2교시의 기울기 역복원용."""
    def __init__(self, h=32):
        super().__init__()
        self.f1 = nn.Linear(784, h)
        self.f2 = nn.Linear(h, 10)

    def forward(self, x):
        return self.f2(Fn.relu(self.f1(x.flatten(1))))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def model_bytes(m, bpe=4):
    return n_params(m) * bpe


# ══════════════════════════════════════════════════════════════════
# 연합 학습 — FedAvg
# ══════════════════════════════════════════════════════════════════
def get_flat(m):
    return torch.cat([p.data.view(-1) for p in m.parameters()])


def set_flat(m, v):
    i = 0
    for p in m.parameters():
        n = p.numel()
        p.data.copy_(v[i:i + n].view_as(p))
        i += n


@torch.no_grad()
def evaluate(m, Xt, Yt, bs=512):
    m.eval()
    ok = 0
    for i in range(0, len(Xt), bs):
        ok += (m(Xt[i:i + bs]).argmax(1) == Yt[i:i + bs]).sum().item()
    return ok / len(Xt)


def local_train(m, X, Y, idx, epochs, lr, bs=32, clip=None, sigma=0.0, gen=None):
    """클라이언트 한 대의 로컬 학습. clip/sigma 가 주어지면 DP-SGD 방식."""
    m.train()
    opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.0)
    n = len(idx)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for s in range(0, n, bs):
            b = idx[perm[s:s + bs].numpy()]
            opt.zero_grad()
            loss = Fn.cross_entropy(m(X[b]), Y[b])
            loss.backward()
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(m.parameters(), clip)
                if sigma > 0:
                    for p in m.parameters():
                        p.grad.add_(torch.randn(p.shape, generator=gen) * sigma * clip / len(b))
            opt.step()
    return m


def fedavg(X, Y, Xt, Yt, parts, rounds=30, epochs=1, lr=0.1, bs=32,
           frac=1.0, seed=0, model_fn=SmallCNN, clip=None, sigma=0.0,
           topk=None, log_every=5, verbose=True):
    """FedAvg — 서버가 평균 내고, 클라이언트가 로컬 학습한다.

    topk 가 주어지면 업데이트의 상위 k 비율만 보낸다(희소화).
    돌려주는 것: 라운드별 (정확도, 누적 업로드 바이트)
    """
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    glob = model_fn()
    P = n_params(glob)
    n_cl = len(parts)
    m_per_round = max(1, int(round(frac * n_cl)))
    hist = []
    up_bytes = 0
    rng = np.random.RandomState(seed)
    for r in range(1, rounds + 1):
        sel = rng.choice(n_cl, m_per_round, replace=False)
        gv = get_flat(glob)
        agg = torch.zeros_like(gv)
        tot = 0
        for c in sel:
            if len(parts[c]) == 0:
                continue
            local = model_fn()
            set_flat(local, gv)
            local_train(local, X, Y, parts[c], epochs, lr, bs, clip, sigma, gen)
            delta = get_flat(local) - gv
            if topk is not None and topk < 1.0:
                k = max(1, int(P * topk))
                thr = delta.abs().kthvalue(P - k + 1).values
                delta = torch.where(delta.abs() >= thr, delta, torch.zeros_like(delta))
                # 희소 전송: 값 4바이트 + 인덱스 4바이트
                up_bytes += k * 8
            else:
                up_bytes += P * 4
            w = len(parts[c])
            agg += delta * w
            tot += w
        if tot:
            set_flat(glob, gv + agg / tot)
        if r % log_every == 0 or r == rounds:
            acc = evaluate(glob, Xt, Yt)
            hist.append({"round": r, "acc": acc, "up_mb": up_bytes / 1048576})
            if verbose:
                print(f"      r{r:3d}  acc {acc:.4f}  누적 업로드 {up_bytes/1048576:8.2f} MB")
    return glob, hist


def centralized(X, Y, Xt, Yt, epochs=10, lr=0.1, bs=32, seed=0, model_fn=SmallCNN):
    """비교 기준 — 데이터를 한곳에 모았을 때."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    m = model_fn()
    idx = np.arange(len(X))
    hist = []
    for e in range(1, epochs + 1):
        local_train(m, X, Y, idx, 1, lr, bs, gen=gen)
        hist.append({"epoch": e, "acc": evaluate(m, Xt, Yt)})
    return m, hist


# ══════════════════════════════════════════════════════════════════
# 기울기 누출 — 해석적 역복원
# ══════════════════════════════════════════════════════════════════
def grad_of(model, x, y):
    """한 배치의 기울기. 이것이 '데이터 대신' 서버로 보내는 것이다."""
    model.zero_grad()
    loss = Fn.cross_entropy(model(x), y)
    g = torch.autograd.grad(loss, list(model.parameters()))
    return [t.detach().clone() for t in g]


def invert_linear(gW, gb, eps=1e-12):
    """선형층의 기울기에서 입력을 되찾는다 — 최적화가 아니라 나눗셈이다.

        z = Wx + b  이면   dL/dW = (dL/dz) xᵀ,   dL/db = dL/dz
        따라서       x = dL/dW[i, :] / dL/db[i]

    |dL/db| 가 가장 큰 행을 쓰면 수치적으로 가장 안정적이다.
    """
    i = int(gb.abs().argmax())
    if gb[i].abs() < eps:
        return None
    return gW[i] / gb[i]


def leak_label(g_last_b):
    """마지막 층 편향 기울기의 부호가 정답 라벨을 알려 준다 (iDLG 관찰).

    소프트맥스 교차엔트로피에서 dL/db_last = p - onehot(y) 이므로
    정답 클래스 성분만 음수가 된다.
    """
    return int(g_last_b.argmin())


def psnr(a, b, peak=None):
    a, b = a.flatten().float(), b.flatten().float()
    mse = ((a - b) ** 2).mean().item()
    if mse < 1e-20:
        return float("inf")
    pk = peak if peak is not None else (a.max() - a.min()).item()
    return 20 * math.log10(pk / math.sqrt(mse))
