# -*- coding: utf-8 -*-
"""7주차 NAS 실습 공용 모듈 — 탐색 공간 정의와 세 가지 측정.

핵심 질문: **연산량(FLOPs)으로 실제 지연을 예측할 수 있는가?**
그래서 학습을 시키기 전에 먼저 파라미터·FLOPs·실측 지연 세 가지를 따로 잰다.
"""
import itertools, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sys import path as _p
_p.insert(0, "/root/ondevice-ai/lab06")
from kd_common import _idx, _prep          # 6주차 데이터 로더 재사용

torch.set_num_threads(2)
DEV = torch.device("cpu")

# ───────────────────── 탐색 공간 ─────────────────────
SPACE = {
    "w0":     [8, 12, 16, 24, 32],      # 첫 스테이지 채널 수
    "depth":  [2, 3, 4],                # 스테이지 개수
    "k":      [3, 5],                   # 커널 크기
    "sep":    [False, True],            # 깊이별 분리 합성곱(depthwise separable) 사용 여부
    "expand": [1.5, 2.0],               # 스테이지마다 채널이 몇 배로 늘어나는가
    "head":   ["gap", "flatten"],       # 전역 평균 풀링 vs 평탄화
}
KEYS = list(SPACE.keys())


def space_size():
    n = 1
    for v in SPACE.values():
        n *= len(v)
    return n


def sample(rng):
    return {k: SPACE[k][rng.randint(len(SPACE[k]))] for k in KEYS}


def geno_str(g):
    return (f"w{g['w0']}·d{g['depth']}·k{g['k']}·"
            f"{'sep' if g['sep'] else 'std'}·e{g['expand']}·{g['head']}")


# ───────────────────── 모델 ─────────────────────
class SepConv(nn.Module):
    """깊이별 분리 합성곱 — depthwise 3x3 + pointwise 1x1. FLOPs 를 크게 줄인다."""

    def __init__(self, cin, cout, k):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, k, padding=k // 2, groups=cin)
        self.pw = nn.Conv2d(cin, cout, 1)

    def forward(self, x):
        return self.pw(self.dw(x))


class Candidate(nn.Module):
    def __init__(self, g, n_cls=10):
        super().__init__()
        self.g = g
        chans, c = [], g["w0"]
        for _ in range(g["depth"]):
            chans.append(int(round(c)))
            c *= g["expand"]
        blocks, cin, size = [], 1, 28
        for cout in chans:
            conv = SepConv(cin, cout, g["k"]) if g["sep"] else \
                nn.Conv2d(cin, cout, g["k"], padding=g["k"] // 2)
            blocks += [conv, nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            cin, size = cout, size // 2
        self.body = nn.Sequential(*blocks)
        self.chans, self.size = chans, size
        if g["head"] == "gap":
            self.fc = nn.Linear(cin, n_cls)
        else:
            self.fc = nn.Linear(cin * size * size, n_cls)

    def forward(self, x):
        x = self.body(x)
        if self.g["head"] == "gap":
            x = F.adaptive_avg_pool2d(x, 1)
        return self.fc(x.flatten(1))


# ───────────────────── 세 가지 측정 ─────────────────────
def n_params(m):
    return sum(p.numel() for p in m.parameters())


def flops_of(g):
    """곱셈-누산(MAC) 횟수를 해석적으로 센다. 배치 1 기준."""
    chans, c = [], g["w0"]
    for _ in range(g["depth"]):
        chans.append(int(round(c)))
        c *= g["expand"]
    macs, cin, size = 0, 1, 28
    k = g["k"]
    for cout in chans:
        out = size                      # padding='same' 이므로 공간 크기 유지
        if g["sep"]:
            macs += cin * k * k * out * out          # depthwise
            macs += cin * cout * out * out           # pointwise
        else:
            macs += cin * cout * k * k * out * out
        cin, size = cout, size // 2
    macs += cin * 10 if g["head"] == "gap" else cin * size * size * 10
    return macs


@torch.no_grad()
def latency_ms(m, x, n=120, warmup=25):
    """배치 1 추론 지연(ms) 중앙값."""
    m.eval()
    for i in range(warmup):
        m(x[i:i + 1])
    ts = []
    for i in range(n):
        t0 = time.perf_counter()
        m(x[i % len(x):i % len(x) + 1])
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))


@torch.no_grad()
def evaluate(m, x, y, bs=1000):
    m.eval()
    ok = 0
    for i in range(0, len(x), bs):
        ok += (m(x[i:i + bs]).argmax(1) == y[i:i + bs]).sum().item()
    return 100.0 * ok / len(x)


def train(m, xtr, ytr, xte, yte, epochs=12, bs=128, lr=2e-3, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    g = torch.Generator().manual_seed(seed + 7)
    for _ in range(epochs):
        m.train()
        perm = torch.randperm(len(xtr), generator=g)
        for i in range(0, len(xtr), bs):
            b = perm[i:i + bs]
            loss = F.cross_entropy(m(xtr[b]), ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    return evaluate(m, xte, yte)


# ───────────────────── 데이터 ─────────────────────
def load(n_train=12000, seed=0):
    d = "/root/data/fmnist"
    xtr = _idx(f"{d}/train-images-idx3-ubyte.gz"); ytr = _idx(f"{d}/train-labels-idx1-ubyte.gz")
    xte = _idx(f"{d}/t10k-images-idx3-ubyte.gz");  yte = _idx(f"{d}/t10k-labels-idx1-ubyte.gz")
    rng = np.random.RandomState(seed)
    per = n_train // 10
    idx = np.concatenate([rng.permutation(np.where(ytr == c)[0])[:per] for c in range(10)])
    idx = rng.permutation(idx)
    return (_prep(xtr[idx]), torch.from_numpy(ytr[idx].astype(np.int64)),
            _prep(xte), torch.from_numpy(yte.astype(np.int64)))


# ───────────────────── 파레토 ─────────────────────
def pareto_front(points):
    """points: [(latency, accuracy, idx)] — 지연은 작을수록, 정확도는 클수록 좋다."""
    front = []
    for lat, acc, i in sorted(points):
        if not front or acc > front[-1][1]:
            front.append((lat, acc, i))
    return front
