# -*- coding: utf-8 -*-
"""MLPerf Tiny v1.0 참조 모델 4종을 PyTorch 로 재구성한다.

출처 구조 설명: MLPerf Tiny Benchmark (Banbury 외, NeurIPS 2021 Datasets & Benchmarks).
파라미터 수가 공개 수치와 맞는지 확인해 재현 여부를 검증한다.
"""
import torch
import torch.nn as nn


# ── 1. Image Classification — ResNet-8 (CIFAR-10 32x32) ─────────────────────
class Basic(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.sc = (nn.Conv2d(cin, cout, 1, stride, bias=False)
                   if (stride != 1 or cin != cout) else nn.Identity())
        self.r = nn.ReLU(inplace=False)

    def forward(self, x):
        y = self.r(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return self.r(y + self.sc(x))


class ResNet8(nn.Module):
    def __init__(self, nc=10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(16), nn.ReLU())
        self.s1 = Basic(16, 16, 1)
        self.s2 = Basic(16, 32, 2)
        self.s3 = Basic(32, 64, 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, nc)

    def forward(self, x):
        x = self.s3(self.s2(self.s1(self.stem(x))))
        return self.fc(self.pool(x).flatten(1))


# ── 2. Keyword Spotting — DS-CNN (MFCC 49x10) ───────────────────────────────
def ds_block(c):
    return nn.Sequential(
        nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False), nn.BatchNorm2d(c), nn.ReLU(),
        nn.Conv2d(c, c, 1, 1, 0, bias=False), nn.BatchNorm2d(c), nn.ReLU())


class DSCNN(nn.Module):
    def __init__(self, nc=12, c=64):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, c, (10, 4), (2, 2), (5, 1), bias=False),
                                  nn.BatchNorm2d(c), nn.ReLU())
        self.body = nn.Sequential(*[ds_block(c) for _ in range(4)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c, nc)

    def forward(self, x):
        x = self.body(self.stem(x))
        return self.fc(self.pool(x).flatten(1))


# ── 3. Visual Wake Words — MobileNetV1 0.25x @ 96x96 ────────────────────────
def dw(cin, cout, s):
    return nn.Sequential(
        nn.Conv2d(cin, cin, 3, s, 1, groups=cin, bias=False), nn.BatchNorm2d(cin), nn.ReLU(),
        nn.Conv2d(cin, cout, 1, 1, 0, bias=False), nn.BatchNorm2d(cout), nn.ReLU())


class MobileNetV1(nn.Module):
    def __init__(self, nc=2, a=0.25):
        super().__init__()
        d = lambda c: max(8, int(c * a))
        cfg = [(64, 1), (128, 2), (128, 1), (256, 2), (256, 1), (512, 2)] + \
              [(512, 1)] * 5 + [(1024, 2), (1024, 1)]
        layers = [nn.Conv2d(3, d(32), 3, 2, 1, bias=False), nn.BatchNorm2d(d(32)), nn.ReLU()]
        cin = d(32)
        for c, s in cfg:
            layers.append(dw(cin, d(c), s))
            cin = d(c)
        self.f = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(cin, nc)

    def forward(self, x):
        return self.fc(self.pool(self.f(x)).flatten(1))


# ── 4. Anomaly Detection — FC AutoEncoder (640-d 입력) ──────────────────────
class FCAE(nn.Module):
    def __init__(self, d=640, h=128, z=8):
        super().__init__()
        def blk(a, b):
            return [nn.Linear(a, b), nn.BatchNorm1d(b), nn.ReLU()]
        self.net = nn.Sequential(
            *blk(d, h), *blk(h, h), *blk(h, h), *blk(h, h), *blk(h, z),
            *blk(z, h), *blk(h, h), *blk(h, h), *blk(h, h), nn.Linear(h, d))

    def forward(self, x):
        return self.net(x)


MODELS = {
    "ResNet-8 (이미지 분류)":     (ResNet8,     (1, 3, 32, 32)),
    "DS-CNN (키워드 인식)":       (DSCNN,       (1, 1, 49, 10)),
    "MobileNetV1 0.25 (사람 감지)": (MobileNetV1, (1, 3, 96, 96)),
    "FC-AutoEncoder (이상 감지)": (FCAE,        (1, 640)),
}


if __name__ == "__main__":
    for name, (cls, shape) in MODELS.items():
        m = cls().eval()
        n = sum(p.numel() for p in m.parameters())
        with torch.no_grad():
            y = m(torch.randn(*shape))
        print(f"{name:32s} params {n:9,d}  in {tuple(shape)} → out {tuple(y.shape)}")
