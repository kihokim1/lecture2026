"""3교시 학생용 — 기울기 하나에서 이미지를 꺼낸다.

연합 학습이 서버로 보내는 것은 '데이터'가 아니라 '기울기'다.
그 기울기로 무엇을 할 수 있는지 직접 해 본다. 여든 줄이면 된다.

    python3 student.py
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torchvision import datasets, transforms

torch.set_num_threads(2)
torch.manual_seed(0)

# ── ① 데이터와 모델 ────────────────────────────────────────────
tf = transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize((0.1307,), (0.3081,))])
ds = datasets.MNIST("/root/lab12/data", train=True, download=True, transform=tf)
X = torch.stack([ds[i][0] for i in range(64)])
Y = torch.tensor([ds[i][1] for i in range(64)])


class MLP(nn.Module):
    """첫 층이 원본 입력을 그대로 받는다 — 이게 전부의 원인이다."""
    def __init__(s, h=32):
        super().__init__()
        s.f1 = nn.Linear(784, h)
        s.f2 = nn.Linear(h, 10)

    def forward(s, x):
        return s.f2(Fn.relu(s.f1(x.flatten(1))))


m = MLP()
print(f"① MLP {sum(p.numel() for p in m.parameters()):,} 파라미터 · MNIST 한 장")


def grads(x, y):
    """이 기기가 서버로 보내는 것 — 데이터가 아니라 이것이다."""
    return [g.detach() for g in
            torch.autograd.grad(Fn.cross_entropy(m(x), y), list(m.parameters()))]


def psnr(a, b):
    mse = ((a.flatten() - b.flatten()) ** 2).mean().item()
    return float("inf") if mse < 1e-20 else \
        20 * math.log10((a.max() - a.min()).item() / math.sqrt(mse))


def invert(gW, gb):
    """dL/dW = (dL/dz)·xᵀ,  dL/db = dL/dz  →  x = dL/dW[i,:] / dL/db[i]"""
    i = int(gb.abs().argmax())          # 나눗셈이 가장 안정적인 행
    return gW[i] / gb[i]


# ── ② 나눗셈 한 번 ──────────────────────────────────────────────
x, y = X[0:1], Y[0:1]
g = grads(x, y)
xh = invert(g[0], g[1])
print(f"\n② 복원 PSNR {psnr(xh, x):.1f} dB · 최대 화소 오차 {float((xh-x.flatten()).abs().max()):.1e}")
print("   반복 최적화가 아닙니다. 나눗셈 한 번입니다.")

# ── ③ 라벨도 샌다 ───────────────────────────────────────────────
ok = sum(int(grads(X[i:i+1], Y[i:i+1])[3].argmin()) == int(Y[i]) for i in range(50))
print(f"\n③ 라벨 복원 {ok}/50 — 마지막 층 편향 기울기가 음수인 자리가 정답입니다")

# ── ④ 배치를 키우면? ────────────────────────────────────────────
print("\n④ 배치를 키우면 — 누출이 사라질까요, 대상이 바뀔까요")
for B in [1, 4, 16, 32]:
    gb = grads(X[:B], Y[:B])
    r = invert(gb[0], gb[1])
    print(f"   B={B:3d}  개별 표본 대비 {psnr(r, X[0]):6.2f} dB")
print("   개별 표본과는 멀어졌습니다. 그런데 무엇과는 가까워졌을까요? (교재 2.2)")

# ── ⑤ 클리핑하면? ───────────────────────────────────────────────
print("\n⑤ 기울기를 클리핑하면")
for C in [1.0, 0.01]:
    g2 = grads(x, y)
    n = torch.sqrt(sum((t ** 2).sum() for t in g2))
    s = min(1.0, C / float(n))
    g2 = [t * s for t in g2]
    print(f"   C={C:<6} 배율 {s:.5f} → PSNR {psnr(invert(g2[0], g2[1]), x):7.2f} dB")
print("   배율이 분자와 분모에서 약분됩니다. 방어력은 정확히 0입니다.")

# ── ⑥ 노이즈를 넣으면? ──────────────────────────────────────────
print("\n⑥ 가우시안 노이즈를 더하면 (C=1.0)")
for sig in [0.0, 0.001, 0.01, 1.0]:
    torch.manual_seed(1)
    g2 = grads(x, y)
    n = torch.sqrt(sum((t ** 2).sum() for t in g2))
    g2 = [t * min(1.0, 1.0 / float(n)) for t in g2]
    if sig:
        g2 = [t + torch.randn_like(t) * sig for t in g2]
    print(f"   σ={sig:<6} PSNR {psnr(invert(g2[0], g2[1]), x):7.2f} dB")

# ── ⑦ 판정 ─────────────────────────────────────────────────────
print("""
⑦ 여기서 멈추지 마십시오.
   σ=0.01 이면 이 공격은 죽습니다. 정확도 손실도 거의 없습니다.
   그런데 그 σ 의 ε 은 1,422,867 입니다 (교재 2.3).
   "이 공격을 막았다" 와 "보장을 갖는다" 는 다른 명제입니다.""")
