"""실험 2 — 기울기에서 무엇이 새어 나오는가.

  "데이터는 기기를 떠나지 않습니다. 기울기만 보냅니다."
  이 문장이 어디까지 참인지 직접 꺼내 본다. 최적화가 아니라 나눗셈 한 번이다.

출력: leak.json (복원 이미지 픽셀 포함)
"""
import json, math
import numpy as np
import torch
import torch.nn.functional as Fn
import fed_common as F

torch.manual_seed(0)
X, Y, Xt, Yt = F.load_mnist(2000, 500)
MEAN, STD = 0.1307, 0.3081


def to255(v):
    """정규화를 되돌려 0~255 화소로."""
    a = (v.detach().reshape(28, 28) * STD + MEAN).clamp(0, 1) * 255
    return a.round().to(torch.uint8).numpy().tolist()


out = {"cfg": {"model": "MLP(784-32-10)", "params": F.n_params(F.MLP())}}
m = F.MLP()

# ══════════ ① 배치 1 — 완전 복원 ══════════
print("① 배치 1 — 기울기 하나에서 입력을 되찾는다")
x, y = X[0:1], Y[0:1]
g = F.grad_of(m, x, y)
rec = F.invert_linear(g[0], g[1])
p = F.psnr(rec, x.flatten())
err = float((rec - x.flatten()).abs().max())
print(f"   PSNR {p:.1f} dB · 최대 화소 오차 {err:.2e} · 라벨 {F.leak_label(g[3])} (정답 {int(y)})")
out["exact"] = {"psnr": None if math.isinf(p) else round(p, 2), "max_err": err,
                "label_pred": F.leak_label(g[3]), "label_true": int(y),
                "orig": to255(x), "rec": to255(rec)}

# ══════════ ② 라벨 누출은 얼마나 잘 맞나 ══════════
print("\n② 라벨 누출 — 마지막 층 편향 기울기의 부호")
ok = 0
N = 300
for i in range(N):
    gg = F.grad_of(m, X[i:i + 1], Y[i:i + 1])
    ok += int(F.leak_label(gg[3]) == int(Y[i]))
out["label_acc"] = ok / N
print(f"   {N}개 중 {ok}개 정확 — {ok/N:.1%}")

# ══════════ ③ 배치를 키우면 ══════════
print("\n③ 배치 크기를 키우면 — 개별 표본 대신 '섞인 것'이 나온다")
out["batch"] = []
for B in [1, 2, 4, 8, 16, 32]:
    xb, yb = X[:B], Y[:B]
    gb_ = F.grad_of(m, xb, yb)
    r = F.invert_linear(gb_[0], gb_[1])
    # 복원된 것은 δ 가중 평균이다 — 그것을 직접 계산해 비교한다
    with torch.no_grad():
        z = m.f1(xb.flatten(1))
        h = Fn.relu(z)
        logits = m.f2(h)
        pr = torch.softmax(logits, 1)
        dlogit = (pr - Fn.one_hot(yb, 10).float()) / B
        dh = dlogit @ m.f2.weight
        dz = dh * (z > 0).float()                    # (B, 32)
    i = int(gb_[1].abs().argmax())
    w = dz[:, i]
    mix = (w[:, None] * xb.flatten(1)).sum(0) / w.sum()
    p_first = F.psnr(r, xb[0].flatten())
    p_mix = F.psnr(r, mix)
    rec = {"B": B,
           "psnr_vs_first": None if math.isinf(p_first) else round(p_first, 2),
           "psnr_vs_mix": None if math.isinf(p_mix) else round(p_mix, 2),
           "img": to255(r)}
    out["batch"].append(rec)
    pf = "∞" if math.isinf(p_first) else f"{p_first:6.2f}"
    pm = "∞" if math.isinf(p_mix) else f"{p_mix:6.2f}"
    print(f"   B={B:3d}  개별 표본 대비 {pf} dB · 가중평균 대비 {pm} dB")

# ══════════ ④ 클리핑만으로는 아무것도 못 막는다 ══════════
print("\n④ 기울기 클리핑만 했을 때 (노이즈 없음)")
out["clip_only"] = []
for C in [10.0, 1.0, 0.1, 0.01]:
    g2 = F.grad_of(m, x, y)
    nrm = torch.sqrt(sum((t ** 2).sum() for t in g2))
    sc = min(1.0, C / float(nrm))
    g2 = [t * sc for t in g2]
    r = F.invert_linear(g2[0], g2[1])
    p2 = F.psnr(r, x.flatten())
    out["clip_only"].append({"C": C, "scale": round(sc, 6),
                             "psnr": None if math.isinf(p2) else round(p2, 2)})
    ps = "∞" if math.isinf(p2) else f"{p2:7.2f}"
    print(f"   C={C:<6} 배율 {sc:.4f} → PSNR {ps} dB")
print("   클리핑은 기울기 전체에 같은 상수를 곱한다. 나눗셈에서 그 상수가 약분된다.")

# ══════════ ⑤ 노이즈를 넣으면 ══════════
print("\n⑤ 가우시안 노이즈를 더하면 (DP-SGD 방식, C=1.0, B=1)")
C = 1.0
out["noise"] = []
for sig in [0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]:
    ps, imgs = [], None
    for t in range(5):
        torch.manual_seed(100 + t)
        g2 = F.grad_of(m, x, y)
        nrm = torch.sqrt(sum((q ** 2).sum() for q in g2))
        sc = min(1.0, C / float(nrm))
        g2 = [q * sc for q in g2]
        if sig > 0:
            g2 = [q + torch.randn_like(q) * sig * C for q in g2]
        r = F.invert_linear(g2[0], g2[1])
        ps.append(F.psnr(r, x.flatten()))
        if t == 0:
            imgs = to255(r)
    fin = [q for q in ps if not math.isinf(q)]
    med = float(np.median(fin)) if fin else float("inf")
    out["noise"].append({"sigma": sig,
                         "psnr": None if math.isinf(med) else round(med, 2),
                         "img": imgs})
    pv = "∞" if math.isinf(med) else f"{med:7.2f}"
    print(f"   σ={sig:<6} PSNR {pv} dB")

json.dump(out, open("leak.json", "w"), ensure_ascii=False)
print("\n→ leak.json")
