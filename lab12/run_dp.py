"""실험 3 — 프라이버시의 값 (차등 프라이버시).

2교시에서 기울기 하나로 이미지를 완전히 복원했다. 노이즈를 넣으면 막힌다.
그러면 그 노이즈는 정확도를 얼마나 깎는가? 그리고 ε 은 얼마인가?

중요 — 이 실험은 **표본 단위 클리핑**을 쓴다(torch.func.vmap).
배치 단위로 자르면 DP-SGD 가 아니고 형식적 보장도 없다.

출력: dp.json
"""
import json, math, time
import numpy as np
import torch
import torch.nn.functional as Fn
from torch.func import functional_call, vmap, grad
import fed_common as F

torch.set_num_threads(2)
N_TR, N_TE, N_CL = 8000, 2000, 10
ROUNDS, BS, LR, ALPHA = 20, 64, 0.25, 1.0
CLIP = 1.0
X, Y, Xt, Yt = F.load_mnist(N_TR, N_TE)
parts = F.dirichlet_split(Y, N_CL, ALPHA, seed=1)
print(f"MLP {F.n_params(F.MLP()):,} 파라미터 · 클라이언트 {N_CL}대 · α={ALPHA} · {ROUNDS}라운드")

_proto = F.MLP()


def make_gradfn(model):
    def loss_fn(p, b, xi, yi):
        o = functional_call(model, (p, b), (xi.unsqueeze(0),))
        return Fn.cross_entropy(o, yi.unsqueeze(0))
    return vmap(grad(loss_fn), in_dims=(None, None, 0, 0))


def dp_local(model, gfn, X, Y, idx, lr, bs, clip, sigma, gen):
    """표본 단위 클리핑 + 가우시안 노이즈 — 진짜 DP-SGD."""
    prm = {k: v.detach() for k, v in model.named_parameters()}
    n = len(idx)
    steps = 0
    perm = torch.randperm(n, generator=gen)
    for s in range(0, n, bs):
        b = idx[perm[s:s + bs].numpy()]
        xb, yb = X[b], Y[b]
        B = len(b)
        g = gfn(prm, {}, xb, yb)                       # 표본별 기울기
        # 표본별 전체 노름으로 클리핑
        sq = sum((v.reshape(B, -1) ** 2).sum(1) for v in g.values())
        fac = (clip / (sq.sqrt() + 1e-6)).clamp(max=1.0)
        upd = {}
        for k, v in g.items():
            vv = (v.reshape(B, -1) * fac[:, None]).sum(0)
            if sigma > 0:
                vv = vv + torch.randn(vv.shape, generator=gen) * sigma * clip
            upd[k] = (vv / B).reshape(prm[k].shape)
        for k in prm:
            prm[k] = prm[k] - lr * upd[k]
        steps += 1
    with torch.no_grad():
        for k, v in model.named_parameters():
            v.data.copy_(prm[k])
    return steps


def run(sigma, seed=1):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    glob = F.MLP()
    gfn = make_gradfn(glob)
    hist, steps_per_client = [], 0
    for r in range(1, ROUNDS + 1):
        gv = F.get_flat(glob)
        agg = torch.zeros_like(gv); tot = 0
        for c in range(N_CL):
            if len(parts[c]) < BS:
                continue
            loc = F.MLP(); F.set_flat(loc, gv)
            st = dp_local(loc, make_gradfn(loc), X, Y, parts[c], LR, BS, CLIP, sigma, gen)
            if c == 0:
                steps_per_client = st
            w = len(parts[c])
            agg += (F.get_flat(loc) - gv) * w; tot += w
        if tot:
            F.set_flat(glob, gv + agg / tot)
        if r % 5 == 0 or r == ROUNDS:
            hist.append({"round": r, "acc": F.evaluate(glob, Xt, Yt)})
    return hist, steps_per_client


def epsilon(sigma, sample_rate, steps, delta=1e-5):
    if sigma <= 0:
        return None
    try:
        from opacus.accountants import RDPAccountant
        a = RDPAccountant()
        for _ in range(steps):
            a.step(noise_multiplier=sigma, sample_rate=sample_rate)
        return float(a.get_epsilon(delta=delta))
    except Exception as e:
        print("   (ε 계산 실패:", e, ")")
        return None


LEAK = json.load(open("leak.json"))
leak_by_sigma = {r["sigma"]: r["psnr"] for r in LEAK["noise"]}

out = {"cfg": {"model": "MLP(784-32-10)", "params": F.n_params(F.MLP()),
               "clients": N_CL, "rounds": ROUNDS, "batch": BS, "lr": LR,
               "alpha": ALPHA, "clip": CLIP, "delta": 1e-5,
               "note": "표본 단위 클리핑(torch.func.vmap) · ε 은 클라이언트 로컬 DP 기준"},
       "rows": []}

print("\nσ 를 바꿔 가며 — 정확도 · ε · 복원 PSNR")
BASE = None
for sig in [0.0, 0.01, 0.1, 0.5, 1.0, 2.0]:
    t0 = time.time()
    h, st = run(sig)
    acc = h[-1]["acc"]
    if sig == 0.0:
        BASE = acc
    n_cl0 = len(parts[0])
    sr = min(1.0, BS / max(n_cl0, 1))
    eps = epsilon(sig, sr, ROUNDS * st)
    row = {"sigma": sig, "acc": acc, "hist": h,
           "eps": None if eps is None else round(eps, 2),
           "acc_drop": None if BASE is None else round(BASE - acc, 4),
           "leak_psnr": leak_by_sigma.get(sig),
           "sec": round(time.time() - t0, 1)}
    out["rows"].append(row)
    es = "보장 없음" if eps is None else f"{eps:9.2f}"
    lp = row["leak_psnr"]
    ls = "복원 완전" if (lp is None or lp > 100) else f"{lp:6.2f} dB"
    print(f"   σ={sig:<5} 정확도 {acc:.4f} (−{(BASE-acc)*100:5.2f}%p) · ε {es} · 복원 {ls}"
          f"  ({time.time()-t0:.0f}s)")

# 클리핑만 했을 때의 비용 (σ=0 행이 그것이다)
out["clip_only_acc"] = out["rows"][0]["acc"]
out["steps_per_client"] = ROUNDS * st
out["sample_rate"] = sr

print("\n요약 — 공격을 막는 것과 보장을 갖는 것은 다르다")
for r in out["rows"]:
    if r["sigma"] == 0:
        continue
    print(f"   σ={r['sigma']:<5} 복원 {'파괴됨' if (r['leak_psnr'] or 0) < 15 else '살아있음'}"
          f" · ε={r['eps']} · 정확도 손실 {r['acc_drop']*100:.2f}%p")

json.dump(out, open("dp.json", "w"), ensure_ascii=False, indent=1)
print("\n→ dp.json")
