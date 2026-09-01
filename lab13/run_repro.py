"""실험 2 — 우리 자신의 실험은 얼마나 흔들리는가.

열두 주 동안 "재현을 요구할 것은 배수가 아니라 구조다" 라고 반복해 말했다.
그 말을 우리 코드로 검증한다.

  A. 같은 코드 · 같은 시드 · 같은 기계 — 반복만 한다  (시간이 얼마나 흔들리나)
  B. 같은 코드 · 다른 시드                            (배수가 얼마나 흔들리나)

두 경우 모두 **구조량**(노드 수 · 왕복 횟수 · 라벨 복원률)과
**규모량**(ms · 커버리지 · PSNR)을 나란히 재서 변동계수(CV)를 비교한다.

출력: repro.json
"""
import json, math, statistics as st, sys, time
import numpy as np

sys.path.insert(0, "/root/lab11")
sys.path.insert(0, "/root/lab12")

N_A, N_B = 8, 8
out = {"cfg": {"runs_A": N_A, "runs_B": N_B}}


def cv(xs):
    """변동계수 (%) — 표준편차 ÷ 평균."""
    xs = [float(x) for x in xs]
    m = st.mean(xs)
    if abs(m) < 1e-12:
        return 0.0
    return 100 * (st.pstdev(xs) / abs(m))


def summarize(name, xs, unit=""):
    xs = [float(x) for x in xs]
    r = {"name": name, "unit": unit, "n": len(xs),
         "mean": round(st.mean(xs), 6), "sd": round(st.pstdev(xs), 6),
         "min": round(min(xs), 6), "max": round(max(xs), 6),
         "cv": round(cv(xs), 4),
         "spread": round(max(xs) - min(xs), 6),
         "ratio": round(max(xs) / min(xs), 4) if min(xs) > 0 else None,
         "values": [round(x, 6) for x in xs]}
    return r


# ══════════════════════════════════════════════════════════════════
# A. 같은 코드 · 같은 시드 · 반복만 — 11주차 가속기 커버리지
# ══════════════════════════════════════════════════════════════════
print("A. 같은 코드·같은 시드·같은 기계 — 8회 반복 (11주차 커버리지 측정)")
import acc_common as A

PATH = "/root/lab08/mbv2.onnx"
m = A.load(PATH)
nodes = A.real_nodes(m)
allow = A.PROFILES["A"]
feed = A.feed_for(PATH)

rows = {"n_real": [], "node_cov": [], "switches": [], "xmb": [],
        "total_ms": [], "time_cov": [], "ceiling": []}
for k in range(N_A):
    t, _ = A.profile(PATH, feed, runs=6, level="none")
    r = A.partition(m, allow, times=t)
    rows["n_real"].append(r["n_nodes"])
    rows["node_cov"].append(r["node_cov"] * 100)
    rows["switches"].append(r["switches"])
    rows["xmb"].append(r["xbytes"] / 1048576)
    rows["total_ms"].append(r["time_total_ms"])
    rows["time_cov"].append(r["time_cov"] * 100)
    c = 1 / (1 - r["time_cov"]) if r["time_cov"] < 1 else float("inf")
    rows["ceiling"].append(c)
    print(f"   {k+1}회  노드 {r['n_nodes']} · 커버리지 {r['node_cov']:.1%} · 왕복 {r['switches']} · "
          f"총 {r['time_total_ms']:6.2f} ms · 시간커버 {r['time_cov']:.2%} · 천장 {c:.2f}배")

out["A"] = {
    "desc": "같은 코드·같은 시드·같은 기계, 8회 반복 (MobileNetV2 · A형 허용목록)",
    "structural": [summarize("실계산 노드 수", rows["n_real"], "개"),
                   summarize("노드 커버리지", rows["node_cov"], "%"),
                   summarize("왕복 횟수", rows["switches"], "회"),
                   summarize("경계 바이트", rows["xmb"], "MB")],
    "magnitude": [summarize("총 추론 시간", rows["total_ms"], "ms"),
                  summarize("시간 커버리지", rows["time_cov"], "%"),
                  summarize("암달 천장", rows["ceiling"], "배")],
}

# ══════════════════════════════════════════════════════════════════
# B. 같은 코드 · 다른 시드 — 12주차 기울기 역복원
# ══════════════════════════════════════════════════════════════════
print("\nB. 같은 코드·다른 시드 — 8회 (12주차 기울기 역복원)")
import torch
import fed_common as FC

X, Y, _, _ = FC.load_mnist(600, 100)
rowsB = {"psnr1": [], "maxerr": [], "label50": [], "psnr_b32_first": [],
         "psnr_b32_mix": [], "psnr_clip": [], "psnr_noise": []}
for k in range(N_B):
    torch.manual_seed(1000 + k)
    mm = FC.MLP()
    x, y = X[0:1], Y[0:1]
    g = FC.grad_of(mm, x, y)
    rec = FC.invert_linear(g[0], g[1])
    p1 = FC.psnr(rec, x.flatten())
    rowsB["psnr1"].append(p1 if not math.isinf(p1) else 200.0)
    rowsB["maxerr"].append(float((rec - x.flatten()).abs().max()))
    ok = sum(int(FC.leak_label(FC.grad_of(mm, X[i:i+1], Y[i:i+1])[3])) == int(Y[i])
             for i in range(50))
    rowsB["label50"].append(ok)
    # 배치 32
    gb = FC.grad_of(mm, X[:32], Y[:32])
    rb = FC.invert_linear(gb[0], gb[1])
    rowsB["psnr_b32_first"].append(FC.psnr(rb, X[0].flatten()))
    import torch.nn.functional as Fn
    with torch.no_grad():
        z = mm.f1(X[:32].flatten(1)); h = Fn.relu(z)
        pr = torch.softmax(mm.f2(h), 1)
        dl = (pr - Fn.one_hot(Y[:32], 10).float()) / 32
        dz = (dl @ mm.f2.weight) * (z > 0).float()
    i = int(gb[1].abs().argmax()); w = dz[:, i]
    mix = (w[:, None] * X[:32].flatten(1)).sum(0) / w.sum()
    pm = FC.psnr(rb, mix)
    rowsB["psnr_b32_mix"].append(pm if not math.isinf(pm) else 200.0)
    # 클리핑 C=0.01
    g2 = FC.grad_of(mm, x, y)
    nrm = torch.sqrt(sum((q ** 2).sum() for q in g2))
    g2 = [q * min(1.0, 0.01 / float(nrm)) for q in g2]
    pc = FC.psnr(FC.invert_linear(g2[0], g2[1]), x.flatten())
    rowsB["psnr_clip"].append(pc if not math.isinf(pc) else 200.0)
    # 노이즈 σ=0.01
    g3 = FC.grad_of(mm, x, y)
    nrm = torch.sqrt(sum((q ** 2).sum() for q in g3))
    g3 = [q * min(1.0, 1.0 / float(nrm)) for q in g3]
    torch.manual_seed(5000 + k)
    g3 = [q + torch.randn_like(q) * 0.01 for q in g3]
    rowsB["psnr_noise"].append(FC.psnr(FC.invert_linear(g3[0], g3[1]), x.flatten()))
    print(f"   시드 {1000+k}  복원 {rowsB['psnr1'][-1]:6.1f} dB · 라벨 {ok}/50 · "
          f"B32(가중평균) {rowsB['psnr_b32_mix'][-1]:6.1f} · 클리핑 {rowsB['psnr_clip'][-1]:6.1f} · "
          f"노이즈 {rowsB['psnr_noise'][-1]:5.1f}")

out["B"] = {
    "desc": "같은 코드·다른 시드 8회 (MLP 784-32-10 · 기울기 역복원)",
    "structural": [summarize("라벨 복원 (50장 중)", rowsB["label50"], "장")],
    "magnitude": [summarize("복원 PSNR (B=1)", rowsB["psnr1"], "dB"),
                  summarize("최대 화소 오차", rowsB["maxerr"], ""),
                  summarize("배치 32 · 가중평균 대비", rowsB["psnr_b32_mix"], "dB"),
                  summarize("클리핑 C=0.01", rowsB["psnr_clip"], "dB"),
                  summarize("노이즈 σ=0.01", rowsB["psnr_noise"], "dB")],
}

# ══════════════════════════════════════════════════════════════════
# 판정 — 구조량과 규모량의 CV 를 가른다
# ══════════════════════════════════════════════════════════════════
S = [r for g in ("A", "B") for r in out[g]["structural"]]
M = [r for g in ("A", "B") for r in out[g]["magnitude"]]
out["verdict"] = {
    "structural_cv_max": round(max(r["cv"] for r in S), 4),
    "magnitude_cv_max": round(max(r["cv"] for r in M), 4),
    "magnitude_cv_mean": round(st.mean([r["cv"] for r in M]), 4),
    "n_structural_exact": sum(1 for r in S if r["cv"] == 0.0),
    "n_structural": len(S),
}

print("\n" + "=" * 78)
print(f"{'양':<26}{'평균':>12}{'최소~최대':>22}{'CV':>10}")
print("-" * 78)
def line(r):
    rng = "{:.3f} ~ {:.3f}".format(r["min"], r["max"])
    print("  {:<24}{:>12.3f}{:>22}{:>9.2f}%".format(r["name"], r["mean"], rng, r["cv"]))


print("  [구조량]")
for r in S:
    line(r)
print("  [규모량]")
for r in M:
    line(r)
v = out["verdict"]
print("-" * 78)
print(f"  구조량 CV 최대 {v['structural_cv_max']}%  ({v['n_structural_exact']}/{v['n_structural']} 개는 완전히 동일)")
print(f"  규모량 CV 최대 {v['magnitude_cv_max']}% · 평균 {v['magnitude_cv_mean']}%")

json.dump(out, open("repro.json", "w"), ensure_ascii=False, indent=1)
print("\n→ repro.json")
