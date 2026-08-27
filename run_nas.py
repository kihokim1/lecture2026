# -*- coding: utf-8 -*-
"""7주차 실측 — 탐색 공간 48개 후보에 대해 파라미터·FLOPs·실측 지연·정확도를 각각 잰다."""
import json, os, time
import numpy as np
import torch
from nas_common import (SPACE, KEYS, space_size, sample, geno_str, Candidate,
                        n_params, flops_of, latency_ms, train, load, pareto_front)

OUT = "/root/ondevice-ai/lab07/nas.json"
N = 48
EPOCHS = 14
R = json.load(open(OUT)) if os.path.exists(OUT) else {"cands": []}

xtr, ytr, xte, yte = load(n_train=10000)
print(f"탐색 공간 {space_size()}개 중 {N}개 표본 · 학습 {len(xtr)}장 / 테스트 {len(xte)}장", flush=True)

# 중복 없이 N개 뽑는다
rng = np.random.RandomState(7)
seen, genos = set(), []
while len(genos) < N:
    g = sample(rng)
    key = tuple(g[k] for k in KEYS)
    if key in seen:
        continue
    seen.add(key); genos.append(g)

done = {c["geno"] for c in R["cands"]}
t_all = time.time()
for i, g in enumerate(genos):
    gs = geno_str(g)
    if gs in done:
        continue
    torch.manual_seed(1000 + i)          # ← 모델 초기값까지 고정 (6주차 교훈)
    m = Candidate(g)
    p, f = n_params(m), flops_of(g)
    lat = latency_ms(m, xte)
    t0 = time.time()
    torch.manual_seed(1000 + i)
    m2 = Candidate(g)
    acc = train(m2, xtr, ytr, xte, yte, epochs=EPOCHS, seed=0)
    R["cands"].append({"i": i, "geno": gs, **g, "params": p, "macs": f,
                       "lat_ms": round(lat, 4), "acc": round(acc, 2),
                       "train_s": round(time.time() - t0, 1)})
    json.dump(R, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(f"[{len(R['cands']):2d}/{N}] {gs:44s} p={p:8,} MACs={f:10,} "
          f"lat={lat:6.3f} acc={acc:5.2f}  ({time.time()-t0:.0f}s)", flush=True)

print(f"\n전체 {time.time()-t_all:.0f}s", flush=True)

# ───────── 분석 ─────────
try:
    from scipy.stats import spearmanr, pearsonr
except Exception:
    spearmanr = pearsonr = None

C = R["cands"]
mac = np.array([c["macs"] for c in C], float)
par = np.array([c["params"] for c in C], float)
lat = np.array([c["lat_ms"] for c in C], float)
acc = np.array([c["acc"] for c in C], float)

ana = {}
if spearmanr:
    ana["spearman"] = {
        "MACs~지연": round(float(spearmanr(mac, lat).statistic), 3),
        "파라미터~지연": round(float(spearmanr(par, lat).statistic), 3),
        "MACs~정확도": round(float(spearmanr(mac, acc).statistic), 3),
        "지연~정확도": round(float(spearmanr(lat, acc).statistic), 3),
    }
    ana["pearson"] = {
        "MACs~지연": round(float(pearsonr(mac, lat).statistic), 3),
        "파라미터~지연": round(float(pearsonr(par, lat).statistic), 3),
    }

# sep 계열과 std 계열을 나눠 보면 왜 어긋나는지 드러난다
for tag, mask in [("sep", np.array([c["sep"] for c in C])),
                  ("std", ~np.array([c["sep"] for c in C]))]:
    if mask.sum() >= 3:
        ana.setdefault("group", {})[tag] = {
            "n": int(mask.sum()),
            "MACs_중앙값": int(np.median(mac[mask])),
            "지연_중앙값_ms": round(float(np.median(lat[mask])), 3),
            "ms_per_MMAC": round(float(np.median(lat[mask] / (mac[mask] / 1e6))), 3),
        }
        if spearmanr and mask.sum() >= 5:
            ana["group"][tag]["spearman_MACs~지연"] = round(float(spearmanr(mac[mask], lat[mask]).statistic), 3)

# FLOPs 는 적은데 더 느린 쌍 — 가장 극적인 것
inv = []
for a in range(len(C)):
    for b in range(len(C)):
        if mac[a] < mac[b] and lat[a] > lat[b]:
            inv.append((float(lat[a] / lat[b]), float(mac[b] / mac[a]), C[a]["geno"], C[b]["geno"]))
inv.sort(reverse=True)
ana["역전_쌍_상위"] = [{"느린쪽": x[2], "빠른쪽": x[3],
                     "지연_배수": round(x[0], 2), "MACs_배수": round(x[1], 2)} for x in inv[:5]]
ana["역전_쌍_비율"] = round(100.0 * len(inv) / (len(C) * (len(C) - 1)), 1)

# 파레토 프론티어
pf = pareto_front([(C[i]["lat_ms"], C[i]["acc"], i) for i in range(len(C))])
ana["pareto"] = [{"geno": C[i]["geno"], "lat_ms": l, "acc": a,
                  "macs": C[i]["macs"], "params": C[i]["params"]} for l, a, i in pf]

# 정확도만 최적화 vs 지연 예산 아래에서 최적화
best_acc = int(np.argmax(acc))
ana["정확도만_최적"] = {"geno": C[best_acc]["geno"], "acc": C[best_acc]["acc"],
                    "lat_ms": C[best_acc]["lat_ms"], "macs": C[best_acc]["macs"]}
for budget in [0.15, 0.20, 0.30]:
    ok = np.where(lat <= budget)[0]
    if len(ok):
        j = ok[int(np.argmax(acc[ok]))]
        ana.setdefault("예산별_최적", {})[f"{budget:.2f}ms"] = {
            "geno": C[j]["geno"], "acc": C[j]["acc"], "lat_ms": C[j]["lat_ms"],
            "정확도_손실_%p": round(float(acc[best_acc] - acc[j]), 2),
            "지연_이득_배": round(float(C[best_acc]["lat_ms"] / C[j]["lat_ms"]), 2)}

# MACs 만 보고 고르면? (MACs 최소 후보의 실제 지연 순위)
j = int(np.argmin(mac))
ana["MACs_최소_후보"] = {"geno": C[j]["geno"], "macs": C[j]["macs"], "lat_ms": C[j]["lat_ms"],
                     "acc": C[j]["acc"],
                     "지연_순위": int((lat < lat[j]).sum()) + 1, "전체": len(C)}
k = int(np.argmin(lat))
ana["지연_최소_후보"] = {"geno": C[k]["geno"], "macs": C[k]["macs"], "lat_ms": C[k]["lat_ms"],
                    "acc": C[k]["acc"], "MACs_순위": int((mac < mac[k]).sum()) + 1}

R["analysis"] = ana
json.dump(R, open(OUT, "w"), indent=1, ensure_ascii=False)
print(json.dumps(ana, ensure_ascii=False, indent=1), flush=True)
