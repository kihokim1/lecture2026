"""실험 1 — 연합 학습은 무엇을 잃는가 (비-IID · 통신량).

  ① 데이터를 모았을 때(중앙집중) 대비 얼마나 손해인가
  ② 데이터가 쏠리면(비-IID) 얼마나 무너지는가
  ③ 목표 정확도에 닿기까지 몇 MB 를 올려야 하는가

출력: fed.json
"""
import json, time
import numpy as np
import torch
import fed_common as F

N_TR, N_TE = 8000, 2000
N_CL, ROUNDS = 10, 30
X, Y, Xt, Yt = F.load_mnist(N_TR, N_TE)
out = {"cfg": {"n_train": N_TR, "n_test": N_TE, "clients": N_CL, "rounds": ROUNDS,
               "model": "SmallCNN", "params": F.n_params(F.SmallCNN()),
               "bytes": F.model_bytes(F.SmallCNN())}}
print(f"모델 {out['cfg']['params']:,} 파라미터 · {out['cfg']['bytes']/1024:.1f} KB")

# ══════════ ① 중앙집중 기준선 ══════════
print("\n① 중앙집중 (데이터를 한곳에 모았을 때)")
t0 = time.time()
_, ch = F.centralized(X, Y, Xt, Yt, epochs=ROUNDS, lr=0.1)
out["central"] = ch
print(f"   {ROUNDS}에폭 후 정확도 {ch[-1]['acc']:.4f}  ({time.time()-t0:.0f}s)")

# ══════════ ② 비-IID 정도를 바꿔 가며 ══════════
print("\n② FedAvg — 디리클레 α 로 쏠림을 조절")
out["alpha"] = []
for a in [100.0, 1.0, 0.5, 0.1]:
    parts = F.dirichlet_split(Y, N_CL, a, seed=1)
    skew = F.label_skew(parts, Y)
    sizes = [len(p) for p in parts]
    print(f"   α={a:<6}  클라이언트당 클래스 수 {skew}  표본 수 {sizes}")
    t0 = time.time()
    _, h = F.fedavg(X, Y, Xt, Yt, parts, rounds=ROUNDS, epochs=1, lr=0.1,
                    seed=1, log_every=5)
    out["alpha"].append({"alpha": a, "classes_per_client": skew, "sizes": sizes,
                         "hist": h, "final": h[-1]["acc"], "sec": round(time.time() - t0, 1)})
    print(f"      → 최종 {h[-1]['acc']:.4f}  ({time.time()-t0:.0f}s)")

# ══════════ ③ 로컬 에폭 — 통신 대 계산의 교환 ══════════
print("\n③ 로컬 에폭 수를 늘리면 라운드가 줄어든다 (α=0.5)")
parts = F.dirichlet_split(Y, N_CL, 0.5, seed=1)
out["epochs"] = []
for e, r in [(1, 30), (5, 30)]:
    t0 = time.time()
    _, h = F.fedavg(X, Y, Xt, Yt, parts, rounds=r, epochs=e, lr=0.1, seed=1,
                    log_every=5, verbose=False)
    out["epochs"].append({"local_epochs": e, "rounds": r, "hist": h,
                          "final": h[-1]["acc"], "up_mb": h[-1]["up_mb"],
                          "sec": round(time.time() - t0, 1)})
    print(f"   E={e}  최종 {h[-1]['acc']:.4f}  업로드 {h[-1]['up_mb']:.1f} MB  "
          f"로컬 계산 {time.time()-t0:.0f}s")

# ══════════ ④ 희소화 — 상위 k%만 보내기 ══════════
print("\n④ 업데이트의 상위 k%만 보내면 (α=0.5, E=1)")
out["topk"] = []
for k in [1.0, 0.1, 0.01]:
    t0 = time.time()
    _, h = F.fedavg(X, Y, Xt, Yt, parts, rounds=ROUNDS, epochs=1, lr=0.1, seed=1,
                    topk=k, log_every=10, verbose=False)
    out["topk"].append({"topk": k, "hist": h, "final": h[-1]["acc"],
                        "up_mb": h[-1]["up_mb"]})
    print(f"   상위 {k*100:5.1f}%  최종 {h[-1]['acc']:.4f}  업로드 {h[-1]['up_mb']:7.2f} MB")

# ══════════ ⑤ 목표 정확도까지의 통신량 ══════════
TARGET = 0.90
print(f"\n⑤ 정확도 {TARGET:.0%}에 닿기까지 올려야 하는 바이트")
rows = []
for tag, h in ([("FedAvg E=1", out["epochs"][0]["hist"]),
                ("FedAvg E=5", out["epochs"][1]["hist"])] +
               [(f"상위 {r['topk']*100:.0f}%", r["hist"]) for r in out["topk"]]):
    hit = next((p for p in h if p["acc"] >= TARGET), None)
    rows.append({"name": tag, "round": hit["round"] if hit else None,
                 "up_mb": round(hit["up_mb"], 3) if hit else None,
                 "final": h[-1]["acc"]})
    if hit:
        print(f"   {tag:<12} r{hit['round']:3d} · {hit['up_mb']:7.2f} MB")
    else:
        print(f"   {tag:<12} 도달 못 함 (최종 {h[-1]['acc']:.4f})")
out["to_target"] = {"target": TARGET, "rows": rows}

# ══════════ ⑥ 현실 규모로 환산 ══════════
B = out["cfg"]["bytes"]
out["scale"] = [
    {"clients": c, "rounds": r,
     "gb": round(2 * B * c * r / 1e9, 3)}
    for c, r in [(10, 30), (100, 100), (1000, 100), (10000, 500)]
]
print("\n⑥ 현실 규모 총 통신량 (다운로드+업로드)")
for s in out["scale"]:
    print(f"   클라이언트 {s['clients']:>6,} × {s['rounds']:>4} 라운드 → {s['gb']:9.2f} GB")

json.dump(out, open("fed.json", "w"), ensure_ascii=False, indent=1)
print("\n→ fed.json")
