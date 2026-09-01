"""실험 1-보충 — 병리적 비-IID (FedAvg 논문 자신의 분할 방식).

디리클레 α 로는 MNIST 가 잘 안 무너진다. FedAvg 논문(McMahan 외, AISTATS 2017 §3)이
쓴 분할을 그대로 재현한다 — 라벨로 정렬해 조각으로 자르고 클라이언트마다 두 조각씩.
"각 클라이언트가 숫자 두 개만 갖는" 상태다.

출력: patho.json
"""
import json, time
import numpy as np
import torch
import fed_common as F

N_TR, N_TE, N_CL, ROUNDS = 8000, 2000, 10, 30
X, Y, Xt, Yt = F.load_mnist(N_TR, N_TE)


def shard_split(Y, n_clients, shards_per_client=2, seed=1):
    """라벨로 정렬 → 균등 조각 → 클라이언트마다 무작위로 k조각."""
    g = np.random.RandomState(seed)
    order = np.argsort(Y.numpy(), kind="stable")
    n_sh = n_clients * shards_per_client
    shards = np.array_split(order, n_sh)
    pick = g.permutation(n_sh)
    return [np.concatenate([shards[s] for s in pick[i * shards_per_client:(i + 1) * shards_per_client]])
            for i in range(n_clients)]


out = {"cfg": {"clients": N_CL, "rounds": ROUNDS, "n_train": N_TR}}
print("병리적 비-IID — FedAvg 논문 §3 의 조각 분할")
out["rows"] = []
for k in [2, 1]:
    parts = shard_split(Y, N_CL, k)
    skew = F.label_skew(parts, Y)
    print(f"\n  클라이언트당 조각 {k}개 → 보유 클래스 수 {skew}")
    t0 = time.time()
    _, h = F.fedavg(X, Y, Xt, Yt, parts, rounds=ROUNDS, epochs=1, lr=0.1,
                    seed=1, log_every=5, verbose=False)
    out["rows"].append({"shards": k, "classes_per_client": skew,
                        "sizes": [len(p) for p in parts], "hist": h,
                        "final": h[-1]["acc"]})
    print(f"     최종 정확도 {h[-1]['acc']:.4f}  ({time.time()-t0:.0f}s)")
    for p in h:
        print(f"       r{p['round']:3d}  {p['acc']:.4f}")

json.dump(out, open("patho.json", "w"), ensure_ascii=False, indent=1)
print("\n→ patho.json")
