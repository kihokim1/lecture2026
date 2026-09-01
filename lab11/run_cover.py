"""실험 1 — 연산자 커버리지와 그래프 분할.

  "가속기가 얼마나 빠른가"가 아니라
  "이 그래프의 몇 %를, 몇 조각으로 받아 주는가"를 잰다.

출력: cover.json
"""
import json, collections
import acc_common as A

MODELS = [
    ("ResNet-18",      "resnet18.onnx"),
    ("MobileNetV2",    "/root/lab08/mbv2.onnx"),
    ("YOLO11n @320",   "/root/lab09/yolo11n_320.onnx"),
    ("Transformer 인코더", "tinyenc.onnx"),
]

out = {"models": []}

for name, path in MODELS:
    print("=" * 60)
    print(name)
    m = A.load(path)
    hist = A.op_hist(m)
    n_all = len(m.graph.node)
    n_real = sum(hist.values())
    feed = A.feed_for(path)

    # 그래프 최적화를 끄고 잰다 — 분할은 '원본 그래프' 위에서 하므로
    # 융합된 그래프의 시간을 원본 노드에 매기면 어긋난다.
    times, _ = A.profile(path, feed, runs=6, level="none")

    rec = {
        "name": name, "path": path,
        "n_all": n_all, "n_real": n_real,
        "hist": dict(hist.most_common()),
        "times": {k: round(v, 4) for k, v in
                  sorted(times.items(), key=lambda kv: -kv[1])},
        "total_ms": round(sum(times.values()), 4),
        "profiles": {},
    }
    print(f"  노드 {n_all}개 중 실계산 {n_real}개 · 총 {rec['total_ms']:.2f} ms")
    print(f"  상위 op: {hist.most_common(6)}")

    for k in "ABC":
        r = A.partition(m, A.PROFILES[k], times=times)
        rs = A.partition(m, A.PROFILES[k], policy="sticky", times=times)
        r["switches_sticky"] = rs["switches"]
        r["xbytes_sticky"] = rs["xbytes"]
        # 무한히 빠른 가속기의 천장 (암달) — 시간 커버리지 기준
        r["ceiling"] = 1.0 / (1.0 - r["time_cov"]) if r["time_cov"] < 1 else float("inf")
        rec["profiles"][k] = r
        print(f"  {k}형: 노드 {r['node_cov']:6.1%} · 시간 {r['time_cov']:6.1%} · "
              f"천장 {r['ceiling']:5.2f}× · 왕복 {r['switches']:3d}"
              f"(정렬개선 {rs['switches']:3d}) · 교차 {r['xbytes']/1048576:6.1f} MB")
        if r["unsupported"]:
            print(f"        미지원 상위: {r['unsupported'][:4]}")

    # ── 커버리지 곡선 — 시간이 큰 op 부터 하나씩 허용목록에 넣는다
    order = [k for k, _ in sorted(times.items(), key=lambda kv: -kv[1])
             if k in hist]
    curve, allow = [], set()
    for op in order:
        allow.add(op)
        r = A.partition(m, allow, times=times)
        curve.append({
            "op": op, "k": len(allow),
            "node_cov": round(r["node_cov"], 5),
            "time_cov": round(r["time_cov"], 5),
            "switches": r["switches"],
            "xmb": round(r["xbytes"] / 1048576, 3),
        })
    rec["curve"] = curve
    print("  곡선(연산자 종류 수 → 시간커버리지 / 왕복):")
    for c in curve:
        print(f"        +{c['op']:<18} k={c['k']:2d} 노드 {c['node_cov']:6.1%} "
              f"시간 {c['time_cov']:6.1%} 왕복 {c['switches']:3d}")
    out["models"].append(rec)

json.dump(out, open("cover.json", "w"), ensure_ascii=False, indent=1)
print("\n→ cover.json")
