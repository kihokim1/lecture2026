"""실험 3 — 경계 비용 (왕복세).

분할된 그래프는 경계마다 텐서를 옮기고, 레이아웃을 바꾸고, 동기화한다.
우리는 NPU 가 없지만 이 세 가지 비용의 '실체'는 CPU 에서 직접 잴 수 있다.

  ① 복사 대역폭 — 실측
  ② 레이아웃 변환(NCHW↔NHWC) — 실제 경계 텐서 모양으로 실측
  ③ ONNX Runtime 이 실제로 끼워 넣은 ReorderInput/ReorderOutput 노드의 실측 시간
     (가속기가 아니라 CPU 인데도 레이아웃 변환 노드가 51개 생긴다)

그리고 ①②를 써서 "가속기가 몇 배 빨라야 이식이 이득인가"를 푼다.

출력: boundary.json
"""
import json, time, collections
import numpy as np
import onnxruntime as ort
import acc_common as A

out = {}

# ══════════ ① 복사 대역폭 ══════════
print("① 복사 대역폭")
bw = []
for mb in [1, 4, 16, 64]:
    n = mb * 1024 * 1024 // 4
    a = np.random.rand(n).astype(np.float32)
    b = np.empty_like(a)
    for _ in range(3):
        np.copyto(b, a)
    ts = []
    for _ in range(15):
        t0 = time.perf_counter()
        np.copyto(b, a)
        ts.append(time.perf_counter() - t0)
    ms = float(np.median(ts)) * 1000
    bw.append({"mb": mb, "ms": round(ms, 4), "gbps": round(mb / 1024 / (ms / 1000), 2)})
    print(f"   {mb:3d} MB · {ms:7.3f} ms · {bw[-1]['gbps']:6.2f} GB/s")
out["copy"] = bw
GBPS = float(np.median([r["gbps"] for r in bw]))
out["copy_gbps"] = round(GBPS, 3)
print(f"   → 중앙값 {GBPS:.2f} GB/s")

# ══════════ ② 레이아웃 변환 ══════════
print("\n② 레이아웃 변환 NCHW→NHWC (실제 특징맵 모양)")
lay = []
for shp in [(1, 16, 160, 160), (1, 32, 80, 80), (1, 64, 40, 40),
            (1, 128, 20, 20), (1, 256, 10, 10), (1, 3, 320, 320)]:
    a = np.random.rand(*shp).astype(np.float32)
    for _ in range(3):
        np.ascontiguousarray(a.transpose(0, 2, 3, 1))
    ts = []
    for _ in range(20):
        t0 = time.perf_counter()
        np.ascontiguousarray(a.transpose(0, 2, 3, 1))
        ts.append(time.perf_counter() - t0)
    ms = float(np.median(ts)) * 1000
    mb = a.nbytes / 1048576
    lay.append({"shape": list(shp), "mb": round(mb, 4), "ms": round(ms, 4),
                "gbps": round(mb / 1024 / (ms / 1000), 2)})
    print(f"   {str(shp):>20} {mb:6.2f} MB · {ms:7.3f} ms · {lay[-1]['gbps']:6.2f} GB/s")
out["layout"] = lay
LGBPS = float(np.median([r["gbps"] for r in lay]))
out["layout_gbps"] = round(LGBPS, 3)
out["layout_penalty"] = round(GBPS / LGBPS, 2)
print(f"   → 중앙값 {LGBPS:.2f} GB/s · 단순 복사 대비 {GBPS/LGBPS:.2f}배 느림")

# ══════════ ③ ORT 가 실제로 끼워 넣은 레이아웃 노드 ══════════
print("\n③ ORT 가 그래프에 끼워 넣은 레이아웃 변환 노드 (실측)")
reo = []
for name, path in [("MobileNetV2", "/root/lab08/mbv2.onnx"),
                   ("YOLO11n @320", "/root/lab09/yolo11n_320.onnx"),
                   ("ResNet-18", "resnet18.onnx")]:
    feed = A.feed_for(path)
    t, _ = A.profile(path, feed, runs=6, level="all")
    tot = sum(t.values())
    r = {k: round(v, 4) for k, v in t.items() if "Reorder" in k or "Transpose" in k}
    rms = sum(r.values())
    reo.append({"name": name, "total_ms": round(tot, 4),
                "reorder_ms": round(rms, 4),
                "share": round(rms / tot, 5) if tot else 0,
                "nodes": r})
    print(f"   {name:14} 총 {tot:6.2f} ms 중 레이아웃 {rms:5.2f} ms ({rms/tot:5.1%}) {r}")
out["reorder"] = reo

# ══════════ ④ INT8 요구 — QDQ 경계세 ══════════
print("\n④ INT8 전용 가속기가 요구하는 QDQ 노드")
qm = A.load("/root/lab09/yolo11n_int8.onnx")
qh = A.op_hist(qm)
fm = A.load("/root/lab09/yolo11n_320.onnx")
fh = A.op_hist(fm)
# 동적 양자화가 실제로 만들어 낸 노드들
QOPS = ["DynamicQuantizeLinear", "QuantizeLinear", "DequantizeLinear",
        "ConvInteger", "MatMulInteger", "Cast"]
nq = sum(qh.get(o, 0) for o in QOPS)
try:
    qfeed = A.feed_for("/root/lab09/yolo11n_int8.onnx")
    qt, _ = A.profile("/root/lab09/yolo11n_int8.onnx", qfeed, runs=6, level="none")
    qms = sum(qt.get(o, 0.0) for o in QOPS if o != "ConvInteger")
    qtot = sum(qt.values())
except Exception as e:
    qt, qms, qtot = {}, 0, 0
    print("   (INT8 프로파일 실패:", e, ")")
out["qdq"] = {
    "fp32_nodes": sum(fh.values()), "int8_nodes": sum(qh.values()),
    "qdq_nodes": nq, "qdq_share_nodes": round(nq / sum(qh.values()), 4),
    "qdq_ms": round(qms, 4), "total_ms": round(qtot, 4),
    "qdq_share_time": round(qms / qtot, 5) if qtot else 0,
    "hist": dict(qh.most_common(14)),
    "fp32_hist": dict(fh.most_common(14)),
    "added_nodes": sum(qh.values()) - sum(fh.values()),
    "node_ratio": round(sum(qh.values()) / sum(fh.values()), 3),
    "qtimes": {k: round(v, 4) for k, v in sorted(qt.items(), key=lambda kv: -kv[1])[:12]},
}
print(f"   FP32 {sum(fh.values())}노드 → INT8 {sum(qh.values())}노드 · "
      f"그중 QDQ {nq}개({nq/sum(qh.values()):.1%}) · 시간 {qms:.2f}/{qtot:.2f} ms ({out['qdq']['qdq_share_time']:.1%})")

# ══════════ ⑤ 손익분기 — 가속기가 몇 배 빨라야 이득인가 ══════════
print("\n⑤ 손익분기 (cover.json 의 실측 분할 결과 + 위 대역폭)")
cov = json.load(open("cover.json"))
rows = []
for m in cov["models"]:
    for k in "ABC":
        p = m["profiles"][k]
        T = m["total_ms"]                       # CPU 전체 시간(ms)
        cov_t = p["time_cov"]
        sw = p["switches"]
        mb = p["xbytes"] / 1048576
        if sw == 0:
            c_ms = 0.0
        else:
            # 경계 1회 비용 = (그 경계를 넘는 바이트를 옮기고 레이아웃을 바꾸는 시간)
            c_ms = (mb / sw) / 1024 / LGBPS * 1000
        def sp(S):
            t = T * (1 - cov_t) + (T * cov_t / S if S != float("inf") else 0.0) + sw * c_ms
            return T / t if t > 1e-9 else float("inf")
        row = {
            "model": m["name"], "profile": k,
            "T_ms": round(T, 3), "time_cov": round(cov_t, 4), "switches": sw,
            "xmb": round(mb, 3), "cost_per_switch_ms": round(c_ms, 5),
            "boundary_ms": round(sw * c_ms, 3),
            "sp2": round(sp(2), 3), "sp5": round(sp(5), 3),
            "sp10": round(sp(10), 3),
            "spinf": (None if sp(float("inf")) == float("inf") else round(sp(float("inf")), 3)),
            "amdahl_only": round(1 / (1 - cov_t), 3) if cov_t < 1 else None,
        }
        # 왕복이 이득을 전부 먹는 지점 (S=∞ 에서 speedup=1)
        row["k_max"] = round(T * cov_t / c_ms) if c_ms > 0 else None
        rows.append(row)
        print(f"   {m['name']:>18} {k}형: 커버 {cov_t:5.1%} 왕복 {sw:3d} "
              f"경계 {sw*c_ms:6.2f} ms | 천장 암달 {row['amdahl_only']} → 실제 {row['spinf']} "
              f"(2배 {row['sp2']} · 10배 {row['sp10']}) k_max {row['k_max']}")
out["breakeven"] = rows

json.dump(out, open("boundary.json", "w"), ensure_ascii=False, indent=1)
print("\n→ boundary.json")
