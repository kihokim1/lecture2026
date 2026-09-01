# -*- coding: utf-8 -*-
"""실험 1 — 실시간 비전 파이프라인에서 시간은 어디로 가는가."""
import json, pathlib, numpy as np, torch
from ultralytics import YOLO
import vis_common as V

D = pathlib.Path("/root/lab09")
IMGS = {p.stem: p.read_bytes() for p in sorted((D / "img").glob("*.jpg"))}
SIZES = [640, 512, 416, 320, 256]
N = 30
res = {}

# ── 해상도별 ONNX 준비 ──────────────────────────────────────────────────────
for sz in SIZES:
    f = D / f"yolo11n_{sz}.onnx"
    if not f.exists():
        m = YOLO("yolo11n.pt")
        out = m.export(format="onnx", imgsz=sz, opset=13, nms=False, dynamic=False)
        pathlib.Path(out).rename(f)
print("ONNX 준비 완료:", [f"{s}²" for s in SIZES])

# ── A. 단계별 시간 (해상도 스윕) ────────────────────────────────────────────
print("\n[A] 단계별 시간 — bus.jpg · 30프레임 · 2스레드")
jpg = IMGS["bus"]
res["stages"] = {}
for sz in SIZES:
    s = V.session(str(D / f"yolo11n_{sz}.onnx"))
    T = V.Timer()
    for _ in range(6):                                   # 워밍업
        V.run_frame(s, jpg, sz, 0.25, V.Timer())
    for _ in range(N):
        r = V.run_frame(s, jpg, sz, 0.25, T)
    st = T.stats()
    tot = sum(v["p50"] for v in st.values())
    inf = st["④ 추론"]["p50"]
    res["stages"][sz] = dict(stat=st, total=tot, infer=inf,
                             other=tot - inf, n_final=r["n_final"], n_cand=r["n_cand"])
    print(f"  {sz}²  전체 {tot:6.1f} ms | 추론 {inf:6.1f} ({inf/tot*100:4.1f} %) | "
          f"나머지 {tot-inf:5.1f} ({(tot-inf)/tot*100:4.1f} %) | 상자 {r['n_final']}")

print("\n  640² 단계별 (p50):")
for k, v in res["stages"][640]["stat"].items():
    print(f"    {k:16s} {v['p50']:7.2f} ms   (p95 {v['p95']:6.2f})")

# ── B. 암달 — 모델만 빠르게 하면 파이프라인은 얼마나 빨라지나 ───────────────
print("\n[B] 암달의 법칙 — 640² 기준, 추론만 k배 빨라졌을 때 전체 배수")
t640 = res["stages"][640]["total"]; i640 = res["stages"][640]["infer"]
res["amdahl"] = []
for k in [1, 2, 4, 8, 1e9]:
    tot = (t640 - i640) + i640 / k
    res["amdahl"].append(dict(k=(None if k > 1e8 else k), total=tot, speedup=t640 / tot))
    lab = "∞ (추론 0초)" if k > 1e8 else f"{k:.0f}배"
    print(f"  추론 {lab:>12s} → 전체 {tot:6.1f} ms · {t640/tot:5.2f}배")

# ── C. 동적 양자화는 여기서 통하는가 (5주차 복습) ───────────────────────────
print("\n[C] 동적 INT8 양자화 — 640²")
q = D / "yolo11n_int8.onnx"
if q.exists():
    s = V.session(str(q)); T = V.Timer()
    for _ in range(6):
        V.run_frame(s, jpg, 640, 0.25, V.Timer())
    for _ in range(N):
        r = V.run_frame(s, jpg, 640, 0.25, T)
    st = T.stats(); tot = sum(v["p50"] for v in st.values())
    res["int8"] = dict(total=tot, infer=st["④ 추론"]["p50"], n_final=r["n_final"],
                       size=q.stat().st_size, size_fp32=(D / "yolo11n_640.onnx").stat().st_size)
    print(f"  파일 {q.stat().st_size/1e6:.1f} MB (FP32 {(D/'yolo11n_640.onnx').stat().st_size/1e6:.1f} MB) "
          f"| 추론 {res['int8']['infer']:.1f} ms (FP32 {i640:.1f}) "
          f"| {i640/res['int8']['infer']:.2f}배")

# ── D. 해상도와 물체 크기 ───────────────────────────────────────────────────
print("\n[D] 해상도를 낮추면 무엇이 먼저 사라지나 (640² 결과를 기준으로 한 상대 비교)")
res["objsize"] = {}
for name, jb in IMGS.items():
    ref = None
    row = {}
    for sz in SIZES:
        s = V.session(str(D / f"yolo11n_{sz}.onnx"))
        r = V.run_frame(s, jb, sz, 0.25, V.Timer())
        areas = ((r["boxes"][:, 2] - r["boxes"][:, 0]) *
                 (r["boxes"][:, 3] - r["boxes"][:, 1])) if len(r["boxes"]) else np.zeros(0)
        row[sz] = dict(n=len(areas),
                       small=int((areas < 32 ** 2).sum()),
                       medium=int(((areas >= 32 ** 2) & (areas < 96 ** 2)).sum()),
                       large=int((areas >= 96 ** 2).sum()))
    res["objsize"][name] = row
    print(f"  {name:22s} " + "  ".join(
        f"{sz}²:{row[sz]['n']:2d}개(소{row[sz]['small']}/중{row[sz]['medium']}/대{row[sz]['large']})"
        for sz in SIZES))

json.dump(res, open(D / "pipeline.json", "w"), ensure_ascii=False, default=float)
print("\n저장: pipeline.json")
