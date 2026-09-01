# -*- coding: utf-8 -*-
"""실험 2 — 지연은 프레임마다 다르고, 큐는 발산하며, FPS는 지연이 아니다."""
import io, json, pathlib, time
import numpy as np
from PIL import Image
import cv2
import vis_common as V

D = pathlib.Path("/root/lab09")
jpg = (D / "img" / "bus.jpg").read_bytes()
res = {}
S640 = V.session(str(D / "yolo11n_640.onnx"))
S320 = V.session(str(D / "yolo11n_320.onnx"))

# ── A. 데이터 의존 지연 — 문턱을 낮추면 무슨 일이 일어나나 ──────────────────
print("[A] 신뢰도 문턱과 후보 상자 수 — 640²")
res["conf"] = []
for conf in [0.25, 0.10, 0.05, 0.01, 0.001]:
    T = V.Timer()
    for _ in range(3):
        V.run_frame(S640, jpg, 640, conf, V.Timer())
    for _ in range(12):
        r = V.run_frame(S640, jpg, 640, conf, T)
    st = T.stats()
    row = dict(conf=conf, n_cand=r["n_cand"], n_final=r["n_final"],
               decode=st["⑤ 상자 디코딩"]["p50"], nms=st["⑥ NMS"]["p50"],
               nms_p99=st["⑥ NMS"]["p99"], infer=st["④ 추론"]["p50"],
               total=sum(v["p50"] for v in st.values()))
    res["conf"].append(row)
    print(f"  conf {conf:<6} 후보 {r['n_cand']:5d}개 → 최종 {r['n_final']:3d}개 | "
          f"디코딩 {row['decode']:6.2f} ms | NMS {row['nms']:7.3f} ms | 전체 {row['total']:6.1f} ms")
worst = res["conf"][-1]; best = res["conf"][0]
print(f"  → 후보가 {worst['n_cand']/max(best['n_cand'],1):.0f}배 늘자 NMS 가 "
      f"{worst['nms']/best['nms']:.0f}배, 전체는 {worst['total']/best['total']:.2f}배")

# ── B. 붐비는 장면을 만들어 본다 (3×3 타일) ─────────────────────────────────
print("\n[B] 같은 모델·같은 해상도인데 장면이 붐비면")
im = np.asarray(Image.open(io.BytesIO(jpg)).convert("RGB"))
h, w = im.shape[:2]
tile = np.asarray(Image.fromarray(im).resize((w // 3, h // 3), Image.BILINEAR))
crowd = np.tile(tile, (3, 3, 1))
buf = io.BytesIO(); Image.fromarray(crowd).save(buf, "JPEG", quality=90)
crowd_jpg = buf.getvalue()
(D / "img_crowd.jpg").write_bytes(crowd_jpg)
res["crowd"] = {}
for tag, jb in [("원본 (물체 5개)", jpg), ("3×3 타일 (물체 45개분)", crowd_jpg)]:
    T = V.Timer()
    for _ in range(3):
        V.run_frame(S640, jb, 640, 0.25, V.Timer())
    for _ in range(12):
        r = V.run_frame(S640, jb, 640, 0.25, T)
    st = T.stats()
    res["crowd"][tag] = dict(n_cand=r["n_cand"], n_final=r["n_final"],
                             nms=st["⑥ NMS"]["p50"], decode=st["⑤ 상자 디코딩"]["p50"],
                             infer=st["④ 추론"]["p50"],
                             total=sum(v["p50"] for v in st.values()))
    x = res["crowd"][tag]
    print(f"  {tag:22s} 후보 {x['n_cand']:4d} → 최종 {x['n_final']:3d} | "
          f"NMS {x['nms']:6.3f} ms | 추론 {x['infer']:5.1f} ms | 전체 {x['total']:5.1f} ms")

# ── C. 전처리를 최적화하면 (PIL → OpenCV) ───────────────────────────────────
print("\n[C] 전처리 구현을 바꾸면 — 640²")


def cv_pre(jb, size):
    arr = np.frombuffer(jb, np.uint8)
    im = cv2.imdecode(arr, cv2.IMREAD_COLOR)                 # BGR
    h, w = im.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    small = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((size, size, 3), 114, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    out[top:top + nh, left:left + nw] = small
    blob = cv2.dnn.blobFromImage(out, 1 / 255.0, swapRB=True)  # 정규화+축변환 한 번에
    return blob, r, left, top


res["pre"] = {}
for tag, fn in [("PIL (교재 구현)", None), ("OpenCV (최적화)", cv_pre)]:
    ts = []
    for _ in range(3 + 30):
        t0 = time.perf_counter()
        if fn is None:
            imr = V.decode(jpg); lb, *_ = V.letterbox(imr, 640); V.normalize(lb)
        else:
            fn(jpg, 640)
        ts.append((time.perf_counter() - t0) * 1000)
    a = np.array(ts[3:])
    res["pre"][tag] = dict(p50=float(np.percentile(a, 50)), p95=float(np.percentile(a, 95)))
    print(f"  {tag:18s} 전처리 p50 {np.percentile(a,50):6.2f} ms  (p95 {np.percentile(a,95):6.2f})")
sp = res["pre"]["PIL (교재 구현)"]["p50"] / res["pre"]["OpenCV (최적화)"]["p50"]
post = 0.91 + 0.10 + 0.04
t640 = res["conf"][0]["total"]
new = t640 - res["pre"]["PIL (교재 구현)"]["p50"] + res["pre"]["OpenCV (최적화)"]["p50"]
res["pre"]["speedup_stage"] = sp
res["pre"]["pipeline_before"] = t640
res["pre"]["pipeline_after"] = new
print(f"  → 전처리 단계는 {sp:.2f}배 빨라지고, 파이프라인 전체는 "
      f"{t640:.1f} → {new:.1f} ms ({t640/new:.2f}배)")

# ── D. FPS 는 지연이 아니다 — 배치를 키우면 ─────────────────────────────────
print("\n[D] 배치를 키우면 처리량은 오르고 지연은 나빠진다 — 320²")
res["batch"] = []
import onnxruntime as ort
from ultralytics import YOLO
dyn = D / "yolo11n_320_dyn.onnx"
if not dyn.exists():
    m = YOLO("yolo11n.pt")
    out = m.export(format="onnx", imgsz=320, opset=13, nms=False, dynamic=True)
    pathlib.Path(out).rename(dyn)
sd = V.session(str(dyn))
imr = V.decode(jpg); lb, *_ = V.letterbox(imr, 320); x1 = V.normalize(lb)
for B in [1, 2, 4, 8]:
    xb = np.repeat(x1, B, 0)
    for _ in range(4):
        sd.run(None, {sd.get_inputs()[0].name: xb})
    ts = []
    for _ in range(10):
        t0 = time.perf_counter()
        sd.run(None, {sd.get_inputs()[0].name: xb})
        ts.append((time.perf_counter() - t0) * 1000)
    a = np.array(ts)
    per = float(np.percentile(a, 50))
    res["batch"].append(dict(B=B, batch_ms=per, per_frame=per / B, fps=1000.0 / (per / B)))
    print(f"  배치 {B}: 한 번에 {per:6.1f} ms | 프레임당 {per/B:5.2f} ms | "
          f"처리량 {1000/(per/B):6.1f} FPS | **마지막 프레임이 기다린 시간 {per:6.1f} ms**")

json.dump(res, open(D / "realtime.json", "w"), ensure_ascii=False, default=float)
print("\n저장: realtime.json")
