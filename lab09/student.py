# -*- coding: utf-8 -*-
"""3교시 실습 — 학생이 그대로 따라 치는 코드. 교재의 출력을 여기서 검증한다."""
import time, urllib.request, pathlib
import numpy as np, onnxruntime as ort
from PIL import Image
import io

D = pathlib.Path("/root/lab09")

# ── 1단계. 준비 ─────────────────────────────────────────────────────────────
sess = ort.InferenceSession(str(D / "yolo11n_640.onnx"), providers=["CPUExecutionProvider"])
IN = sess.get_inputs()[0].name
jpeg = (D / "img" / "bus.jpg").read_bytes()
print(f"입력 {sess.get_inputs()[0].shape} → 출력 {sess.get_outputs()[0].shape}")

# ── 2단계. 일곱 단계를 함수로 쪼갠다 ────────────────────────────────────────
def s1_decode(b):
    return np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))

def s2_letterbox(im, size=640):
    h, w = im.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    small = np.asarray(Image.fromarray(im).resize((nw, nh), Image.BILINEAR))
    out = np.full((size, size, 3), 114, np.uint8)
    t, l = (size - nh) // 2, (size - nw) // 2
    out[t:t + nh, l:l + nw] = small
    return out, r, l, t

def s3_normalize(im):
    return np.ascontiguousarray((im.astype(np.float32) / 255.0).transpose(2, 0, 1)[None])

def s4_infer(x):
    return sess.run(None, {IN: x})[0]

def s5_decode_boxes(pred, conf=0.25):
    p = pred[0]
    sc_all = p[4:]
    cls, sc = sc_all.argmax(0), sc_all.max(0)
    k = sc > conf
    if not k.any():
        return np.zeros((0, 4), np.float32), np.zeros(0), np.zeros(0, np.int64)
    xywh = p[:4, k].T
    xy, wh = xywh[:, :2], xywh[:, 2:]
    return np.concatenate([xy - wh / 2, xy + wh / 2], 1).astype(np.float32), sc[k], cls[k]

def s6_nms(boxes, scores, cls, iou=0.45):
    if len(boxes) == 0:
        return np.zeros(0, np.int64)
    b = boxes + cls.astype(np.float32)[:, None] * 8192.0   # 클래스별로 분리
    x1, y1, x2, y2 = b.T
    area = (x2 - x1) * (y2 - y1)
    order, keep = scores.argsort()[::-1], []
    while order.size:
        i = order[0]; keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou_v = inter / (area[i] + area[order[1:]] - inter + 1e-9)
        order = order[1:][iou_v <= iou]
    return np.array(keep, np.int64)

def s7_unletterbox(boxes, r, l, t, W, H):
    b = boxes.copy()
    b[:, [0, 2]] = ((b[:, [0, 2]] - l) / r).clip(0, W)
    b[:, [1, 3]] = ((b[:, [1, 3]] - t) / r).clip(0, H)
    return b

# ── 3단계. 단계별로 잰다 ────────────────────────────────────────────────────
def one_frame(rec=None):
    def tick(name, fn, *a):
        t0 = time.perf_counter(); out = fn(*a)
        if rec is not None:
            rec.setdefault(name, []).append((time.perf_counter() - t0) * 1000)
        return out
    im = tick("① JPEG 디코드", s1_decode, jpeg)
    H, W = im.shape[:2]
    lb, r, l, t = tick("② 레터박스", s2_letterbox, im)
    x = tick("③ 정규화", s3_normalize, lb)
    pred = tick("④ 추론", s4_infer, x)
    boxes, sc, cl = tick("⑤ 상자 디코딩", s5_decode_boxes, pred)
    k = tick("⑥ NMS", s6_nms, boxes, sc, cl)
    fin = tick("⑦ 좌표 복원", s7_unletterbox, boxes[k], r, l, t, W, H)
    return len(boxes), len(k)

for _ in range(6):
    one_frame()                      # 워밍업
rec = {}
for _ in range(30):
    n_cand, n_final = one_frame(rec)

print(f"\n후보 {n_cand}개 → 최종 {n_final}개\n")
p50 = {k: float(np.percentile(v, 50)) for k, v in rec.items()}
total = sum(p50.values())
for k, v in p50.items():
    print(f"  {k:14s} {v:7.2f} ms   ({v/total*100:5.1f} %)")
print(f"  {'합계':14s} {total:7.2f} ms   = {1000/total:.1f} FPS")

# ── 4단계. 암달의 벽 ────────────────────────────────────────────────────────
inf = p50["④ 추론"]
print("\n[암달] 추론만 빨라진다면")
for k in [2, 4, 8, None]:
    t = (total - inf) + (0 if k is None else inf / k)
    lab = "0초가 되면" if k is None else f"{k}배 빨라지면"
    print(f"  추론이 {lab:12s} → 전체 {t:5.1f} ms · {total/t:.2f}배")

# ── 5단계. 큐 시뮬레이션 ────────────────────────────────────────────────────
def simulate(svc_ms, policy, fps=30.0, dur=10.0, jitter=0.15, seed=0):
    rng = np.random.default_rng(seed)
    arr = np.arange(0, dur, 1 / fps)
    q, now, ages, dropped, i = [], 0.0, [], 0, 0
    while i < len(arr) or q:
        while i < len(arr) and arr[i] <= now:
            if policy == "latest" and q:
                dropped += len(q); q = []
            q.append(arr[i]); i += 1
        if not q:
            now = arr[i]; continue
        born = q.pop(0)
        now = max(now, born) + max(svc_ms * rng.lognormal(0, jitter), 1.0) / 1000
        ages.append((now - born) * 1000)
    a = np.array(ages)
    return len(a), dropped, float(np.percentile(a, 50)), float(a[-1])

print("\n[큐] 30 fps 카메라 · 10초")
for svc, tag in [(total, "640² (현재)"), (18.1, "320² (해상도를 낮추면)")]:
    for pol, nm in [("queue", "다 쌓는다  "), ("latest", "최신만 남긴다")]:
        n, d, a50, alast = simulate(svc, pol)
        print(f"  {tag:22s} {nm} → 처리 {n:3d}장 · 버림 {d:3d}장 | "
              f"나이 p50 {a50:7.1f} ms · 마지막 {alast:7.1f} ms")
