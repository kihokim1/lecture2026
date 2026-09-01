# -*- coding: utf-8 -*-
"""실시간 비전 파이프라인을 **단계별로 쪼개서** 직접 만든다.

모델은 YOLO11n(ONNX, 원시 출력 1×84×N). 디코딩과 NMS 를 우리가 직접 짜야
그 비용을 따로 잴 수 있다. 라이브러리가 다 해 주면 어디에 시간이 갔는지 못 본다.
"""
import time
import numpy as np
import onnxruntime as ort
from PIL import Image
import io

# ── 단계 0. JPEG 디코드 ─────────────────────────────────────────────────────
def decode(jpeg_bytes):
    """카메라/파일에서 온 JPEG 바이트 → RGB 배열 (H, W, 3) uint8."""
    return np.asarray(Image.open(io.BytesIO(jpeg_bytes)).convert("RGB"))


# ── 단계 1. 레터박스 리사이즈 ───────────────────────────────────────────────
def letterbox(im, size):
    """가로세로 비를 유지한 채 size×size 에 맞추고 남는 곳을 회색으로 채운다."""
    h, w = im.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    small = np.asarray(Image.fromarray(im).resize((nw, nh), Image.BILINEAR))
    out = np.full((size, size, 3), 114, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    out[top:top + nh, left:left + nw] = small
    return out, r, left, top


def naive_resize(im, size):
    """비를 무시하고 늘려 버리는 구현 — 비교용."""
    return np.asarray(Image.fromarray(im).resize((size, size), Image.BILINEAR))


# ── 단계 2. 정규화 · 축 변환 ────────────────────────────────────────────────
def normalize(im):
    """HWC uint8 → NCHW float32 [0,1]. 여기가 의외로 비싸다."""
    x = im.astype(np.float32) / 255.0
    x = x.transpose(2, 0, 1)[None]
    return np.ascontiguousarray(x)


# ── 단계 4. 디코딩 (원시 출력 → 상자) ───────────────────────────────────────
def decode_boxes(pred, conf):
    """pred: (1, 4+C, N) → (남은 상자, 점수, 클래스). 상자는 xyxy."""
    p = pred[0]                             # (84, N)
    scores_all = p[4:]                      # (80, N)
    cls = scores_all.argmax(0)
    scores = scores_all.max(0)
    keep = scores > conf
    if not keep.any():
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64)
    xywh = p[:4, keep].T
    xy, wh = xywh[:, :2], xywh[:, 2:]
    boxes = np.concatenate([xy - wh / 2, xy + wh / 2], 1)
    return boxes.astype(np.float32), scores[keep], cls[keep]


# ── 단계 5. NMS (직접 구현 — 비용을 보려면 직접 짜야 한다) ──────────────────
def nms(boxes, scores, iou_thr=0.45):
    """고전적인 탐욕 NMS. 남은 상자 수에 대해 대략 제곱으로 커진다."""
    if len(boxes) == 0:
        return np.zeros(0, np.int64)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return np.array(keep, np.int64)


def nms_per_class(boxes, scores, cls, iou_thr=0.45):
    """클래스별로 따로 억제한다(표준 방식). 좌표에 클래스 오프셋을 더해 한 번에."""
    if len(boxes) == 0:
        return np.zeros(0, np.int64)
    off = cls.astype(np.float32)[:, None] * 8192.0
    return nms(boxes + off, scores, iou_thr)


# ── 단계 6. 좌표 되돌리기 ───────────────────────────────────────────────────
def unletterbox(boxes, r, left, top, W, H):
    b = boxes.copy()
    b[:, [0, 2]] = (b[:, [0, 2]] - left) / r
    b[:, [1, 3]] = (b[:, [1, 3]] - top) / r
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, W)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, H)
    return b


# ── 세션 ────────────────────────────────────────────────────────────────────
def session(path, threads=2):
    o = ort.SessionOptions()
    o.intra_op_num_threads = threads
    o.inter_op_num_threads = 1
    return ort.InferenceSession(path, o, providers=["CPUExecutionProvider"])


class Timer:
    """단계별 시간을 모으는 도구. 나노초 해상도 타이머를 쓴다."""
    def __init__(self):
        self.rec = {}

    def __call__(self, name):
        return _Ctx(self, name)

    def add(self, name, dt):
        self.rec.setdefault(name, []).append(dt * 1000.0)

    def stats(self):
        out = {}
        for k, v in self.rec.items():
            a = np.array(v)
            out[k] = dict(mean=a.mean(), p50=np.percentile(a, 50),
                          p95=np.percentile(a, 95), p99=np.percentile(a, 99),
                          mx=a.max(), n=len(a))
        return out


class _Ctx:
    def __init__(self, t, name):
        self.t, self.name = t, name

    def __enter__(self):
        self.s = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.t.add(self.name, time.perf_counter() - self.s)
        return False


def run_frame(sess, jpeg, size, conf, T, iou=0.45):
    """한 프레임을 끝까지 처리하며 각 단계를 잰다."""
    with T("① JPEG 디코드"):
        im = decode(jpeg)
    H, W = im.shape[:2]
    with T("② 레터박스 리사이즈"):
        lb, r, left, top = letterbox(im, size)
    with T("③ 정규화·축 변환"):
        x = normalize(lb)
    with T("④ 추론"):
        pred = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    with T("⑤ 상자 디코딩"):
        boxes, scores, cls = decode_boxes(pred, conf)
    n_cand = len(boxes)
    with T("⑥ NMS"):
        k = nms_per_class(boxes, scores, cls, iou)
    with T("⑦ 좌표 복원"):
        final = unletterbox(boxes[k], r, left, top, W, H) if len(k) else boxes[k]
    return dict(n_cand=n_cand, n_final=len(k), boxes=final,
                scores=scores[k] if len(k) else scores, cls=cls[k] if len(k) else cls)
