# -*- coding: utf-8 -*-
"""실험 3 — 큐는 발산하고, 구현은 알고리즘보다 크게 작용한다."""
import json, pathlib, time
import numpy as np
import vis_common as V

D = pathlib.Path("/root/lab09")
res = {}

# ── A. 큐 시뮬레이션 — 카메라는 기다려 주지 않는다 ──────────────────────────
# 실측 지연(p50/p95)을 그대로 넣는다.
P = json.load(open(D / "pipeline.json"))
LAT = {int(k): (v["stat"]["④ 추론"]["p50"], v["total"],
                sum(x["p95"] for x in v["stat"].values()))
       for k, v in P["stages"].items()}
FPS_IN, DUR = 30.0, 10.0        # 30 fps 카메라, 10초


def simulate(total_p50, total_p95, policy, fps=FPS_IN, dur=DUR, seed=0):
    """policy: 'queue'(다 쌓는다) 또는 'latest'(최신만 남기고 버린다)."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / fps
    arrivals = np.arange(0, dur, dt)
    # 지연은 프레임마다 흔들린다 — p50~p95 사이의 로그정규 근사
    sigma = max((total_p95 - total_p50) / 1.645 / max(total_p50, 1e-9), 1e-3)
    q, now, ages, done, dropped = [], 0.0, [], 0, 0
    i = 0
    while i < len(arrivals) or q:
        while i < len(arrivals) and arrivals[i] <= now:
            if policy == "latest" and q:
                dropped += len(q); q = []
            q.append(arrivals[i]); i += 1
        if not q:
            now = arrivals[i]; continue
        born = q.pop(0)
        svc = max(total_p50 * float(rng.lognormal(0, sigma)), 1.0) / 1000.0
        start = max(now, born)
        now = start + svc
        ages.append((now - born) * 1000.0)      # 글래스-투-글래스 지연
        done += 1
    a = np.array(ages)
    return dict(processed=done, dropped=dropped,
                age_p50=float(np.percentile(a, 50)), age_p95=float(np.percentile(a, 95)),
                age_max=float(a.max()), age_last=float(a[-1]),
                fps_out=done / dur, ages=a.tolist())


print("[A] 30 fps 카메라 · 10초 — 큐 정책에 따라 무슨 일이 일어나나")
res["queue"] = {}
for sz in [640, 320]:
    _, t50, t95 = LAT[sz]
    for pol, nm in [("queue", "다 쌓는다"), ("latest", "최신만 남긴다")]:
        r = simulate(t50, t95, pol)
        res["queue"][f"{sz}_{pol}"] = {k: v for k, v in r.items() if k != "ages"}
        res["queue"][f"{sz}_{pol}"]["ages"] = r["ages"]
        print(f"  {sz}² ({t50:5.1f} ms) · {nm:12s} → 처리 {r['processed']:3d}장 "
              f"({r['fps_out']:5.1f} fps) · 버림 {r['dropped']:3d}장 | "
              f"프레임 나이 p50 {r['age_p50']:7.1f} ms · 마지막 {r['age_last']:8.1f} ms")

_, t50_640, t95_640 = LAT[640]
res["queue"]["input_fps"] = FPS_IN
res["queue"]["capacity_640"] = 1000.0 / t50_640
res["queue"]["capacity_320"] = 1000.0 / LAT[320][1]
print(f"  → 640² 의 처리 능력은 {1000/t50_640:.1f} fps, 입력은 30 fps. "
      f"모자란 만큼 큐가 매초 {30 - 1000/t50_640:.1f}장씩 쌓인다.")

# ── B. 같은 알고리즘, 다른 구현 ─────────────────────────────────────────────
print("\n[B] NMS — 같은 알고리즘을 두 가지로 짜면")


def nms_naive(boxes, scores, cls, iou_thr=0.45):
    """교과서를 그대로 옮긴 파이썬 이중 루프. 결과는 같고 속도만 다르다."""
    keep = []
    for c in np.unique(cls):
        idx = [int(i) for i in np.where(cls == c)[0]]
        idx.sort(key=lambda i: -scores[i])
        while idx:
            i = idx.pop(0)
            keep.append(i)
            rest = []
            for j in idx:
                ax1, ay1, ax2, ay2 = boxes[i]
                bx1, by1, bx2, by2 = boxes[j]
                ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
                iy = max(0.0, min(ay2, by2) - max(ay1, by1))
                inter = ix * iy
                ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
                if inter / (ua + 1e-9) <= iou_thr:
                    rest.append(j)
            idx = rest
    return np.array(sorted(keep), np.int64)


S = V.session(str(D / "yolo11n_640.onnx"))
jpg = (D / "img" / "bus.jpg").read_bytes()
im = V.decode(jpg); lb, *_ = V.letterbox(im, 640); x = V.normalize(lb)
pred = S.run(None, {S.get_inputs()[0].name: x})[0]
res["nms_impl"] = []
for conf in [0.25, 0.01, 0.001]:
    b, sc, cl = V.decode_boxes(pred, conf)
    tv, tn = [], []
    for _ in range(7):
        t0 = time.perf_counter(); kv = V.nms_per_class(b, sc, cl); tv.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); kn = nms_naive(b, sc, cl); tn.append(time.perf_counter() - t0)
    v, n = np.median(tv) * 1000, np.median(tn) * 1000
    same = len(kv) == len(kn)
    res["nms_impl"].append(dict(conf=conf, n=len(b), vec=v, naive=n, ratio=n / v,
                                keep_vec=len(kv), keep_naive=len(kn), same=bool(same)))
    print(f"  후보 {len(b):4d}개 | 벡터화 {v:7.3f} ms | 파이썬 루프 {n:8.3f} ms | "
          f"{n/v:6.1f}배 | 남은 상자 {len(kv)} vs {len(kn)}")

json.dump(res, open(D / "queue.json", "w"), ensure_ascii=False, default=float)
print("\n저장: queue.json")
