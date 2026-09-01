"""실험 2 — 컴파일러가 하는 일 (융합 · 레이아웃 · 빌드 시간).

가속기 툴체인은 '미리 컴파일'한다. 우리는 GPU/NPU 가 없지만,
ONNX Runtime 의 그래프 최적화 단계가 정확히 같은 종류의 일을 한다.
끄고 켜 가며 재면 컴파일러가 무엇을 하는지 직접 보인다.

출력: compile.json
"""
import json, os, collections
import acc_common as A

MODELS = [
    ("ResNet-18",         "resnet18.onnx"),
    ("MobileNetV2",       "/root/lab08/mbv2.onnx"),
    ("YOLO11n @320",      "/root/lab09/yolo11n_320.onnx"),
    ("Transformer 인코더", "tinyenc.onnx"),
]
LEVELS = ["none", "basic", "ext", "all"]
LEVEL_KR = {"none": "끔", "basic": "기본", "ext": "확장", "all": "전부"}

out = {"models": [], "levels": LEVELS}
os.makedirs("opt", exist_ok=True)

for name, path in MODELS:
    print("=" * 62)
    print(name)
    feed = A.feed_for(path)
    rec = {"name": name, "levels": {}}
    base_hist = A.op_hist(A.load(path, infer=False))
    rec["orig_nodes"] = sum(base_hist.values())
    rec["orig_hist"] = dict(base_hist.most_common())

    for lv in LEVELS:
        t = A.session_ms(path, lv, feed)
        o = f"opt/{name.split()[0].replace('-', '')}_{lv}.onnx"
        try:
            h = A.optimized_graph(path, lv, o)
        except Exception as e:
            h = collections.Counter()
            print("   (최적화 그래프 덤프 실패:", e, ")")
        t["nodes"] = sum(h.values())
        t["hist"] = dict(h.most_common(14))
        rec["levels"][lv] = t
        print(f"  {LEVEL_KR[lv]:>3}: 노드 {t['nodes']:4d} · 빌드 {t['build_ms']:8.1f} ms · "
              f"첫추론 {t['first_ms']:7.2f} ms · 정상 {t['steady_ms']:6.2f} ms")

    n0 = rec["levels"]["none"]["nodes"]
    n1 = rec["levels"]["all"]["nodes"]
    s0 = rec["levels"]["none"]["steady_ms"]
    s1 = rec["levels"]["all"]["steady_ms"]
    b0 = rec["levels"]["none"]["build_ms"]
    b1 = rec["levels"]["all"]["build_ms"]
    rec["fuse_ratio"] = n0 / n1 if n1 else None
    rec["speedup"] = s0 / s1 if s1 else None
    rec["build_ratio"] = b1 / b0 if b0 else None
    # 컴파일 비용을 몇 번 추론해야 회수하나
    gain = s0 - s1
    rec["amortize_runs"] = (b1 - b0) / gain if gain > 0 else None
    print(f"  → 노드 {n0}→{n1} ({rec['fuse_ratio']:.2f}배 축약) · "
          f"속도 {rec['speedup']:.2f}배 · 빌드 {rec['build_ratio']:.2f}배 · "
          f"회수 {rec['amortize_runs'] if rec['amortize_runs'] is None else round(rec['amortize_runs'])}회")

    # 사라진 op / 새로 생긴 op — 융합의 직접 증거
    h0 = rec["levels"]["none"]["hist"]
    ha = A.op_hist(A.load(f"opt/{name.split()[0].replace('-', '')}_all.onnx", infer=False))
    gone = {k: v for k, v in base_hist.items() if ha.get(k, 0) < v}
    new = {k: v for k, v in ha.items() if base_hist.get(k, 0) < v}
    rec["gone"] = dict(sorted(gone.items(), key=lambda kv: -kv[1]))
    rec["new"] = dict(sorted(new.items(), key=lambda kv: -kv[1]))
    print(f"     줄거나 사라진 op: {list(rec['gone'].items())[:6]}")
    print(f"     새로 생긴 op    : {list(rec['new'].items())[:6]}")
    out["models"].append(rec)

json.dump(out, open("compile.json", "w"), ensure_ascii=False, indent=1)
print("\n→ compile.json")
