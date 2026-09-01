"""11주차 공통 모듈 — 가속기 연산자 커버리지 · 그래프 분할 · 경계 비용.

핵심 질문은 "가속기가 얼마나 빠른가"가 아니라
"가속기가 이 그래프의 무엇을, 몇 조각으로 받아 주는가"이다.
"""
import json, os, time, collections
import numpy as np
import onnx
from onnx import shape_inference
import onnxruntime as ort

# ══════════════════════════════════════════════════════════════════
# 가속기 프로파일 — 허용 연산자 목록(allowlist)
#
#  실제 벤더 목록을 그대로 옮기지 않는다. 벤더 목록은 SDK 판마다 바뀌고
#  공개 문서와도 어긋나는 일이 잦기 때문이다. 대신 "세대별 능력 등급"을
#  세 가지로 정의한다. 이 등급은 재현 가능하고, 수업에서 학생이 자기
#  가속기의 문서를 보고 직접 고쳐 쓸 수 있다.
# ══════════════════════════════════════════════════════════════════
PROFILE_A = {  # 합성곱 코어 — 1세대 NPU/DLA 급
    "Conv", "Relu", "MaxPool", "AveragePool", "GlobalAveragePool",
    "Add", "Gemm", "MatMul", "Flatten", "Reshape", "BatchNormalization",
}
PROFILE_B = PROFILE_A | {  # + 원소별 · 형상 조작
    "Mul", "Sigmoid", "Concat", "Split", "Resize", "Sub", "Div",
    "Clip", "HardSigmoid", "HardSwish", "LeakyRelu", "Pad", "ConvTranspose",
}
PROFILE_C = PROFILE_B | {  # + 축약 · 인덱싱 · 자료형
    "Transpose", "Slice", "Softmax", "ReduceMean", "ReduceMax", "ReduceSum",
    "Pow", "Sqrt", "Exp", "Erf", "Where", "Expand", "Gather", "Shape",
    "Unsqueeze", "Squeeze", "Cast", "Constant", "ConstantOfShape", "Identity",
}
PROFILES = {"A": PROFILE_A, "B": PROFILE_B, "C": PROFILE_C}
PROFILE_LABEL = {
    "A": "A형 · 합성곱 코어",
    "B": "B형 · 합성곱+원소별",
    "C": "C형 · 광범위",
}

BPE = {  # onnx TensorProto elem_type → bytes
    1: 4, 2: 1, 3: 1, 4: 2, 5: 2, 6: 4, 7: 8, 9: 1,
    10: 2, 11: 8, 12: 4, 13: 8, 16: 2,
}


# ══════════════════════════════════════════════════════════════════
# 그래프 읽기
# ══════════════════════════════════════════════════════════════════
def load(path, infer=True):
    m = onnx.load(path)
    if infer:
        try:
            m = shape_inference.infer_shapes(m, strict_mode=False)
        except Exception:
            pass
    return m


def initializer_names(model):
    return {i.name for i in model.graph.initializer}


def real_nodes(model):
    """상수 노드를 뺀 '실제로 계산하는' 노드만 돌려준다.

    Constant / 상수만 먹는 Identity 는 런타임에 사라진다. 이걸 세면
    커버리지가 부풀려진다 — 8주차에서 겪은 것과 같은 함정이다.
    """
    const = initializer_names(model)
    const |= {n.output[0] for n in model.graph.node if n.op_type == "Constant"}
    out = []
    for n in model.graph.node:
        if n.op_type == "Constant":
            continue
        ins = [i for i in n.input if i]
        if n.op_type == "Identity" and ins and all(i in const for i in ins):
            const.add(n.output[0])
            continue
        out.append(n)
    return out


def op_hist(model):
    c = collections.Counter(n.op_type for n in real_nodes(model))
    return c


def tensor_bytes(model):
    """그래프의 모든 값이름 → 바이트 수. shape 를 모르면 None."""
    out = {}
    vis = list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output)
    for v in vis:
        t = v.type.tensor_type
        if not t.elem_type:
            continue
        n = 1
        ok = True
        for d in t.shape.dim:
            if d.HasField("dim_value") and d.dim_value > 0:
                n *= d.dim_value
            else:
                ok = False
                break
        out[v.name] = n * BPE.get(t.elem_type, 4) if ok else None
    return out


# ══════════════════════════════════════════════════════════════════
# 위상 정렬 — 타이브레이크 정책이 왕복 횟수를 바꾼다
# ══════════════════════════════════════════════════════════════════
def topo(nodes, model, policy="default", dev=None):
    """Kahn 위상 정렬.

    policy="default"  원래 노드 순서를 유지 (ONNX 파일 순서 = 내보낸 순서)
    policy="sticky"   준비된 노드 중 '지금 장치와 같은 장치' 노드를 먼저 —
                      컴파일러의 스케줄링이 하는 일과 같다.
    """
    idx = {id(n): k for k, n in enumerate(nodes)}
    producer = {}
    for n in nodes:
        for o in n.output:
            if o:
                producer[o] = n
    preds = {id(n): set() for n in nodes}
    succs = {id(n): set() for n in nodes}
    for n in nodes:
        for i in n.input:
            p = producer.get(i)
            if p is not None and id(p) != id(n):
                preds[id(n)].add(id(p))
                succs[id(p)].add(id(n))
    byid = {id(n): n for n in nodes}
    indeg = {k: len(v) for k, v in preds.items()}
    ready = [k for k, d in indeg.items() if d == 0]
    order, cur = [], None
    while ready:
        if policy == "sticky" and cur is not None:
            same = [k for k in ready if dev and dev.get(k if isinstance(k,int) else id(byid[k])) == cur]
            pick = min(same or ready, key=lambda k: idx[k])
        else:
            pick = min(ready, key=lambda k: idx[k])
        ready.remove(pick)
        order.append(byid[pick])
        cur = dev.get(pick) if dev else None
        for s in succs[pick]:
            indeg[s] -= 1
            if indeg[s] == 0:
                ready.append(s)
    return order


# ══════════════════════════════════════════════════════════════════
# 분할 — 허용목록으로 그래프를 자른다
# ══════════════════════════════════════════════════════════════════
def partition(model, allow, policy="default", times=None):
    """가속기 허용목록으로 그래프를 분할한다.

    돌려주는 것:
      n_nodes      실제 계산 노드 수
      n_acc        가속기가 받아 주는 노드 수
      node_cov     노드 커버리지
      time_cov     시간 커버리지 (times 가 있을 때, 실측 기반)
      switches     호스트↔가속기 전환 횟수 = 왕복 횟수
      blocks       연속 실행 블록 목록 [(dev, 노드수), ...]
      xbytes       경계를 넘는 텐서의 총 바이트 (중복 제거)
      unsupported  미지원 op 별 (개수, 시간ms)
    """
    nodes = real_nodes(model)
    dev = {id(n): ("acc" if n.op_type in allow else "cpu") for n in nodes}
    order = topo(nodes, model, policy, dev)

    blocks, switches, cur = [], 0, None
    for n in order:
        d = dev[id(n)]
        if d != cur:
            if cur is not None:
                switches += 1
            blocks.append([d, 0])
            cur = d
        blocks[-1][1] += 1

    # 경계를 넘는 텐서
    tb = tensor_bytes(model)
    producer = {}
    for n in nodes:
        for o in n.output:
            if o:
                producer[o] = n
    crossed, unknown = set(), 0
    for n in nodes:
        for i in n.input:
            p = producer.get(i)
            if p is not None and dev[id(p)] != dev[id(n)]:
                crossed.add(i)
    xbytes = 0
    for t in crossed:
        b = tb.get(t)
        if b is None:
            unknown += 1
        else:
            xbytes += b

    n_acc = sum(1 for n in nodes if dev[id(n)] == "acc")
    res = {
        "n_nodes": len(nodes),
        "n_acc": n_acc,
        "node_cov": n_acc / len(nodes),
        "switches": switches,
        "n_blocks": len(blocks),
        "acc_blocks": sum(1 for b in blocks if b[0] == "acc"),
        "xtensors": len(crossed),
        "xbytes": xbytes,
        "xunknown": unknown,
    }
    if times:
        # times 는 op_type 별 '합계' ms 이므로 노드마다 더하면 안 된다.
        present = {n.op_type for n in nodes}
        tot = sum(v for k, v in times.items() if k in present)
        acc = sum(v for k, v in times.items() if k in present and k in allow)
        res["time_excluded_ms"] = sum(v for k, v in times.items() if k not in present)
        res["time_total_ms"] = tot
        res["time_acc_ms"] = acc
        res["time_cov"] = acc / tot if tot else 0.0
        un = collections.Counter()
        for n in nodes:
            if dev[id(n)] == "cpu":
                un[n.op_type] += 1
        res["unsupported"] = sorted(
            ([k, un[k], round(times.get(k, 0.0), 3)] for k in un),
            key=lambda r: -r[2],
        )
    return res


# ══════════════════════════════════════════════════════════════════
# 실측 — ORT 노드별 프로파일링
# ══════════════════════════════════════════════════════════════════
def profile(path, feed, runs=8, level="all"):
    """노드별 실행 시간을 실제로 잰다. op_type → 총 ms 를 돌려준다."""
    lv = {
        "none": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "ext": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }[level]
    so = ort.SessionOptions()
    so.enable_profiling = True
    so.graph_optimization_level = lv
    so.intra_op_num_threads = 2
    s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    for _ in range(2):
        s.run(None, feed)          # 워밍업 — 1주차에서 배운 그대로
    for _ in range(runs):
        s.run(None, feed)
    pf = s.end_profiling()
    ev = json.load(open(pf))
    os.remove(pf)
    per_op, per_node = collections.Counter(), collections.Counter()
    nruns = 0
    for e in ev:
        if e.get("cat") != "Node" or not e["name"].endswith("_kernel_time"):
            continue
        op = e["args"].get("op_name", "?")
        per_op[op] += e["dur"] / 1000.0
        per_node[e["name"]] += e["dur"] / 1000.0
    # 워밍업 2회 + 본 runs 회 가 모두 기록된다 → 실행 횟수로 나눈다
    tot_runs = runs + 2
    for k in per_op:
        per_op[k] /= tot_runs
    return dict(per_op), tot_runs


def session_ms(path, level, feed, reps=3):
    """세션 생성(=컴파일) 시간과 첫 추론·정상 추론 시간."""
    lv = {
        "none": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "ext": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }[level]
    build, first, steady = [], [], []
    for _ in range(reps):
        so = ort.SessionOptions()
        so.graph_optimization_level = lv
        so.intra_op_num_threads = 2
        t0 = time.perf_counter()
        s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        build.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        s.run(None, feed)
        first.append((time.perf_counter() - t0) * 1000)
        for _ in range(3):
            s.run(None, feed)
        ts = []
        for _ in range(10):
            t0 = time.perf_counter()
            s.run(None, feed)
            ts.append((time.perf_counter() - t0) * 1000)
        steady.append(float(np.median(ts)))
        del s
    return {
        "build_ms": float(np.median(build)),
        "first_ms": float(np.median(first)),
        "steady_ms": float(np.median(steady)),
    }


def optimized_graph(path, level, out):
    """최적화된 그래프를 파일로 뽑아 노드 수를 센다 — 융합의 직접 증거."""
    lv = {
        "none": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "ext": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }[level]
    so = ort.SessionOptions()
    so.graph_optimization_level = lv
    so.optimized_model_filepath = out
    so.intra_op_num_threads = 2
    ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    m = onnx.load(out)
    return op_hist(m)


def feed_for(path):
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    f = {}
    for i in s.get_inputs():
        shp = [d if isinstance(d, int) else 1 for d in i.shape]
        f[i.name] = np.random.rand(*shp).astype(np.float32)
    return f
