"""3교시 학생용 — 내 모델이 가속기에 몇 % 올라가는지 재 본다.

여덟 걸음이면 판정이 나온다. GPU 도 NPU 도 필요 없다.

    python3 student.py                       # 기본 MobileNetV2
    python3 student.py /경로/내모델.onnx      # 내 모델
"""
import sys, json, collections
import numpy as np
import onnx, onnxruntime as ort

PATH = sys.argv[1] if len(sys.argv) > 1 else "/root/lab08/mbv2.onnx"

# 내 가속기의 허용 연산자 목록. 벤더 문서를 보고 여기를 고치는 것이 과제다.
ALLOW = {
    "Conv", "Relu", "MaxPool", "AveragePool", "GlobalAveragePool",
    "Add", "Gemm", "MatMul", "Flatten", "Reshape", "BatchNormalization",
}

# ── ① 그래프를 연다 ────────────────────────────────────────────
m = onnx.load(PATH)
const = {i.name for i in m.graph.initializer}
const |= {n.output[0] for n in m.graph.node if n.op_type == "Constant"}
nodes = [n for n in m.graph.node
         if n.op_type != "Constant"
         and not (n.op_type == "Identity" and n.input and all(i in const for i in n.input))]
hist = collections.Counter(n.op_type for n in nodes)
print(f"① 전체 노드 {len(m.graph.node)}개 · 실제 계산하는 노드 {len(nodes)}개")
print(f"   상위 연산자: {hist.most_common(6)}")

# ── ② 노드별 시간을 실제로 잰다 ────────────────────────────────
so = ort.SessionOptions()
so.enable_profiling = True
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL  # 원본 그래프로
so.intra_op_num_threads = 2
s = ort.InferenceSession(PATH, so, providers=["CPUExecutionProvider"])
feed = {i.name: np.random.rand(*[d if isinstance(d, int) else 1 for d in i.shape])
        .astype(np.float32) for i in s.get_inputs()}
for _ in range(2):
    s.run(None, feed)                      # 워밍업 — 1주차에서 배운 그대로
for _ in range(6):
    s.run(None, feed)
import os
ev = json.load(open(s.end_profiling()))
[os.remove(f) for f in os.listdir(".") if f.endswith(".json") and "profile" in f]
t = collections.Counter()
for e in ev:
    if e.get("cat") == "Node" and e["name"].endswith("_kernel_time"):
        t[e["args"]["op_name"]] += e["dur"] / 1000.0 / 8      # 워밍업 2 + 본 6
t = {k: v for k, v in t.items() if k in hist}
T = sum(t.values())
print(f"\n② 총 {T:.2f} ms · 상위: {sorted(t.items(), key=lambda kv: -kv[1])[:4]}")

# ── ③ 노드 커버리지와 시간 커버리지 ────────────────────────────
n_acc = sum(1 for n in nodes if n.op_type in ALLOW)
node_cov = n_acc / len(nodes)
time_cov = sum(v for k, v in t.items() if k in ALLOW) / T
print(f"\n③ 노드 커버리지 {node_cov:.1%}  ·  시간 커버리지 {time_cov:.1%}")
print("   두 숫자가 다르면, 성능을 정하는 것은 아래쪽입니다.")

# ── ④ 무엇이 막고 있나 — 시간 순 ───────────────────────────────
blockers = sorted(((k, hist[k], t.get(k, 0)) for k in hist if k not in ALLOW),
                  key=lambda r: -r[2])
print("\n④ 미지원 연산자 (시간이 큰 순):")
for k, c, ms in blockers[:5]:
    print(f"     {k:<22} {c:4d}개 · {ms:6.3f} ms · 전체의 {ms/T:5.1%}")
if not blockers:
    print("     없음 — 전부 올라갑니다.")

# ── ⑤ 몇 번 왕복하나 ───────────────────────────────────────────
prod = {o: n for n in nodes for o in n.output if o}
indeg, succ = {}, collections.defaultdict(set)
for n in nodes:
    ps = {id(prod[i]) for i in n.input if i in prod and prod[i] is not n}
    indeg[id(n)] = len(ps)
    for p in ps:
        succ[p].add(id(n))
byid = {id(n): n for n in nodes}
rank = {id(n): k for k, n in enumerate(nodes)}
ready = [k for k, d in indeg.items() if d == 0]
order = []
while ready:
    p = min(ready, key=lambda k: rank[k])
    ready.remove(p)
    order.append(byid[p])
    for x in succ[p]:
        indeg[x] -= 1
        if indeg[x] == 0:
            ready.append(x)
sw, cur = 0, None
for n in order:
    d = n.op_type in ALLOW
    if cur is not None and d != cur:
        sw += 1
    cur = d
print(f"\n⑤ 호스트↔가속기 왕복 {sw}회 (블록 {sw+1}개)")

# ── ⑥ 경계를 넘는 바이트 ───────────────────────────────────────
mi = onnx.shape_inference.infer_shapes(m, strict_mode=False)
BPE = {1: 4, 2: 1, 3: 1, 6: 4, 7: 8, 9: 1, 10: 2, 11: 8}
size = {}
for v in list(mi.graph.value_info) + list(mi.graph.input) + list(mi.graph.output):
    tt = v.type.tensor_type
    dims = [d.dim_value for d in tt.shape.dim]
    if tt.elem_type and all(d > 0 for d in dims) and dims:
        size[v.name] = int(np.prod(dims)) * BPE.get(tt.elem_type, 4)
cross = {i for n in nodes for i in n.input
         if i in prod and (prod[i].op_type in ALLOW) != (n.op_type in ALLOW)}
xmb = sum(size.get(c, 0) for c in cross) / 1048576
print(f"\n⑥ 경계를 넘는 텐서 {len(cross)}개 · {xmb:.1f} MB")

# ── ⑦ 손익분기 — 가속기가 몇 배 빨라야 하나 ────────────────────
LGBPS = 9.87          # 레이아웃 변환 대역폭 (2교시 실측). 자기 기기에서 다시 재도 좋다
c_ms = (xmb / sw) / 1024 / LGBPS * 1000 if sw else 0.0


def speedup(S):
    return T / (T * (1 - time_cov) + (T * time_cov / S if S else 0) + sw * c_ms)


print(f"\n⑦ 왕복 1회 {c_ms*1000:.1f} µs · 왕복세 합계 {sw*c_ms:.2f} ms")
print(f"   암달 천장   {1/(1-time_cov):6.2f}배   ← 왕복을 무시하면")
print(f"   실제 천장   {speedup(None):6.2f}배   ← 왕복세를 넣으면")
for S in (2, 5, 10):
    print(f"   가속기 {S:2d}배 → {speedup(S):5.2f}배")

# ── ⑧ 판정 ─────────────────────────────────────────────────────
print("\n⑧ 판정")
if time_cov > 0.95 and sw <= 8:
    print("   올려도 좋습니다 — 커버리지가 높고 조각이 적습니다.")
elif time_cov > 0.9:
    print(f"   조각이 문제입니다({sw}회). 먼저 '{blockers[0][0]}' 지원 여부를 벤더에 확인하십시오.")
else:
    print(f"   커버리지가 부족합니다({time_cov:.0%}). 가속기보다 모델을 먼저 고치십시오 —")
    print(f"   '{blockers[0][0]}' 하나가 시간의 {blockers[0][2]/T:.0%}를 잡고 있습니다.")
print("\n   ※ 이 모형은 데이터 이동만 셉니다. 실제 가속기는 왕복마다 커널 실행·동기화")
print("      고정비(보통 수십 µs)가 더 붙으므로, 위 값은 낙관적인 쪽입니다.")
