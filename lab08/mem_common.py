# -*- coding: utf-8 -*-
"""ONNX 그래프에서 **활성값(activation) 메모리**를 실제로 계산한다.

MCU 배포에서 바닥나는 것은 두 가지다.
  - Flash  ← 가중치(initializer). 모델 크기.
  - SRAM   ← 중간 결과(activation). 모델 크기와 무관하게 결정된다.

이 파일은 후자를 텐서 **수명(lifetime)** 기준으로 계산한다.
"""
import onnx
from onnx import shape_inference, TensorProto

ELEM = {  # 원소 하나당 바이트
    TensorProto.FLOAT: 4, TensorProto.FLOAT16: 2, TensorProto.DOUBLE: 8,
    TensorProto.INT64: 8, TensorProto.INT32: 4, TensorProto.INT8: 1,
    TensorProto.UINT8: 1, TensorProto.BOOL: 1,
}


def load(path):
    m = onnx.load(path)
    return shape_inference.infer_shapes(m)


def _dims(vi):
    """value_info 에서 정적 shape 를 뽑는다. 동적 축은 1로 본다."""
    t = vi.type.tensor_type
    if not t.HasField("shape"):
        return None
    out = []
    for d in t.shape.dim:
        if d.HasField("dim_value"):
            out.append(d.dim_value)
        else:
            out.append(1)          # 배치 등 동적 축 → 1
    return out


def shapes_of(model):
    """이름 → (shape, elem_type). 그래프 입력·출력·중간값 전부."""
    g = model.graph
    tab = {}
    for vi in list(g.input) + list(g.value_info) + list(g.output):
        d = _dims(vi)
        if d is not None:
            tab[vi.name] = (d, vi.type.tensor_type.elem_type)
    return tab


def nbytes(shape, elem_type, bpe=None):
    n = 1
    for s in shape:
        n *= max(int(s), 1)
    return n * (bpe if bpe is not None else ELEM.get(elem_type, 4))


def constants_of(model):
    """상수 전파(Constant Folding) — 입력이 전부 상수인 노드의 출력도 상수다.

    torch.onnx.export 는 가중치를 Identity 노드로 한 번 흘려보내는 경우가 있다.
    이것을 걸러내지 않으면 **가중치를 SRAM 으로 잘못 세게 된다**.
    (3주차에서 배운 Constant Folding 을 여기서 직접 쓴다.)
    """
    g = model.graph
    const = {i.name for i in g.initializer}
    const |= {n.output[0] for n in g.node if n.op_type == "Constant"}
    changed = True
    while changed:
        changed = False
        for n in g.node:
            ins = [i for i in n.input if i]
            if ins and all(i in const for i in ins):
                for o in n.output:
                    if o and o not in const:
                        const.add(o)
                        changed = True
    return const


# 출력 버퍼를 입력 버퍼 위에 덮어쓸 수 있는 연산 (모양이 같을 때)
INPLACEABLE = {"Relu", "Clip", "LeakyRelu", "Sigmoid", "Tanh", "Elu", "HardSigmoid",
               "HardSwish", "Add", "Mul", "Sub", "Div", "Abs", "Neg", "Exp", "Log",
               "Sqrt", "Erf", "Softplus", "PRelu", "BatchNormalization", "Identity"}


def analyze(model, bpe=1, inplace=False):
    """활성 텐서의 크기·수명을 계산한다.

    bpe:     원소당 바이트. INT8 배포를 가정하면 1.
    inplace: True 면 원소별 연산의 출력을 입력 버퍼에 별칭(alias)한다.
    """
    g = model.graph
    init = constants_of(model)
    tab = shapes_of(model)

    # --- Flash: 가중치 총량 -------------------------------------------------
    flash = 0
    for i in g.initializer:
        n = 1
        for s in i.dims:
            n *= max(int(s), 1)
        flash += n * bpe

    # --- 활성 텐서 목록 -----------------------------------------------------
    graph_in = [i.name for i in g.input if i.name not in init]
    graph_out = [o.name for o in g.output]

    size = {}
    for name, (sh, et) in tab.items():
        if name in init:
            continue
        size[name] = nbytes(sh, et, bpe)

    N = len(g.node)
    birth, death = {}, {}
    for nm in graph_in:
        birth[nm] = -1
    for k, node in enumerate(g.node):
        for o in node.output:
            if o and o not in init:
                birth.setdefault(o, k)
        for i in node.input:
            if i and i not in init:
                death[i] = k                    # 마지막으로 소비되는 노드
    for nm in graph_out:
        death[nm] = N                           # 끝까지 살아 있다

    live_tensors = [t for t in birth if t in size and size[t] > 0]
    for t in live_tensors:
        death.setdefault(t, birth[t])           # 아무도 안 쓰면 즉시 사망

    # --- in-place 별칭: 원소별 연산의 출력을 입력 버퍼에 덮어쓴다 ----------
    alias = {}
    n_alias = 0

    def root(t):
        while t in alias:
            t = alias[t]
        return t

    if inplace:
        outs = set(graph_out)
        for k, node in enumerate(g.node):
            if node.op_type not in INPLACEABLE or len(node.output) != 1:
                continue
            o = node.output[0]
            if o not in size or o in outs:
                continue
            cand = [i for i in node.input
                    if i and i not in init and i in size
                    and size[i] == size[o] and death.get(i) == k and i not in graph_in]
            if not cand:
                continue
            r = root(cand[0])
            alias[o] = r
            death[r] = max(death[r], death.get(o, k))   # 수명을 합친다
            n_alias += 1
        live_tensors = [t for t in live_tensors if t not in alias]

    # --- ① 재사용 없음: 모든 활성값을 동시에 들고 있기 ----------------------
    sum_all = sum(size[t] for t in live_tensors)

    # --- ② 교과서식 "입력 버퍼 + 출력 버퍼" 추정 ---------------------------
    inout = 0
    per_node = []
    for k, node in enumerate(g.node):
        a = sum(size.get(i, 0) for i in node.input if i not in init)
        b = sum(size.get(o, 0) for o in node.output if o not in init)
        per_node.append((k, node.op_type, a + b))
        inout = max(inout, a + b)

    # --- ③ 하한: 어느 시점에 살아 있는 텐서 합의 최댓값 --------------------
    watermark, wm_at = 0, -1
    live_profile = []
    for k in range(N + 1):
        s = sum(size[t] for t in live_tensors
                if birth[t] < k <= death[t] or (birth[t] == k <= death[t]))
        live_profile.append(s)
        if s > watermark:
            watermark, wm_at = s, k

    # --- ④ 실제 플래너: 크기 큰 순 탐욕 오프셋 배치 (TFLM 방식) ------------
    order = sorted(live_tensors, key=lambda t: -size[t])
    placed = []          # (offset, end, birth, death)
    offsets = {}
    peak = 0
    for t in order:
        b, d, sz = birth[t], death[t], size[t]
        conflicts = sorted(
            (o, o + s) for (o, s, ob, od) in placed
            if not (od < b or d < ob)           # 수명이 겹치는 것만
        )
        off = 0
        for lo, hi in conflicts:
            if off + sz <= lo:
                break
            off = max(off, hi)
        offsets[t] = off
        placed.append((off, sz, b, d))
        peak = max(peak, off + sz)

    return dict(
        flash=flash, sum_all=sum_all, inout=inout,
        watermark=watermark, watermark_at=wm_at, greedy=peak,
        n_nodes=N, size=size, birth=birth, death=death,
        per_node=per_node, live_profile=live_profile, n_alias=n_alias,
    )


def kb(x):
    return x / 1024.0


def fmt(r):
    return (f"Flash(가중치) {kb(r['flash']):9.1f} KB | "
            f"재사용없음 {kb(r['sum_all']):9.1f} | "
            f"입출력추정 {kb(r['inout']):8.1f} | "
            f"하한 {kb(r['watermark']):8.1f} | "
            f"실제플래너 {kb(r['greedy']):8.1f} KB")
