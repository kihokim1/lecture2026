# -*- coding: utf-8 -*-
"""3교시 실습 — 학생이 그대로 따라 치는 코드. 실제 출력을 확보하기 위해 여기서 검증한다."""
import onnx, torch, torchvision
from onnx import shape_inference

# ── 1단계. 모델을 ONNX 로 내보낸다 ─────────────────────────────────────────
HW = 224
m = torchvision.models.mobilenet_v2(weights=None).eval()
torch.onnx.export(m, torch.randn(1, 3, HW, HW), "mbv2.onnx",
                  input_names=["input"], opset_version=13, dynamo=False)
model = shape_inference.infer_shapes(onnx.load("mbv2.onnx"))
g = model.graph
print(f"노드 {len(g.node)}개, 가중치 {len(g.initializer)}개")

# ── 2단계. 상수를 걸러낸다 (3주차 Constant Folding) ────────────────────────
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
                    const.add(o); changed = True
print(f"상수로 판정된 텐서 {len(const)}개")

# ── 3단계. 텐서 크기 (INT8 = 1바이트/원소) ─────────────────────────────────
def nelem(vi):
    d = vi.type.tensor_type.shape.dim
    n = 1
    for x in d:
        n *= max(x.dim_value, 1)
    return n

size = {vi.name: nelem(vi) for vi in
        list(g.input) + list(g.value_info) + list(g.output)
        if vi.name not in const}
flash = sum(int(torch.tensor(list(i.dims)).prod()) if i.dims else 1 for i in g.initializer)
print(f"활성 텐서 {len(size)}개 | Flash(가중치) {flash/1024:.1f} KB")

# ── 4단계. 수명 — 언제 태어나 언제 죽나 ────────────────────────────────────
N = len(g.node)
birth = {i.name: -1 for i in g.input if i.name not in const}
death = {}
for k, n in enumerate(g.node):
    for o in n.output:
        if o in size:
            birth.setdefault(o, k)
    for i in n.input:
        if i in size:
            death[i] = k
for o in g.output:
    death[o.name] = N
live = [t for t in birth if t in size and size[t] > 0]
for t in live:
    death.setdefault(t, birth[t])
print(f"수명을 매긴 텐서 {len(live)}개")

# ── 5단계. 세 가지 방식으로 최대 메모리를 잰다 ─────────────────────────────
naive = sum(size[t] for t in live)

peak_life, at = 0, -1
for k in range(N + 1):
    s = sum(size[t] for t in live if birth[t] <= k <= death[t])
    if s > peak_life:
        peak_life, at = s, k

INPLACE = {"Relu", "Clip", "Add", "Mul", "Sigmoid", "Tanh", "BatchNormalization"}
alias, gout = {}, {o.name for o in g.output}
def root(t):
    while t in alias:
        t = alias[t]
    return t
for k, n in enumerate(g.node):
    if n.op_type not in INPLACE or len(n.output) != 1:
        continue
    o = n.output[0]
    if o not in size or o in gout:
        continue
    c = [i for i in n.input if i in size and size[i] == size[o]
         and death.get(i) == k and birth.get(i, -1) >= 0]
    if c:
        r = root(c[0]); alias[o] = r
        death[r] = max(death[r], death.get(o, k))
live2 = [t for t in live if t not in alias]
peak_ip = max(sum(size[t] for t in live2 if birth[t] <= k <= death[t])
              for k in range(N + 1))

print()
print(f"① 재사용 없음      {naive/1024:9.1f} KB")
print(f"② 수명 기반        {peak_life/1024:9.1f} KB   ({naive/peak_life:.2f}배 감소, 노드 {at})")
print(f"③ ② + in-place     {peak_ip/1024:9.1f} KB   ({naive/peak_ip:.2f}배 감소, 별칭 {len(alias)}개)")

# ── 6단계. 예산 판정 ───────────────────────────────────────────────────────
SRAM_KB, FLASH_KB = 320, 1024
print()
print(f"[STM32F746 판정] Flash {flash/1024:.0f}/{FLASH_KB} KB "
      f"({flash/1024/FLASH_KB:.1f}배)  |  SRAM {peak_ip/1024:.0f}/{SRAM_KB} KB "
      f"({peak_ip/1024/SRAM_KB:.1f}배)")

# ── 7단계. 배터리 계산기 ───────────────────────────────────────────────────
def life_years(period_s, t_inf=0.1, i_a=3.3e-3, i_s=3.16e-6, cap_mah=225.0):
    self_a = cap_mah * 0.01 / 8766.0 * 1e-3          # 연 1% 자기방전
    i = (i_a * t_inf + i_s * (period_s - t_inf)) / period_s + self_a
    return (cap_mah / 1000) / i / 24 / 365

print()
for T in [1, 60, 3600]:
    a, b = life_years(T), life_years(T, t_inf=0.05)
    print(f"주기 {T:>5}초 : {a:6.2f}년 → 추론 절반이면 {b:6.2f}년  ({b/a:.3f}배)")
