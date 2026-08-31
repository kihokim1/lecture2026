# 08주차 3교시. 내 모델이 이 MCU에 들어가는지 직접 재 보기

> **오늘의 질문** — 1·2교시에서 나온 숫자들은 전부 계산해서 얻은 것이다. **그 계산을 직접 해 보자.** MCU 보드는 없어도 된다. 필요한 것은 ONNX 파일 하나와 파이썬뿐이며, 오늘 만들 40줄짜리 스크립트가 MCUNetV2 논문에 실린 값을 **킬로바이트까지** 재현한다.

---

## 실험 설계

| | 내용 |
|---|---|
| **가설** | MCU 배포를 막는 것은 모델 파일 크기가 아니라 **최대 활성값 메모리**이며, 그 값은 모델을 바꾸지 않고 **메모리 배치만 바꿔도** 크게 달라진다. |
| **측정 대상** | ① 가중치 총량(Flash) ② 재사용 없는 활성값 합 ③ 수명 기반 최대 메모리 ④ in-place 적용 후 최대 메모리 |
| **검증 방법** | 우리 ③④ 값을 MCUNet(5.3배 초과)과 MCUNetV2(1,372 kB)의 보고값과 대조한다. |
| **필요한 것** | `torch`, `torchvision`, `onnx`. **MCU 보드 불필요.** |
| **타당성 위협** | 우리는 순수 텐서 메모리만 센다. 실제 런타임은 여기에 스택·정렬 여백·연산자 임시 버퍼를 더한다. 그래서 우리 값은 **하한**이다. |

---

## 3.1 ONNX 그래프를 열고 상수를 걸러내기 (12분)

### 준비

```bash
pip install torch torchvision onnx
```

### 1단계 — 모델을 ONNX 로 내보낸다

```python
import onnx, torch, torchvision
from onnx import shape_inference

HW = 224
m = torchvision.models.mobilenet_v2(weights=None).eval()
torch.onnx.export(m, torch.randn(1, 3, HW, HW), "mbv2.onnx",
                  input_names=["input"], opset_version=13, dynamo=False)

model = shape_inference.infer_shapes(onnx.load("mbv2.onnx"))
g = model.graph
print(f"노드 {len(g.node)}개, 가중치 {len(g.initializer)}개")
```

```
노드 209개, 가중치 67개
```

`shape_inference` 가 핵심이다. ONNX 파일에는 **입력과 출력의 모양만** 적혀 있고 중간 텐서의 모양은 비어 있다. 이 함수가 그래프를 훑으며 모든 중간 텐서의 모양을 채워 준다. 이게 없으면 크기를 잴 수 없다.

### 2단계 — 상수를 걸러낸다

여기서 **함정 하나**를 먼저 넘어야 한다. 노드가 209개인데 우리가 아는 MobileNetV2의 연산 층은 100개 남짓이다. 나머지 109개는 무엇인가?

`torch.onnx.export` 는 가중치 일부를 `Identity` 노드로 한 번 흘려보낸다. 이것들을 안 걸러내면 **가중치를 SRAM으로 잘못 세게 된다.**

해법은 3주차에서 배운 **상수 접기(Constant Folding)** 다. 입력이 전부 상수인 노드의 출력도 상수다. 이 규칙을 더 이상 변하지 않을 때까지 반복한다.

```python
const  = {i.name for i in g.initializer}
const |= {n.output[0] for n in g.node if n.op_type == "Constant"}

changed = True
while changed:                       # 변화가 없을 때까지 되풀이한다
    changed = False
    for n in g.node:
        ins = [i for i in n.input if i]
        if ins and all(i in const for i in ins):     # 입력이 전부 상수라면
            for o in n.output:
                if o and o not in const:
                    const.add(o); changed = True     # 출력도 상수다
print(f"상수로 판정된 텐서 {len(const)}개")
```

```
상수로 판정된 텐서 176개
```

> **왜 `while` 인가.** 한 번만 훑으면 안 된다. `A(상수) → B → C` 구조에서 B를 상수로 판정한 뒤에야 C를 판정할 수 있는데, 노드 순서가 그 반대일 수 있기 때문이다. 이런 **고정점(fixed point) 반복**은 컴파일러 최적화의 기본 패턴이며, 11주차 툴체인에서 다시 만난다.

### 3단계 — 텐서 크기를 잰다

```python
def nelem(vi):                       # value_info 에서 원소 개수를 뽑는다
    n = 1
    for x in vi.type.tensor_type.shape.dim:
        n *= max(x.dim_value, 1)
    return n

# INT8 배포 가정 → 원소 하나 = 1바이트. FP32 라면 4를 곱한다.
size = {vi.name: nelem(vi) for vi in
        list(g.input) + list(g.value_info) + list(g.output)
        if vi.name not in const}

flash = sum(int(torch.tensor(list(i.dims)).prod()) for i in g.initializer)
print(f"활성 텐서 {len(size)}개 | Flash(가중치) {flash/1024:.1f} KB")
```

```
활성 텐서 101개 | Flash(가중치) 3393.6 KB
```

209개 노드에서 **활성 텐서 101개**만 남았다. 1교시에 인용한 3,394 KB가 여기서 나온 값이다.

> 한 줄 정리: ONNX 그래프에서 활성값을 세려면 **모양 추론**과 **상수 접기**를 먼저 해야 한다. 이 두 단계를 빼먹으면 가중치를 SRAM으로 잘못 센다.

---

## 3.2 텐서의 수명을 매기고 최대 메모리를 재기 (16분)

### 4단계 — 언제 태어나 언제 죽나

```python
N = len(g.node)
birth = {i.name: -1 for i in g.input if i.name not in const}   # 그래프 입력은 -1에 태어남
death = {}

for k, n in enumerate(g.node):
    for o in n.output:
        if o in size:
            birth.setdefault(o, k)      # 이 노드가 출력하면 여기서 태어난다
    for i in n.input:
        if i in size:
            death[i] = k                # 덮어쓰므로 '마지막' 소비 노드가 남는다

for o in g.output:
    death[o.name] = N                   # 최종 출력은 끝까지 산다

live = [t for t in birth if t in size and size[t] > 0]
for t in live:
    death.setdefault(t, birth[t])       # 아무도 안 쓰는 텐서는 즉시 죽는다
```

`death[i] = k` 한 줄이 이 실습에서 가장 영리한 부분이다. 노드를 순서대로 훑으며 **덮어쓰기** 때문에, 반복이 끝나면 자동으로 **마지막으로 소비한 노드**가 남는다.

### 5단계 — 세 가지 방식으로 재기

```python
# ① 재사용 없음 — 전부 동시에 들고 있다
naive = sum(size[t] for t in live)

# ② 수명 기반 — 각 시점에 살아 있는 것만 더하고, 그 최댓값을 취한다
peak_life, at = 0, -1
for k in range(N + 1):
    s = sum(size[t] for t in live if birth[t] <= k <= death[t])
    if s > peak_life:
        peak_life, at = s, k
```

②의 세 줄이 이번 주 전체의 핵심이다. **"각 시점에 살아 있는 것만 더한다"** — 이 한 문장이 12 MB를 2.3 MB로 만든다.

```python
# ③ in-place — 원소별 연산의 출력을 입력 자리에 덮어쓴다
INPLACE = {"Relu", "Clip", "Add", "Mul", "Sigmoid", "Tanh", "BatchNormalization"}
alias, gout = {}, {o.name for o in g.output}

def root(t):                       # 별칭을 따라가 최종 주인을 찾는다
    while t in alias:
        t = alias[t]
    return t

for k, n in enumerate(g.node):
    if n.op_type not in INPLACE or len(n.output) != 1:
        continue
    o = n.output[0]
    if o not in size or o in gout:
        continue
    # 조건: 모양(크기)이 같고, 그 입력이 바로 이 노드에서 죽고, 그래프 입력이 아니다
    c = [i for i in n.input if i in size and size[i] == size[o]
         and death.get(i) == k and birth.get(i, -1) >= 0]
    if c:
        r = root(c[0]); alias[o] = r
        death[r] = max(death[r], death.get(o, k))    # 수명을 합친다

live2 = [t for t in live if t not in alias]
peak_ip = max(sum(size[t] for t in live2 if birth[t] <= k <= death[t])
              for k in range(N + 1))
```

세 조건을 하나씩 확인하자. 하나라도 빠지면 **틀린 결과가 나오는데 오류는 안 난다.**

| 조건 | 왜 필요한가 |
|---|---|
| `size[i] == size[o]` | 크기가 다르면 덮어쓸 수 없다 |
| `death.get(i) == k` | 입력이 이 노드 뒤에도 쓰인다면(잔차 연결!) 덮어쓰면 **값이 깨진다** |
| `birth.get(i,-1) >= 0` | 그래프 입력 버퍼는 호출자 소유라 우리가 못 건드린다 |

### 결과

```python
print(f"① 재사용 없음      {naive/1024:9.1f} KB")
print(f"② 수명 기반        {peak_life/1024:9.1f} KB   ({naive/peak_life:.2f}배 감소, 노드 {at})")
print(f"③ ② + in-place     {peak_ip/1024:9.1f} KB   ({naive/peak_ip:.2f}배 감소, 별칭 {len(alias)}개)")
```

```
① 재사용 없음        12846.1 KB
② 수명 기반           2352.0 KB   (5.46배 감소, 노드 51)
③ ② + in-place        1470.0 KB   (8.74배 감소, 별칭 45개)
```

**모델은 한 글자도 안 바꿨다.** 배치 방식만 바꿔 8.74배가 줄었다.

> 한 줄 정리: 수명을 매기는 데 12줄, 최대 메모리를 재는 데 5줄이면 된다. 그 17줄이 **모델을 안 바꾸고 8.74배**를 만든다.

---

## 3.3 판정하고, 논문과 대조하기 (12분)

### 6단계 — 예산 판정

```python
SRAM_KB, FLASH_KB = 320, 1024        # STM32F746
print(f"[STM32F746 판정] Flash {flash/1024:.0f}/{FLASH_KB} KB "
      f"({flash/1024/FLASH_KB:.1f}배)  |  SRAM {peak_ip/1024:.0f}/{SRAM_KB} KB "
      f"({peak_ip/1024/SRAM_KB:.1f}배)")
```

```
[STM32F746 판정] Flash 3394/1024 KB (3.3배)  |  SRAM 1470/320 KB (4.6배)
```

### 논문과 맞춰 보기 — 여기가 오늘의 정점이다

우리가 40줄로 얻은 값과, 논문에 실린 값을 나란히 놓자.

| | 우리 측정 | 논문 보고 | 차이 |
|---|---:|---:|---|
| int8 MobileNetV2 의 SRAM 초과 배수 | **4.6배** | **5.3배** (MCUNet [1]) | 논문이 14 % 크다 |
| MobileNetV2 앞쪽 최대 메모리 | **1,372.0 KB** (6번 노드) | **1,372 kB** (MCUNetV2 [2]) | **완전 일치** |

두 번째 줄을 다시 보라. **킬로바이트 단위까지 같다.** 우리 전체 최댓값 1,470 KB와 논문 값이 다른 것은 재는 **경계**가 다르기 때문이다 — 논문은 블록 단위로, 우리는 노드 단위로 쟀다.

첫 번째 줄의 14 % 차이는 **설명할 수 있는 차이**다. 우리는 텐서만 셌고, 실제 런타임은 여기에 다음을 더한다.

- 연산자 임시 버퍼(합성곱의 im2col 버퍼가 대표적이다)
- 메모리 정렬을 위한 여백
- 인터프리터 자료구조와 스택

그래서 **우리 값은 항상 하한**이다. 실무에서 "우리 계산으로는 300 KB니까 320 KB에 들어간다"고 말하면 위험하다.

> **재현 검증이 무엇인지 보여 주는 대목이다.** 숫자가 맞았다는 것보다 **차이가 설명된다**는 것이 중요하다. 완전히 일치하는 값 하나(1,372)와, 방향과 크기가 설명되는 차이 하나(4.6 대 5.3). 이 두 가지가 함께 있을 때 비로소 "재현했다"고 말할 수 있다.

### 7단계 — 배터리 계산기

```python
def life_years(period_s, t_inf=0.1, i_a=3.3e-3, i_s=3.16e-6, cap_mah=225.0):
    """period_s 마다 t_inf 동안 추론할 때의 배터리 수명(년).
    i_a : nRF52840 CoreMark @64MHz, 플래시 실행, DC/DC, 3V  → 3.3 mA
    i_s : System ON, 256kB RAM 유지, RTC 기상               → 3.16 uA
    cap : Panasonic CR2032 공칭 225 mAh, 자기방전 연 1.0%
    """
    self_a = cap_mah * 0.01 / 8766.0 * 1e-3
    i = (i_a * t_inf + i_s * (period_s - t_inf)) / period_s + self_a
    return (cap_mah / 1000) / i / 24 / 365

for T in [1, 60, 3600]:
    a, b = life_years(T), life_years(T, t_inf=0.05)
    print(f"주기 {T:>5}초 : {a:6.2f}년 → 추론 절반이면 {b:6.2f}년  ({b/a:.3f}배)")
```

```
주기     1초 :   0.08년 → 추론 절반이면   0.15년  (1.980배)
주기    60초 :   2.88년 → 추론 절반이면   4.17년  (1.446배)
주기  3600초 :   7.32년 → 추론 절반이면   7.42년  (1.013배)
```

2교시의 손잡이 역전이 세 줄로 확인된다. **1초 주기에서 1.98배, 1시간 주기에서 1.013배.**

![실습 파이프라인 — ONNX 내보내기부터 예산 판정까지 일곱 단계와 각 단계의 산출물](../assets/w08_p3_pipeline_12.png)

> 한 줄 정리: 40줄짜리 스크립트가 MCUNetV2의 1,372 kB를 **킬로바이트까지 재현**했고, MCUNet의 5.3배와는 설명 가능한 14 % 차이를 보였다. **차이가 설명되는 것이 일치보다 중요하다.**

---

## 3.4 과제 (5분 안내)

### 필수 과제 — 「내 모델의 세 개의 숫자」

**IMRaD 형식 2~3쪽.** 실행 로그와 코드를 부록에 붙일 것.

1. **모델 세 개를 고른다.** 하나는 반드시 오늘 안 다룬 모델이어야 한다(예: `resnet18`, `squeezenet1_1`, `shufflenet_v2_x0_5`, 또는 여러분 연구 주제의 모델).
2. 각 모델에 대해 **Flash / ② 수명 기반 SRAM / ③ in-place 후 SRAM** 세 숫자를 재고, STM32F746(320 KB / 1 MB) 예산에 대해 판정하라.
3. **in-place 이득이 모델마다 다른 이유를 설명하라.** 이득이 1.00배인 모델이 있다면, 그 모델의 최대 지점에 어떤 연산이 있는지 밝히고 그것으로 설명할 것. (힌트: 최대가 발생한 노드 번호를 출력해 `g.node[at].op_type` 을 확인하라.)
4. **입력 해상도를 절반으로 낮춰** 다시 재라. Flash와 SRAM 중 어느 쪽이 얼마나 변했는지 표로 제시하고, 그 결과가 2.3의 주장과 맞는지 판정하라.
5. **타당성 위협**을 최소 두 개 적어라. 우리 계산이 실제 기기 값보다 작게 나오는 이유를 항목별로 쓸 것.

### 선택 과제 A — 「단편화를 만들어 보기」

우리 ②는 각 시점의 **합**을 구했다. 실제 할당기는 텐서를 **오프셋에 배치**하므로, 수명이 겹치지 않아도 조각난 빈틈 때문에 합보다 더 쓸 수 있다(**단편화**). TFLM은 크기 큰 순으로 정렬해 빈틈에 끼워 넣는 탐욕 방식을 쓴다.

탐욕 오프셋 할당기를 구현하고, ②의 하한과 실제 배치 결과를 비교하라. **MobileNetV2에서는 둘이 같게 나올 것이다.** 왜 그런지 설명하고, **둘이 달라지는 그래프 구조**를 하나 만들어 보라(힌트: 수명이 서로 엇갈리게 겹치는 텐서 세 개면 충분하다).

### 선택 과제 B — 「문지기를 설계하라」

2.5의 캐스케이드에서 문지기 모델을 실제로 골라 보자.

1. 1교시의 DS-CNN을 문지기로 쓸 때, 그 모델의 SRAM과 연산량을 재라.
2. 문지기 추론 시간을 연산량에 비례한다고 가정하고(2.6의 MicroNets 근거를 인용할 것) 본 모델 대비 몇 배 빠를지 추정하라.
3. 그 추정값으로 손익분기 통과율 $p^\*$ 를 다시 계산하라.
4. **2.6의 네 가지 조건 중 여러분의 추정이 위반한 것이 있는가?** 있다면 그것이 결론을 얼마나 흔드는지 논하라.

### 평가 기준

| 항목 | 배점 |
|---|---:|
| 세 숫자를 정확히 재고 판정했는가 | 30 |
| in-place 이득 차이를 **최대 지점의 연산자**로 설명했는가 | 25 |
| 해상도 실험의 결과 해석 (Flash 불변을 짚었는가) | 20 |
| 타당성 위협을 구체적으로 적었는가 | 15 |
| 재현 가능성 (코드·환경·로그) | 10 |

---

## 3교시 정리
- ONNX 그래프에서 활성값을 세려면 **모양 추론 → 상수 접기 → 수명 → 배치** 네 단계를 거친다. 상수 접기를 빼면 가중치를 SRAM으로 잘못 센다.
- 최대 메모리를 재는 핵심은 세 줄이다 — **각 시점에 살아 있는 텐서만 더하고, 그 최댓값을 취한다.**
- in-place 별칭에는 조건이 셋 있고, **`death[i] == k` 를 빼먹으면 잔차 연결이 있는 모델에서 값이 조용히 깨진다.**
- 우리 40줄이 MCUNetV2의 1,372 kB를 재현했고, MCUNet의 5.3배와는 14 % 차이가 났다. **그 차이는 런타임 버퍼로 설명된다.**
- 우리 계산은 언제나 **하한**이다. 실무 판정에는 여유를 두어야 한다.

> **교수님을 위한 Tip** — 3.2의 in-place 조건 세 개 중 **`death.get(i) == k` 를 일부러 빼고 돌려 보이십시오.** 값은 더 작게(더 좋아 보이게) 나오고 오류는 안 납니다. "이 코드의 어디가 틀렸는가"를 찾게 하면, 잔차 연결이 있는 모델에서 왜 값이 깨지는지를 학생들이 스스로 도달합니다. 이번 학기에서 **틀린 최적화가 좋아 보이는** 사례를 직접 만들어 볼 수 있는 몇 안 되는 지점입니다.

### 더 읽어보기
- [1] J. Lin, W.-M. Chen, Y. Lin, J. Cohn, C. Gan, and S. Han, "MCUNet: Tiny deep learning on IoT devices," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 33, 2020.
- [2] J. Lin, W.-M. Chen, H. Cai, C. Gan, and S. Han, "Memory-efficient patch-based inference for tiny deep learning," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 34, 2021.
- [4] R. David *et al.*, "TensorFlow Lite Micro: Embedded machine learning for TinyML systems," in *Proc. Machine Learning and Systems (MLSys)*, vol. 3, 2021, pp. 800–811. — **§4.4.2 의 메모리 플래너가 선택 과제 A의 정답지다.**
- [8] L. Lai, N. Suda, and V. Chandra, "CMSIS-NN: Efficient neural network kernels for Arm Cortex-M CPUs," *arXiv:1801.06601*, 2018.
