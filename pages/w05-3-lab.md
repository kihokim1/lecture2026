# 5주차 3교시. 양자화 적용과 크기·속도·정확도 비교

**실습 목표** — PyTorch로 모델을 INT8로 양자화해 **모델 크기가 실제로 1/4로 주는 것**과 속도 변화를 측정하고, 정확도 트레이드오프는 개념적으로 함께 짚는다.

> 준비물: 1주차 환경(`odai`) + `torch`. 이번엔 CPU만으로 충분하다.

---

## 3.1 양자화 적용 (20분)

가장 간단한 **동적 양자화(Dynamic Quantization)** 를 Linear 레이어로 구성된 작은 모델에 적용한다. 동적 양자화는 가중치를 미리 INT8로 바꾸고 활성값은 추론 시점에 처리하므로, 보정 데이터 없이 한 줄로 적용된다.

```python
import torch, torch.nn as nn, os

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 10))
    def forward(self, x): return self.net(x)

fp32 = Net().eval()

# 동적 양자화: Linear 레이어의 가중치를 INT8로 (torch.ao.quantization 도 동일)
int8 = torch.quantization.quantize_dynamic(fp32, {nn.Linear}, dtype=torch.qint8)
print(int8)     # Linear → DynamicQuantizedLinear 로 바뀐 것을 확인
```

---

## 3.2 크기·속도·정확도 비교 (20분)

**모델 크기** — 상태 사전(state_dict)을 저장해 파일 크기를 잰다.

```python
def size_kb(m, path):
    torch.save(m.state_dict(), path)
    return os.path.getsize(path) / 1024

print(f"FP32 : {size_kb(fp32,'fp32.pt'):.1f} KB")
print(f"INT8 : {size_kb(int8,'int8.pt'):.1f} KB   (약 1/4)")
```

**추론 속도** — 큰 배치로 두 모델의 지연을 비교한다.

```python
import time
x = torch.randn(256, 512)

def bench(m, runs=50):
    m.eval()
    with torch.no_grad():
        m(x)                                  # 워밍업
        t0 = time.perf_counter()
        for _ in range(runs): m(x)
        return (time.perf_counter()-t0)/runs*1000   # ms

print(f"FP32 : {bench(fp32):.2f} ms")
print(f"INT8 : {bench(int8):.2f} ms")
```

![FP32 vs INT8 (예시) — 모델 크기는 약 1/4, 추론 지연은 정수 연산 가속 시 감소, 정확도는 보정으로 거의 유지](../assets/w05_p3_bench_06.png)

관찰의 핵심:

1. **크기**는 거의 확실히 약 1/4로 준다(가중치가 32→8비트). 이것이 양자화의 가장 확실한 이득이다.
2. **속도**는 하드웨어·연산 종류에 따라 다르다. 정수 연산 가속(SIMD/NPU)이 받쳐줄 때 빨라진다 — 노트북 CPU에서는 큰 행렬일수록 이득이 잘 보인다.
3. **정확도**는 약간 떨어질 수 있으나, 2교시의 Calibration·Per-channel·QAT로 최소화한다.

> 관찰 포인트: "크기는 항상 준다, 속도는 하드웨어에 달렸다, 정확도는 방어할 수 있다" — 이 셋을 분리해서 보는 것이 양자화를 이해하는 핵심이다.

---

## 3.3 과제 (10분 안내)

1. **3지표 비교표** — 본인 모델(위 예제 또는 더 큰 Linear 모델)에서 FP32/INT8의 **크기·추론시간(평균·표준편차)** 을 측정해 표로 제출하고, 크기 절감률(%)을 계산한다.
2. **속도 해석** — INT8이 기대만큼 안 빨라졌다면(또는 빨라졌다면) 그 이유를 2교시의 SIMD·정수 유닛 관점으로 3~4문장 서술한다.
3. **PTQ vs QAT 조사** — 본인 도메인 모델에 PTQ와 QAT 중 무엇이 적합할지, 정확도·비용을 근거로 한 문단으로 정리한다.
4. **다음 주 예습** — 6주차(지식 증류)에서는 값을 줄이는 대신 **큰 모델의 지식을 작은 모델에 전수**한다. "정답(hard label)만이 아니라 큰 모델의 확률분포(soft label)를 배우면 왜 더 잘 되는가?"를 미리 생각해 온다.

> 교수님을 위한 Tip: 크기 절감(1/4)은 거의 항상 재현되지만 속도는 환경마다 다르다. 학생들이 "왜 크기는 줄었는데 속도는 그대로죠?"라고 물으면, 그 CPU가 INT8 연산을 실제로 가속하는지(백엔드·연산 종류)를 함께 따져보게 하라. 4주차와 같은 "압축 ≠ 속도" 통찰이 다시 확인된다.

---

### 3교시 정리
- 동적 양자화로 모델을 INT8로 바꾸고 크기가 약 1/4로 줆을 확인했다.
- 크기·속도·정확도를 분리해 트레이드오프를 관찰했다.
- 다음 주부터는 '값을 줄이는' 경량화를 넘어 '지식을 전수하는' 지식 증류로 넘어간다.
