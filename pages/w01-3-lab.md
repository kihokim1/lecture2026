# 01주차 3교시. 첫 온디바이스 추론 관찰

**실습 목표** — 개발 환경을 준비하고, "같은 추론"을 클라우드와 로컬에서 각각 수행해 **지연의 차이와 그 원인**을 직접 눈으로 확인한다. 1주차 실습은 코딩 실력보다 *관찰과 해석*에 목적이 있다.

> 준비물: 노트북(Windows/macOS/Linux) 또는 실습 보드. GPU는 필요 없다.

---

## 3.1 환경 준비 (15분)

파이썬 가상환경을 만들고 온디바이스 추론 런타임을 설치한다. 여기서는 설치가 간단하고 크로스 플랫폼인 **ONNX Runtime**을 사용한다.

```bash
# 1) 가상환경 생성
python -m venv odai
source odai/bin/activate        # Windows: odai\Scripts\activate

# 2) 온디바이스 추론 런타임 및 모델 준비용 도구 설치
pip install onnxruntime numpy torch torchvision
```

설치 확인:

```python
import onnxruntime as ort
print("ONNX Runtime:", ort.__version__)
print("사용 가능한 실행 공급자:", ort.get_available_providers())
# CPUExecutionProvider가 보이면 온디바이스(로컬 CPU) 추론 준비 완료
```

> `get_available_providers()`는 **설치된 빌드가 지원하는 실행 공급자(Execution Provider)** 목록을 돌려준다. 기본 `onnxruntime`(CPU 빌드)에서는 `CPUExecutionProvider`만 나타난다. 젯슨처럼 GPU를 쓰려면 `onnxruntime-gpu`(또는 Jetson 전용 빌드)를 설치해야 `CUDAExecutionProvider`가 함께 나타난다. 2주차에서 이 공급자가 곧 가속기 선택임을 다룬다.

이어서 실습에 쓸 작은 이미지 분류 모델(MobileNetV2)을 ONNX로 내보낸다. (조교가 `mobilenetv2.onnx`를 사전 배포한 경우 이 단계는 건너뛴다.)

```python
import torch, torchvision

model = torchvision.models.mobilenet_v2(weights="DEFAULT").eval()
dummy = torch.randn(1, 3, 224, 224)             # NCHW, 224x224 RGB 1장
torch.onnx.export(model, dummy, "mobilenetv2.onnx",
                  input_names=["input"], output_names=["logits"],
                  opset_version=13)
print("저장 완료: mobilenetv2.onnx")
```

---

## 3.2 클라우드 vs 온디바이스 추론 지연 관찰 (25분)

작은 이미지 분류 모델(예: MobileNet)을 **로컬에서** 실행하고, 추론에 걸린 시간을 측정한다.

```python
import time, numpy as np, onnxruntime as ort

sess = ort.InferenceSession("mobilenetv2.onnx",
                            providers=["CPUExecutionProvider"])
name = sess.get_inputs()[0].name
x = np.random.rand(1, 3, 224, 224).astype(np.float32)   # 더미 입력 1장

# 워밍업(최초 1회는 메모리 로딩 때문에 느리므로 제외)
sess.run(None, {name: x})

# 온디바이스 추론 시간 측정 (30회 각각 기록 → 평균·표준편차)
samples = []
for _ in range(30):
    t0 = time.perf_counter()
    sess.run(None, {name: x})
    samples.append((time.perf_counter() - t0) * 1000)   # ms
samples = np.array(samples)
print(f"온디바이스 추론: 평균 {samples.mean():.1f} ms  ±{samples.std():.1f} ms/장")
```

이제 개념적으로, 같은 한 장을 **클라우드 API**로 보냈다고 하자. 이때의 총 지연은 다음의 합이다.

```
클라우드 총 지연 = 전처리 + 업링크 전송 + 서버 대기/추론 + 다운링크 전송
```

로컬 추론이 순수 연산 시간만 든다면, 클라우드는 여기에 **왕복 네트워크 시간**이 얹힌다. 아래는 두 경우를 나란히 둔 관찰 예시다.

![실습 관찰: 동일 추론의 응답 지연 비교(예시). Cloud API는 전처리+네트워크 왕복+서버 추론으로 약 210ms, 온디바이스는 약 15ms. 지연의 대부분은 연산이 아니라 데이터 이동에서 발생한다](../assets/w01_p3_latency_bench_06.png)

> 관찰 포인트: 위 값은 개념 이해용 예시다. 실제 수업에서는 각자 측정한 `on_device_ms`를 적고, 사용 중인 클라우드 API의 응답 시간(브라우저 개발자도구의 Network 탭이나 `curl -w "%{time_total}"`)과 비교한다. 핵심 결론은 **"지연의 지배적 요인은 연산이 아니라 데이터 이동"**이라는 점이다.

---

## 3.3 과제 (10분 안내)

1. **측정 리포트** — 본인 기기에서 `on_device_ms`를 측정하고, 임의의 공개 이미지 분류 API 응답 시간과 비교한 표를 제출한다(각 3회 측정, 평균·표준편차 포함).
2. **동인 매핑** — 1교시의 4대 동인(Latency/Privacy/Reliability/Bandwidth) 중, 본인이 관심 있는 서비스가 온디바이스로 갈 때 얻는 이득을 1~2문장으로 서술한다.
3. **다음 주 예습** — `get_available_providers()` 결과를 캡처해 온다. 2주차에서 각 공급자가 어떤 하드웨어(CPU/GPU/NPU)에 매핑되는지 분석한다.

> 교수님을 위한 Tip: 실습의 성패는 "환경 설치"에서 갈린다. `torch`/`torchvision` 설치와 모델 내보내기(3.1)가 네트워크 사정으로 느릴 수 있으니, `mobilenetv2.onnx`를 미리 내보내 강의 자료로 함께 배포하면 실습이 매끄럽다. 첫 주는 설치 트러블슈팅 시간을 넉넉히 잡고, 설치 완료 자체를 출석 체크로 삼는 것을 권한다.

---

### 3교시 정리
- 온디바이스 추론 환경을 설치하고 첫 추론을 실행했다.
- 클라우드와 온디바이스의 지연 차이를 측정·관찰했다.
- 그 차이의 근본 원인이 **데이터 이동**임을 확인했고, 이것이 앞으로 배울 경량화·가속 기술의 출발점이다.
