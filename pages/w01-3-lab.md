# 01주차 3교시. 실험: 온디바이스 추론 지연의 실증적 관찰

**실험 목표** — 1교시의 지연 분해 모델을 **실측으로 검증**한다. 같은 추론을 로컬과 클라우드 경로로 수행해 지연을 측정하고, 그 차이의 원인을 모델의 항으로 설명한다. 1주차 실습은 코딩 실력보다 *측정 프로토콜과 해석*에 목적이 있다.

> 준비물: 노트북(Windows/macOS/Linux) 또는 실습 보드. GPU는 필요 없다.

**연구 질문과 가설** — 학술 실험답게 질문을 먼저 명시한다.

- **RQ1.** 동일 모델·동일 입력의 단대단 지연에서, 클라우드 경로와 온디바이스 경로의 차이는 얼마이며 어느 항이 지배하는가?
- **가설 H1.** 클라우드 경로 지연의 지배 항은 서버의 연산 시간이 아니라 **네트워크 전송·전파·대기**$(D/B + T_{prop} + T_{queue})$일 것이다.
- **가설 H2.** 온디바이스 지연은 클라우드 지연보다 **분산(변동폭)이 작을** 것이다(결정적 지연).

---

## 3.1 실험 환경 구축 (15분)

파이썬 가상환경을 만들고 온디바이스 추론 런타임을 설치한다. 설치가 간단하고 크로스 플랫폼인 **ONNX Runtime**을 사용한다.

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

이어서 실험 대상 모델(MobileNetV2[1])을 ONNX로 내보낸다. (조교가 `mobilenetv2.onnx`를 사전 배포한 경우 이 단계는 건너뛴다.)

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

## 3.2 측정 프로토콜과 실측 (25분)

지연 측정에는 함정이 많다. 학술적으로 방어 가능한 결과를 위해 세 가지 프로토콜을 지킨다.

1. **워밍업 제외** — 최초 1회는 모델 적재·메모리 할당이 섞인 **콜드 스타트**이므로 표본에서 제외한다.
2. **반복 측정(n=30)** — 단발 측정은 우연에 좌우된다. 30회를 각각 기록해 분포를 본다.
3. **평균만 보고하지 않는다** — 지연 분포는 오른쪽 꼬리가 긴 경우가 많아 평균이 왜곡된다. **중앙값(p50)과 꼬리 지연(p95)** 을 함께 보고한다. 실서비스 SLA가 참조하는 값도 꼬리 지연이다.

```python
import time, numpy as np, onnxruntime as ort

sess = ort.InferenceSession("mobilenetv2.onnx",
                            providers=["CPUExecutionProvider"])
name = sess.get_inputs()[0].name
x = np.random.rand(1, 3, 224, 224).astype(np.float32)   # 더미 입력 1장

# 프로토콜 1: 워밍업(콜드 스타트 제외)
sess.run(None, {name: x})

# 프로토콜 2·3: n=30 반복 → 평균·표준편차·중앙값·p95
samples = []
for _ in range(30):
    t0 = time.perf_counter()
    sess.run(None, {name: x})
    samples.append((time.perf_counter() - t0) * 1000)   # ms
s = np.array(samples)
print(f"온디바이스 추론 (n=30)")
print(f"  평균 {s.mean():.1f} ms  ±{s.std():.1f}")
print(f"  중앙값(p50) {np.percentile(s,50):.1f} ms   p95 {np.percentile(s,95):.1f} ms")
```

이제 같은 한 장을 **클라우드 API**로 보냈다고 하자. 1교시의 분해식 그대로다.

```
클라우드 총 지연 = 전처리 + 업링크 전송 + 전파/대기 + 서버 추론 + 다운링크 전송
```

로컬 추론이 순수 연산 시간만 든다면, 클라우드는 여기에 **왕복 네트워크 시간**이 얹힌다. 아래는 두 경우를 나란히 둔 관찰 예시다.

![실습 관찰: 동일 추론의 응답 지연 비교(예시). Cloud API는 전처리+네트워크 왕복+서버 추론으로 약 210ms, 온디바이스는 약 15ms. 지연의 대부분은 연산이 아니라 데이터 이동에서 발생한다](../assets/w01_p3_latency_bench_06.png)

> 관찰 포인트: 위 값은 개념 이해용 예시다. 실제 수업에서는 각자 측정한 값을 적고, 사용 중인 클라우드 API의 응답 시간과 비교한다. H2(분산 비교)를 검증하려면 클라우드 쪽도 같은 프로토콜로 반복 측정해야 한다:
>
> ```bash
> for i in $(seq 30); do curl -s -o /dev/null -w "%{time_total}\n" <API_URL>; done
> ```
> H1 검증의 핵심: 클라우드 총 지연에서 서버 추론 시간(로컬 측정값으로 근사)을 빼 보면, 나머지 — 즉 **데이터 이동** — 가 지배적임이 드러난다.

---

## 3.3 결과 분석과 타당성 위협 (Threats to Validity) (5분)

측정 결과를 보고할 때는 결과를 흔들 수 있는 요인도 함께 명시하는 것이 학술적 관행이다.

- **열 스로틀링(thermal throttling)** — 반복 측정 중 기기가 뜨거워지면 후반 표본이 느려진다. (초반 15회 vs 후반 15회 평균을 비교해 보라.)
- **백그라운드 부하** — 다른 프로세스가 CPU를 점유하면 꼬리 지연(p95)이 튄다.
- **네트워크 조건** — 클라우드 측정은 시간대·망 혼잡에 민감하므로, 측정 시각과 네트워크 종류(Wi-Fi/유선/LTE)를 기록한다.
- **표본 크기** — n=30은 수업용 최소치다. p95 같은 꼬리 통계는 n이 작을수록 불안정하므로, 보고서에는 n과 산포(표준편차 또는 p95)를 반드시 병기한다.

> 한 줄 정리: 숫자 하나가 아니라 "조건 + 분포"가 실험 결과다 — 이 습관이 11주차 프로파일링과 14주차 캡스톤 평가(측정 근거 필수)의 기초가 된다.

---

## 3.4 과제: 미니 실험 보고서 (5분 안내)

측정을 **1쪽 실험 보고서(IMRaD 축약형)** 로 제출한다.

1. **방법(Method)** — 기기 사양, n, 워밍업 처리, 측정 시각·네트워크 조건.
2. **결과(Results)** — 온디바이스 vs 클라우드 API의 평균±표준편차·p50·p95 비교 표 (각 3회 이상 세션).
3. **분석(Discussion)** — H1·H2가 지지되는가? 클라우드 지연에서 데이터 이동 항이 차지하는 비율 추정.
4. **동인 매핑** — 1교시 4대 동인 중, 본인 관심 서비스가 온디바이스로 갈 때 얻는 이득 1~2문장.
5. **다음 주 예습** — `get_available_providers()` 결과를 캡처해 온다. 2주차에서 각 공급자가 어떤 하드웨어(CPU/GPU/NPU)에 매핑되는지 분석한다.

> 교수님을 위한 Tip: 실습의 성패는 "환경 설치"에서 갈린다. `torch`/`torchvision` 설치와 모델 내보내기(3.1)가 네트워크 사정으로 느릴 수 있으니, `mobilenetv2.onnx`를 미리 내보내 강의 자료로 배포하면 매끄럽다. 첫 주는 설치 트러블슈팅 시간을 넉넉히 잡고, 설치 완료 자체를 출석 체크로 삼는 것을 권한다.

---

### 3교시 정리
- 연구 질문(RQ1)과 가설(H1·H2)을 세우고, 워밍업 제외·반복 측정·꼬리 지연 보고의 프로토콜로 실측했다.
- 클라우드와 온디바이스의 지연 차이의 지배 요인이 **데이터 이동**임을 실증했다.
- 타당성 위협을 명시하는 보고 습관을 익혔다 — 이것이 이후 모든 주차의 측정 실습과 기말 캡스톤의 기본기다.

### 참고문헌
[1] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, "MobileNetV2: Inverted Residuals and Linear Bottlenecks," in *Proc. IEEE CVPR*, 2018, pp. 4510–4520.
