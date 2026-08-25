# 03주차. 온디바이스AI 프레임워크 및 추론 엔진

> Inference Engines & Frameworks
> 주당 3시간 · 3교시 × 약 50분 · 이론 70% / 실습 30%

## 학습 목표
- 학습용 프레임워크(PyTorch/TF)와 추론 전용 엔진의 차이를 설명할 수 있다.
- ONNX가 상호운용성(Interoperability) 표준으로서 하는 역할을 이해한다.
- 정적 그래프 최적화(Operator Fusion, Constant Folding, Dead Code Elimination)의 원리를 설명할 수 있다.
- TFLite·ExecuTorch·TVM의 내부 구조와 선택 기준, Delegate/컴파일러 접근의 차이를 비교할 수 있다.
- PyTorch→ONNX 변환과 그래프 최적화 수준 변화가 추론에 미치는 영향을 직접 측정할 수 있다.

## 교시 구성

| 교시 | 성격 | 하위 목차 | 배정(분) |
|:---:|------|-----------|:---:|
| **1교시** | 개론 | 1.1 학습 프레임워크 vs 추론 엔진 / 1.2 모델 표현식과 ONNX / 1.3 왜 '그래프'로 다루는가 | 50 |
| **2교시** | 심화 | 2.1 정적 그래프 최적화 3종 / 2.2 TFLite 아키텍처(FlatBuffer·Delegate) / 2.3 ExecuTorch·TVM / 2.4 IoT 프로토콜 연동 | 60 |
| **3교시** | 실습 | 3.1 PyTorch→ONNX 변환·시각화 / 3.2 그래프 최적화 수준별 지연 비교 / 3.3 과제 | 50 |

## 그림 목록
- `w03_p1_train_vs_infer_01.png` — 학습 프레임워크 vs 추론 엔진
- `w03_p1_onnx_hub_02.png` — 상호운용성 허브로서의 ONNX
- `w03_p2_graph_opt_03.png` — Operator Fusion(Conv+BN+ReLU 융합)
- `w03_p2_tflite_delegate_04.png` — TFLite Interpreter와 Delegate
- `w03_p2_frameworks_05.png` — TFLite·PyTorch/ExecuTorch·TVM 비교
- `w03_p3_optlevel_bench_06.png` — 그래프 최적화 수준별 지연(실습, 예시)

## 선수 연결
- 2주차 3교시에서 예고한 **Delegate**(연산을 가속기로 위임)를 2.2에서 본격적으로 다룬다.
- 1주차에서 만든 PyTorch→ONNX 변환 흐름을 3교시 실습에서 재사용한다.
