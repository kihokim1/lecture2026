# 13주차. 연합 학습 및 프라이버시 보호형 AIoT

> Federated Learning & Privacy-Preserving AI
> 주당 3시간 · 이론 70% / 실습 30% · 교시당 약 50~60분

## 학습 목표
- 중앙 집중식 학습의 프라이버시 한계와 연합 학습(FL)의 동기를 설명할 수 있다.
- FedAvg 알고리즘(Local SGD + 가중치 평균화)과 Non-IID 데이터 문제를 이해한다.
- 통신 오버헤드와 그 효율화(Gradient 압축·클라이언트 선택)를 설명할 수 있다.
- 온디바이스 파인튜닝(Memory-efficient, LoRA/PEFT)의 원리를 이해한다.
- Differential Privacy·Secure Aggregation·TEE 등 프라이버시·보안 기법을 비교할 수 있다.

## 교시 구성

| 교시 | 성격 | 하위 목차 | 배정(분) |
|:---:|------|-----------|:---:|
| **1교시** | 개론 | 1.1 왜 연합 학습인가 / 1.2 핵심 알고리즘 FedAvg / 1.3 통신 오버헤드와 효율화 | 55 |
| **2교시** | 심화 | 2.1 온디바이스 파인튜닝(LoRA/PEFT) / 2.2 Differential Privacy / 2.3 Secure Aggregation과 하드웨어 보안 / 2.4 정리 | 60 |
| **3교시** | 실습 | 3.1 FedAvg 시뮬레이션 / 3.2 Non-IID 영향 관찰 / 3.3 과제 | 50 |

## 그림 목록
- `w13_p1_centralized_vs_fl_01.png` — 중앙 집중 vs 연합 학습
- `w13_p1_fedavg_02.png` — FedAvg 순환(모델 배포→로컬 학습→가중치 취합)
- `w13_p1_noniid_03.png` — Non-IID 데이터 문제
- `w13_p2_lora_peft_04.png` — 온디바이스 파인튜닝(LoRA/PEFT)
- `w13_p2_privacy_05.png` — Differential Privacy·Secure Aggregation·TEE
- `w13_p3_fedavg_sim_06.png` — FedAvg 시뮬레이션 수렴(실습, 예시)

## 선수 연결
- 1주차 4대 동인 중 Privacy가 학습 단계로 확장된다.
- 11주차 On-device LoRA가 온디바이스 파인튜닝으로 심화된다.
- 온디바이스 AI의 종착지: 추론(Inference)을 넘어 현장 학습(Training)으로.
