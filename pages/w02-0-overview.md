# 02주차. 온디바이스 AI 하드웨어 아키텍처

> Hardware Architectures for Edge AI
> 주당 3시간 · 3교시 × 약 50분 · 이론 70% / 실습 30%

## 학습 목표
- 폰 노이만 구조의 병목(Memory Wall)이 AI 연산에서 왜 치명적인지 설명할 수 있다.
- CPU·GPU·DSP·NPU의 연산 특성과 역할 분담(이기종 컴퓨팅)을 비교할 수 있다.
- Systolic Array 등 도메인 특화 아키텍처가 데이터 재사용으로 전력·성능을 얻는 원리를 설명할 수 있다.
- 임베디드 메모리 계층·타일링·전성비(Performance per Watt)의 관점에서 엣지 디바이스를 선택할 수 있다.
- 본인 기기의 실행 공급자·스레드 설정이 추론 지연에 미치는 영향을 직접 측정·해석할 수 있다.

## 교시 구성

| 교시 | 성격 | 하위 목차 | 배정(분) |
|:---:|------|-----------|:---:|
| **1교시** | 개론 | 1.1 폰 노이만 병목과 Memory Wall / 1.2 가속기별 연산 특징(CPU·GPU·DSP·NPU) / 1.3 이기종 컴퓨팅의 등장 | 55 |
| **2교시** | 심화 | 2.1 도메인 특화 아키텍처와 Systolic Array / 2.2 임베디드 메모리 계층과 Tiling / 2.3 전성비와 DVFS / 2.4 엣지 디바이스 비교 | 60 |
| **3교시** | 실습 | 3.1 내 기기의 하드웨어·공급자 확인 / 3.2 스레드·공급자에 따른 지연 측정 / 3.3 과제 | 50 |

## 그림 목록
- `w02_p1_memory_wall_01.png` — 연산 성능 vs 메모리 대역폭 격차(Memory Wall)
- `w02_p1_accelerators_02.png` — CPU·GPU·DSP·NPU 특성 비교
- `w02_p2_systolic_array_03.png` — Systolic Array 데이터 흐름
- `w02_p2_memory_hierarchy_04.png` — 임베디드 메모리 계층 + Tiling
- `w02_p2_edge_devices_05.png` — 주요 엣지 디바이스 전성비 비교
- `w02_p3_threads_bench_06.png` — 실습: 스레드 수에 따른 추론 지연(예시)

## 선수 연결
- 1주차 과제 "get_available_providers() 캡처"를 3.1에서 하드웨어 매핑으로 분석한다.
- 1주차 4계층 스택 중 **Hardware Level**을 이번 주에 깊게 다룬다.
