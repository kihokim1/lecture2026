# 08주차. TinyML 및 초저전력 온디바이스 AI

> TinyML & Ultra-low Power AI
> 주당 3시간 · 3교시 × 약 50분 · 이론 70% / 실습 30%

## 학습 목표
- TinyML의 정의와 MCU 기반 생태계(Always-on)를 설명할 수 있다.
- RAM 수백 KB 환경의 극한 자원 제약(Memory Wall extreme, Compute Gap)을 이해한다.
- TinyML 소프트웨어 스택(TFLM, CMSIS-NN)의 역할을 설명할 수 있다.
- 에너지 소모 분석과 Duty Cycling, Depthwise Separable Convolution의 절감 원리를 이해한다.
- Peak Memory 개념과 MCU 배포 파이프라인(xxd 포함)을 설명할 수 있다.

## 교시 구성

| 교시 | 성격 | 하위 목차 | 배정(분) |
|:---:|------|-----------|:---:|
| **1교시** | 개론 | 1.1 TinyML의 정의와 생태계 / 1.2 극한 자원 제약 / 1.3 TinyML 소프트웨어 스택 | 55 |
| **2교시** | 심화 | 2.1 에너지 소모 분석과 Duty Cycling / 2.2 Depthwise Separable Convolution / 2.3 Peak Memory와 In-place / 2.4 최신 연구(Battery-less) | 60 |
| **3교시** | 실습 | 3.1 배포 파이프라인 / 3.2 모델을 펌웨어로(xxd) / 3.3 과제 | 50 |

## 그림 목록
- `w08_p1_tinyml_spectrum_01.png` — 컴퓨팅 스펙트럼(Cloud→MCU)과 자원 규모
- `w08_p1_mcu_constraints_02.png` — MCU의 자원 제약
- `w08_p1_sw_stack_03.png` — TinyML 소프트웨어 스택
- `w08_p2_duty_cycling_04.png` — Duty Cycling 전력 프로파일
- `w08_p2_depthwise_05.png` — 표준 Conv vs Depthwise Separable Conv
- `w08_p3_deploy_06.png` — MCU 배포 파이프라인과 Peak Memory

## 선수 연결
- 2주차의 저전력 MCU(Cortex-M)와 Memory Wall이 극단으로 나타나는 환경.
- 5주차 양자화(정수 INT8)가 TinyML의 사실상 필수 전제.
- 7주차 중간고사 이후, 후반부(응용) 첫 주차.
