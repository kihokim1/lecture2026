# 09주차. 엣지 기반 시각 지능 구현 (Computer Vision on the Edge)

> Computer Vision on the Edge
> 주당 3시간 · 3교시 × 약 50분 · 이론 70% / 실습 30%

## 학습 목표
- 모바일·엣지에 최적화된 경량 백본(MobileNet, EfficientNet-Edge)의 구조를 설명할 수 있다.
- 실시간 객체 탐지(YOLO, SSD)의 경량화와 후처리(NMS) 병목을 이해한다.
- 경량 시맨틱 세그멘테이션(DeepLabV3+, Atrous Convolution)의 원리를 설명할 수 있다.
- 엣지 가속 SDK(TensorRT, SNPE)의 역할과 선택 기준을 비교할 수 있다.
- 실시간 비전 파이프라인(전처리→추론→후처리)과 Zero-copy·FPS 측정을 설계할 수 있다.

## 교시 구성

| 교시 | 성격 | 하위 목차 | 배정(분) |
|:---:|------|-----------|:---:|
| **1교시** | 개론 | 1.1 경량 백본 네트워크 / 1.2 실시간 객체 탐지 / 1.3 Anchor-free와 NMS | 55 |
| **2교시** | 심화 | 2.1 경량 시맨틱 세그멘테이션 / 2.2 엣지 가속 SDK(TensorRT·SNPE) / 2.3 Vision Transformer on Edge / 2.4 정리 | 60 |
| **3교시** | 실습 | 3.1 실시간 비전 파이프라인 / 3.2 FPS·병목 측정 / 3.3 과제 | 50 |

## 그림 목록
- `w09_p1_backbones_01.png` — 경량 백본(MobileNet 진화·EfficientNet-Edge)
- `w09_p1_detection_02.png` — 실시간 객체 탐지(YOLO·SSD)
- `w09_p1_nms_03.png` — NMS 후처리와 Anchor-free
- `w09_p2_segmentation_04.png` — 경량 세그멘테이션(DeepLabV3+·Atrous)
- `w09_p2_accel_sdk_05.png` — 엣지 가속 SDK 비교
- `w09_p3_pipeline_06.png` — 실시간 비전 파이프라인(Zero-copy·FPS)

## 선수 연결
- 8주차 Depthwise Separable Convolution이 MobileNet 백본의 핵심으로 재등장.
- 3주차 추론 엔진·11주차 하드웨어 가속기 프로그래밍으로 이어지는 TensorRT·SNPE 실무.
