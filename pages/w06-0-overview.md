# 06주차. 모델 경량화 III: 지식 증류

> Knowledge Distillation
> 주당 3시간 · 이론 70% / 실습 30% · 교시당 약 50~60분

## 학습 목표
- 지식 증류(KD)의 동기와 Teacher-Student 구조를 설명할 수 있다.
- Soft target·Temperature·Dark Knowledge의 의미와 정보 이득을 이해한다.
- 증류 손실 함수(KL-Divergence + Cross-Entropy, α·β 가중)를 수식으로 설계할 수 있다.
- Response/Feature/Relation 기반 지식과 FitNets·Self·Cross-Modal 증류를 비교할 수 있다.
- PyTorch로 KD를 구현해 "같은 크기의 학생 모델이 KD로 더 좋아지는가"를 측정할 수 있다.

## 교시 구성

| 교시 | 성격 | 하위 목차 | 배정(분) |
|:---:|------|-----------|:---:|
| **1교시** | 개론 | 1.1 왜 지식 증류인가 / 1.2 Soft target과 Temperature / 1.3 지식의 형태 | 50 |
| **2교시** | 심화 | 2.1 증류 손실 함수 설계 / 2.2 Feature 기반 증류(FitNets) / 2.3 Self·Online 증류 / 2.4 Cross-Modal 증류(AIoT) | 60 |
| **3교시** | 실습 | 3.1 KD 구현 / 3.2 KD 유무 성능 비교 / 3.3 과제·중간 프로젝트 점검 | 50 |

## 그림 목록
- `w06_p1_teacher_student_01.png` — Teacher-Student 구조
- `w06_p1_soft_vs_hard_02.png` — Hard label vs Soft label(Dark Knowledge)
- `w06_p1_temperature_03.png` — Softmax with Temperature
- `w06_p1_knowledge_types_04.png` — Response/Feature/Relation 기반 지식
- `w06_p2_loss_05.png` — 증류 손실 함수(KL + CE, α·β)
- `w06_p3_bench_06.png` — KD 유무에 따른 정확도(실습, 예시)

## 선수 연결
- 4주차(프루닝)·5주차(양자화)가 '기존 모델을 줄이는' 방식이었다면, 6주차는 '작은 모델을 더 잘 가르치는' 알고리즘적 경량화.
- 8주차 중간 프로젝트에서 Pruning·Quantization·KD 중 2개 이상을 조합하는 전략의 마지막 퍼즐.
