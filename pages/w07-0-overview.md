# 7주차. Neural Architecture Search & Hardware-aware Design

> NAS & 하드웨어 인지형 설계
> 주당 3시간 · 이론 70% / 실습 30% · 교시당 약 50~60분

## 학습 목표
- 사람이 설계한 아키텍처의 한계를 이해하고 NAS의 등장 배경을 설명할 수 있다.
- NAS의 3대 구성 요소(탐색 공간·탐색 전략·성능 예측)를 구분할 수 있다.
- DARTS의 연속 완화(continuous relaxation) 아이디어를 설명할 수 있다.
- Hardware-aware NAS(지연을 목적 함수에 산입)와 Pareto 최적성을 이해한다.
- ProxylessNAS·Once-for-All의 핵심 기여를 비교하고, 탐색 도구를 체험할 수 있다.

## 교시 구성

| 교시 | 성격 | 하위 목차 | 배정(분) |
|:---:|------|-----------|:---:|
| **1교시** | 개론 | 1.1 NAS의 등장 배경 / 1.2 NAS의 3대 요소 / 1.3 탐색 전략 개관 | 55 |
| **2교시** | 심화 | 2.1 Differentiable NAS(DARTS) / 2.2 Multi-Objective & Hardware-aware / 2.3 ProxylessNAS·Once-for-All / 2.4 한계와 연구 동향 | 60 |
| **3교시** | 실습 | 3.1 탐색 공간·제약 정의 / 3.2 NAS 도구 체험 / 3.3 과제·중간고사 안내 | 50 |

## 그림 목록
- `w07_p1_manual_vs_nas_01.png` — 수동 설계 vs NAS 자동 탐색
- `w07_p1_three_components_02.png` — NAS 3대 요소
- `w07_p1_search_strategies_03.png` — 탐색 전략(RL·EA·Gradient)
- `w07_p2_darts_04.png` — DARTS 연속 완화
- `w07_p2_pareto_05.png` — 정확도-지연 Pareto front
- `w07_p2_ofa_06.png` — Once-for-All(supernet → 서브넷 추출)

## 선수 연결
- 4~6주차의 경량화가 '주어진 모델을 다루는' 것이었다면, 7주차는 '모델 구조 자체를 탐색'한다.
- 2주차의 하드웨어 지연이 목적 함수로 들어오는 Hardware-aware 설계로 완성된다.
- **8주차 중간고사 범위(1~7주차)의 마지막 주차.**
