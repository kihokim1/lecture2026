# 07주차 1교시. NAS와 하드웨어 인지형 설계

**강의 목표** — 아키텍처 설계를 자동 탐색 문제로 정식화하고, 정확도만이 아니라 **실측 지연을 목적 함수에 넣는** 하드웨어 인지형 설계까지 한 교시로 조망한다.

4~6주차의 경량화(프루닝·양자화·지식 증류)는 **주어진 모델을 다루는** 기법이었다. 7주차의 **Neural Architecture Search(NAS)** 는 한 단계 더 나아가 **모델 구조 그 자체를 자동으로 탐색**한다. 1주차에서 세운 제약 최적화 정식화와 파레토 프론티어가, 여기서 실제 알고리즘으로 구현되는 셈이다.

> **이번 주차 운영** — 1교시는 위 내용의 강의이고, **2·3교시에는 중간고사(범위 1~6주차)** 를 치른다. 오늘 배우는 NAS는 시험 범위가 아니다.

---

## 1.1 NAS의 등장 배경 (8분)

![수동 설계 vs NAS — 사람이 설계-학습-평가를 반복하는 Trial-and-Error 대신, NAS는 정의된 탐색 공간에서 알고리즘이 최적 구조를 자동으로 찾는다](../assets/w07_p1_manual_vs_nas_01.png)

ResNet·MobileNet 같은 유명 아키텍처는 대부분 **전문가의 직관과 시행착오**로 설계됐다. 이 **Manual Design** 에는 두 가지 한계가 있다.

- **고비용의 Trial-and-Error** — 설계 → 학습 → 평가를 사람이 반복해야 하므로 느리고 비싸다.
- **기기별 최적 구조가 다름** — 갤럭시와 아이폰, 젯슨과 MCU는 메모리 계층·가속기가 달라, 한 모델이 모든 기기에서 최적일 수 없다(2주차의 하드웨어 다양성).

Zoph와 Le는 순환 신경망 컨트롤러가 아키텍처를 생성하고 그 성능을 보상으로 학습하는 방식으로, **아키텍처 설계 자체를 학습 문제로 바꿀 수 있음**을 보였다[1]. 사람이 "탐색할 범위"만 정하면 알고리즘이 그 안에서 구조를 찾는다.

> 한 줄 정리: NAS는 사람의 직관·시행착오에 의존하던 아키텍처 설계를 자동 탐색 문제로 바꾼다.

---

## 1.2 NAS의 3대 요소 (12분)

Elsken 등의 정리에 따르면 NAS는 세 구성 요소로 분해된다[2].

![NAS의 3대 요소 — Search Space(무엇을 고를 수 있나), Search Strategy(어떻게 찾나), Performance Estimation(얼마나 좋은지 추정)](../assets/w07_p1_three_components_02.png)

- **탐색 공간(Search Space)** — 모델이 가질 수 있는 모든 구조의 집합. 어떤 연산(Conv, Depthwise Conv, Pooling 등)과 연결·깊이를 허용할지 정의한다. 공간이 넓으면 자유롭지만 탐색이 어려워진다.
- **탐색 전략(Search Strategy)** — 그 넓은 공간에서 좋은 후보를 **어떻게 찾을지**의 방법. 강화학습(RL), 진화 알고리즘(EA), 미분 가능 탐색(Gradient-based)이 대표적이다.
- **성능 예측(Performance Estimation)** — 후보 구조가 얼마나 좋은지 추정하는 방법. 모든 후보를 끝까지 학습시키면 비용이 폭발하므로, 조기 종료·가중치 공유·성능 예측기로 **빠르게 추정**한다.

이 셋이 "정의된 공간에서 → 전략으로 후보를 뽑고 → 성능을 추정"하는 순환을 돈다. **성능 예측이 NAS 비용의 핵심 변수**라는 점을 기억해 두자 — §1.5의 Once-for-All이 바로 이 항을 공략한다.

> 한 줄 정리: NAS = 탐색 공간(무엇을) × 탐색 전략(어떻게) × 성능 예측(얼마나 좋은지)의 조합이다[2].

---

## 1.3 탐색 전략 개관 (10분)

![탐색 전략 — 강화학습(RL)은 컨트롤러가 구조를 생성하고 보상으로 학습, 진화 알고리즘(EA)은 변이·선택으로 세대를 개선, 미분 가능(Gradient) 방식은 구조를 연속으로 완화해 경사하강법으로 탐색](../assets/w07_p1_search_strategies_03.png)

- **강화학습(RL)** — 컨트롤러가 구조를 생성하고, 그 성능을 **보상**으로 받아 더 좋은 구조를 생성하도록 학습한다[1]. 강력하지만 탐색 비용이 매우 크다.
- **진화 알고리즘(EA)** — 후보 구조 집단을 **변이·선택**하며 세대를 거쳐 개선한다. 병렬화가 쉽고 탐색 공간이 유연하다.
- **미분 가능 탐색(Gradient-based)** — 이산적인 "어떤 연산?" 선택을 **연속적으로 완화(continuous relaxation)** 해 경사하강법으로 구조를 학습한다. DARTS[3]가 대표적으로, 각 연산에 가중치 $\alpha$ 를 두고 그 소프트맥스 가중합으로 후보를 표현한 뒤 $\alpha$ 를 학습하고, 마지막에 가장 큰 $\alpha$ 의 연산만 남긴다. 탐색 비용을 크게 낮춰 NAS 대중화의 전환점이 됐다.

> 한 줄 정리: 이산적 구조 선택을 연속 가중치로 완화한 미분 가능 탐색이 NAS를 실용 단계로 옮겨 놓았다[3]. (DARTS의 수식 전개는 본 주차 심화 자료 §2.1, 탐색 불안정성 논의는 §2.4 참조.)

---

## 1.4 Hardware-aware NAS와 파레토 프론티어 (12분)

초기 NAS는 오직 **정확도**만 목표로 했다. 그러나 온디바이스에서는 **지연·전력**도 똑같이 중요하다. 1주차에서 세운 제약 최적화 문제가 여기서 되살아난다.

![정확도-지연 Pareto Front — 정확도와 지연의 트레이드오프 위에서, 더 나은 접점이 없는 최적해들의 경계(Pareto Front)를 찾는다. 기기 제약에 따라 그 위의 다른 점을 선택한다](../assets/w07_p2_pareto_05.png)

- **파레토 최적성(Pareto Optimality)** — "정확도를 더 높이려면 지연이 늘고, 지연을 줄이려면 정확도가 준다"는 트레이드오프에서, 어느 한쪽을 손해 없이 개선할 수 없는 최적해들의 경계가 **파레토 프론티어**다. 모바일용·서버용 모델은 이 경계 위의 서로 다른 점이다. **NAS는 이 경계를 자동으로 탐색하는 기법**이다(1주차 2교시 §2.1).
- **지연을 목적 함수에 넣기** — MnasNet은 정확도와 지연을 함께 담은 다목적 보상을 사용한다[4]. 목표 지연 $T$ 에 대해

$$\text{maximize}\quad ACC(m) \times \left[\frac{LAT(m)}{T}\right]^{w}$$

  형태로, 지연이 $T$ 를 넘으면 벌점이 붙도록 가중치 $w\ (w<0)$ 를 잡는다. **핵심은 시뮬레이션이 아니라 실제 기기에서 측정한 지연**을 쓴다는 점이다. MnasNet은 실제 스마트폰에서 지연을 측정해 탐색에 반영했다.
- **ProxylessNAS** — 작은 대리(proxy) 과제가 아니라 **타깃 과제와 타깃 하드웨어에서 직접** 탐색한다[5]. 기기별 실측 지연을 미분 가능한 형태로 모델링해 목적 함수에 넣는다.

2주차의 "같은 모델도 하드웨어마다 실행 효율이 다르다"가 여기서 **설계 목적 함수**로 들어온다. 그리고 1주차에서 배운 교훈 — FLOPs가 줄어도 실제 지연은 그대로일 수 있다 — 이 실측 지연을 고집해야 하는 이유다.

> 한 줄 정리: Hardware-aware NAS는 정확도와 함께 '실제 기기의 실측 지연'을 목적 함수에 넣어, 파레토 프론티어 위에서 기기 맞춤 구조를 찾는다[4][5].

---

## 1.5 Once-for-All: 학습 1회, 배포 다수 (8분)

기기마다 NAS를 다시 돌리면 비용이 기기 수에 비례해 폭발한다. **Once-for-All(OFA)** 은 이 문제를 정면으로 해결한다[6].

![Once-for-All — 거대 Supernet을 한 번 학습해두면, MCU·모바일·엣지GPU 등 기기별 제약에 맞는 서브넷을 재학습 없이 골라 배포할 수 있다](../assets/w07_p2_ofa_06.png)

모든 후보를 포함한 거대 **Supernet을 한 번만 학습**한 뒤, 다양한 하드웨어 제약에 맞는 **서브 네트워크를 추가 학습 없이 추출**한다. 서브넷들이 서로 간섭하지 않도록 커널 크기·깊이·너비를 점진적으로 축소하며 학습하는 **progressive shrinking** 이 핵심 기법이다.

이는 1주차 2교시에서 말한 **"학습 1회, 배포 다수(train once, deploy many)"** 의 구현이다. 성능 예측(§1.2의 세 번째 요소) 비용을 supernet의 공유 가중치로 상환한 것이라고도 볼 수 있다.

> 교수님을 위한 Tip: 마지막에 "논문 수치만큼 내 젯슨/라즈베리파이에서도 전성비가 나올까?"를 던져라. 시뮬레이션과 실제 임베디드 하드웨어의 괴리를 짚는 감각이 11주차 프로파일링과 기말 캡스톤으로 이어진다. 다만 곧바로 시험이 이어지므로 길게 끌지 말고 한 문장 질문으로 남겨두는 편이 좋다.

---

### 7주차 핵심 키워드
- **Search Space** — 모델이 가질 수 있는 모든 구조의 집합.
- **Supernet** — 모든 후보 연산을 포함한 거대 네트워크.
- **Hardware-aware** — 실제 측정 지연·전력을 설계 지표로 삼는 방식.
- **Pareto Front** — 정확도와 효율 사이에서 어느 한쪽을 손해 없이 개선할 수 없는 최적해들의 경계.

### 1교시 복습 질문
1. Manual Design의 두 가지 근본 한계는?
2. NAS 3대 요소를 각각 한 문장으로 정의하라.
3. Hardware-aware NAS에서 '실측 지연'을 고집해야 하는 이유를 1주차의 FLOPs 교훈과 연결해 설명하라.
4. OFA가 "학습 1회, 배포 다수"를 가능하게 한 방식은?

### 읽기 자료
- **필수** — Elsken et al., "Neural Architecture Search: A Survey"[2]의 3대 요소 부분.
- **권장** — Cai et al., "Once-for-All"[6]; Tan et al., "MnasNet"[4]. DARTS의 수식 전개는 본 주차 심화 자료 §2.1.

### 참고문헌
[1] B. Zoph and Q. V. Le, "Neural Architecture Search with Reinforcement Learning," in *Proc. ICLR*, 2017.
[2] T. Elsken, J. H. Metzen, and F. Hutter, "Neural Architecture Search: A Survey," *Journal of Machine Learning Research*, vol. 20, no. 55, pp. 1–21, 2019.
[3] H. Liu, K. Simonyan, and Y. Yang, "DARTS: Differentiable Architecture Search," in *Proc. ICLR*, 2019.
[4] M. Tan, B. Chen, R. Pang, V. Vasudevan, M. Sandler, A. Howard, and Q. V. Le, "MnasNet: Platform-Aware Neural Architecture Search for Mobile," in *Proc. IEEE CVPR*, 2019, pp. 2820–2828.
[5] H. Cai, L. Zhu, and S. Han, "ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware," in *Proc. ICLR*, 2019.
[6] H. Cai, C. Gan, T. Wang, Z. Zhang, and S. Han, "Once-for-All: Train One Network and Specialize It for Efficient Deployment," in *Proc. ICLR*, 2020.
