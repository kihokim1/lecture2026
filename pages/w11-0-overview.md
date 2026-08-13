# 11주차. 엣지 기반 언어 지능 및 On-Device LLM

> NLP on the Edge
> 주당 3시간 · 이론 70% / 실습 30% · 교시당 약 50~60분

## 학습 목표
- Transformer의 온디바이스 병목(Attention의 $O(N^2)$, KV Cache)을 설명할 수 있다.
- 경량 언어 모델(DistilBERT·TinyBERT·MobileBERT)과 온디바이스 음성 인식(Whisper)을 이해한다.
- 소형 거대 언어 모델(sLLM)과 4비트 양자화(AWQ·GPTQ)의 필요성을 설명할 수 있다.
- Speculative Decoding 등 생성 가속 기법의 원리를 이해한다.
- On-Device LLM 추론 엔진(llama.cpp·MLC LLM)과 토큰 생성 속도의 메모리 대역폭 종속성을 설명할 수 있다.

## 교시 구성

| 교시 | 성격 | 하위 목차 | 배정(분) |
|:---:|------|-----------|:---:|
| **1교시** | 개론 | 1.1 Transformer의 온디바이스 병목 / 1.2 경량 BERT / 1.3 온디바이스 음성 인식 | 55 |
| **2교시** | 심화 | 2.1 sLLM 트렌드 / 2.2 극단적 양자화(AWQ·GPTQ) / 2.3 Speculative Decoding / 2.4 추론 엔진과 On-device LoRA | 60 |
| **3교시** | 실습 | 3.1 KV Cache 메모리 추정 / 3.2 토큰 생성 속도 관찰 / 3.3 과제 | 50 |

## 그림 목록
- `w11_p1_attention_bottleneck_01.png` — Attention의 $O(N^2)$ 병목
- `w11_p1_kv_cache_02.png` — KV Cache 메모리 증가
- `w11_p1_compact_bert_03.png` — 경량 BERT(DistilBERT·TinyBERT·MobileBERT)
- `w11_p2_quant_4bit_04.png` — 4비트 양자화(AWQ·GPTQ) 메모리 절감
- `w11_p2_speculative_05.png` — Speculative Decoding
- `w11_p3_bench_06.png` — 토큰 생성 속도와 메모리 대역폭

## 선수 연결
- 5주차 양자화가 여기서 4비트(INT4)·FP8로 극단화된다.
- 10주차의 Transformer(ViT) 병목이 언어 모델에서 본격적으로 다뤄진다.
- 13주차 On-device 파인튜닝(LoRA)로 이어진다.
