# -*- coding: utf-8 -*-
"""3교시 실습 — 학생이 그대로 따라 치는 코드. 교재의 출력을 여기서 검증한다."""
import time
import numpy as np
import torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.set_num_threads(2)
MID = "HuggingFaceTB/SmolLM2-135M-Instruct"
tok = AutoTokenizer.from_pretrained(MID)
m = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float32).eval()
cfg = m.config
BYTES = sum(p.numel() * p.element_size() for p in m.parameters())
print(f"파라미터 {sum(p.numel() for p in m.parameters()):,} · 가중치 {BYTES/1e6:.1f} MB (FP32)")
print(f"층 {cfg.num_hidden_layers} · hidden {cfg.hidden_size} · "
      f"질의 헤드 {cfg.num_attention_heads} · KV 헤드 {cfg.num_key_value_heads}")


# ── 1단계. 프리필과 디코드를 갈라 놓는다 ────────────────────────────────────
@torch.no_grad()
def split_measure(n_prompt=128, n_new=24):
    base = tok("The quick brown fox jumps over the lazy dog. ", return_tensors="pt").input_ids
    ids = base.repeat(1, n_prompt // base.shape[1] + 2)[:, :n_prompt]

    t0 = time.perf_counter()                       # ── 프리필
    out = m(input_ids=ids, use_cache=True)
    ttft = (time.perf_counter() - t0) * 1000
    nxt, cache = out.logits[:, -1:].argmax(-1), out.past_key_values

    steps = []                                     # ── 디코드
    for i in range(n_new):
        t0 = time.perf_counter()
        out = m(input_ids=nxt, past_key_values=cache, use_cache=True,
                cache_position=torch.tensor([n_prompt + i]))
        steps.append((time.perf_counter() - t0) * 1000)
        cache, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
    return ttft, float(np.median(steps)), n_prompt


split_measure(64, 3)                               # 워밍업
ttft, tpot, np_ = split_measure()
print(f"\n[1] 프리필 — {np_} 토큰을 {ttft:.1f} ms 에  →  {np_/(ttft/1000):.0f} tok/s")
print(f"    디코드 — 토큰 하나에 {tpot:.2f} ms  →  {1000/tpot:.1f} tok/s")
print(f"    같은 모델·같은 CPU 인데 처리량이 {(np_/(ttft/1000))/(1000/tpot):.0f}배 차이 난다")
print(f"    대역폭 환산: {(1000/tpot) * BYTES / 1e9:.2f} GB/s")

# ── 2단계. 대역폭 가설을 검증한다 ───────────────────────────────────────────
q = torch.ao.quantization.quantize_dynamic(m, {nn.Linear}, dtype=torch.qint8).eval()
import io
buf = io.BytesIO(); torch.save(q.state_dict(), buf); BQ = buf.tell()
m_fp32, m = m, q                                   # 잠시 바꿔 끼운다
split_measure(64, 3)
ttft_q, tpot_q, _ = split_measure()
m = m_fp32
print(f"\n[2] 동적 INT8 양자화")
print(f"    FP32  {BYTES/1e6:6.1f} MB | TPOT {tpot:6.2f} ms | 디코드 {1000/tpot:5.1f} tok/s")
print(f"    INT8  {BQ/1e6:6.1f} MB | TPOT {tpot_q:6.2f} ms | 디코드 {1000/tpot_q:5.1f} tok/s")
print(f"    → 바이트 {BYTES/BQ:.2f}배 감소 · 속도 {tpot/tpot_q:.2f}배 향상")
print(f"    → 9주차 탐지 모델에서는 같은 기법이 1.52배 느려지게 만들었다")

# ── 3단계. KV 캐시를 계산한다 ───────────────────────────────────────────────
hd = cfg.hidden_size // cfg.num_attention_heads
per_token = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * hd * 4
print(f"\n[3] KV 캐시 — 토큰당 {per_token/1024:.1f} KB (FP32)")
print(f"    (2 × 층 {cfg.num_hidden_layers} × KV헤드 {cfg.num_key_value_heads} "
      f"× 헤드차원 {hd} × 4바이트)")
for seq, bsz in [(2048, 1), (2048, 8), (8192, 1), (8192, 32)]:
    kv = per_token * seq * bsz
    print(f"    문맥 {seq:5d} · 배치 {bsz:2d} → {kv/1e6:8.1f} MB  ({kv/BYTES:5.2f}× 가중치)")
print(f"    MHA 였다면 {cfg.num_attention_heads//cfg.num_key_value_heads}배였을 것이다")

# ── 4단계. 내 언어로 예산을 세운다 ──────────────────────────────────────────
EN = ("On-device inference removes the round trip to the server. The model runs "
      "where the data is produced, so the answer does not have to travel.")
KO = ("온디바이스 추론은 서버까지 다녀오는 왕복을 없앤다. 모델이 데이터가 만들어지는 "
      "곳에서 돌기 때문에 답이 이동할 필요가 없다.")
print(f"\n[4] 토큰 대 단어 — 내 토크나이저로 직접 센다")
wpt = {}
for tag, s in [("영어", EN), ("한국어", KO)]:
    nt, nw = len(tok(s).input_ids), len(s.split())
    wpt[tag] = nw / nt
    print(f"    {tag:4s} 단어 {nw:3d} · 토큰 {nt:3d} → 단어당 토큰 {nt/nw:.2f}")
print(f"\n    묵독 238 wpm 을 따라가려면 (Brysbaert 2019)")
for tag in ["영어", "한국어"]:
    need = 238 / 60 / wpt[tag]
    got = 1000 / tpot
    print(f"    {tag:4s} 필요 {need:5.1f} tok/s · 실측 {got:5.1f} tok/s → "
          f"{got/need:.2f}배 {'여유 있다' if got >= need else '못 따라간다'}")
