# -*- coding: utf-8 -*-
"""실험 — On-Device LLM 의 지연은 두 개의 다른 연산으로 이루어져 있다."""
import io, json, time
import numpy as np
import torch, torch.nn as nn
import llm_common as L

res = {}
tok, m = L.load()
cfg = m.config
B32 = L.model_bytes(m)
print(f"모델 SmolLM2-135M · 파라미터 {sum(p.numel() for p in m.parameters()):,} · "
      f"가중치 {B32/1e6:.1f} MB (fp32)")
print(f"층 {cfg.num_hidden_layers} · hidden {cfg.hidden_size} · "
      f"질의 헤드 {cfg.num_attention_heads} · KV 헤드 {cfg.num_key_value_heads} "
      f"(GQA {cfg.num_attention_heads // cfg.num_key_value_heads}:1)")
res["cfg"] = dict(params=sum(p.numel() for p in m.parameters()), bytes=B32,
                  layers=cfg.num_hidden_layers, hidden=cfg.hidden_size,
                  heads=cfg.num_attention_heads, kv_heads=cfg.num_key_value_heads,
                  vocab=cfg.vocab_size, max_pos=cfg.max_position_embeddings)

# ── A. 프리필과 디코드는 같은 모델의 다른 얼굴 ─────────────────────────────
print("\n[A] 프리필과 디코드 — 같은 모델, 같은 CPU")
ids = L.make_ids(tok, 128)
L.run(m, ids, 3)
r = L.run(m, ids, 24)
res["split"] = {k: v for k, v in r.items() if k != "steps"}
res["split"]["steps"] = r["steps"]
print(f"  프리필 — 128 토큰을 {r['ttft']:.1f} ms 에  →  {r['prefill_tps']:.0f} tok/s")
print(f"  디코드 — 토큰 하나에 {r['tpot']:.2f} ms  →  {r['decode_tps']:.1f} tok/s")
print(f"  같은 모델·같은 CPU 인데 처리량이 {r['prefill_tps']/r['decode_tps']:.1f}배 차이 난다")

# ── B. 디코드는 대역폭에 묶여 있다 — 양자화로 확인 ──────────────────────────
print("\n[B] 동적 INT8 양자화 — 9주차와 반대 결과가 나오는가")
q = torch.ao.quantization.quantize_dynamic(m, {nn.Linear}, dtype=torch.qint8).eval()


def ser_bytes(mm):
    b = io.BytesIO(); torch.save(mm.state_dict(), b); return b.tell()


BQ = ser_bytes(q)
L.run(q, ids, 3)
rq = L.run(q, ids, 24)
res["quant"] = dict(fp32=dict(bytes=B32, **{k: v for k, v in r.items() if k != "steps"}),
                    int8=dict(bytes=BQ, **{k: v for k, v in rq.items() if k != "steps"}))
print(f"  FP32  {B32/1e6:6.1f} MB | TPOT {r['tpot']:6.2f} ms | 디코드 {r['decode_tps']:6.2f} tok/s "
      f"| 프리필 {r['prefill_tps']:5.0f} tok/s")
print(f"  INT8  {BQ/1e6:6.1f} MB | TPOT {rq['tpot']:6.2f} ms | 디코드 {rq['decode_tps']:6.2f} tok/s "
      f"| 프리필 {rq['prefill_tps']:5.0f} tok/s")
print(f"  → 디코드 {r['tpot']/rq['tpot']:.2f}배 빨라짐 (9주차 탐지 모델에서는 1.52배 느려졌다)")
print(f"  → 대역폭 환산: FP32 {r['decode_tps']*B32/1e9:.2f} GB/s · INT8 {rq['decode_tps']*BQ/1e9:.2f} GB/s")
res["quant"]["bw_fp32"] = r["decode_tps"] * B32 / 1e9
res["quant"]["bw_int8"] = rq["decode_tps"] * BQ / 1e9

# ── C. 프롬프트가 길어지면 TTFT 는 어떻게 되나 ──────────────────────────────
print("\n[C] 프롬프트 길이와 TTFT · 문맥 길이와 TPOT")
res["ctx"] = []
for n in [64, 128, 256, 512, 1024, 2048]:
    x = L.make_ids(tok, n)
    L.run(m, x, 2)
    rr = L.run(m, x, 12)
    kvb = L.kv_bytes(cfg, n + 12)
    res["ctx"].append(dict(prompt=n, ttft=rr["ttft"], prefill_tps=rr["prefill_tps"],
                           tpot=rr["tpot"], decode_tps=rr["decode_tps"], kv=kvb))
    print(f"  프롬프트 {n:5d}  TTFT {rr['ttft']:8.1f} ms ({rr['prefill_tps']:5.0f} tok/s) | "
          f"TPOT {rr['tpot']:6.2f} ms ({rr['decode_tps']:5.1f} tok/s) | "
          f"KV {kvb/1e6:6.1f} MB")

# ── D. KV 캐시는 언제 가중치를 넘어서나 ─────────────────────────────────────
print("\n[D] KV 캐시 vs 가중치 — 교차점")
res["kv"] = []
for n in [512, 1024, 2048, 4096, 8192]:
    for bsz in [1, 8, 32]:
        kb = L.kv_bytes(cfg, n, bsz)
        res["kv"].append(dict(seq=n, batch=bsz, kv=kb, ratio=kb / B32))
mha = L.kv_bytes(cfg, 8192) * cfg.num_attention_heads / cfg.num_key_value_heads
cross = B32 / (L.kv_bytes(cfg, 1) )
print(f"  GQA {cfg.num_attention_heads}:{cfg.num_key_value_heads} 기준 토큰 1개당 KV = "
      f"{L.kv_bytes(cfg,1)/1024:.1f} KB (fp32)")
print(f"  가중치({B32/1e6:.0f} MB)를 넘어서는 지점 = 배치1에서 {cross:,.0f} 토큰 "
      f"(최대 문맥 {cfg.max_position_embeddings} 의 {cross/cfg.max_position_embeddings:.1f}배)")
for n in [2048, 8192]:
    for bsz in [1, 8, 32]:
        kb = L.kv_bytes(cfg, n, bsz)
        print(f"    문맥 {n:5d} · 배치 {bsz:2d}  KV {kb/1e6:8.1f} MB "
              f"({kb/B32:5.2f}× 가중치)")
res["kv_per_token"] = L.kv_bytes(cfg, 1)
res["kv_crossover_tokens"] = cross
res["kv_mha_would_be"] = mha
print(f"  만약 GQA 가 아니라 MHA 였다면 같은 문맥에서 "
      f"{cfg.num_attention_heads // cfg.num_key_value_heads}배인 {mha/1e6:.1f} MB")

# ── E. 배치 — 9주차와 반대로 여기서는 이득이다 ──────────────────────────────
print("\n[E] 배치 디코드 — 9주차 결론이 뒤집히는가")
res["batch"] = []
base = L.make_ids(tok, 128)
for bsz in [1, 2, 4, 8]:
    x = base.repeat(bsz, 1)
    with torch.no_grad():
        nxt, cache, _ = L.prefill(m, x)
        L.decode_steps(m, nxt, cache, 2, x.shape[1])
        nxt, cache, tpre = L.prefill(m, x)
        ts, _, _ = L.decode_steps(m, nxt, cache, 12, x.shape[1])
    step = float(np.median(ts))
    res["batch"].append(dict(B=bsz, step_ms=step, per_token=step / bsz,
                             tps=bsz * 1000 / step, ttft=tpre))
    print(f"  배치 {bsz}: 한 스텝 {step:6.2f} ms | 토큰당 {step/bsz:5.2f} ms | "
          f"처리량 {bsz*1000/step:6.1f} tok/s | 사용자가 느끼는 TPOT {step:6.2f} ms")

json.dump(res, open("/root/lab10/llm.json", "w"), ensure_ascii=False, default=float)
print("\n저장: llm.json")
