# -*- coding: utf-8 -*-
"""On-Device LLM 을 **프리필과 디코드로 쪼개서** 잰다.

generate() 한 번으로 재면 두 개의 완전히 다른 연산이 한 숫자에 섞인다.
- 프리필(prefill): 프롬프트 N개 토큰을 한 번에 — 행렬×행렬, 연산에 묶인다
- 디코드(decode): 토큰을 하나씩 — 행렬×벡터, 메모리 대역폭에 묶인다
"""
import time, json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.set_num_threads(2)
MID = "HuggingFaceTB/SmolLM2-135M-Instruct"


def load(mid=MID, dtype=torch.float32):
    tok = AutoTokenizer.from_pretrained(mid)
    m = AutoModelForCausalLM.from_pretrained(mid, dtype=dtype).eval()
    return tok, m


def model_bytes(m):
    """가중치가 차지하는 실제 바이트."""
    return sum(p.numel() * p.element_size() for p in m.parameters())


def kv_bytes(cfg, seq, batch=1, bytes_per=4):
    """KV 캐시 크기 = 2(K,V) × 층 × KV헤드 × 헤드차원 × 길이 × 배치 × 바이트."""
    hd = cfg.hidden_size // cfg.num_attention_heads
    kvh = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    return 2 * cfg.num_hidden_layers * kvh * hd * seq * batch * bytes_per


@torch.no_grad()
def prefill(m, ids):
    """프롬프트 전체를 한 번에 통과시킨다. 반환: (다음 토큰, 캐시, 걸린 시간 ms)"""
    t0 = time.perf_counter()
    out = m(input_ids=ids, use_cache=True)
    dt = (time.perf_counter() - t0) * 1000
    nxt = out.logits[:, -1:].argmax(-1)
    return nxt, out.past_key_values, dt


@torch.no_grad()
def decode_steps(m, nxt, cache, n, cur_len):
    """토큰을 하나씩 n개 만든다. 반환: 각 단계의 ms 리스트."""
    ts = []
    for i in range(n):
        pos = torch.tensor([[cur_len + i]])
        t0 = time.perf_counter()
        out = m(input_ids=nxt, past_key_values=cache, use_cache=True,
                cache_position=pos.flatten())
        ts.append((time.perf_counter() - t0) * 1000)
        cache = out.past_key_values
        nxt = out.logits[:, -1:].argmax(-1)
    return ts, nxt, cache


@torch.no_grad()
def run(m, ids, n_new, warm=False):
    """한 번의 생성을 프리필/디코드로 나눠 잰다."""
    nxt, cache, t_pre = prefill(m, ids)
    ts, _, _ = decode_steps(m, nxt, cache, n_new, ids.shape[1])
    a = np.array(ts)
    return dict(prompt=ids.shape[1], n_new=n_new,
                ttft=t_pre,                                  # 첫 토큰까지의 시간
                prefill_tps=ids.shape[1] / (t_pre / 1000),   # 프리필 처리량
                tpot=float(np.median(a)),                    # 토큰당 시간
                tpot_p95=float(np.percentile(a, 95)),
                decode_tps=1000 / float(np.median(a)),       # 디코드 처리량
                total=t_pre + a.sum(),
                steps=ts)


def make_ids(tok, n):
    """길이 n 짜리 프롬프트를 만든다(내용은 무의미, 길이만 통제)."""
    base = tok("The quick brown fox jumps over the lazy dog. ", return_tensors="pt").input_ids
    reps = int(np.ceil(n / base.shape[1])) + 1
    return base.repeat(1, reps)[:, :n]
