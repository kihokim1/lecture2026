# -*- coding: utf-8 -*-
"""실험 보충 — 사람이 읽는 속도와 견주면 얼마나 빠르면 되는가.

주의: "1 토큰 ≈ 0.75 단어" 는 OpenAI 토크나이저·영어 기준 값이다.
     우리 모델의 토크나이저로 **직접 세어** 확인한다. 한국어는 특히 다르다.
"""
import json
import numpy as np
import torch
import llm_common as L

tok, m = L.load()
out = {}

EN = ("On-device inference removes the round trip to the server. The model runs "
      "where the data is produced, so the answer does not have to travel. This "
      "changes the latency budget, the privacy story, and the cost structure all "
      "at once. But it also means the device must hold the weights in memory and "
      "read them back for every single token it produces.")
KO = ("온디바이스 추론은 서버까지 다녀오는 왕복을 없앤다. 모델이 데이터가 만들어지는 "
      "곳에서 돌기 때문에 답이 이동할 필요가 없다. 이것은 지연 예산과 프라이버시 "
      "이야기와 비용 구조를 한꺼번에 바꾼다. 그러나 그 대가로 기기는 가중치를 메모리에 "
      "들고 있어야 하고, 토큰 하나를 만들 때마다 그것을 다시 읽어야 한다.")

print("[F] 토큰 대 단어 비율 — 직접 세어 본다 (SmolLM2 토크나이저, 어휘 49,152)")
for tag, s in [("영어", EN), ("한국어", KO)]:
    n_tok = len(tok(s).input_ids)
    n_word = len(s.split())
    out[tag] = dict(tokens=n_tok, words=n_word, wpt=n_word / n_tok, tpw=n_tok / n_word)
    print(f"  {tag:4s} 단어 {n_word:3d}개 · 토큰 {n_tok:3d}개 → "
          f"토큰당 단어 {n_word/n_tok:.3f} · 단어당 토큰 {n_tok/n_word:.2f}")
print(f"  → OpenAI 도움말의 '토큰당 0.75 단어' 는 영어·자기네 토크나이저 기준이다. "
      f"우리 측정은 영어 {out['영어']['wpt']:.2f} · 한국어 {out['한국어']['wpt']:.2f}")
print(f"  → 같은 내용을 한국어로 쓰면 토큰이 {out['한국어']['tpw']/out['영어']['tpw']:.1f}배 든다 "
      f"(단어당 기준)")

# ── 필요한 디코드 속도 ──────────────────────────────────────────────────────
print("\n[G] 읽는 속도를 따라가려면 초당 몇 토큰이 필요한가")
RATES = [("묵독 · 비소설", 238), ("묵독 · 소설", 260), ("소리 내어 읽기", 183)]
out["budget"] = []
for nm, wpm in RATES:
    for lang in ["영어", "한국어"]:
        need = wpm / 60 / out[lang]["wpt"]
        out["budget"].append(dict(mode=nm, wpm=wpm, lang=lang, need_tps=need))
        print(f"  {nm:14s} {wpm} wpm · {lang:4s} → {need:5.1f} tok/s 필요")

# ── 실제 생성 결과 (교재에 그대로 싣는다) ───────────────────────────────────
print("\n[H] 실제 생성 — 모델이 정말 말이 되는 답을 내는가")
msgs = [{"role": "user", "content": "What is on-device AI? Answer in one sentence."}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                              tokenize=True, return_tensors="pt")
if not hasattr(ids, "shape"):
    ids = ids["input_ids"]
with torch.no_grad():
    g = m.generate(ids, max_new_tokens=48, do_sample=False,
                   pad_token_id=tok.eos_token_id)
text = tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True)
out["sample"] = dict(prompt_tokens=int(ids.shape[1]), text=text)
print(f"  프롬프트 {ids.shape[1]} 토큰 →")
print("  " + text.strip().replace("\n", "\n  "))

# ── 우리 실측 디코드 속도와 견주면 ─────────────────────────────────────────
D = json.load(open("/root/lab10/llm.json"))
mine = D["split"]["decode_tps"]
print(f"\n[I] 우리 실측 디코드 {mine:.1f} tok/s 와 견주면")
for r in out["budget"]:
    if r["mode"] == "묵독 · 비소설":
        v = mine / r["need_tps"]
        print(f"  {r['lang']:4s} — 필요 {r['need_tps']:5.1f} tok/s · 실측 {mine:.1f} tok/s → "
              f"{v:.2f}배 {'여유 있다' if v >= 1 else '못 따라간다'}")
        r["ratio_vs_measured"] = v
out["measured_decode_tps"] = mine

json.dump(out, open("/root/lab10/budget.json", "w"), ensure_ascii=False, indent=1)
print("\n저장: budget.json")
