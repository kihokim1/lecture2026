"""실험 1 — 서지 자체가 흔들린다.

  "이 논문 인용해 오세요" 라고 하면 학생은 제목으로 검색한다.
  그런데 (a) 그 논문이 학회에 실린 적이 없거나 (b) 실렸는데 제목이 다르거나
  (c) 애초에 검색이 엉뚱한 논문을 잡으면?

DBLP 를 원전으로 삼아 이 과목이 인용한 논문 전부를 두 가지 방식으로 조회한다.
  ① 제목만으로 매칭 (학생이 실제로 하는 방식)
  ② 제목 + 제1저자 성으로 매칭 (올바른 방식)
두 결과가 얼마나 다른지가 이 실험의 첫 번째 결과다.

DBLP 는 게재 형태를 명시적으로 분류한다 —
  "Informal and Other Publications" = CoRR(arXiv) 프리프린트
  "Conference and Workshop Papers" / "Journal Articles" = 게재본

출력: biblio.json
"""
import json, re, time, unicodedata, urllib.parse, urllib.request
from papers import PAPERS

UA = {"User-Agent": "ondevice-ai-course/1.0 (graduate lecture material)"}
THRESH = 0.62

# 앞선 주차의 인용 검증에서 손으로 확인한 "게재본 제목이 arXiv 제목과 다른" 논문.
# 제목으로만 검색하면 게재본을 영영 못 찾으므로, 여기서 별칭으로 다시 조회한다.
ALIASES = {
    "LLM.int8()": "GPT3.int8(): 8-bit Matrix Multiplication for Transformers at Scale",
    "GPTQ": "OPTQ: Accurate Quantization for Generative Pre-trained Transformers",
}

# 제목이 너무 일반적이거나 색인이 다른 경우의 재조회 질의
RETRY = {
    "Deep Compression": "Deep Compression Compressing Deep Neural Networks Pruning Huffman Coding",
    "Model Compression": "Bucilua Caruana Niculescu-Mizil Model compression",
    "Zhao 비-IID": "Zhao Li Lai Suda Civin Chandra Federated Learning with Non-IID Data",
    "DP-SGD": "Abadi Chu Goodfellow McMahan Mironov Talwar Zhang Deep Learning with Differential Privacy",
    "DLG": "Zhu Liu Han Deep Leakage from Gradients NeurIPS",
    "역전 공격 평가": "Huang Gupta Song Li Arora Evaluating Gradient Inversion Attacks and Defenses",
    "CMSIS-NN": "Lai Suda Chandra CMSIS-NN Efficient Neural Network Kernels Arm Cortex-M",
}

# DBLP 는 전산 분야 색인이다. 타 분야 논문은 여기 없다 — 그것도 결과다.
OUT_OF_SCOPE = {"Brysbaert 2019": "심리학 저널(J. Memory and Language) — DBLP 는 전산 분야만 색인한다"}


def norm(s):
    s = unicodedata.normalize("NFKD", s or "").lower()
    return re.sub(r"[^a-z0-9]+", "", s)


def sim(a, b):
    """정규화 제목 유사도 — 포함 관계면 1.0, 아니면 문자 3-gram 자카드."""
    a, b = norm(a), norm(b)
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return 1.0
    ga = {a[i:i + 3] for i in range(len(a) - 2)}
    gb = {b[i:i + 3] for i in range(len(b) - 2)}
    return len(ga & gb) / max(len(ga | gb), 1)


def authors_of(info):
    a = info.get("authors", {}).get("author", [])
    if isinstance(a, dict):
        a = [a]
    out = []
    for x in a:
        t = x.get("text", "") if isinstance(x, dict) else str(x)
        out.append(re.sub(r"\s+\d+$", "", t))
    return out


def has_author(info, surname):
    sn = norm(surname)
    return any(sn in norm(a) for a in authors_of(info))


def dblp(title, h=15):
    u = ("https://dblp.org/search/publ/api?q=" +
         urllib.parse.quote(title) + f"&format=json&h={h}")
    for k in range(4):
        try:
            r = urllib.request.Request(u, headers=UA)
            return json.loads(urllib.request.urlopen(r, timeout=30).read().decode())
        except Exception:
            if k == 3:
                raise
            time.sleep(3 * (k + 1))


def classify(t):
    return {"Informal and Other Publications": "preprint",
            "Conference and Workshop Papers": "conf",
            "Journal Articles": "journal",
            "Books and Theses": "book",
            "Parts in Books or Collections": "book"}.get(t, "other")


def pick(cands):
    """게재본 우선(학회·저널 > 단행본, 그 안에서 연도 빠른 것), 없으면 프리프린트."""
    order = {"conf": 0, "journal": 0, "book": 1}
    pub = sorted([c for c in cands if c["kind"] in order],
                 key=lambda c: (order[c["kind"]], c.get("year") or "9999"))
    pre = sorted([c for c in cands if c["kind"] == "preprint"],
                 key=lambda c: c.get("year") or "9999")
    return pub, pre


out = {"papers": [], "thresh": THRESH}
print(f"DBLP 조회 — {len(PAPERS)}편  (제목만 매칭 vs 제목+제1저자 매칭)\n")
print(f"  {'주':>2} {'논문':<26} {'형태':<9} {'게재처':<22} {'연도':<5} 비고")
print("  " + "-" * 96)

for wk, short, title, au in PAPERS:
    if short in OUT_OF_SCOPE:
        print(f"  {wk:>2} {short:<26} {'-':<9} {'-':<22} {'-':<5} 색인 범위 밖 — {OUT_OF_SCOPE[short]}")
        out["papers"].append({"week": wk, "short": short, "query": title, "author": au,
                              "out_of_scope": True, "reason": OUT_OF_SCOPE[short],
                              "not_found": True, "naive_wrong": False,
                              "preprint_only": False, "title_changed": False,
                              "cited_differs": False, "lag_years": None,
                              "published": None, "preprint": None,
                              "loose_best": None, "strict_best": None})
        continue
    try:
        hits = dblp(title)["result"]["hits"].get("hit", [])
    except Exception as e:
        print(f"  {wk:>2} {short:<26} 조회 실패 {type(e).__name__}")
        out["papers"].append({"week": wk, "short": short, "query": title,
                              "author": au, "error": type(e).__name__})
        time.sleep(1.5)
        continue

    loose, strict = [], []
    for h in hits:
        i = h["info"]
        t = (i.get("title") or "").rstrip(".")
        s = sim(title, t)
        if s < THRESH:
            continue
        c = {"title": t, "kind": classify(i.get("type", "")),
             "venue": i.get("venue"), "year": i.get("year"),
             "sim": round(s, 3), "doi": i.get("doi"), "url": i.get("url"),
             "authors": authors_of(i)[:4]}
        loose.append(c)
        if has_author(i, au):
            strict.append(c)

    # ① 제목 검색으로 게재본을 못 찾았으면 별칭(확인된 게재본 제목)으로 재조회
    alias_used = None
    if short in ALIASES and not any(c["kind"] in ("conf", "journal") for c in strict):
        time.sleep(1.5)
        try:
            for h in dblp(ALIASES[short])["result"]["hits"].get("hit", []):
                i = h["info"]
                t = (i.get("title") or "").rstrip(".")
                if sim(ALIASES[short], t) >= THRESH and has_author(i, au):
                    strict.append({"title": t, "kind": classify(i.get("type", "")),
                                   "venue": i.get("venue"), "year": i.get("year"),
                                   "sim": round(sim(ALIASES[short], t), 3),
                                   "doi": i.get("doi"), "url": i.get("url"),
                                   "authors": authors_of(i)[:4], "via_alias": True})
                    alias_used = ALIASES[short]
        except Exception:
            pass
    # ② 아무것도 못 찾았으면 저자 이름을 넣은 질의로 재조회
    retry_used = None
    if not strict and short in RETRY:
        time.sleep(1.5)
        try:
            for h in dblp(RETRY[short])["result"]["hits"].get("hit", []):
                i = h["info"]
                t = (i.get("title") or "").rstrip(".")
                if sim(title, t) >= THRESH and has_author(i, au):
                    strict.append({"title": t, "kind": classify(i.get("type", "")),
                                   "venue": i.get("venue"), "year": i.get("year"),
                                   "sim": round(sim(title, t), 3),
                                   "doi": i.get("doi"), "url": i.get("url"),
                                   "authors": authors_of(i)[:4], "via_retry": True})
                    retry_used = RETRY[short]
        except Exception:
            pass

    lpub, lpre = pick(loose)
    spub, spre = pick(strict)
    lbest = (lpub or lpre or [None])[0]
    sbest = (spub or spre or [None])[0]

    rec = {"week": wk, "short": short, "query": title, "author": au,
           "loose_best": lbest, "strict_best": sbest,
           "n_loose": len(loose), "n_strict": len(strict),
           "published": spub[0] if spub else None,
           "preprint": spre[0] if spre else None,
           "alias_used": alias_used, "retry_used": retry_used}
    rec["title_search_failed"] = bool(alias_used or retry_used)

    # ① 제목만 검색이 엉뚱한 논문을 잡았는가
    rec["naive_wrong"] = bool(lbest) and (not sbest or lbest["title"] != sbest["title"])
    rec["not_found"] = sbest is None
    # ② 프리프린트 전용인가
    rec["preprint_only"] = (not spub) and bool(spre)
    # ③ 프리프린트와 게재본의 제목이 다른가
    rec["title_changed"] = False
    if spub and spre:
        a, b = norm(spre[0]["title"]), norm(spub[0]["title"])
        rec["title_changed"] = not (a == b or a in b or b in a)
    # ④ 우리가 적은 제목과 게재본 제목이 다른가
    rec["cited_differs"] = False
    if spub:
        a, b = norm(title), norm(spub[0]["title"])
        rec["cited_differs"] = not (a == b or a in b or b in a)
    # ⑤ arXiv → 게재 시차
    rec["lag_years"] = None
    if spub and spre:
        try:
            rec["lag_years"] = int(spub[0]["year"]) - int(spre[0]["year"])
        except Exception:
            pass

    out["papers"].append(rec)

    notes = []
    if alias_used:
        notes.append("제목 검색 실패 → 게재본 제목으로 재조회")
    if retry_used:
        notes.append("제목 검색 실패 → 저자 질의로 재조회")
    if rec["not_found"]:
        notes.append("DBLP 에서 못 찾음")
    if rec["naive_wrong"]:
        notes.append(f"제목만 검색은 오답 → \"{lbest['title'][:38]}\"")
    if rec["preprint_only"]:
        notes.append("프리프린트 전용")
    if rec["title_changed"]:
        notes.append(f"제목 바뀜 → \"{spub[0]['title'][:38]}\"")
    elif rec["cited_differs"]:
        notes.append(f"게재본 제목 다름 → \"{spub[0]['title'][:38]}\"")
    b = sbest or {}
    print(f"  {wk:>2} {short:<26} {b.get('kind','-'):<9} "
          f"{str(b.get('venue','-'))[:22]:<22} {str(b.get('year','-')):<5} "
          f"{' · '.join(notes)}")
    time.sleep(1.5)

# ══════════ 집계 ══════════
ps = [p for p in out["papers"] if "error" not in p]
n = len(ps)
found = [p for p in ps if not p["not_found"]]
kinds = {}
for p in found:
    k = p["published"]["kind"] if p["published"] else "preprint"
    kinds[k] = kinds.get(k, 0) + 1
lags = [p["lag_years"] for p in ps if p["lag_years"] is not None and p["lag_years"] >= 0]


def L(key):
    return [p for p in ps if p.get(key)]


out["summary"] = {
    "n": n, "n_found": len(found), "n_failed": len(out["papers"]) - n,
    "kinds": kinds,
    "title_search_failed": len(L("title_search_failed")),
    "out_of_scope": len(L("out_of_scope")),
    "naive_wrong": len(L("naive_wrong")),
    "naive_wrong_pct": round(100 * len(L("naive_wrong")) / n, 1) if n else 0,
    "not_found": len(L("not_found")),
    "preprint_only": len(L("preprint_only")),
    "preprint_only_pct": round(100 * len(L("preprint_only")) / len(found), 1) if found else 0,
    "title_changed": len(L("title_changed")),
    "cited_differs": len(L("cited_differs")),
    "lag_mean": round(sum(lags) / len(lags), 2) if lags else None,
    "lag_max": max(lags) if lags else None,
    "lag_dist": {str(v): lags.count(v) for v in sorted(set(lags))},
    "lists": {
        "naive_wrong": [{"short": p["short"], "got": p["loose_best"]["title"],
                         "want": (p["strict_best"] or {}).get("title")} for p in L("naive_wrong")],
        "preprint_only": [{"short": p["short"], "week": p["week"],
                           "year": p["preprint"]["year"]} for p in L("preprint_only")],
        "title_changed": [{"short": p["short"], "pre": p["preprint"]["title"],
                           "pub": p["published"]["title"],
                           "venue": p["published"]["venue"]} for p in L("title_changed")],
        "not_found": [p["short"] for p in L("not_found") if not p.get("out_of_scope")],
        "out_of_scope": [{"short": p["short"], "reason": p["reason"]} for p in L("out_of_scope")],
        "title_search_failed": [{"short": p["short"], "week": p["week"],
                                 "found": (p["strict_best"] or {}).get("title"),
                                 "venue": (p["strict_best"] or {}).get("venue")}
                                for p in L("title_search_failed")],
    },
}

s = out["summary"]
print("\n" + "=" * 98)
print(f"조회 {n}편 · DBLP 에서 찾음 {s['n_found']}편")
print(f"  게재 형태            {s['kinds']}")
print(f"  제목만 검색이 오답    {s['naive_wrong']}편 ({s['naive_wrong_pct']}%)")
print(f"  제목 검색이 실패      {s['title_search_failed']}편 (별칭·저자 질의로 재조회해 찾음)")
print(f"  색인 범위 밖          {s['out_of_scope']}편")
print(f"  프리프린트 전용       {s['preprint_only']}편 ({s['preprint_only_pct']}%)")
print(f"  제목이 바뀐 논문      {s['title_changed']}편")
print(f"  arXiv→게재 평균 시차  {s['lag_mean']}년 (최대 {s['lag_max']}년) · 분포 {s['lag_dist']}")

print("\n[제목만 검색하면 엉뚱한 논문이 잡히는 것]")
for x in s["lists"]["naive_wrong"]:
    print(f"   · {x['short']}\n       잡힌 것: {x['got'][:74]}\n       맞는 것: {str(x['want'])[:74]}")
print("\n[제목으로 검색해서는 게재본을 못 찾은 것 — 재조회로 찾음]")
for x in s["lists"]["title_search_failed"]:
    print(f"   · {x['week']:>2}주차  {x['short']}  →  {str(x['found'])[:56]} ({x['venue']})")
print("\n[프리프린트 전용 — 학회/저널 게재 이력 없음]")
for x in s["lists"]["preprint_only"]:
    print(f"   · {x['week']:>2}주차  {x['short']} ({x['year']})")
print("\n[제목이 바뀐 논문 — arXiv 제목으로 인용하면 게재본을 못 찾는다]")
for x in s["lists"]["title_changed"]:
    print(f"   · {x['short']} ({x['venue']})\n       arXiv : {x['pre'][:74]}\n       게재본: {x['pub'][:74]}")
if s["lists"]["not_found"]:
    print("\n[DBLP 에서 못 찾음]")
    for x in s["lists"]["not_found"]:
        print(f"   · {x}")

json.dump(out, open("biblio.json", "w"), ensure_ascii=False, indent=1)
print("\n→ biblio.json")
