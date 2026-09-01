"""실험 1-보충 — 자동 조회는 어디까지 맞는가.

run_biblio.py 는 DBLP 를 기계적으로 조회한다. 그런데 4~12주차 동안 우리는
같은 논문들을 **손으로** 확인했다(1차 출처의 PDF·게재처 페이지를 직접 열어서).

두 결과를 맞대 본다. 자동 조회가 몇 편에서 틀리는지가 이 실험의 결론이다.

출력: verify.json
"""
import json

BIB = json.load(open("biblio.json"))

# 4~12주차 인용 검증에서 **손으로 확인한** 서지. 근거는 각 주차 교재의 참고문헌.
#   kind: conf | journal | preprint (게재 이력 없음)
GROUND = {
    "Deep Compression":        ("conf", "ICLR", "2016", ""),
    "Han 2015 (연결 학습)":      ("conf", "NIPS", "2015", ""),
    "Model Compression":       ("conf", "KDD", "2006", ""),
    "Hinton 증류":              ("preprint", "NIPS 워크숍", "2015",
                                "NIPS 2014 딥러닝 워크숍 발표 · 논문집 게재 없음"),
    "DistilBERT":              ("preprint", "CoRR", "2019", ""),
    "MobileNets":              ("preprint", "CoRR", "2017", ""),
    "CMSIS-NN":                ("preprint", "CoRR", "2018", ""),
    "Hello Edge":              ("preprint", "CoRR", "2017", ""),
    "MCUNetV2":                ("conf", "NeurIPS", "2021",
                                "게재 제목에 \"MCUNetV2:\" 접두어가 없다"),
    "Amdahl 1967":             ("conf", "AFIPS", "1967", ""),
    "LLM.int8()":              ("conf", "NeurIPS", "2022",
                                "게재 제목은 \"GPT3.int8()\""),
    "GPTQ":                    ("conf", "ICLR", "2023", "게재 제목은 \"OPTQ\""),
    "MQA (Shazeer)":           ("preprint", "CoRR", "2019", ""),
    "추측 표집":                  ("preprint", "CoRR", "2023", ""),
    "Brysbaert 2019":          ("journal", "J. Memory and Language", "2019",
                                "심리학 저널 — DBLP 색인 범위 밖"),
    "Glow":                    ("preprint", "CoRR", "2018", ""),
    "Krishnamoorthi 백서":       ("preprint", "CoRR", "2018", ""),
    "Nagel 백서":                ("preprint", "CoRR", "2021", ""),
    "DP-SGD":                  ("conf", "CCS", "2016", ""),
    "DLG":                     ("conf", "NeurIPS", "2019", ""),
    "iDLG":                    ("preprint", "CoRR", "2020", ""),
    "Zhao 비-IID":              ("preprint", "CoRR", "2018", ""),
    "Hsu 디리클레":               ("preprint", "CoRR", "2019", ""),
    "Gboard":                  ("preprint", "CoRR", "2018", ""),
    "Apple DP 분석":             ("preprint", "CoRR", "2017", ""),
    "역전 공격 평가":               ("conf", "NeurIPS", "2021", ""),
    "FL 서베이":                  ("journal", "Found. Trends Mach. Learn.", "2021",
                                "프리프린트가 아니라 정식 출판 monograph"),
    "SA 우회":                   ("conf", "CCS", "2022",
                                "제목은 \"…via Model Inconsistency\""),
    "Phong (해석적 역산)":         ("journal", "IEEE TIFS", "2018", ""),
    "SCAFFOLD":                ("conf", "ICML", "2020",
                                "arXiv v1 제목에 \"On-Device\" 가 있었다"),
    "FedProx":                 ("conf", "MLSys", "2020",
                                "arXiv v1 제목은 \"On the Convergence of…\""),
    "Triton":                  ("conf", "MAPL@PLDI", "2019", "본회의가 아니라 워크숍"),
}

auto = {p["short"]: p for p in BIB["papers"]}
rows, agree, disagree, missed = [], 0, 0, 0
for short, (gk, gv, gy, note) in GROUND.items():
    a = auto.get(short)
    if a is None:
        continue
    pub, pre = a.get("published"), a.get("preprint")
    ak = pub["kind"] if pub else ("preprint" if pre else "없음")
    av = (pub or pre or {}).get("venue", "-")
    ay = (pub or pre or {}).get("year", "-")
    # 게재 형태만 같으면 "일치"로 볼 수 없다 — DBLP 가 같은 형태의 **다른 논문**을
    # 잡아 오는 일이 있다. 연도까지 맞아야 같은 논문으로 인정한다.
    ok = (ak == gk) and (str(ay) == str(gy))
    if ak == "없음":
        missed += 1
        verdict = "못 찾음"
    elif ok:
        agree += 1
        verdict = "일치"
    else:
        disagree += 1
        verdict = "불일치"
    rows.append({"short": short, "week": a["week"],
                 "auto_kind": ak, "auto_venue": av, "auto_year": ay,
                 "true_kind": gk, "true_venue": gv, "true_year": gy,
                 "verdict": verdict, "note": note,
                 "title_search_failed": bool(a.get("title_search_failed"))})

n = len(rows)
out = {"n_checked": n, "agree": agree, "disagree": disagree, "missed": missed,
       "accuracy": round(100 * agree / n, 1) if n else 0,
       "error_rate": round(100 * (disagree + missed) / n, 1) if n else 0,
       "rows": rows,
       "biblio_summary": BIB["summary"]}

print(f"손으로 확인한 {n}편에 대해 자동 조회를 대조한다\n")
print(f"  {'논문':<24}{'자동 조회':<28}{'실제 (손 확인)':<28}판정")
print("  " + "-" * 92)
for r in sorted(rows, key=lambda r: (r["verdict"] == "일치", r["short"])):
    a = f"{r['auto_kind']} · {str(r['auto_venue'])[:14]} {r['auto_year']}"
    t = f"{r['true_kind']} · {str(r['true_venue'])[:14]} {r['true_year']}"
    mark = {"일치": " ", "불일치": "✗", "못 찾음": "?"}[r["verdict"]]
    print(f"  {r['short']:<24}{a:<28}{t:<28}{mark} {r['verdict']}")
    if r["note"]:
        print(f"    └ {r['note']}")

print("\n" + "=" * 94)
print(f"  일치 {agree} · 불일치 {disagree} · 못 찾음 {missed}  →  "
      f"자동 조회 정확도 {out['accuracy']}% · 오류율 {out['error_rate']}%")

s = BIB["summary"]
print(f"\n[전체 {s['n']}편 자동 조회 요약]")
print(f"  게재 형태            {s['kinds']}")
print(f"  프리프린트 전용       {s['preprint_only']}편 ({s['preprint_only_pct']}%)")
print(f"  제목만 검색이 오답    {s['naive_wrong']}편")
print(f"  제목 검색 실패        {s['title_search_failed']}편 (별칭·저자 질의로 재조회)")
print(f"  제목이 바뀐 논문      {s['title_changed']}편")
print(f"  arXiv→게재 평균 시차  {s['lag_mean']}년 (최대 {s['lag_max']}년)")

json.dump(out, open("verify.json", "w"), ensure_ascii=False, indent=1)
print("\n→ verify.json")
