"""3교시 학생용 — 논문 하나의 서지를 확인한다.

여섯 질문 중 ⑥(어디에 실렸나)만 자동화할 수 있다. 나머지 다섯은 논문을 열어야 한다.
이 도구는 그 한 걸음을 대신해 주고, **나머지 다섯을 직접 채우라고 표를 내민다.**

    python3 student.py "논문 제목"
    python3 student.py "논문 제목" 제1저자성
"""
import json, re, sys, time, unicodedata, urllib.parse, urllib.request

UA = {"User-Agent": "ondevice-ai-course/1.0 (graduate seminar)"}
TITLE = sys.argv[1] if len(sys.argv) > 1 else "Deep Leakage from Gradients"
AUTHOR = sys.argv[2] if len(sys.argv) > 2 else None


def norm(s):
    s = unicodedata.normalize("NFKD", s or "").lower()
    return re.sub(r"[^a-z0-9]+", "", s)


def sim(a, b):
    a, b = norm(a), norm(b)
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return 1.0
    ga = {a[i:i + 3] for i in range(len(a) - 2)}
    gb = {b[i:i + 3] for i in range(len(b) - 2)}
    return len(ga & gb) / max(len(ga | gb), 1)


def authors_of(i):
    a = i.get("authors", {}).get("author", [])
    if isinstance(a, dict):
        a = [a]
    return [re.sub(r"\s+\d+$", "", x.get("text", "") if isinstance(x, dict) else str(x))
            for x in a]


KIND = {"Informal and Other Publications": "프리프린트 (동료심사 없음)",
        "Conference and Workshop Papers": "학회·워크숍 논문",
        "Journal Articles": "저널 논문",
        "Books and Theses": "단행본·학위논문",
        "Parts in Books or Collections": "단행본 수록"}


def dblp(q, h=15):
    u = ("https://dblp.org/search/publ/api?q=" + urllib.parse.quote(q) +
         f"&format=json&h={h}")
    for k in range(3):
        try:
            r = urllib.request.Request(u, headers=UA)
            return json.loads(urllib.request.urlopen(r, timeout=30).read().decode())
        except Exception:
            if k == 2:
                raise
            time.sleep(3)


print(f'조회: "{TITLE}"' + (f'  (제1저자 {AUTHOR})' if AUTHOR else "  (저자 미지정)"))
print("=" * 78)

hits = dblp(TITLE)["result"]["hits"].get("hit", [])
cands = []
for h in hits:
    i = h["info"]
    t = (i.get("title") or "").rstrip(".")
    s = sim(TITLE, t)
    if s < 0.55:
        continue
    au = authors_of(i)
    cands.append({"title": t, "kind": i.get("type", ""), "venue": i.get("venue"),
                  "year": i.get("year"), "sim": s, "authors": au,
                  "doi": i.get("doi"), "url": i.get("url"),
                  "au_ok": (AUTHOR is None) or
                           any(norm(AUTHOR) in norm(a) for a in au)})

if not cands:
    print("\n⚠ 제목으로 못 찾았습니다. 셋 중 하나입니다 —")
    print("   ① 제목이 너무 일반적이라 결과에 묻혔다  (저자 이름을 넣어 다시 검색)")
    print("   ② 게재본 제목이 arXiv 제목과 다르다     (게재처 페이지에서 확인)")
    print("   ③ DBLP 색인 범위 밖이다                (전산 분야가 아닌 저널)")
    print("\n   → 세 경우 모두 **사람이 확인해야 합니다.** 못 찾았다고 없는 논문이 아닙니다.")
    sys.exit(0)

pub = [c for c in cands if c["au_ok"] and c["kind"] in
       ("Conference and Workshop Papers", "Journal Articles")]
pre = [c for c in cands if c["au_ok"] and c["kind"] == "Informal and Other Publications"]
pub.sort(key=lambda c: c.get("year") or "9999")
pre.sort(key=lambda c: c.get("year") or "9999")

print("\n① 찾은 항목")
for c in cands[:8]:
    mark = " " if c["au_ok"] else "✗"
    print(f"  {mark} {str(c['year']):<5} {KIND.get(c['kind'], c['kind'])[:18]:<20} "
          f"{str(c['venue'])[:20]:<20} {c['title'][:44]}")
    if not c["au_ok"]:
        print(f"      └ 제1저자 불일치 → {', '.join(c['authors'][:3])}")

if AUTHOR and any(not c["au_ok"] for c in cands):
    n = sum(1 for c in cands if not c["au_ok"])
    print(f"\n  ⚠ 제목은 비슷한데 저자가 다른 항목이 {n}건 있습니다. "
          f"제목만으로 검색하면 이런 것이 잡힙니다.")

print("\n② 판정")
if pub:
    b = pub[0]
    print(f"  게재본  : {KIND.get(b['kind'])} · {b['venue']} {b['year']}")
    print(f"  제목    : {b['title']}")
    if b.get("doi"):
        print(f"  DOI     : {b['doi']}")
    if pre and norm(pre[0]["title"]) != norm(b["title"]):
        print(f"\n  ⚠ arXiv 제목과 게재본 제목이 다릅니다")
        print(f"      arXiv : {pre[0]['title']}")
        print(f"      게재본: {b['title']}")
        print(f"      → **게재본 제목으로 인용해야 합니다.**")
    if pre:
        try:
            lag = int(b["year"]) - int(pre[0]["year"])
            print(f"\n  arXiv {pre[0]['year']} → 게재 {b['year']} ({lag}년)")
        except Exception:
            pass
elif pre:
    b = pre[0]
    print(f"  **프리프린트입니다** — {b['venue']} {b['year']}")
    print(f"  제목    : {b['title']}")
    print("  → 학회·저널 게재 이력을 찾지 못했습니다.")
    print("  → 인용은 \"arXiv:XXXX.XXXXX\" 로. **\"in Proc. …\" 로 쓰면 안 됩니다.**")
    print("  → 다만 DBLP 가 못 찾은 것일 수도 있습니다(우리 실측 오류율 15.6%).")
    print("     게재처 페이지·저자 홈페이지를 한 번 더 확인하십시오.")
else:
    print("  제1저자가 일치하는 항목이 없습니다. 저자명을 확인하거나 저자 없이 다시 돌리십시오.")

print("""
③ 여기까지가 자동으로 되는 전부입니다 — 여섯 질문 중 ⑥ 하나뿐입니다.
   나머지 다섯은 논문을 열어야 합니다.

   ┌─────────────────────────────────────────────────────────────────┐
   │ ① 무엇과 비교했나?      → 실험 절의 baseline 문장을 그대로 옮길 것 │
   │ ② 어떤 조건에서?        → 하드웨어·데이터셋·배치·분할·버전        │
   │ ③ 그 숫자가 논문에 있나? → 인용하려는 값을 본문/표에서 직접 찾을 것│
   │ ④ 논문이 그 말을 했나?   → 해당 절 전체를 읽을 것                 │
   │ ⑤ 이게 이 논문이 처음인가?→ 관련 연구 절에서 선행 연구를 확인할 것 │
   │ ⑥ 어디에 실렸나?        → 위에서 확인함 (그래도 눈으로 재확인)    │
   └─────────────────────────────────────────────────────────────────┘""")
