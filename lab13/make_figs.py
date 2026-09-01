# -*- coding: utf-8 -*-
"""13주차 그림 — corpus.py · biblio.json · verify.json · repro.json 기반.

주의: 유니코드 위첨자(ᵀ ⁻⁵ ²⁰)는 Noto Sans CJK KR 에 글리프가 없다.
      평문 표기(x^T, 1e-5, 2^20)만 쓴다.
"""
import json, pathlib, sys, collections

sys.path.insert(0, "/root/lab13")
from corpus import CORPUS, TYPES, CHANGED_CONCLUSION  # noqa: E402

D = pathlib.Path("/root/lab13")
BI = json.load(open(D / "biblio.json"))
VE = json.load(open(D / "verify.json"))
RE = json.load(open(D / "repro.json"))
OUT = pathlib.Path("/root/ondevice-ai/img/week13"); OUT.mkdir(parents=True, exist_ok=True)

HEAD = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        'font-family="\'Segoe UI\',Arial,sans-serif">\n'
        '  <rect width="{w}" height="{h}" fill="#f8fafc"/>\n')
F = {}
TEAL, AMB, RED, BLUE = "#028090", "#e4711b", "#dc2626", "#2563eb"
GRN, PUR, PINK = "#16a34a", "#7c3aed", "#be185d"
INK, MUT, LINE = "#0f172a", "#64748b", "#cbd5e1"


def fig(name, w, h, body):
    F[name] = HEAD.format(w=w, h=h) + body + "\n</svg>\n"


def tw(s, size):
    """한글 1.0em, 라틴 0.56em 근사 — 범례·배지 폭 계산용."""
    return sum(size * (1.0 if ord(c) > 0x2000 else 0.56) for c in s)


def esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def title(x, y, t, sub=None):
    s = (f'  <text x="{x}" y="{y}" text-anchor="middle" font-size="19" '
         f'font-weight="700" fill="{INK}">{esc(t)}</text>\n')
    if sub:
        s += (f'  <text x="{x}" y="{y+22}" text-anchor="middle" font-size="13.5" '
              f'fill="{MUT}">{esc(sub)}</text>\n')
    return s


def note(x, y, w, h, head, body, bg="#edf6f4", ln="#9fd6cc", hc="#0b4a48"):
    s = f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{bg}" stroke="{ln}"/>\n'
    s += (f'  <text x="{x+w/2}" y="{y+26}" text-anchor="middle" font-size="14.5" '
          f'font-weight="700" fill="{hc}">{esc(head)}</text>\n')
    for i, l in enumerate(body):
        s += (f'  <text x="{x+w/2}" y="{y+50+i*20}" text-anchor="middle" '
              f'font-size="12.5" fill="#334155">{esc(l)}</text>\n')
    return s


def legend(x, y, items, size=12.5):
    s, cx = "", x
    for c, lab in items:
        s += f'  <rect x="{cx}" y="{y-9}" width="13" height="13" rx="3" fill="{c}"/>\n'
        s += f'  <text x="{cx+19}" y="{y+2}" font-size="{size}" fill="#334155">{esc(lab)}</text>\n'
        cx += 19 + tw(lab, size) + 22
    return s


def legend_w(items, size=12.5):
    return sum(19 + tw(l, size) + 22 for _, l in items) - 22


MK = ('<marker id="{i}" markerWidth="9" markerHeight="9" refX="7" refY="3.5" '
      'orient="auto" markerUnits="userSpaceOnUse">'
      '<path d="M0,0 L7,3.5 L0,7 Z" fill="{c}"/></marker>')


def card(x, y, w, h, hd, col, bg, lines, hsize=15, lsize=12.5, lh=19):
    s = (f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="11" fill="{bg}" '
         f'stroke="{col}" stroke-width="1.8"/>\n')
    s += (f'  <text x="{x+14}" y="{y+25}" font-size="{hsize}" font-weight="700" '
          f'fill="{col}">{esc(hd)}</text>\n')
    for i, l in enumerate(lines):
        s += (f'  <text x="{x+14}" y="{y+48+i*lh}" font-size="{lsize}" '
              f'fill="#334155">{esc(l)}</text>\n')
    return s


# ══════════════════════════════════════════════════════════════════════
# 01. 열두 주간의 정정 56건 — 유형과 주차 분포
# ══════════════════════════════════════════════════════════════════════
byt = collections.Counter(c[1] for c in CORPUS)
byw = collections.Counter(c[0] for c in CORPUS)
N = len(CORPUS)

b = title(540, 30, f"열두 주 동안의 정정 {N}건 — 어디서, 어떤 종류로",
          "매주 독립 검토자가 1차 출처를 대조해 나온 것 (실측)")

b += (f'  <text x="52" y="82" font-size="14.5" font-weight="700" '
      f'fill="{INK}">유형별</text>\n')
COLS = {"조건": TEAL, "서지": AMB, "무주장": RED, "귀속": PUR, "기준선": BLUE, "범위": PINK}
order = [t for t in TYPES if byt[t]]
order.sort(key=lambda t: -byt[t])
BX, BY, BW = 132, 96, 300
mx = max(byt[t] for t in order)
for i, t in enumerate(order):
    yy = BY + i * 34
    v = byt[t]
    b += (f'  <text x="{BX-10}" y="{yy+17}" text-anchor="end" font-size="13.5" '
          f'font-weight="700" fill="{COLS[t]}">{t}</text>\n')
    b += (f'  <rect x="{BX}" y="{yy}" width="{BW*v/mx:.1f}" height="24" rx="4" '
          f'fill="{COLS[t]}"/>\n')
    b += (f'  <text x="{BX+BW*v/mx+9:.1f}" y="{yy+17}" font-size="13" '
          f'font-weight="700" fill="{INK}">{v}건 · {100*v/N:.1f}%</text>\n')

b += (f'  <text x="560" y="82" font-size="14.5" font-weight="700" '
      f'fill="{INK}">주차별</text>\n')
WX, WY, WH = 566, 100, 176
ws = list(range(min(byw), max(byw) + 1))
mw = max(byw.values())
cw = 44
for i, w in enumerate(ws):
    xx = WX + i * (cw + 16)
    v = byw[w]
    hh = WH * v / mw
    col = RED if v == mw else TEAL
    if v:
        b += (f'  <rect x="{xx}" y="{WY+WH-hh:.1f}" width="{cw}" height="{hh:.1f}" '
              f'rx="5" fill="{col}"/>\n')
    b += (f'  <text x="{xx+cw/2}" y="{WY+WH-hh-8:.1f}" text-anchor="middle" '
          f'font-size="13.5" font-weight="700" fill="{col if v else LINE}">{v}</text>\n')
    b += (f'  <text x="{xx+cw/2}" y="{WY+WH+20}" text-anchor="middle" '
          f'font-size="12.5" fill="{MUT}">{w}주</text>\n')
b += (f'  <line x1="{WX-8}" y1="{WY+WH}" x2="{WX+len(ws)*(cw+16)-16}" y2="{WY+WH}" '
      f'stroke="{LINE}" stroke-width="1.4"/>\n')
b += (f'  <text x="{WX+(len(ws)*(cw+16)-16)/2}" y="{WY+WH+44}" text-anchor="middle" '
      f'font-size="12.5" fill="{MUT}">1~5주 0건 · 6~12주 평균 {N/6:.1f}건</text>\n')

b += note(52, 340, 976, 116, "이 56건은 무명 논문이 아니라 이 분야의 대표 논문들에서 나왔다",
          ["교재에 인용하려던 것들이다 — 즉 가장 유명하고 가장 많이 인용되는 논문들이다.",
           "1~5주가 0건인 것은 그때 검증 절차를 아직 넣지 않았기 때문이고, 12주차가 19건인 것은 그 주제가 부정확해서가",
           "아니라 그때쯤 무엇을 확인해야 하는지 알게 됐기 때문이다. 뒤로 갈수록 늘어나는 것은 그래서다."])
fig("w13_p1_corpus_01", 1080, 476, b)

# ══════════════════════════════════════════════════════════════════════
# 02. 조건이 잘리는 여섯 가지 패턴
# ══════════════════════════════════════════════════════════════════════
PAT = [
    ("① 조건", TEAL, "수치는 맞는데 하드웨어·데이터셋·",
     "배치·분할·버전이 잘려 나갔다",
     "\"SSD 는 59 FPS\" → 배치 8 의 값. 배치 1 은 46"),
    ("② 서지", AMB, "프리프린트를 게재 논문으로,",
     "또는 제목·게재처가 실제와 다르다",
     "GPTQ → 게재본 제목은 OPTQ (ICLR 2023)"),
    ("③ 무주장", RED, "그 논문은 그런 말을",
     "한 적이 없다",
     "Amdahl 원논문에는 수식이 하나도 없다"),
    ("④ 귀속", PUR, "다른 논문(또는 후속 논문)의",
     "공을 잘못 돌렸다",
     "선형층 역산의 최초는 Phong 외 TIFS 2018"),
    ("⑤ 기준선", BLUE, "수치는 맞는데 비교 대상이",
     "인용된 것과 다르다",
     "\"Ansor 3.8배\" → AutoTVM 대비는 1.02~1.8배"),
    ("⑥ 범위", PINK, "서로 다른 실험의 값을",
     "하나의 범위로 이어 붙였다",
     "\"TVM 1.2~3.8배\" → 그런 범위는 논문에 없다"),
]
b = title(500, 30, "조건이 잘리는 여섯 가지 패턴",
          "56건을 분류하면 여섯 개로 수렴한다 — 이것이 논문을 열 때의 도구다")
for i, (nm, col, d1, d2, ex) in enumerate(PAT):
    x = 44 + (i % 2) * 468
    y = 72 + (i // 2) * 118
    v = byt[nm[1:].strip()]
    b += (f'  <rect x="{x}" y="{y}" width="444" height="102" rx="12" fill="#ffffff" '
          f'stroke="{col}" stroke-width="1.9"/>\n')
    b += f'  <rect x="{x}" y="{y}" width="7" height="102" rx="3" fill="{col}"/>\n'
    b += (f'  <text x="{x+22}" y="{y+28}" font-size="16.5" font-weight="700" '
          f'fill="{col}">{nm}</text>\n')
    pct = f"{v}건 · {100*v/N:.1f}%"
    pw = tw(pct, 13) + 20
    b += (f'  <rect x="{x+444-pw-14}" y="{y+12}" width="{pw}" height="23" rx="11.5" '
          f'fill="{col}" opacity="0.13"/>\n')
    b += (f'  <text x="{x+444-14-pw/2}" y="{y+28}" text-anchor="middle" font-size="13" '
          f'font-weight="700" fill="{col}">{pct}</text>\n')
    b += (f'  <text x="{x+22}" y="{y+50}" font-size="12.8" fill="#334155">{esc(d1)}</text>\n')
    b += (f'  <text x="{x+22}" y="{y+68}" font-size="12.8" fill="#334155">{esc(d2)}</text>\n')
    b += (f'  <text x="{x+22}" y="{y+90}" font-size="12.2" fill="{MUT}">{esc(ex)}</text>\n')
b += note(44, 438, 912, 78,
          "가장 확인하기 어려운 것은 ③ 무주장이다",
          ["조건 누락은 논문을 열면 보인다. 그런데 \"논문이 하지 않은 말\"은 논문을 열어도 안 보인다 — 없는 것을 확인하려면 전체를 읽어야 한다."],
          bg="#fef3e8", ln="#f0b27a", hc="#9a4b06")
fig("w13_p1_patterns_02", 1000, 536, b)

# ══════════════════════════════════════════════════════════════════════
# 03. 정정이 결론을 바꾼 네 사례
# ══════════════════════════════════════════════════════════════════════
FLIP = [
    (8, "nRF52840 조건 정정", "손익분기 7.0분", "손익분기 1.74분",
     "듀티 사이클 설계가 달라진다", "LDO/DC-DC · RAM 유지 여부"),
    (11, "TVM 범위를 분리", "1.2 ~ 3.8배", "모바일 1.2 ~ 1.6배",
     "가속기 도입 판단이 달라진다", "서버 GPU 와 모바일 GPU 는 다른 실험"),
    (12, "\"비-IID 55%\" 를 조건과 함께", "정확도 55% 하락", "MNIST 6.5 ~ 11.3%",
     "우리 실측 7.05%p 가 정상값이 된다", "55% 는 KWS 특정 분할의 값"),
    (12, "역산의 귀속을 바로잡음", "2020년대 최신 공격", "최초 출처 2018년",
     "\"최근 공격\" 이 아니라 8년 된 결과", "Phong 외 TIFS 2018 §3 관찰 (O1)"),
]
b = title(614, 30, "정정이 결론 자체를 바꾼 네 사례",
          f"56건 중 대부분은 숫자를 다듬는 수준이었다. 결론이 뒤집힌 것은 {len(FLIP)}건 (실측)")
b += f'  <defs>{MK.format(i="fa", c=AMB)}</defs>\n'
for i, (w, what, before, after, effect, why) in enumerate(FLIP):
    y = 76 + i * 106
    b += (f'  <rect x="44" y="{y}" width="1140" height="92" rx="11" fill="#ffffff" '
          f'stroke="{LINE}"/>\n')
    b += f'  <circle cx="82" cy="{y+34}" r="21" fill="{TEAL}"/>\n'
    b += (f'  <text x="82" y="{y+40}" text-anchor="middle" font-size="16" '
          f'font-weight="700" fill="#ffffff">{w}주</text>\n')
    b += (f'  <text x="118" y="{y+29}" font-size="14.5" font-weight="700" '
          f'fill="{INK}">{esc(what)}</text>\n')
    b += (f'  <text x="118" y="{y+52}" font-size="12.2" fill="{MUT}">{esc(why)}</text>\n')
    bx = 440
    b += (f'  <rect x="{bx}" y="{y+16}" width="176" height="38" rx="8" fill="#fee2e2" '
          f'stroke="#f0a3a3"/>\n')
    b += (f'  <text x="{bx+88}" y="{y+40}" text-anchor="middle" font-size="13.5" '
          f'font-weight="700" fill="{RED}">{esc(before)}</text>\n')
    b += (f'  <line x1="{bx+186}" y1="{y+35}" x2="{bx+218}" y2="{y+35}" stroke="{AMB}" '
          f'stroke-width="2.6" marker-end="url(#fa)"/>\n')
    b += (f'  <rect x="{bx+228}" y="{y+16}" width="188" height="38" rx="8" fill="#dcfce7" '
          f'stroke="#86d3a4"/>\n')
    b += (f'  <text x="{bx+322}" y="{y+40}" text-anchor="middle" font-size="13.5" '
          f'font-weight="700" fill="#15803d">{esc(after)}</text>\n')
    b += (f'  <text x="{bx+88}" y="{y+76}" text-anchor="middle" font-size="12" '
          f'fill="{MUT}">흔히 도는 말</text>\n')
    b += (f'  <text x="{bx+322}" y="{y+76}" text-anchor="middle" font-size="12" '
          f'fill="{MUT}">1차 출처 확인 후</text>\n')
    b += (f'  <text x="{bx+438}" y="{y+40}" font-size="12.6" font-weight="700" '
          f'fill="{AMB}">→ {esc(effect)}</text>\n')
b += note(44, 508, 1140, 74, "네 건 모두 실무 결정을 바꾼다",
          ["듀티 사이클 설계 · 가속기 도입 판단 · 실측값의 정상 여부 · 위험 평가. 숫자를 다듬는 것과 결론이 뒤집히는 것은 다른 일이다."],
          bg="#fef3e8", ln="#f0b27a", hc="#9a4b06")
fig("w13_p1_flip_03", 1228, 602, b)

# ══════════════════════════════════════════════════════════════════════
# 04. 인용 71편의 게재 형태
# ══════════════════════════════════════════════════════════════════════
S = BI["summary"]
K = S["kinds"]
NP = S["n"]
kinds = [("학회 논문", K["conf"], TEAL),
         ("저널 논문", K["journal"], BLUE),
         ("프리프린트 (게재 이력 없음)", K["preprint"], AMB),
         ("DBLP 에서 못 찾음", S["not_found"], MUT)]
assert sum(k[1] for k in kinds) == NP, sum(k[1] for k in kinds)
b = title(600, 30, f"우리가 인용한 {NP}편의 게재 형태 — 넷 중 하나는 프리프린트다",
          "DBLP 조회 · 제목 + 제1저자 성 (실측)")

# 71칸 격자
GX, GY, CS, PER = 60, 84, 26, 18
seq = []
for lab, v, col in kinds:
    seq += [col] * v
for i, col in enumerate(seq):
    x = GX + (i % PER) * CS
    y = GY + (i // PER) * CS
    b += (f'  <rect x="{x}" y="{y}" width="{CS-5}" height="{CS-5}" rx="4" '
          f'fill="{col}"/>\n')
b += (f'  <text x="{GX}" y="{GY-12}" font-size="12.5" fill="{MUT}">칸 하나가 논문 한 편</text>\n')

LX = GX + PER * CS + 42
for i, (lab, v, col) in enumerate(kinds):
    y = GY + 14 + i * 26
    b += f'  <rect x="{LX}" y="{y-13}" width="15" height="15" rx="3.5" fill="{col}"/>\n'
    b += (f'  <text x="{LX+23}" y="{y}" font-size="13.5" fill="#334155">{esc(lab)}</text>\n')
    b += (f'  <text x="{LX+400}" y="{y}" text-anchor="end" font-size="13.5" '
          f'font-weight="700" fill="{col}">{v}편</text>\n')

BANDY = GY + 4 * CS + 24
b += (f'  <rect x="{GX}" y="{BANDY}" width="1080" height="54" rx="10" '
      f'fill="#fef3e8" stroke="#f0b27a"/>\n')
b += (f'  <text x="{GX+540}" y="{BANDY+34}" text-anchor="middle" '
      f'font-size="15.5" font-weight="700" fill="#9a4b06">'
      f'프리프린트 {K["preprint"]}편 — DBLP 가 찾은 {S["n_found"]}편 중 {S["preprint_only_pct"]}%'
      f' (전체 {NP}편 기준 {100*K["preprint"]/NP:.1f}%)</text>\n')

b += card(GX, BANDY + 68, 1080, 82, "그것도 이 분야의 표준 인용들이다", AMB, "#ffffff",
          ["MobileNets · DistilBERT · Glow · Hello Edge · CMSIS-NN · "
           "Krishnamoorthi 백서 · Nagel 백서 · MQA · 추측 표집"])

b += note(GX, BANDY + 166, 1080, 96, "프리프린트를 인용하지 말라는 뜻이 아니다",
          ["인용해도 된다. 다만 \"in Proc. …\" 이 아니라 \"arXiv:XXXX.XXXXX\" 로 써야 한다.",
           f"게재된 {S['lag_dist']['0']+S['lag_dist']['1']+S['lag_dist']['2']}편의 arXiv → 게재 시차는 평균 {S['lag_mean']}년 (같은 해 {S['lag_dist']['0']} · 1년 {S['lag_dist']['1']} · 2년 {S['lag_dist']['2']}) — 지금 적어 둔 인용은 제출 전에 다시 확인해야 한다."])
fig("w13_p1_kinds_04", 1200, BANDY + 288, b)

# ══════════════════════════════════════════════════════════════════════
# 05. arXiv 제목과 게재본 제목이 다르다
# ══════════════════════════════════════════════════════════════════════
TCH = [("LLM.int8(): 8-bit Matrix Multiplication…",
        "GPT3.int8(): 8-bit Matrix Multiplication…", "NeurIPS 2022", "이름 자체가 바뀜"),
       ("GPTQ: Accurate Post-Training Quantization…",
        "OPTQ: Accurate Quantization…", "ICLR 2023", "이름 자체가 바뀜"),
       ("SCAFFOLD: … for On-Device Federated Learning",
        "SCAFFOLD: … for Federated Learning", "ICML 2020", "\"On-Device\" 가 빠짐"),
       ("On the Convergence of Federated Optimization…",
        "Federated Optimization in Heterogeneous Networks", "MLSys 2020", "제목 전면 교체")]
b = title(530, 30, "게재되면서 제목이 바뀐다 — 인용 71편 중 4편",
          "제목만으로 검색하면 게재본을 영영 못 찾는다 (실측)")
b += f'  <defs>{MK.format(i="tb", c=BLUE)}</defs>\n'
b += (f'  <text x="230" y="72" text-anchor="middle" font-size="14" font-weight="700" '
      f'fill="{MUT}">arXiv 제목 (널리 인용되는 쪽)</text>\n')
b += (f'  <text x="740" y="72" text-anchor="middle" font-size="14" font-weight="700" '
      f'fill="{TEAL}">게재본 제목 (인용해야 하는 쪽)</text>\n')
for i, (pre, pub, ven, why) in enumerate(TCH):
    y = 90 + i * 84
    b += (f'  <rect x="40" y="{y}" width="380" height="52" rx="9" fill="#f1f5f9" '
          f'stroke="{LINE}"/>\n')
    b += (f'  <text x="230" y="{y+32}" text-anchor="middle" font-size="12.6" '
          f'fill="{MUT}">{esc(pre)}</text>\n')
    b += (f'  <line x1="432" y1="{y+26}" x2="486" y2="{y+26}" stroke="{BLUE}" '
          f'stroke-width="2.6" marker-end="url(#tb)"/>\n')
    b += (f'  <text x="459" y="{y+16}" text-anchor="middle" font-size="11.5" '
          f'fill="{BLUE}">게재</text>\n')
    b += (f'  <rect x="498" y="{y}" width="484" height="52" rx="9" fill="#e6f4f2" '
          f'stroke="{TEAL}" stroke-width="1.7"/>\n')
    b += (f'  <text x="740" y="{y+32}" text-anchor="middle" font-size="12.8" '
          f'font-weight="700" fill="#0b4a48">{esc(pub)}</text>\n')
    b += (f'  <text x="998" y="{y+22}" font-size="12.5" font-weight="700" '
          f'fill="{INK}">{esc(ven)}</text>\n')
    b += (f'  <text x="998" y="{y+42}" font-size="11.8" fill="{MUT}">{esc(why)}</text>\n')
b += note(40, 430, 1160, 96,
          "LLM.int8() 과 GPTQ 는 제목으로 검색하면 게재본을 못 찾는다",
          ["우리 스크립트도 못 찾았다 — 앞선 주차에서 손으로 확인해 둔 게재본 제목을 별칭으로 넣어 다시 조회해서야 찾았다.",
           "즉 이 실패는 자동으로 발견되지 않는다. 이미 답을 아는 사람만 찾을 수 있다."],
          bg="#fee2e2", ln="#f0a3a3", hc="#991b1b")
fig("w13_p1_titles_05", 1240, 546, b)

# ══════════════════════════════════════════════════════════════════════
# 06. 자동 조회 84.4%
# ══════════════════════════════════════════════════════════════════════
V = VE["summary"] if "summary" in VE else VE
n_chk, agree, dis, mis = V["n_checked"], V["agree"], V["disagree"], V["missed"]
b = title(520, 30, f"자동 서지 조회의 정확도 — 손으로 확인한 {n_chk}편과 맞대 보면",
          f"여섯 편 중 한 편꼴로 틀린다 (정확도 {V['accuracy']}% · 오류율 {V['error_rate']}%)")
GX, GY, CS, PER = 56, 92, 40, 8
cells = [GRN] * agree + [RED] * dis + [MUT] * mis
for i, col in enumerate(cells):
    x = GX + (i % PER) * CS
    y = GY + (i // PER) * CS
    b += (f'  <rect x="{x}" y="{y}" width="{CS-7}" height="{CS-7}" rx="6" fill="{col}"/>\n')
b += legend(GX, GY + 4 * CS + 20,
            [(GRN, f"일치 {agree}"), (RED, f"불일치 {dis}"), (MUT, f"못 찾음 {mis}")])
b += (f'  <rect x="{GX}" y="{GY+4*CS+42}" width="{PER*CS-7}" height="56" rx="10" '
      f'fill="#fee2e2" stroke="#f0a3a3"/>\n')
b += (f'  <text x="{GX+(PER*CS-7)/2}" y="{GY+4*CS+78}" text-anchor="middle" '
      f'font-size="19" font-weight="700" fill="#991b1b">오류율 {V["error_rate"]}%</text>\n')

RX = GX + PER * CS + 40
b += (f'  <text x="{RX}" y="{GY-8}" font-size="14.5" font-weight="700" '
      f'fill="{INK}">무엇이, 왜 틀렸나 — 원인이 넷 다 다르다</text>\n')
ROWS = [("Han 2015 (연결 학습)", "프리프린트로 판정", "NIPS 2015 게재", "검색 상위에 안 뜸", RED),
        ("DLG", "conf · ICHI 2021", "conf · NeurIPS 2019", "성 \"Zhu\" 가 흔해 엉뚱한 논문 통과", RED),
        ("Deep Compression", "못 찾음", "ICLR 2016", "게재본 색인 매칭 실패", MUT),
        ("Model Compression", "못 찾음", "KDD 2006", "제목이 일반적이라 결과에 묻힘", MUT),
        ("Brysbaert 2019", "못 찾음", "J. Memory and Language 2019", "DBLP 는 전산 분야만 색인", MUT)]
b += (f'  <text x="{RX}" y="{GY+18}" font-size="12" fill="{MUT}">논문</text>\n')
b += (f'  <text x="{RX+192}" y="{GY+18}" font-size="12" fill="{MUT}">자동 조회</text>\n')
b += (f'  <text x="{RX+352}" y="{GY+18}" font-size="12" fill="{MUT}">실제</text>\n')
b += (f'  <text x="{RX+588}" y="{GY+18}" font-size="12" fill="{MUT}">실패 원인</text>\n')
for i, (nm, auto, real, why, col) in enumerate(ROWS):
    y = GY + 30 + i * 40
    b += (f'  <rect x="{RX-10}" y="{y}" width="{870}" height="34" rx="7" '
          f'fill="{"#ffffff" if i%2==0 else "#f1f5f9"}" stroke="{LINE}"/>\n')
    b += (f'  <text x="{RX}" y="{y+22}" font-size="12.6" font-weight="700" '
          f'fill="{INK}">{esc(nm)}</text>\n')
    b += (f'  <text x="{RX+192}" y="{y+22}" font-size="12.4" fill="{col}">{esc(auto)}</text>\n')
    b += (f'  <text x="{RX+352}" y="{y+22}" font-size="12.4" font-weight="700" '
          f'fill="#15803d">{esc(real)}</text>\n')
    b += (f'  <text x="{RX+588}" y="{y+22}" font-size="12.2" fill="{MUT}">{esc(why)}</text>\n')

b += note(56, 356, 1276, 96, "서지 확인은 자동화되지 않는다",
          [f"① 제목만으로 검색 — 71편 중 {S['naive_wrong']}편에서 엉뚱한 논문이 잡혔다 ({S['naive_wrong_pct']}%).",
           f"② 제목 + 제1저자 성으로 검색 — 엉뚱한 논문은 걸러졌지만, 손으로 확인한 {n_chk}편 중 {dis+mis}편이 여전히 틀렸다 ({V['error_rate']}%).",
           "마지막 한 걸음 — 게재처 페이지를 눈으로 확인하는 일 — 은 사람이 해야 한다."],
          bg="#fef3e8", ln="#f0b27a", hc="#9a4b06")
fig("w13_p1_auto_06", 1392, 486, b)

# ══════════════════════════════════════════════════════════════════════
# 07. 구조량과 규모량의 변동계수
# ══════════════════════════════════════════════════════════════════════
SS = [r for g in ("A", "B") for r in RE[g]["structural"]]
MM = [r for g in ("A", "B") for r in RE[g]["magnitude"]]
MM = [r for r in MM if r["name"] != "최대 화소 오차"]
MM.sort(key=lambda r: r["cv"])
VD = RE["verdict"]

b = title(560, 30, "같은 코드를 여덟 번 돌리면 무엇이 흔들리는가",
          "A: 같은 시드 8회(11주차 커버리지) · B: 다른 시드 8회(12주차 역복원) — 변동계수 CV")
AX, AY, AW = 300, 88, 620
mxcv = max(r["cv"] for r in MM)
b += (f'  <text x="{AX-232}" y="{AY-14}" font-size="14.5" font-weight="700" '
      f'fill="{TEAL}">구조량 — 세는 것</text>\n')
rows = [(r, TEAL) for r in SS] + [(None, None)] + [(r, AMB) for r in MM]
yy = AY
for r, col in rows:
    if r is None:
        b += (f'  <line x1="{AX-236}" y1="{yy+6}" x2="{AX+AW+96}" y2="{yy+6}" '
              f'stroke="{LINE}" stroke-dasharray="5 4"/>\n')
        yy += 26
        b += (f'  <text x="{AX-232}" y="{yy+2}" font-size="14.5" font-weight="700" '
              f'fill="{AMB}">규모량 — 재는 것</text>\n')
        yy += 16
        continue
    nm = r["name"] + (f' ({r["unit"]})' if r["unit"] else "")
    b += (f'  <text x="{AX-14}" y="{yy+16}" text-anchor="end" font-size="12.8" '
          f'fill="#334155">{esc(nm)}</text>\n')
    wpx = AW * r["cv"] / mxcv if mxcv else 0
    if r["cv"] == 0.0:
        b += (f'  <circle cx="{AX}" cy="{yy+12}" r="4.5" fill="none" stroke="{col}" '
              f'stroke-width="2"/>\n')
        b += (f'  <text x="{AX+16}" y="{yy+17}" font-size="13" font-weight="700" '
              f'fill="{col}">CV 0.00%  (막대 없음)   —   여덟 번 모두 소수점까지 동일: '
              f'{r["min"]:g}{r["unit"]}</text>\n')
    else:
        b += (f'  <rect x="{AX}" y="{yy+3}" width="{max(wpx,3):.1f}" height="18" rx="4" '
              f'fill="{col}"/>\n')
        b += (f'  <text x="{AX+max(wpx,3)+9:.1f}" y="{yy+17}" font-size="12.6" '
              f'font-weight="700" fill="{INK}">{r["cv"]:.2f}%</text>\n')
        b += (f'  <text x="{AX+max(wpx,3)+62:.1f}" y="{yy+17}" font-size="12" '
              f'fill="{MUT}">{r["min"]:.2f} ~ {r["max"]:.2f}</text>\n')
    yy += 27
b += (f'  <line x1="{AX}" y1="{AY-4}" x2="{AX}" y2="{yy-6}" stroke="{LINE}" '
      f'stroke-width="1.4"/>\n')

yy += 12
b += note(64, yy, 540, 96, "구조량",
          [f"{VD['n_structural_exact']}/{VD['n_structural']} 개가 CV 0.00%",
           "노드 수 · 커버리지 · 왕복 횟수 · 경계 바이트 · 라벨 복원 수"])
b += note(624, yy, 540, 96, "규모량",
          [f"막대로 그린 것 중 최대 CV {max(r['cv'] for r in MM):.2f}% · 전체 최대 {VD['magnitude_cv_max']}% · 평균 {VD['magnitude_cv_mean']}%",
           "ms · PSNR · 천장 배수 — 시드·기계·버전에 따라 흔들린다"],
          bg="#fef3e8", ln="#f0b27a", hc="#9a4b06")
b += (f'  <text x="614" y="{yy+152}" text-anchor="middle" font-size="14" '
      f'font-weight="700" fill="{INK}">'
      f'재현을 요구할 것은 배수가 아니라 구조다 — 이 문장은 수사가 아니라 측정 결과였다</text>\n')
b += (f'  <text x="614" y="{yy+176}" text-anchor="middle" font-size="12.8" fill="{MUT}">'
      f'※ 복원 PSNR 이 160 dB 대인 것은 오타가 아니다 — 나눗셈 한 번으로 float32 표현 한계까지 복원된다(12주차 2.1).</text>\n')
b += (f'  <text x="614" y="{yy+198}" text-anchor="middle" font-size="12.8" fill="{MUT}">'
      f'※ 전체 최대 {VD["magnitude_cv_max"]}% 는 「최대 화소 오차」 항목의 값이다 — 평균이 1e-7 수준이라 막대로 그리면 오독되므로 표에서만 보고한다</text>\n')
fig("w13_p2_repro_07", 1228, yy + 222, b)

# ══════════════════════════════════════════════════════════════════════
# 08. 같은 코드, 다른 시드 — 2.85배 범위
# ══════════════════════════════════════════════════════════════════════
NZ = [r for r in RE["B"]["magnitude"] if r["name"].startswith("노이즈")][0]
vals = NZ["values"]
b = title(520, 30, "같은 코드에 시드만 바꿨을 때 — 노이즈 σ=0.01 의 복원 PSNR",
          f"8회 · {NZ['min']:.2f} ~ {NZ['max']:.2f} dB · {NZ['ratio']:.2f}배 범위 (실측)")
PX, PY, PW, PH = 96, 96, 800, 220
lo, hi = 5.0, 25.0


def yv(v):
    return PY + PH - PH * (v - lo) / (hi - lo)


for g in range(5, 26, 5):
    b += (f'  <line x1="{PX}" y1="{yv(g):.1f}" x2="{PX+PW}" y2="{yv(g):.1f}" '
          f'stroke="{LINE}" stroke-dasharray="4 4"/>\n')
    b += (f'  <text x="{PX-12}" y="{yv(g)+5:.1f}" text-anchor="end" font-size="12" '
          f'fill="{MUT}">{g}</text>\n')
b += (f'  <text x="{PX}" y="{PY-12}" font-size="11.8" fill="{MUT}">'
      f'※ 세로축은 0 이 아니라 5 dB 부터 그렸다</text>\n')
b += (f'  <text x="{PX-52}" y="{PY+PH/2}" text-anchor="middle" font-size="12.5" '
      f'fill="{MUT}" transform="rotate(-90 {PX-52} {PY+PH/2})">복원 PSNR (dB)</text>\n')
b += (f'  <rect x="{PX}" y="{yv(NZ["max"]):.1f}" width="{PW}" '
      f'height="{yv(NZ["min"])-yv(NZ["max"]):.1f}" fill="{AMB}" opacity="0.09"/>\n')
b += (f'  <line x1="{PX}" y1="{yv(NZ["mean"]):.1f}" x2="{PX+PW}" y2="{yv(NZ["mean"]):.1f}" '
      f'stroke="{AMB}" stroke-width="2" stroke-dasharray="7 4"/>\n')
b += (f'  <text x="{PX+PW+10}" y="{yv(NZ["mean"])+5:.1f}" font-size="12.5" '
      f'font-weight="700" fill="{AMB}">평균 {NZ["mean"]:.2f}</text>\n')
for i, v in enumerate(vals):
    x = PX + PW * (i + 0.5) / len(vals)
    col = RED if v == NZ["min"] else (BLUE if v == NZ["max"] else TEAL)
    b += f'  <circle cx="{x:.1f}" cy="{yv(v):.1f}" r="9" fill="{col}"/>\n'
    lw = tw(f"{v:.2f}", 12.5) + 12
    b += (f'  <rect x="{x-lw/2:.1f}" y="{yv(v)-30:.1f}" width="{lw:.1f}" height="18" '
          f'rx="5" fill="#f8fafc" opacity="0.94"/>\n')
    b += (f'  <text x="{x:.1f}" y="{yv(v)-17:.1f}" text-anchor="middle" font-size="12.5" '
          f'font-weight="700" fill="{col}">{v:.2f}</text>\n')
    b += (f'  <text x="{x:.1f}" y="{PY+PH+22}" text-anchor="middle" font-size="12" '
          f'fill="{MUT}">시드 {1000+i}</text>\n')
b += (f'  <line x1="{PX}" y1="{PY+PH}" x2="{PX+PW}" y2="{PY+PH}" stroke="{LINE}" '
      f'stroke-width="1.4"/>\n')

b += card(64, 366, 556, 116, "논문 A 라면 이렇게 쓴다", RED, "#fee2e2",
          ["\"σ=0.01 의 노이즈로 복원 공격을 막을 수 있다.\"",
           f"(PSNR {NZ['min']:.1f} dB — 형체가 남지 않는다)", "",
           "시드 1000 을 뽑았다."])
b += card(640, 366, 556, 116, "논문 B 라면 이렇게 쓴다", BLUE, "#e0edfd",
          ["\"σ=0.01 로는 부족하다. 여전히 형체가 남는다.\"",
           f"(PSNR {NZ['max']:.1f} dB — 윤곽이 보인다)", "",
           "시드 1005 를 뽑았다."])
b += note(64, 498, 1132, 74, "둘 다 맞다. 같은 코드에 시드만 다르다",
          ["그리고 논문에는 대개 시드가 안 적혀 있다. 한 점이 아니라 곡선 전체를 보고하는 것이 유일한 방어다 — 12주차가 σ 를 0 부터 2.0 까지 보고한 이유다."],
          bg="#fef3e8", ln="#f0b27a", hc="#9a4b06")
fig("w13_p2_spread_08", 1260, 592, b)

# ══════════════════════════════════════════════════════════════════════
# 09. 논문을 열 때 던질 여섯 질문
# ══════════════════════════════════════════════════════════════════════
Q = [("①", "무엇과 비교했나?", "실험 절의 baseline 문장", "기준선", BLUE, "5분"),
     ("②", "어떤 조건에서?", "표 각주 · 실험 설정 절", "조건", TEAL, "5분"),
     ("③", "이 숫자가 논문에 그대로 있나?", "본문·표에서 그 값 찾기", "범위", PINK, "5분"),
     ("④", "논문이 그 말을 했나?", "해당 절 전체 읽기", "무주장", RED, "절 하나"),
     ("⑤", "이게 이 논문이 처음인가?", "관련 연구 절 · 선행 연구", "귀속", PUR, "관련 연구 절"),
     ("⑥", "어디에 실렸나?", "DBLP · 게재처 페이지", "서지", AMB, "검색 + 눈")]
b = title(520, 30, "논문을 열 때 던질 여섯 질문",
          "1교시의 여섯 유형을 그대로 뒤집으면 체크리스트가 된다")
b += (f'  <text x="118" y="76" font-size="12.5" fill="{MUT}">질문</text>\n')
b += (f'  <text x="620" y="76" text-anchor="start" font-size="12.5" fill="{MUT}">어디를 보나</text>\n')
b += (f'  <text x="884" y="76" font-size="12.5" fill="{MUT}">막는 유형</text>\n')
b += (f'  <text x="1000" y="76" text-anchor="start" font-size="12.5" fill="{MUT}">드는 시간</text>\n')
for i, (n, q, where, typ, col, cost) in enumerate(Q):
    y = 88 + i * 58
    b += (f'  <rect x="48" y="{y}" width="1064" height="48" rx="10" fill="#ffffff" '
          f'stroke="{col}" stroke-width="1.6"/>\n')
    b += f'  <circle cx="88" cy="{y+24}" r="17" fill="{col}"/>\n'
    b += (f'  <text x="88" y="{y+31}" text-anchor="middle" font-size="17" '
          f'font-weight="700" fill="#ffffff">{n}</text>\n')
    b += (f'  <text x="118" y="{y+30}" font-size="15" font-weight="700" '
          f'fill="{INK}">{esc(q)}</text>\n')
    b += (f'  <text x="620" y="{y+30}" font-size="12.6" fill="{MUT}">{esc(where)}</text>\n')
    bwp = tw(typ, 13) + 22
    b += (f'  <rect x="880" y="{y+11}" width="{bwp}" height="26" rx="13" fill="{col}" '
          f'opacity="0.14"/>\n')
    b += (f'  <text x="{880+bwp/2}" y="{y+29}" text-anchor="middle" font-size="13" '
          f'font-weight="700" fill="{col}">{esc(typ)}</text>\n')
    b += (f'  <text x="1000" y="{y+30}" font-size="12.6" fill="#334155">{esc(cost)}</text>\n')
b += note(48, 446, 528, 96, "구조 주장 — 믿고 가져간다",
          ["\"A 가 B 보다 빠르다\" · \"X 를 지원하면 왕복이 없어진다\"",
           "방향은 잘 안 뒤집힌다. 논문의 기여는 대개 여기 있다."])
b += note(584, 446, 528, 96, "규모 주장 — 조건과 함께만",
          ["\"3.8배 빠르다\" · \"PSNR 164 dB\"",
           "초록에 실리는 것은 이쪽이다. 배수는 재현하려 하지 않는다."],
          bg="#fef3e8", ln="#f0b27a", hc="#9a4b06")
b += (f'  <text x="580" y="570" text-anchor="middle" font-size="13.5" '
      f'font-weight="700" fill="{INK}">'
      f'전부 다 할 필요는 없다 — 인용의 무게에 맞춘다. 한 줄 배경 인용이면 ⑥만, 결론의 근거로 쓰면 ①~⑥ 전부.</text>\n')
fig("w13_p2_checklist_09", 1160, 596, b)

# ══════════════════════════════════════════════════════════════════════
# 10. 열두 주가 남긴 미결 문제
# ══════════════════════════════════════════════════════════════════════
OPEN = [(8, "SRAM 계산은 하한이다 — 실제와 얼마나 차이 나나?", "실제 MCU 에 올려 재야 안다", TEAL),
        (9, "전처리를 GPU 로 옮기면 암달 천장 3.71배가 어디까지?", "분모가 바뀌면 다시 재야 한다", TEAL),
        (10, "135M 모델의 결과가 7B 로 확장되나?", "방법은 크기 무관이지만 배수는 아니다", TEAL),
        (11, "커널 실행 고정비 c0 를 넣으면 이식 판정이 어디서 뒤집히나?", "실제 보드에서 c0 를 재야 안다", BLUE),
        (11, "왕복 비단조성의 일반 조건은?", "그래프 구조와의 관계가 아직 정식화 안 됨", BLUE),
        (12, "합성곱에서 역복원은 얼마나 어려운가?", "\"어려운 것\" 과 \"불가능한 것\" 은 다르다", PUR),
        (12, "연합 학습 업데이트의 법적 지위", "개인정보보호위원회 유권해석이 없다", PUR)]
b = title(520, 30, "열두 주가 남긴, 아직 안 풀린 문제들",
          "매주 재고 명시적으로 못 푼 채 남긴 것 — 세미나와 캡스톤의 재료다")
for i, (w, q, why, col) in enumerate(OPEN):
    y = 76 + i * 52
    b += (f'  <rect x="48" y="{y}" width="1016" height="44" rx="9" fill="#ffffff" '
          f'stroke="{LINE}"/>\n')
    b += f'  <rect x="48" y="{y}" width="6" height="44" rx="3" fill="{col}"/>\n'
    b += f'  <circle cx="88" cy="{y+22}" r="16" fill="{col}" opacity="0.15"/>\n'
    b += (f'  <text x="88" y="{y+27}" text-anchor="middle" font-size="13" '
          f'font-weight="700" fill="{col}">{w}주</text>\n')
    b += (f'  <text x="118" y="{y+27}" font-size="13.4" fill="{INK}">{esc(q)}</text>\n')
    b += (f'  <text x="1052" y="{y+27}" text-anchor="end" font-size="12" '
          f'fill="{MUT}">{esc(why)}</text>\n')
y = 76 + len(OPEN) * 52 + 8
b += (f'  <rect x="48" y="{y}" width="1016" height="102" rx="11" fill="#fef3e8" '
      f'stroke="{AMB}" stroke-width="2"/>\n')
b += (f'  <rect x="48" y="{y}" width="6" height="102" rx="3" fill="{AMB}"/>\n')
b += (f'  <text x="88" y="{y+30}" font-size="13" font-weight="700" fill="{AMB}">전반</text>\n')
b += (f'  <text x="140" y="{y+30}" font-size="14.5" font-weight="700" fill="#9a4b06">'
      f'대리 지표의 유효성이 기기와 조건에 달렸다면, 언제 어떤 지표가 유효한지 판정하는 방법은?</text>\n')
b += (f'  <text x="140" y="{y+56}" font-size="12.5" fill="#334155">'
      f'7주차에서 FLOPs 는 데스크톱 CPU 의 나쁜 대리 지표였고, 8주차에서 MCU 위의 지연은 연산 수에 선형이었다.</text>\n')
b += (f'  <text x="140" y="{y+78}" font-size="12.5" fill="#334155">'
      f'두 논문은 모순이 아니다 — 유효성이 지표의 성질이 아니라 기기와 조건의 성질이기 때문이다. 미리 판정하는 방법은 아직 없다.</text>\n')
fig("w13_p2_open_10", 1112, y + 130, b)

# ══════════════════════════════════════════════════════════════════════
# 11. 인용 71편의 게재처 분포
# ══════════════════════════════════════════════════════════════════════
GMAP = [
    ("CoRR (arXiv)", ["CoRR"], "프리프린트 — 게재 이력 없음", AMB),
    ("NeurIPS · ICML · ICLR · AISTATS",
     ["NeurIPS", "NIPS", "NeurIPS Datasets and Benchmarks", "ICML", "ICLR", "AISTATS"],
     "기계학습 본류 — 모델 · 알고리즘 · 이론", TEAL),
    ("CVPR · ECCV · ICCV", ["CVPR", "ECCV", "ICCV"],
     "컴퓨터 비전 — 검출 · 경량 백본", BLUE),
    ("OSDI · SOSP · PACT · AFIPS",
     ["OSDI", "SOSP", "PACT", "AFIPS Spring Joint Computing Conference"],
     "시스템 · 아키텍처 — 서빙 · 스케줄링", GRN),
    ("CCS · EuroS&P · TIFS",
     ["CCS", "EuroS&amp;P", "IEEE Trans. Inf. Forensics Secur."],
     "보안 · 프라이버시 — 공격 · 방어 · 프로토콜", PUR),
    ("MLSys", ["MLSys"], "기계학습 시스템", "#0891b2"),
    ("저널 (CACM · FnT · TPDS)",
     ["Commun. ACM", "Found. Trends Mach. Learn.",
      "IEEE Trans. Parallel Distributed Syst."], "저널 — 장문 · 서베이", MUT),
    ("그 밖의 학회 (EMNLP · ICHI · MobiSys 워크숍)",
     ["EMNLP", "ICHI", "AdaAIoTSys@MobiSys"], "인접 분야", "#94a3b8"),
    ("PLDI · MAPL", ["PLDI", "MAPL@PLDI"], "프로그래밍 언어 — 컴파일러 · 커널", PINK),
]
_vc = collections.Counter()
for _p in BI["papers"]:
    if _p["not_found"]:
        continue
    _e = _p["published"] or _p["preprint"] or _p["strict_best"] or _p["loose_best"]
    _vc[_e["venue"]] += 1
GRP = [(nm, sum(_vc[v] for v in vs), kind, col) for nm, vs, kind, col in GMAP]
GRP.sort(key=lambda g: -g[1])
GRP.append(("DBLP 에서 못 찾음", S["not_found"], "본문 1.3 의 실패 사례", LINE))
assert sum(g[1] for g in GRP) == NP, sum(g[1] for g in GRP)

b = title(640, 30, f"우리가 인용한 {NP}편은 어디에 실렸나 — 한 학회의 주제가 아니다",
          f"게재처 {len(_vc)}곳에 흩어져 있다 (실측 · 우리 인용 목록 기준)")
BX, BY, BW = 396, 82, 400
mg = max(g[1] for g in GRP)
for i, (nm, v, kind, col) in enumerate(GRP):
    y = BY + i * 44
    b += (f'  <text x="{BX-14}" y="{y+19}" text-anchor="end" font-size="13.4" '
          f'font-weight="700" fill="{INK}">{esc(nm)}</text>\n')
    b += (f'  <rect x="{BX}" y="{y}" width="{BW*v/mg:.1f}" height="27" rx="5" fill="{col}"/>\n')
    b += (f'  <text x="{BX+BW*v/mg+10:.1f}" y="{y+19}" font-size="13.4" '
          f'font-weight="700" fill="{col}">{v}편</text>\n')
    b += (f'  <text x="{BX+BW+58}" y="{y+19}" font-size="12.4" fill="{MUT}">{esc(kind)}</text>\n')
    if nm == "MLSys":
        b += (f'  <text x="{BX+BW+188}" y="{y+19}" font-size="12.2" font-weight="700" '
              f'fill="#0891b2">← 이 과목의 중심 · MicroNets · FedProx · Pope · AWQ</text>\n')
b += (f'  <line x1="{BX}" y1="{BY-6}" x2="{BX}" y2="{BY+len(GRP)*44-10}" '
      f'stroke="{LINE}" stroke-width="1.4"/>\n')
b += (f'  <text x="{BX-14}" y="{BY+len(GRP)*44+14}" text-anchor="end" font-size="13" '
      f'font-weight="700" fill="{INK}">합계</text>\n')
b += (f'  <text x="{BX+10}" y="{BY+len(GRP)*44+14}" font-size="13" font-weight="700" '
      f'fill="{INK}">{NP}편</text>\n')
b += (f'  <text x="{BX+90}" y="{BY+len(GRP)*44+14}" font-size="12.8" fill="{MUT}">'
      f'※ 묶음은 게재 형태가 아니라 주제 기준이다 — 예컨대 TIFS 는 저널이지만 「보안·프라이버시」 로 묶었다</text>\n')

MY = BY + len(GRP) * 44 + 26
b += (f'  <text x="60" y="{MY+20}" font-size="14.5" font-weight="700" '
      f'fill="{INK}">여러분의 관심에 따라 먼저 볼 곳</text>\n')
MAP = [("모델 경량화 · NAS", "NeurIPS · ICML · ICLR · CVPR", TEAL),
       ("추론 시스템 · 서빙", "MLSys · OSDI · SOSP", GRN),
       ("컴파일러 · 커널", "PLDI · CGO · MAPL", PINK),
       ("TinyML · MCU", "MLSys · 임베디드 학회", AMB),
       ("프라이버시 · 연합 학습", "CCS · S&P · EuroS&P · NeurIPS", PUR)]
for i, (k, v, col) in enumerate(MAP):
    x = 60 + (i % 3) * 380
    y = MY + 34 + (i // 3) * 52
    b += (f'  <rect x="{x}" y="{y}" width="356" height="42" rx="9" fill="#ffffff" '
          f'stroke="{col}" stroke-width="1.5"/>\n')
    b += (f'  <text x="{x+14}" y="{y+18}" font-size="12.4" fill="{MUT}">{esc(k)}</text>\n')
    b += (f'  <text x="{x+14}" y="{y+35}" font-size="13" font-weight="700" '
          f'fill="{col}">{esc(v)}</text>\n')
b += note(60, MY + 144, 1300, 74, "이 분포는 우리 인용 목록의 분포이지 분야 전체의 분포가 아니다",
          [f"교재를 쓰며 고른 {NP}편이라 선택 편향이 있다. 여러분 주제로 같은 집계를 다시 하면 다른 지도가 나온다 — 그것이 3교시 선택 과제다."])
fig("w13_p2_venues_11", 1420, MY + 246, b)

# ══════════════════════════════════════════════════════════════════════
# 12. 3교시 산출물 — 조건표 한 장
# ══════════════════════════════════════════════════════════════════════
ROWS12 = [("논문", "저자 · 제목 · 게재처와 연도 (프리프린트면 arXiv 번호)", "⑥", AMB, 10),
          ("가져온 숫자", "인용하려는 값 하나", "", MUT, 0),
          ("논문의 어디에", "표 N · 그림 N · §N.N — 못 찾았으면 그렇게 적을 것", "③", PINK, 30),
          ("무엇과 비교", "baseline 문장을 그대로 옮긴다", "①", BLUE, 30),
          ("어떤 조건", "하드웨어 · 데이터셋 · 설정 · 버전", "②", TEAL, "①과 합산"),
          ("논문이 한 말인가", "인용된 서술과 논문 서술의 차이", "④", RED, 20),
          ("이 논문이 처음인가", "관련 연구 절에서 확인한 선행 연구", "⑤", PUR, "④와 합산"),
          ("판정", "여섯 유형 중 어디에 해당하는가 (또는 \"이상 없음\")", "", MUT, 0),
          ("한 문장", "조건을 붙여 다시 쓴 문장 — 이것이 진짜 산출물", "", GRN, 10)]
b = title(540, 30, "3교시 산출물 — 조건표 한 장",
          "발표는 10분, 자료는 이 표 한 장이면 된다. 요약도 발표 기술도 배점에 없다.")
TY = 78
b += (f'  <text x="64" y="{TY-8}" font-size="12" fill="{MUT}">항목</text>\n')
b += (f'  <text x="270" y="{TY-8}" font-size="12" fill="{MUT}">무엇을 적나</text>\n')
b += (f'  <text x="836" y="{TY-8}" font-size="12" fill="{MUT}">질문</text>\n')
b += (f'  <text x="978" y="{TY-8}" text-anchor="end" font-size="12" fill="{MUT}">'
      f'배점 (합계 100)</text>\n')
for i, (k, v, q, col, pt) in enumerate(ROWS12):
    y = TY + i * 46
    bg = "#ffffff" if i % 2 == 0 else "#f1f5f9"
    if k == "한 문장":
        bg = "#dcfce7"
    b += (f'  <rect x="52" y="{y}" width="964" height="40" rx="8" fill="{bg}" '
          f'stroke="{LINE}"/>\n')
    b += (f'  <text x="70" y="{y+26}" font-size="13.4" font-weight="700" '
          f'fill="{INK}">{esc(k)}</text>\n')
    b += (f'  <text x="270" y="{y+26}" font-size="12.6" fill="#334155">{esc(v)}</text>\n')
    if q:
        b += f'  <circle cx="850" cy="{y+20}" r="14" fill="{col}"/>\n'
        b += (f'  <text x="850" y="{y+26}" text-anchor="middle" font-size="14" '
              f'font-weight="700" fill="#ffffff">{q}</text>\n')
    if isinstance(pt, int) and pt:
        b += (f'  <text x="978" y="{y+26}" text-anchor="end" font-size="13" '
              f'font-weight="700" fill="{MUT}">{pt}점</text>\n')
    elif isinstance(pt, str):
        b += (f'  <text x="978" y="{y+26}" text-anchor="end" font-size="11.8" '
              f'fill="{MUT}">{esc(pt)}</text>\n')
    else:
        b += (f'  <text x="978" y="{y+26}" text-anchor="end" font-size="11.8" '
              f'fill="{MUT}">배점 없음</text>\n')
Y2 = TY + len(ROWS12) * 46 + 10
b += (f'  <rect x="52" y="{Y2}" width="964" height="122" rx="11" fill="#e6f4f2" '
      f'stroke="{TEAL}" stroke-width="1.9"/>\n')
b += (f'  <text x="72" y="{Y2+28}" font-size="14" font-weight="700" fill="#0b4a48">'
      f'「한 문장」 의 예 — 이대로 자기 논문에 옮겨 쓸 수 있어야 한다</text>\n')
for i, l in enumerate([
        "\"TVM 은 서버급 GPU(Titan X)에서 MXNet·TensorFlow·TF-XLA 대비 종단간 1.6~3.8배,",
        "모바일 GPU(Mali-T860MP4)에서 ARM Compute Library 대비 1.2~1.6배의 속도 향상을 보고했다(OSDI 2018 §6.1·§6.3).",
        "이 둘은 다른 하드웨어·다른 기준선의 수치이므로 하나의 범위로 합쳐 인용해서는 안 된다.\""]):
    b += (f'  <text x="72" y="{Y2+56+i*24}" font-size="12.8" '
          f'fill="#334155">{esc(l)}</text>\n')
b += (f'  <text x="534" y="{Y2+152}" text-anchor="middle" font-size="13.5" '
      f'font-weight="700" fill="{AMB}">'
      f'"이상 없음" 도 훌륭한 결과다 — 억지로 흠을 찾지 말 것. 확인했다는 것 자체가 결과다.</text>\n')
fig("w13_p3_seminar_12", 1068, Y2 + 180, b)

# ══════════════════════════════════════════════════════════════════════
for k, v in F.items():
    (OUT / f"{k}.svg").write_text(v, encoding="utf-8")
    print(f"  {k}.svg  ({len(v)//1024} KB)")
print(f"\n→ {len(F)} figures in {OUT}")
