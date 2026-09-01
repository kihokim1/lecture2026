# -*- coding: utf-8 -*-
"""14주차 그림 — audit.json · rubric.json 기반.

주의: 유니코드 위첨자(ᵀ ⁻⁵ ²⁰)는 Noto Sans CJK KR 에 글리프가 없다.
      평문 표기만 쓴다.
검사: 그림에 들어가는 합계·비율은 아래에서 assert 로 자동 검증한다.
"""
import json, pathlib

D = pathlib.Path("/root/lab14")
AU = json.load(open(D / "audit.json"))
RU = json.load(open(D / "rubric.json"))
OUT = pathlib.Path("/root/ondevice-ai/img/week14"); OUT.mkdir(parents=True, exist_ok=True)

HEAD = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        'font-family="\'Segoe UI\',Arial,sans-serif">\n'
        '  <rect width="{w}" height="{h}" fill="#f8fafc"/>\n')
F = {}
TEAL, AMB, RED, BLUE = "#028090", "#e4711b", "#dc2626", "#2563eb"
GRN, PUR, PINK, CYAN = "#16a34a", "#7c3aed", "#be185d", "#0891b2"
INK, MUT, LINE = "#0f172a", "#64748b", "#cbd5e1"


def fig(name, w, h, body):
    F[name] = HEAD.format(w=w, h=h) + body + "\n</svg>\n"


def tw(s, size):
    return sum(size * (1.0 if ord(c) > 0x2000 else 0.56) for c in s)


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


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


def card(x, y, w, h, hd, col, bg, lines, hsize=15, lsize=12.5, lh=19):
    s = (f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="11" fill="{bg}" '
         f'stroke="{col}" stroke-width="1.8"/>\n')
    s += (f'  <text x="{x+14}" y="{y+25}" font-size="{hsize}" font-weight="700" '
          f'fill="{col}">{esc(hd)}</text>\n')
    for i, l in enumerate(lines):
        s += (f'  <text x="{x+14}" y="{y+48+i*lh}" font-size="{lsize}" '
              f'fill="#334155">{esc(l)}</text>\n')
    return s


MK = ('<marker id="{i}" markerWidth="9" markerHeight="9" refX="7" refY="3.5" '
      'orient="auto" markerUnits="userSpaceOnUse">'
      '<path d="M0,0 L7,3.5 L0,7 Z" fill="{c}"/></marker>')

S, H, AP = AU["summary"], AU["hand"], AU["appendix"]
ROWS = AU["rows"]

# ══════════════ 자동 정합성 검사 ══════════════
assert sum(r["claims"] for r in ROWS) + AP["claims"] == S["claims"], "주차별+부록 수치문장 합 ≠ 총계"
assert sum(r["cond"] for r in ROWS) + AP["cond"] == S["cond"], "주차별+부록 조건 합 ≠ 총계"
assert sum(r["cites"] for r in ROWS) + AP["cites"] == S["cites"], "주차별+부록 인용 합 ≠ 총계"
assert H["n_claims"] + H["n_notclaim"] == H["n"], "손 라벨 분해 합 불일치"
assert H["agree"] + H["fp"] + H["fn"] == H["n_claims"], "일치+FP+FN ≠ 주장 수"
assert abs(H["accuracy"] + H["error_rate"] - 100) < 0.05, "정확도+오류율 ≠ 100"
assert abs(sum(c["share"] for c in RU["contrib"]) - 100) < 0.5, "기여 합 ≠ 100"
assert sum(c["w"] for c in RU["contrib"]) == 100, "루브릭 배점 합 ≠ 100"
print("  ✓ 정합성 검사 통과")

# ══════════════════════════════════════════════════════════════════
# 01. 교재 자가 감사
# ══════════════════════════════════════════════════════════════════
b = title(808, 30, f"우리 교재 {AU['cfg']['n_pages']}편을 우리 기준으로 채점하면",
          "수치가 든 산문 문장 중 같은 문장에 조건 표지가 있는 비율 (자동 집계 · 실측)")
BX, BY, BW = 96, 96, 620
rows = [r for r in ROWS if r["claims"]]
if AP["claims"]:
    rows = rows + [{"week": "부록", "claims": AP["claims"], "cond": AP["cond"],
                    "pct": round(100 * AP["cond"] / AP["claims"], 1)}]
mx = max(r["pct"] for r in rows)
AVGX = BX + BW * S["cond_pct"] / 50
b += (f'  <line x1="{AVGX:.1f}" y1="{BY-14}" x2="{AVGX:.1f}" y2="{BY+len(rows)*31-4}" '
      f'stroke="{RED}" stroke-width="2" stroke-dasharray="6 4"/>\n')
b += (f'  <text x="{AVGX:.1f}" y="{BY-22}" text-anchor="middle" font-size="12.5" '
      f'font-weight="700" fill="{RED}">전체 {S["cond_pct"]}%</text>\n')
for i, r in enumerate(rows):
    y = BY + i * 31
    col = AMB if r["pct"] == mx else TEAL
    b += (f'  <text x="{BX-12}" y="{y+17}" text-anchor="end" font-size="13" '
          f'fill="{INK}">{r["week"]}{"" if r["week"] == "부록" else "주"}</text>\n')
    b += (f'  <rect x="{BX}" y="{y}" width="{BW*r["pct"]/50:.1f}" height="23" rx="4" '
          f'fill="{col}"/>\n')
    lx = BX + BW * r["pct"] / 50 + 10
    lw = tw(f'{r["pct"]}%', 12.6) + 10
    b += (f'  <rect x="{lx-5:.1f}" y="{y+3}" width="{lw:.1f}" height="18" rx="4" '
          f'fill="#f8fafc"/>\n')
    b += (f'  <text x="{lx:.1f}" y="{y+17}" font-size="12.6" '
          f'font-weight="700" fill="{col}">{r["pct"]}%</text>\n')
    b += (f'  <text x="{BX+BW+150}" y="{y+17}" text-anchor="end" font-size="12" '
          f'fill="{MUT}">{r["cond"]} / {r["claims"]}문장</text>\n')
b += (f'  <line x1="{BX}" y1="{BY-8}" x2="{BX}" y2="{BY+len(rows)*31-4}" '
      f'stroke="{LINE}" stroke-width="1.4"/>\n')

CX = BX + BW + 196
for i, (hd, v, sub, col) in enumerate([
        ("수치가 든 산문 문장", f'{S["claims"]}개', f'그중 조건 표지 {S["cond"]}개', INK),
        ("전체 조건 명시율", f'{S["cond_pct"]}%', "셋 중 둘은 문장만 떼면 조건이 없다", AMB),
        ("인용 서지 완전성", f'{S["cites_pct"]}%', f'{S["cites_ok"]} / {S["cites"]}건 (게재처+연도)', TEAL),
        ("「재현 정보」 블록", f'{S["repro_weeks"]} / {S["n_weeks"]}주',
         f'실험 코드는 {S["code_weeks"]} / {S["n_weeks"]}주', MUT)]):
    y = BY - 8 + i * 84
    b += (f'  <rect x="{CX}" y="{y}" width="330" height="72" rx="10" fill="#ffffff" '
          f'stroke="{LINE}"/>\n')
    b += f'  <rect x="{CX}" y="{y}" width="6" height="72" rx="3" fill="{col}"/>\n'
    b += (f'  <text x="{CX+20}" y="{y+24}" font-size="12.5" fill="{MUT}">{esc(hd)}</text>\n')
    b += (f'  <text x="{CX+20}" y="{y+50}" font-size="21" font-weight="700" '
          f'fill="{col}">{esc(v)}</text>\n')
    b += (f'  <text x="{CX+150}" y="{y+50}" font-size="11.8" fill="{MUT}">{esc(sub)}</text>\n')

NY = BY + len(rows) * 31 + 16
b += note(96, NY, 1424, 96, "조건이 없는 것이 아니라, 조건이 문장 밖에 있다",
          ["조건은 절 머리의 「실험 설계」 표와 주차 개요의 「재현 정보」 블록에 있다. 문장은 그것을 전제하고 짧게 쓴다.",
           "그런데 인용은 문장 단위로 잘려 나간다 — 13주차 ① 조건 유형(42.9%)이 여기서 설명된다."])
fig("w14_p1_audit_01", 1616, NY + 128, b)

# ══════════════════════════════════════════════════════════════════
# 02. 자동 감사 vs 손 라벨
# ══════════════════════════════════════════════════════════════════
b = title(560, 30, f"그 감사 자체는 얼마나 틀리는가 — 손으로 라벨한 {H['n']}문장",
          f"13주차 서지 조회의 오류율 15.6% · 여기서는 {H['error_rate']}% (실측)")
GX, GY, CS, PER = 96, 96, 44, 7
cells = ([("NC", LINE)] * H["n_notclaim"] + [("A", GRN)] * H["agree"] +
         [("FP", RED)] * H["fp"] + [("FN", AMB)] * H["fn"])
assert len(cells) == H["n"]
for i, (_, col) in enumerate(cells):
    x = GX + (i % PER) * CS
    y = GY + (i // PER) * CS
    b += (f'  <rect x="{x}" y="{y}" width="{CS-8}" height="{CS-8}" rx="7" fill="{col}"/>\n')
b += legend(GX, GY + 4 * CS + 14,
            [(LINE, f"주장 아님 {H['n_notclaim']}"), (GRN, f"일치 {H['agree']}")])
b += legend(GX, GY + 4 * CS + 40,
            [(RED, f"거짓양성 {H['fp']}"), (AMB, f"거짓음성 {H['fn']}")])

RX = GX + PER * CS + 46
STEPS = [(f'표본 {H["n"]}문장', "자동 감사가 \"수치 주장\"으로 센 문장", MUT, None),
         (f'− 주장 아님 {H["n_notclaim"]}개 ({H["notclaim_pct"]}%)',
          "비유 · 정의 · 구조 서술 · 진행 안내", RED, "자동 감사는 분모부터 부풀린다"),
         (f'= 실제 수치 주장 {H["n_claims"]}개', "여기서부터가 판정 대상", INK, None),
         (f'정확도 {H["accuracy"]}% · 오류율 {H["error_rate"]}%',
          f'일치 {H["agree"]} · 거짓양성 {H["fp"]} · 거짓음성 {H["fn"]}', AMB, None),
         (f'사람이 센 조건 명시율 {H["hand_cond_pct"]}%',
          f'자동 추정 {S["cond_pct"]}% 보다 높다 — 분모가 줄었기 때문', TEAL,
          "그래도 절반이 안 된다")]
for i, (hd, sub, col, tail) in enumerate(STEPS):
    y = GY - 10 + i * 62
    b += (f'  <rect x="{RX}" y="{y}" width="700" height="52" rx="10" fill="#ffffff" '
          f'stroke="{LINE}"/>\n')
    b += f'  <rect x="{RX}" y="{y}" width="6" height="52" rx="3" fill="{col}"/>\n'
    b += (f'  <text x="{RX+20}" y="{y+23}" font-size="15" font-weight="700" '
          f'fill="{col}">{esc(hd)}</text>\n')
    b += (f'  <text x="{RX+20}" y="{y+42}" font-size="12" fill="{MUT}">{esc(sub)}</text>\n')
    if tail:
        b += (f'  <text x="{RX+686}" y="{y+31}" text-anchor="end" font-size="12.4" '
              f'font-weight="700" fill="{col}">{esc(tail)}</text>\n')

b += note(96, 400, 1304, 96, "자동화는 후보를 좁혀 줄 뿐 판정을 대신하지 않는다",
          ["13주차의 자동 서지 조회는 15.6%, 이 자가 감사는 16.7% 틀렸다. 도구가 달라도 비슷한 자리에서 멈춘다.",
           "그래서 selfcheck.py 의 출력은 「고칠 목록」이 아니라 「설명할 목록」이다 — 고치거나, 왜 해당 없는지 한 줄 적거나."],
          bg="#fef3e8", ln="#f0b27a", hc="#9a4b06")
fig("w14_p1_handcheck_02", 1496, 528, b)

# ══════════════════════════════════════════════════════════════════
# 03. 여덟 칸 격자
# ══════════════════════════════════════════════════════════════════
AXES = [("정확도", "원본 대비 얼마나 지켰나", "학습 정확도를 보고한다", TEAL),
        ("지연", "타깃 기기에서 얼마나 빨라졌나", "노트북에서 잰 값을 보고한다", BLUE),
        ("메모리", "피크가 얼마나 줄었나", "모델 크기와 혼동한다 (8주차)", PUR),
        ("모델 크기", "파일이 얼마나 줄었나", "압축 전 크기로 계산한다", CYAN)]
CONDS = [("하드웨어", "Raspberry Pi 4 · 4스레드 · 전원 연결 · 발열 안정 후"),
         ("데이터", "검증셋 2,000장 · 클래스 균형 · 전처리 포함 여부"),
         ("설정", "배치 1 · 입력 224² · 반복 30회 · 워밍업 5회 제외"),
         ("버전", "onnxruntime 1.18.0 · 커밋 a1b2c3d")]
b = title(560, 30, "발표가 끝났을 때 채워져 있어야 하는 여덟 칸",
          "네 축(무엇을) × 값·조건(어떻게) — 비어 있는 칸이 곧 질문이다")
TX, TY = 96, 88
b += (f'  <text x="{TX+16}" y="{TY+22}" font-size="13" font-weight="700" '
      f'fill="{MUT}">축</text>\n')
b += (f'  <text x="{TX+206}" y="{TY+22}" font-size="13" font-weight="700" '
      f'fill="{MUT}">무엇을 보고하나 (값)</text>\n')
b += (f'  <text x="{TX+606}" y="{TY+22}" font-size="13" font-weight="700" '
      f'fill="{MUT}">흔한 실패</text>\n')
for i, (nm, what, fail, col) in enumerate(AXES):
    y = TY + 36 + i * 54
    b += (f'  <rect x="{TX}" y="{y}" width="1008" height="46" rx="9" fill="#ffffff" '
          f'stroke="{col}" stroke-width="1.6"/>\n')
    b += f'  <rect x="{TX}" y="{y}" width="7" height="46" rx="3" fill="{col}"/>\n'
    b += (f'  <text x="{TX+22}" y="{y+29}" font-size="15" font-weight="700" '
          f'fill="{col}">{esc(nm)}</text>\n')
    b += (f'  <text x="{TX+206}" y="{y+29}" font-size="12.8" fill="#334155">{esc(what)}</text>\n')
    b += (f'  <text x="{TX+606}" y="{y+29}" font-size="12.4" fill="{RED}">{esc(fail)}</text>\n')
CY = TY + 36 + 4 * 54 + 18
b += (f'  <text x="{TX+16}" y="{CY+16}" font-size="13" font-weight="700" '
      f'fill="{MUT}">그 네 숫자에 붙는 네 조건 — 표 머리에 한 줄로</text>\n')
for i, (nm, ex) in enumerate(CONDS):
    y = CY + 30 + i * 44
    b += (f'  <rect x="{TX}" y="{y}" width="1008" height="36" rx="8" fill="#edf6f4" '
          f'stroke="#9fd6cc"/>\n')
    b += (f'  <text x="{TX+22}" y="{y+24}" font-size="13.5" font-weight="700" '
          f'fill="#0b4a48">{esc(nm)}</text>\n')
    b += (f'  <text x="{TX+206}" y="{y+24}" font-size="12.6" fill="#334155">{esc(ex)}</text>\n')
GY2 = CY + 30 + 4 * 44 + 16
b += (f'  <rect x="{TX}" y="{GY2}" width="1008" height="82" rx="11" fill="#fef3e8" '
      f'stroke="{AMB}" stroke-width="2"/>\n')
b += (f'  <text x="{TX+504}" y="{GY2+32}" text-anchor="middle" font-size="16" '
      f'font-weight="700" fill="#9a4b06">'
      f'그리고 그 넷을 잇는 한 장 — 프로파일링 (게이트)</text>\n')
b += (f'  <text x="{TX+504}" y="{GY2+60}" text-anchor="middle" font-size="13" '
      f'fill="#334155">'
      f'"어디가 병목이었고, 무엇을 바꿨고, 그래서 그 병목이 줄었는가" — 최적화 전후 표가 둘 다 있어야 한다</text>\n')
fig("w14_p1_prove_03", 1200, GY2 + 112, b)

# ══════════════════════════════════════════════════════════════════
# 04. 타깃 등급 A~D
# ══════════════════════════════════════════════════════════════════
TG = [("A", "임베디드 보드", "Jetson · Raspberry Pi · MCU", "보유 시",
       "전부. 전력·발열까지 보는 유일한 등급", TEAL),
      ("B", "안드로이드 폰", "부록 A · 팀에 한 대면 충분", "0원",
       "ARM CPU · GPU delegate · 연산자별 프로파일링", AMB),
      ("C", "무료 ARM 클라우드", "Ampere A1 등", "0원",
       "x86 과 다른 아키텍처의 지연·메모리", BLUE),
      ("D", "제약 프로파일 노트북", "스레드·해상도를 고정한 가상 타깃", "0원",
       "개선율만. 아키텍처 차이는 못 봄", PUR)]
b = title(560, 30, "타깃 기기 네 등급 — 배점 차이는 없다",
          "평가하는 것은 어떤 장비를 살 수 있었는가가 아니라 측정을 얼마나 정직하게 했는가")
for i, (g, nm, sub, cost, see, col) in enumerate(TG):
    y = 82 + i * 92
    b += (f'  <rect x="96" y="{y}" width="1008" height="80" rx="12" fill="#ffffff" '
          f'stroke="{col}" stroke-width="1.9"/>\n')
    b += f'  <circle cx="146" cy="{y+40}" r="26" fill="{col}"/>\n'
    b += (f'  <text x="146" y="{y+49}" text-anchor="middle" font-size="24" '
          f'font-weight="700" fill="#ffffff">{g}</text>\n')
    b += (f'  <text x="192" y="{y+34}" font-size="17" font-weight="700" '
          f'fill="{INK}">{esc(nm)}</text>\n')
    b += (f'  <text x="192" y="{y+58}" font-size="12.4" fill="{MUT}">{esc(sub)}</text>\n')
    cw = tw(cost, 14) + 34
    b += (f'  <rect x="{540-cw/2}" y="{y+24}" width="{cw}" height="32" rx="16" '
          f'fill="{col}" opacity="0.14"/>\n')
    b += (f'  <text x="540" y="{y+46}" text-anchor="middle" font-size="14" '
          f'font-weight="700" fill="{col}">{esc(cost)}</text>\n')
    b += (f'  <text x="626" y="{y+46}" font-size="12.8" fill="#334155">{esc(see)}</text>\n')
    if g == "B":
        b += (f'  <text x="1086" y="{y+22}" text-anchor="end" font-size="12" '
              f'font-weight="700" fill="{AMB}">← 기본값 권장</text>\n')
b += note(96, 462, 1008, 100, "A~D 사이에 배점 차이를 두지 않는 이유",
          ["D 를 고른 팀도 조건을 고정하고 개선율을 일관되게 보고했다면 만점을 받을 수 있다.",
           "반대로 A 를 고르고도 조건을 안 밝히면 감점된다. 발표 첫 슬라이드에 어느 등급을 왜 골랐는지 한 줄을 넣는다."])
fig("w14_p1_target_04", 1200, 592, b)

# ══════════════════════════════════════════════════════════════════
# 05. 루브릭 배점과 변동 기여
# ══════════════════════════════════════════════════════════════════
C = sorted(RU["contrib"], key=lambda x: -x["w"])
b = title(846, 30, "루브릭 100점 — 배점이 곧 채점자 오판의 지분이다",
          "한 항목을 한 수준 오판했을 때 총점이 움직이는 크기 (실측 · 몬테카를로)")
BX, BY, BW = 400, 92, 480
COLS = [TEAL, BLUE, PUR, CYAN, AMB, MUT]
for i, c in enumerate(C):
    y = BY + i * 56
    col = AMB if "프로파일링" in c["name"] else COLS[i % len(COLS)]
    b += (f'  <text x="{BX-16}" y="{y+20}" text-anchor="end" font-size="13.4" '
          f'font-weight="700" fill="{INK}">{esc(c["name"])}</text>\n')
    b += (f'  <text x="{BX-16}" y="{y+40}" text-anchor="end" font-size="12" '
          f'fill="{MUT}">{c["w"]}점</text>\n')
    b += (f'  <rect x="{BX}" y="{y+4}" width="{BW*c["w"]/20:.1f}" height="34" rx="6" '
          f'fill="{col}"/>\n')
    b += (f'  <text x="{BX+BW*c["w"]/20+12:.1f}" y="{y+27}" font-size="13.4" '
          f'font-weight="700" fill="{col}">{c["shift"]:.2f}점 이동</text>\n')
    b += (f'  <text x="{BX+BW+360}" y="{y+27}" text-anchor="end" font-size="12.6" '
          f'fill="{MUT}">총점 변동 기여 {c["share"]}%</text>\n')
    if "프로파일링" in c["name"]:
        b += (f'  <text x="{BX+BW+380}" y="{y+27}" font-size="12.4" font-weight="700" '
              f'fill="{AMB}">← 배점은 15점이지만 게이트다</text>\n')
b += (f'  <line x1="{BX}" y1="{BY-6}" x2="{BX}" y2="{BY+len(C)*56-14}" '
      f'stroke="{LINE}" stroke-width="1.4"/>\n')
NY = BY + len(C) * 56 + 4
b += note(96, NY, 1500, 100, "기여는 배점에 정확히 비례한다 — 당연해 보이지만 함의가 있다",
          ["20점짜리 항목 하나를 잘못 읽으면 총점이 4점 움직인다. 상위 두 팀의 참값 격차가 4점이면 항목 하나의 오판으로 순위가 뒤집힌다.",
           "배점을 크게 준 항목은 그만큼 채점자 오판의 지분도 크게 준 것이다."])
fig("w14_p2_rubric_05", 1692, NY + 130, b)

# ══════════════════════════════════════════════════════════════════
# 06. 채점자 수와 총점 오차
# ══════════════════════════════════════════════════════════════════
G = RU["graders"]
b = title(560, 30, "채점자를 한 명 더 두면 총점 오차가 얼마나 줄어드는가",
          f"6항목 루브릭 · 항목당 오판 확률 q={RU['cfg']['q']} · 7팀 몬테카를로 (실측)")
PX, PY, PW, PH = 150, 110, 800, 260
mxv = max(G[str(k)]["max"] if str(k) in G else G[k]["max"] for k in G)


def gv(k):
    return G[k] if k in G else G[str(k)]


for t in range(0, 25, 5):
    yy = PY + PH - PH * t / 24
    b += (f'  <line x1="{PX}" y1="{yy:.1f}" x2="{PX+PW}" y2="{yy:.1f}" '
          f'stroke="{LINE}" stroke-dasharray="4 4"/>\n')
    b += (f'  <text x="{PX-12}" y="{yy+5:.1f}" text-anchor="end" font-size="12" '
          f'fill="{MUT}">{t}</text>\n')
b += (f'  <text x="{PX-56}" y="{PY+PH/2}" text-anchor="middle" font-size="12.5" '
      f'fill="{MUT}" transform="rotate(-90 {PX-56} {PY+PH/2})">총점 오차 (점)</text>\n')
SER = [("평균 절대 오차", "mae", TEAL), ("95분위", "p95", AMB), ("최대", "max", RED)]
for gi, k in enumerate(["1", "2", "3"]):
    g = gv(k)
    cx = PX + PW * (gi + 0.5) / 3
    for si, (lab, key, col) in enumerate(SER):
        w = 62
        x = cx - 1.5 * w - 8 + si * (w + 8)
        h = PH * g[key] / 24
        b += (f'  <rect x="{x:.1f}" y="{PY+PH-h:.1f}" width="{w}" height="{h:.1f}" '
              f'rx="5" fill="{col}"/>\n')
        b += (f'  <text x="{x+w/2:.1f}" y="{PY+PH-h-8:.1f}" text-anchor="middle" '
              f'font-size="12.5" font-weight="700" fill="{col}">{g[key]:.2f}</text>\n')
    b += (f'  <text x="{cx:.1f}" y="{PY+PH+24}" text-anchor="middle" font-size="14" '
          f'font-weight="700" fill="{INK}">채점자 {k}명</text>\n')
b += (f'  <line x1="{PX}" y1="{PY+PH}" x2="{PX+PW}" y2="{PY+PH}" stroke="{LINE}" '
      f'stroke-width="1.4"/>\n')
b += legend(PX, PY + PH + 52, [(c, l) for l, _, c in SER])

b += card(1000, 108, 380, 118, "가장 큰 이득은 두 번째 채점자에 있다", TEAL, "#ffffff",
          [f'95분위 {gv("1")["p95"]:.2f} → {gv("2")["p95"]:.2f}점 '
           f'({gv("1")["p95"]/gv("2")["p95"]:.2f}배 감소)',
           f'3명으로 늘려도 {gv("2")["p95"]:.2f} → {gv("3")["p95"]:.2f}점',
           "이론값 1/√n 보다 완만하다 — 수준이 0.4~1.0 으로 절단되기 때문"])
b += card(1000, 244, 380, 126, "채점자 1명이면", RED, "#fee2e2",
          ["스무 번 중 한 번은 참값에서",
           f'{gv("1")["p95"]:.2f}점 이상 벗어난다.',
           "100점 만점에서 10.5점은",
           "등급 하나 반이다."])
b += note(96, 452, 1288, 96, "그래서 이 과목은 모든 팀을 최소 2명이 채점한다",
          ["두 채점자는 상의하지 않고 각자 기입한다 — 상의하면 독립 잡음이 아니게 되어 2명을 쓴 이득이 사라진다.",
           "합의는 등급 경계 ±3점에 걸린 팀에 대해서만 사후에 한다."])
fig("w14_p2_noise_06", 1480, 580, b)

# ══════════════════════════════════════════════════════════════════
# 07. 참값 격차와 순위 뒤바뀜
# ══════════════════════════════════════════════════════════════════
FL = [f for f in RU["flip"] if f["gap_items"] >= 1][:5]
b = title(560, 30, "몇 점 차이부터가 진짜 차이인가",
          "두 팀의 참값 격차별 순위 뒤바뀜 확률 · 채점자 2명 (실측)")
PX, PY, PW, PH = 140, 106, 860, 250
mxp = 25.0
for t in range(0, 26, 5):
    yy = PY + PH - PH * t / mxp
    b += (f'  <line x1="{PX}" y1="{yy:.1f}" x2="{PX+PW}" y2="{yy:.1f}" '
          f'stroke="{LINE}" stroke-dasharray="4 4"/>\n')
    b += (f'  <text x="{PX-12}" y="{yy+5:.1f}" text-anchor="end" font-size="12" '
          f'fill="{MUT}">{t}%</text>\n')
y5 = PY + PH - PH * 5 / mxp
b += (f'  <line x1="{PX}" y1="{y5:.1f}" x2="{PX+PW}" y2="{y5:.1f}" stroke="{RED}" '
      f'stroke-width="2.2" stroke-dasharray="8 4"/>\n')
b += (f'  <rect x="{PX+PW-116}" y="{y5-22}" width="112" height="19" rx="5" '
      f'fill="#f8fafc"/>\n')
b += (f'  <text x="{PX+PW-8}" y="{y5-8:.1f}" text-anchor="end" font-size="12.5" '
      f'font-weight="700" fill="{RED}">허용선 5%</text>\n')
for i, f in enumerate(FL):
    cx = PX + PW * (i + 0.5) / len(FL)
    h = PH * min(f["flip_pct"], mxp) / mxp
    hh = max(h, 6.0)
    col = RED if f["flip_pct"] >= 5 else GRN
    op = "" if h >= 6 else ' opacity="0.45"'
    b += (f'  <rect x="{cx-46:.1f}" y="{PY+PH-hh:.1f}" width="92" height="{hh:.1f}" '
          f'rx="5" fill="{col}"{op}/>\n')
    b += (f'  <text x="{cx:.1f}" y="{PY+PH-hh-9:.1f}" text-anchor="middle" '
          f'font-size="14" font-weight="700" fill="{col}">{f["flip_pct"]}%</text>\n')
    b += (f'  <text x="{cx:.1f}" y="{PY+PH+24}" text-anchor="middle" font-size="13.5" '
          f'font-weight="700" fill="{INK}">{f["gap_mean"]:.1f}점</text>\n')
    b += (f'  <text x="{cx:.1f}" y="{PY+PH+44}" text-anchor="middle" font-size="11.8" '
          f'fill="{MUT}">{f["gap_items"]}개 항목 우세</text>\n')
b += (f'  <line x1="{PX}" y1="{PY+PH}" x2="{PX+PW}" y2="{PY+PH}" stroke="{LINE}" '
      f'stroke-width="1.4"/>\n')
b += (f'  <text x="{PX+PW/2}" y="{PY+PH+70}" text-anchor="middle" font-size="12.5" '
      f'fill="{MUT}">두 팀의 참값 격차</text>\n')
b += (f'  <text x="{PX}" y="{PY+PH+92}" font-size="11.8" fill="{MUT}">'
      f'※ 5% 미만 막대는 최소 높이로 흐리게 그렸다 — 길이가 아니라 숫자로 읽을 것</text>\n')

safe = RU["safe_gap"]
b += card(1040, 106, 400, 132, "이 루브릭의 실제 해상도", AMB, "#fef3e8",
          [f'뒤바뀜을 5% 아래로 내리려면',
           f'참값 격차가 {safe["gap_mean"]:.1f}점 이상이어야 한다.', "",
           "100점 만점이지만 읽을 수 있는",
           "눈금은 약 10점이다."], lh=20)
b += card(1040, 254, 400, 102, "5주차가 여기서 돌아온다", TEAL, "#ffffff",
          ["자의 눈금보다 가는 것을",
           "읽었다고 말하면 안 된다.",
           "1~2점 차이로 순위를 매기지 않는다."], lh=20)
b += note(96, 486, 1344, 96, "그래서 총점이 아니라 등급으로 확정한다",
          ["등급 경계 ±3점 안에 걸린 팀만 두 채점자가 함께 다시 본다. 그 밖의 팀은 총점 소수점을 비교하지 않는다.",
           "그리고 최우수 한 팀을 뽑지 않는다 — 1등 적중은 77.6%로, 넷 중 하나는 바뀐다."])
fig("w14_p2_flip_07", 1536, 614, b)

# ══════════════════════════════════════════════════════════════════
# 08. 규모량 vs 구조량
# ══════════════════════════════════════════════════════════════════
ST = RU["structure"]
b = title(560, 30, "총점은 흔들려도 「누가 잘했는가」 는 덜 흔들린다",
          "13주차의 구조량·규모량 구분이 채점에도 그대로 나타난다 (실측 · 채점자 2명)")
LEFT = [("총점 상대 오차", f'{ST["score_err_pct"]}%', "몇 점인가", AMB)]
RIGHT = [("순위 상관 (스피어만)", f'{ST["spearman_mean"]:.3f}', "누가 위인가", TEAL),
         ("상위 3팀 집합 일치", f'{ST["top3_overlap"]}%', "어느 팀들이 위인가", TEAL),
         ("1등 적중", f'{ST["top1_hit"]}%', "누가 1등인가", RED)]
b += (f'  <rect x="96" y="88" width="560" height="300" rx="13" fill="#fef3e8" '
      f'stroke="{AMB}" stroke-width="2"/>\n')
b += (f'  <text x="376" y="124" text-anchor="middle" font-size="19" font-weight="700" '
      f'fill="#9a4b06">규모량 — 재는 것</text>\n')
b += (f'  <text x="376" y="150" text-anchor="middle" font-size="13" '
      f'fill="{MUT}">채점자·시점에 따라 흔들린다</text>\n')
for i, (nm, v, sub, col) in enumerate(LEFT):
    b += (f'  <text x="376" y="242" text-anchor="middle" font-size="52" '
          f'font-weight="700" fill="{AMB}">{esc(v)}</text>\n')
    b += (f'  <text x="376" y="278" text-anchor="middle" font-size="15" '
          f'fill="{INK}">{esc(nm)}</text>\n')
b += (f'  <text x="376" y="330" text-anchor="middle" font-size="13" '
      f'fill="{MUT}">채점자 1명이면 95분위 오차가 10.50점 —</text>\n')
b += (f'  <text x="376" y="354" text-anchor="middle" font-size="13" '
      f'fill="{MUT}">등급 하나 반이 우연으로 움직인다</text>\n')

b += (f'  <rect x="688" y="88" width="616" height="300" rx="13" fill="#edf6f4" '
      f'stroke="{TEAL}" stroke-width="2"/>\n')
b += (f'  <text x="996" y="124" text-anchor="middle" font-size="19" font-weight="700" '
      f'fill="#0b4a48">구조량 — 세는 것</text>\n')
b += (f'  <text x="996" y="150" text-anchor="middle" font-size="13" '
      f'fill="{MUT}">경계 근처의 한두 팀만 자리를 바꾼다</text>\n')
for i, (nm, v, sub, col) in enumerate(RIGHT):
    y = 178 + i * 68
    b += (f'  <rect x="712" y="{y}" width="568" height="56" rx="9" fill="#ffffff" '
          f'stroke="{LINE}"/>\n')
    b += (f'  <text x="736" y="{y+24}" font-size="14" font-weight="700" '
          f'fill="{INK}">{esc(nm)}</text>\n')
    b += (f'  <text x="736" y="{y+45}" font-size="11.8" fill="{MUT}">{esc(sub)}</text>\n')
    b += (f'  <text x="1256" y="{y+38}" text-anchor="end" font-size="26" '
          f'font-weight="700" fill="{col}">{esc(v)}</text>\n')
b += note(96, 408, 1208, 100, "그래서 「최우수 한 팀」 을 뽑지 않는다",
          ["상위 3팀 집합은 86.6% 유지되지만 1등 적중은 77.6%다 — 넷 중 하나는 1등이 바뀐다.",
           "상위 그룹으로 표창하는 것이 이 도구의 해상도 안에서 할 수 있는 말이다."],
          bg="#fee2e2", ln="#f0a3a3", hc="#991b1b")
fig("w14_p2_structure_08", 1400, 538, b)

# ══════════════════════════════════════════════════════════════════
# 09. 발표일 운영표
# ══════════════════════════════════════════════════════════════════
b = title(560, 30, "발표일 3시간 운영 — 팀당 20분",
          "오리엔테이션 10분 → 팀 발표 20분 × 7팀 = 140분 → 종합 강평 15분")
BX, BY, BW2 = 96, 96, 1008
SEG = [("오리엔테이션", 10, MUT, "1교시 요약 · 여덟 칸 격자 배포 · 채점 규칙 공지"),
       ("팀 발표 (7팀)", 140, TEAL, "팀당 20분 — 아래 분해"),
       ("종합 강평", 15, AMB, "팀마다 「가장 잘한 측정 하나」와 「다음에 개선할 조건 하나」")]
tot = sum(s[1] for s in SEG)
x = BX
for nm, m, col, sub in SEG:
    w = BW2 * m / tot
    b += (f'  <rect x="{x:.1f}" y="{BY}" width="{w-4:.1f}" height="56" rx="8" fill="{col}"/>\n')
    if tw(nm, 14) + 16 <= w - 4:
        b += (f'  <text x="{x+w/2-2:.1f}" y="{BY+27}" text-anchor="middle" font-size="14" '
              f'font-weight="700" fill="#ffffff">{esc(nm)}</text>\n')
        b += (f'  <text x="{x+w/2-2:.1f}" y="{BY+46}" text-anchor="middle" font-size="12" '
              f'fill="#eaf6f4">{m}분</text>\n')
    else:
        b += (f'  <text x="{x+w/2-2:.1f}" y="{BY+35}" text-anchor="middle" font-size="13" '
              f'font-weight="700" fill="#ffffff">{m}분</text>\n')
        b += (f'  <text x="{x+w/2-2:.1f}" y="{BY-10}" text-anchor="middle" font-size="12.5" '
              f'font-weight="700" fill="{col}">{esc(nm)}</text>\n')
    x += w
b += (f'  <text x="{BX}" y="{BY+78}" font-size="12" fill="{MUT}">0분</text>\n')
b += (f'  <text x="{BX+BW2}" y="{BY+78}" text-anchor="end" font-size="12" '
      f'fill="{MUT}">165분 (여유 15분)</text>\n')

TY2 = BY + 104
b += (f'  <text x="{BX}" y="{TY2}" font-size="15" font-weight="700" '
      f'fill="{INK}">팀당 20분의 분해</text>\n')
SUB = [("발표", 12, TEAL, "① 문제·타깃 정의(등급 A~D와 이유)  ② 적용 기법과 이유  ③ 네 축 × 네 조건 + 프로파일링"),
       ("질의응답", 6, BLUE, "청중 2문항 + 교수 1문항 — ①~③은 모든 팀에 반드시 묻는다"),
       ("채점·전환", 2, MUT, "채점자 2명이 상의하지 않고 각자 기입")]
x = BX
for nm, m, col, sub in SUB:
    w = BW2 * m / 20
    b += (f'  <rect x="{x:.1f}" y="{TY2+16}" width="{w-4:.1f}" height="50" rx="8" '
          f'fill="{col}" opacity="0.16" stroke="{col}" stroke-width="1.6"/>\n')
    b += (f'  <text x="{x+w/2-2:.1f}" y="{TY2+40}" text-anchor="middle" font-size="14" '
          f'font-weight="700" fill="{col}">{esc(nm)}</text>\n')
    b += (f'  <text x="{x+w/2-2:.1f}" y="{TY2+58}" text-anchor="middle" font-size="12" '
          f'fill="{MUT}">{m}분</text>\n')
    x += w
for i, (nm, m, col, sub) in enumerate(SUB):
    y = TY2 + 86 + i * 34
    b += f'  <circle cx="{BX+12}" cy="{y+8}" r="5" fill="{col}"/>\n'
    b += (f'  <text x="{BX+30}" y="{y+13}" font-size="12.6" fill="#334155">{esc(sub)}</text>\n')

QY = TY2 + 86 + 3 * 34 + 12
b += (f'  <text x="{BX}" y="{QY+16}" font-size="15" font-weight="700" '
      f'fill="{INK}">질의응답에서 반드시 나와야 하는 세 질문 (13주차 여섯 질문의 발표판)</text>\n')
QQ = [("①", "무엇과 비교한 개선율입니까?", "기준선 — 원본 FP32 인가, 다른 최적화본인가", BLUE),
      ("②", "어느 기기에서, 어떤 설정으로 쟀습니까?", "네 조건", TEAL),
      ("③", "그 숫자가 슬라이드 어느 표에 있습니까?", "말로만 하는 숫자를 거른다", PINK)]
for i, (n, q, why, col) in enumerate(QQ):
    y = QY + 30 + i * 48
    b += (f'  <rect x="{BX}" y="{y}" width="1008" height="40" rx="9" fill="#ffffff" '
          f'stroke="{col}" stroke-width="1.6"/>\n')
    b += f'  <circle cx="{BX+28}" cy="{y+20}" r="15" fill="{col}"/>\n'
    b += (f'  <text x="{BX+28}" y="{y+26}" text-anchor="middle" font-size="15" '
          f'font-weight="700" fill="#ffffff">{n}</text>\n')
    b += (f'  <text x="{BX+58}" y="{y+26}" font-size="14" font-weight="700" '
          f'fill="{INK}">{esc(q)}</text>\n')
    b += (f'  <text x="{BX+996}" y="{y+26}" text-anchor="end" font-size="12.2" '
          f'fill="{MUT}">{esc(why)}</text>\n')
EY = QY + 30 + 3 * 48 + 10
b += note(BX, EY, 1008, 74, "청중이 ①~③을 못 물으면 교수가 대신 묻는다",
          ["이 세 질문이 안 나오면 그 팀의 발표는 검증되지 않은 채로 끝난다. 처음 두세 팀에서 시범을 보이면 나머지는 학생들이 알아서 묻는다."])
fig("w14_p3_timeline_09", 1200, EY + 104, b)

# ══════════════════════════════════════════════════════════════════
# 10. 제출 체크리스트와 결과 요약표
# ══════════════════════════════════════════════════════════════════
b = title(600, 30, "제출물 다섯 가지와 결과 요약표 한 장",
          "발표 자료의 중심은 이 표 한 장이다 — 표 머리의 조건 한 줄이 없으면 어떤 숫자도 의미가 없다")
SUBM = [("1", "발표 슬라이드", "첫 장에 타깃 등급과 이유 · 네 축 × 네 조건 · 프로파일링", TEAL),
        ("2", "코드·설정 + README", "코드를 안 짠 사람이 빈 폴더에서 돌려 봤는가", BLUE),
        ("3", "프로파일링 원자료", "최적화 전후 둘 다. 연산자별 또는 단계별", AMB),
        ("4", "결과 요약표", "아래 형식. 표 머리에 조건 한 줄", PUR),
        ("5", "selfcheck.py 출력", "걸린 항목마다 고쳤거나, 왜 해당 없는지 한 줄", CYAN)]
for i, (n, nm, why, col) in enumerate(SUBM):
    y = 82 + i * 50
    b += (f'  <rect x="96" y="{y}" width="1160" height="42" rx="9" fill="#ffffff" '
          f'stroke="{col}" stroke-width="1.6"/>\n')
    b += f'  <circle cx="128" cy="{y+21}" r="15" fill="{col}"/>\n'
    b += (f'  <text x="128" y="{y+27}" text-anchor="middle" font-size="15" '
          f'font-weight="700" fill="#ffffff">{n}</text>\n')
    b += (f'  <text x="158" y="{y+27}" font-size="14.5" font-weight="700" '
          f'fill="{INK}">{esc(nm)}</text>\n')
    b += (f'  <text x="1244" y="{y+27}" text-anchor="end" font-size="12.4" '
          f'fill="{MUT}">{esc(why)}</text>\n')

TY3 = 82 + 5 * 50 + 18
b += (f'  <rect x="96" y="{TY3}" width="1160" height="42" rx="8" fill="#edf6f4" '
      f'stroke="#9fd6cc"/>\n')
b += (f'  <text x="676" y="{TY3+27}" text-anchor="middle" font-size="12.6" '
      f'font-weight="700" fill="#0b4a48">'
      f'측정 조건 — Raspberry Pi 4B(4 GB) · 4스레드 · 전원 연결 · 발열 안정 후 · '
      f'배치 1 · 입력 224² · 30회 반복(워밍업 5회 제외) · onnxruntime 1.18.0 · 커밋 a1b2c3d</text>\n')
TBL = [("축", "원본", "최적화", "변화", None),
       ("정확도 (검증셋 2,000장, top-1)", "91.4%", "90.8%", "−0.6%p", TEAL),
       ("지연 (중앙값 / p95)", "412 / 468 ms", "129 / 151 ms", "3.19배 / 3.10배", BLUE),
       ("메모리 (피크 RSS)", "148 MB", "61 MB", "2.43배", PUR),
       ("모델 크기 (파일)", "14.2 MB", "3.8 MB", "3.74배", CYAN)]
for i, (a, c1, c2, c3, col) in enumerate(TBL):
    y = TY3 + 48 + i * 40
    hd = col is None
    b += (f'  <rect x="96" y="{y}" width="1160" height="36" rx="7" '
          f'fill="{"#0e4a44" if hd else ("#ffffff" if i%2 else "#f1f5f9")}" '
          f'stroke="{LINE}"/>\n')
    tc = "#ffffff" if hd else INK
    b += (f'  <text x="120" y="{y+24}" font-size="12.8" '
          f'font-weight="{700 if hd else 400}" fill="{tc}">{esc(a)}</text>\n')
    for j, v in enumerate([c1, c2, c3]):
        xx = 660 + j * 200
        vc = "#ffffff" if hd else (col if j == 2 else MUT)
        b += (f'  <text x="{xx}" y="{y+24}" text-anchor="end" font-size="12.8" '
              f'font-weight="{700 if (hd or j==2) else 400}" fill="{vc}">{esc(v)}</text>\n')

RY = TY3 + 48 + 5 * 40 + 14
RULES = ["표 머리에 조건 한 줄", "평균이 아니라 중앙값과 p95",
         "메모리와 모델 크기를 분리", "정확도는 검증셋에서"]
for i, r in enumerate(RULES):
    x = 96 + i * 292
    b += (f'  <rect x="{x}" y="{RY}" width="276" height="42" rx="9" fill="#fef3e8" '
          f'stroke="{AMB}" stroke-width="1.5"/>\n')
    b += (f'  <text x="{x+138}" y="{RY+27}" text-anchor="middle" font-size="12.8" '
          f'font-weight="700" fill="#9a4b06">{esc(r)}</text>\n')
b += note(96, RY + 58, 1160, 74, "\"이상 없음\" 도 결과다",
          ["기법을 적용했는데 안 빨라졌다면 그 사실과 이유를 보고한다. 안 된 이유를 설명한 팀이, 된 척한 팀보다 높은 점수를 받는다."])
fig("w14_p3_checklist_10", 1352, RY + 164, b)

# ══════════════════════════════════════════════════════════════════
for k, v in F.items():
    (OUT / f"{k}.svg").write_text(v, encoding="utf-8")
    print(f"  {k}.svg  ({len(v)//1024} KB)")
print(f"\n→ {len(F)} figures in {OUT}")
