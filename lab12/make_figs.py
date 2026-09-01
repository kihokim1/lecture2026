# -*- coding: utf-8 -*-
"""12주차 그림 — 개념 도해 + fed/patho/leak/dp.json 기반 실측 차트."""
import json, math, pathlib

D = pathlib.Path("/root/lab12")
FD = json.load(open(D / "fed.json"))
PA = json.load(open(D / "patho.json"))
LK = json.load(open(D / "leak.json"))
DP = json.load(open(D / "dp.json"))
OUT = pathlib.Path("/root/ondevice-ai/img/week12"); OUT.mkdir(parents=True, exist_ok=True)

HEAD = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        'font-family="\'Segoe UI\',Arial,sans-serif">\n'
        '  <rect width="{w}" height="{h}" fill="#f8fafc"/>\n')
F = {}
TEAL, AMB, RED, BLUE = "#028090", "#e4711b", "#dc2626", "#2563eb"
GRN, PUR = "#16a34a", "#7c3aed"
INK, MUT, LINE = "#0f172a", "#64748b", "#cbd5e1"


def fig(name, w, h, body):
    F[name] = HEAD.format(w=w, h=h) + body + "\n</svg>\n"


def tw(s, size):
    return sum(size * (1.0 if ord(c) > 0x2000 else 0.56) for c in s)


def title(x, y, t, sub=None):
    s = (f'  <text x="{x}" y="{y}" text-anchor="middle" font-size="19" '
         f'font-weight="700" fill="{INK}">{t}</text>\n')
    if sub:
        s += (f'  <text x="{x}" y="{y+22}" text-anchor="middle" font-size="13.5" '
              f'fill="{MUT}">{sub}</text>\n')
    return s


def note(x, y, w, h, head, body, bg="#edf6f4", ln="#9fd6cc", hc="#0b4a48"):
    s = f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{bg}" stroke="{ln}"/>\n'
    s += (f'  <text x="{x+w/2}" y="{y+26}" text-anchor="middle" font-size="14.5" '
          f'font-weight="700" fill="{hc}">{head}</text>\n')
    for i, l in enumerate(body):
        s += (f'  <text x="{x+w/2}" y="{y+50+i*20}" text-anchor="middle" '
              f'font-size="12.5" fill="#334155">{l}</text>\n')
    return s


def legend(x, y, items, size=12.5):
    s, cx = "", x
    for c, lab in items:
        s += f'  <rect x="{cx}" y="{y-9}" width="13" height="13" rx="3" fill="{c}"/>\n'
        s += f'  <text x="{cx+19}" y="{y+2}" font-size="{size}" fill="#334155">{lab}</text>\n'
        cx += 19 + tw(lab, size) + 22
    return s


def SUP(e, size=12.5):
    """유니코드 위첨자(ᵀ ⁻⁵ ²⁰ …)는 Noto CJK 에 글리프가 없다. tspan 으로 올린다."""
    return (f'<tspan font-size="{size*0.7:.1f}" dy="-{size*0.34:.1f}">{e}</tspan>'
            f'<tspan font-size="{size:.1f}" dy="{size*0.34:.1f}"> </tspan>')


MK = ('<marker id="{i}" markerWidth="9" markerHeight="9" refX="7" refY="3.5" '
      'orient="auto" markerUnits="userSpaceOnUse">'
      '<path d="M0,0 L7,3.5 L0,7 Z" fill="{c}"/></marker>')


def digit(x, y, arr, px=5.4, cap=None, capcol=None, frame=LINE):
    """28x28 화소 배열을 SVG 로 그린다. 배경(0)은 그리지 않아 파일이 작아진다."""
    s = (f'  <rect x="{x-1}" y="{y-1}" width="{28*px+2}" height="{28*px+2}" '
         f'fill="#000000" stroke="{frame}" stroke-width="1.4"/>\n')
    for r in range(28):
        row = arr[r]
        c = 0
        while c < 28:
            v = row[c]
            c2 = c
            while c2 + 1 < 28 and row[c2 + 1] == v:
                c2 += 1
            if v > 4:
                s += (f'  <rect x="{x+c*px:.1f}" y="{y+r*px:.1f}" '
                      f'width="{(c2-c+1)*px:.1f}" height="{px:.1f}" '
                      f'fill="rgb({v},{v},{v})"/>\n')
            c = c2 + 1
    if cap:
        s += (f'  <text x="{x+14*px}" y="{y+28*px+18}" text-anchor="middle" '
              f'font-size="12.5" font-weight="700" fill="{capcol or MUT}">{cap}</text>\n')
    return s


# ═════════ 01. 왜 연합 학습인가 ═════════
b = title(430, 30, "데이터를 모으지 않고 배운다 — 그리고 무엇이 남는가",
          "11주차의 경계 비용이 여기서 네트워크 규모로 커진다")
for x, hd, c, bg, items in [
    (44, "중앙집중 학습", RED, "#fee2e2",
     ["원본 데이터를 서버로 보낸다", "정확도 최고 (오늘 실측 기준선)",
      "규제·동의·유출 위험이 전부 서버에", "→ 의료·금융에서 애초에 불가능한 경우"]),
    (452, "연합 학습", TEAL, "#e6f4f2",
     ["데이터는 기기에 남는다", "모델 파라미터/기울기만 오간다",
      "비-IID·통신량이라는 새 비용", "→ 그런데 기울기는 안전한가? (2교시)"])]:
    b += f'  <rect x="{x}" y="76" width="384" height="168" rx="13" fill="{bg}" stroke="{c}" stroke-width="2.2"/>\n'
    b += (f'  <text x="{x+192}" y="106" text-anchor="middle" font-size="19" '
          f'font-weight="700" fill="{c}">{hd}</text>\n')
    for i, t in enumerate(items):
        b += (f'  <text x="{x+192}" y="{136+i*26}" text-anchor="middle" '
              f'font-size="13" fill="#334155">{t}</text>\n')
b += (f'  <text x="430" y="172" text-anchor="middle" font-size="26" '
      f'font-weight="700" fill="{MUT}">→</text>\n')
b += note(44, 264, 792, 106, "이번 주에 새로 생기는 비용 세 가지",
          ["① 통계적 비용 — 데이터가 기기마다 쏠린다(비-IID)",
           "② 통신 비용 — 라운드마다 모델을 내려받고 올린다. 11주차의 경계 비용이 네트워크로 커진 것",
           "③ 프라이버시 비용 — \"기울기만 보낸다\"가 안전을 뜻하지 않는다는 것을 2교시에 실측한다"])
fig("w12_p1_why_01", 880, 394, b)

# ═════════ 02. FedAvg 한 라운드 ═════════
b = title(430, 30, "FedAvg — 한 라운드에 무슨 일이 일어나는가",
          "McMahan 외, AISTATS 2017 · 이 용어가 처음 등장한 논문")
SX, SY = 216, 84
b += f'  <rect x="{SX}" y="{SY}" width="150" height="52" rx="10" fill="#0e4a44"/>\n'
b += (f'  <text x="{SX+75}" y="{SY+32}" text-anchor="middle" font-size="16" '
      f'font-weight="700" fill="#ffffff">서버</text>\n')
CY = 288
for i in range(4):
    cx = 60 + i * 128
    b += f'  <rect x="{cx}" y="{CY}" width="112" height="52" rx="10" fill="#e6f4f2" stroke="{TEAL}" stroke-width="2"/>\n'
    b += (f'  <text x="{cx+56}" y="{CY+22}" text-anchor="middle" font-size="13.5" '
          f'font-weight="700" fill="{TEAL}">기기 {i+1}</text>\n')
    b += (f'  <text x="{cx+56}" y="{CY+40}" text-anchor="middle" font-size="10.5" '
          f'fill="{MUT}">로컬 데이터 (안 나감)</text>\n')
    b += (f'  <path d="M{SX+40} {SY+54} L{cx+38} {CY-6}" stroke="{BLUE}" '
          f'stroke-width="1.8" marker-end="url(#d)" stroke-dasharray="5 4"/>\n')
    b += (f'  <path d="M{cx+78} {CY-6} L{SX+104} {SY+54}" stroke="{AMB}" '
          f'stroke-width="2" marker-end="url(#a)"/>\n')
b += f'  <rect x="596" y="76" width="256" height="188" rx="12" fill="#ffffff" stroke="{LINE}"/>\n'
STEPS = [("①", "서버가 현재 모델을 내려보낸다", BLUE),
         ("②", "각 기기가 자기 데이터로", TEAL),
         ("", "  로컬 학습 (E 에폭)", TEAL),
         ("③", "모델 변화량 Δ 만 올려 보낸다", AMB),
         ("④", "서버가 표본 수로 가중 평균", "#0e4a44")]
for i, (n, t, c) in enumerate(STEPS):
    yy = 108 + i * 30
    if n:
        b += f'  <text x="614" y="{yy}" font-size="13.5" font-weight="700" fill="{c}">{n}</text>\n'
    b += f'  <text x="638" y="{yy}" font-size="12.5" fill="#334155">{t}</text>\n'
b += (f'  <text x="724" y="286" text-anchor="middle" font-size="12.5" font-weight="700" '
      f'fill="{INK}">원본 데이터는 기기를 떠나지 않는다</text>\n')
b += (f'  <text x="724" y="308" text-anchor="middle" font-size="12.5" font-weight="700" '
      f'fill="{RED}">← 이 문장이 2교시의 표적이다</text>\n')
b += note(44, 366, 808, 100, "라운드마다 오가는 바이트",
          [f"모델 {FD['cfg']['params']:,} 파라미터 × 4 바이트 = {FD['cfg']['bytes']/1024:.1f} KB",
           "라운드당 = 내려받기 + 올리기 = 모델 크기 × 2 × 참여 기기 수",
           "이것이 곧 통신 비용이고, 1.3 에서 실측한다"])
b += '  <defs>' + MK.format(i="d", c=BLUE) + MK.format(i="a", c=AMB) + '</defs>\n'
fig("w12_p1_fedavg_02", 896, 490, b)

# ═════════ 03. 비-IID ═════════
b = title(500, 30, "데이터가 쏠리면 무너진다 — 그런데 얼마나?",
          "MNIST · SmallCNN · 클라이언트 10대 · 30라운드 (실측)")
rows = [("중앙집중 (데이터를 모았을 때)", FD["central"][-1]["acc"], "#94a3b8", "10")]
for a in FD["alpha"]:
    lab = f"FedAvg · 디리클레 α={a['alpha']:g}"
    cl = a["classes_per_client"]
    rows.append((lab, a["final"], TEAL, f"{min(cl)}~{max(cl)}"))
for r in PA["rows"]:
    k = r["shards"]
    cl = r["classes_per_client"]
    rows.append((f"FedAvg · 병리적 분할 (조각 {k}개)", r["final"], RED, f"{min(cl)}~{max(cl)}"))
XL, XR, PT, RH = 300, 740, 78, 44
lo = min(r[1] for r in rows) - 0.03
for i, (lab, acc, c, cls) in enumerate(rows):
    y = PT + i * RH
    b += (f'  <text x="{XL-12}" y="{y+22}" text-anchor="end" font-size="13" '
          f'font-weight="700" fill="#334155">{lab}</text>\n')
    w = max((acc - lo) / (1.0 - lo) * (XR - XL), 3)
    b += f'  <rect x="{XL}" y="{y+4}" width="{w:.1f}" height="26" rx="3" fill="{c}"/>\n'
    b += (f'  <text x="{XL+w+9:.1f}" y="{y+23}" font-size="13.5" font-weight="700" '
          f'fill="{c}">{acc:.2%}</text>\n')
    b += (f'  <text x="{XR+150}" y="{y+23}" font-size="12" fill="{MUT}">'
          f'보유 클래스 {cls}개</text>\n')
b += f'  <line x1="{XL}" y1="{PT-4}" x2="{XL}" y2="{PT+len(rows)*RH-6}" stroke="#94a3b8" stroke-width="1.5"/>\n'
b += (f'  <text x="{XL}" y="{PT+len(rows)*RH+16}" font-size="11.5" fill="{MUT}">'
      f'가로축 시작점 {lo:.0%} (차이를 보이기 위한 절단 축)</text>\n')
DROP = FD["central"][-1]["acc"] - PA["rows"][-1]["final"]
b += note(44, PT + len(rows) * RH + 32, 916, 122, "MNIST 는 잘 안 무너진다 — 그게 정보다",
          [f"디리클레 α=0.1 까지 내려도 하락은 {(FD['central'][-1]['acc']-FD['alpha'][-1]['final'])*100:.1f}%p 에 그친다",
           f"기기마다 클래스를 하나만 주는 병리적 분할에서야 {DROP*100:.1f}%p 가 무너진다",
           "문헌도 같다 — 비-IID 하락은 MNIST 6.5~11.3%, KWS 54.5% 로 과제마다 다르다",
           "\"비-IID 면 55% 떨어진다\" 는 특정 데이터셋·특정 분할의 값이다"],
          bg="#fff7ed", ln="#fbbf24", hc="#92400e")
fig("w12_p1_noniid_03", 1004, PT + len(rows) * RH + 170, b)

# ═════════ 04. 통신 ═════════
b = title(430, 30, "통신이 진짜 비용이다",
          "정확도 90% 에 닿기까지 올려야 하는 바이트 (실측)")
tt = FD["to_target"]
XL, PT, RH, BW = 250, 82, 46, 380
mx = max((r["up_mb"] or 0) for r in tt["rows"]) or 1
for i, r in enumerate(tt["rows"]):
    y = PT + i * RH
    b += (f'  <text x="{XL-12}" y="{y+22}" text-anchor="end" font-size="13" '
          f'font-weight="700" fill="#334155">{r["name"]}</text>\n')
    if r["up_mb"] is None:
        b += (f'  <text x="{XL+6}" y="{y+22}" font-size="13" font-weight="700" '
              f'fill="{RED}">30라운드 안에 도달 못 함 (최종 {r["final"]:.2%})</text>\n')
        continue
    w = max(r["up_mb"] / mx * BW, 3)
    c = GRN if r["up_mb"] < mx * 0.3 else TEAL
    b += f'  <rect x="{XL}" y="{y+4}" width="{w:.1f}" height="26" rx="3" fill="{c}"/>\n'
    b += (f'  <text x="{XL+w+9:.1f}" y="{y+23}" font-size="13" font-weight="700" '
          f'fill="{c}">{r["up_mb"]:.2f} MB</text>\n')
    b += (f'  <text x="{XL+BW+118}" y="{y+23}" font-size="12" fill="{MUT}">'
          f'{r["round"]}라운드</text>\n')
b += f'  <line x1="{XL}" y1="{PT-2}" x2="{XL}" y2="{PT+len(tt["rows"])*RH-10}" stroke="#94a3b8" stroke-width="1.5"/>\n'
yy = PT + len(tt["rows"]) * RH + 10
b += f'  <text x="44" y="{yy+14}" font-size="14" font-weight="700" fill="{INK}">현실 규모로 환산하면 (내려받기+올리기 합계)</text>\n'
for i, s in enumerate(FD["scale"]):
    y2 = yy + 34 + i * 26
    b += (f'  <text x="60" y="{y2}" font-size="12.8" fill="#334155">'
          f'기기 {s["clients"]:,}대 × {s["rounds"]}라운드</text>\n')
    col = RED if s["gb"] > 100 else (AMB if s["gb"] > 10 else MUT)
    b += (f'  <text x="330" y="{y2}" font-size="12.8" font-weight="700" fill="{col}">'
          f'{s["gb"]:,.1f} GB</text>\n')
b += note(452, yy + 16, 384, 152, "로컬 에폭은 왜 답이 아니었나",
          ["E=5 는 계산을 4.6배 쓰고도",
           "라운드를 못 줄여 통신은 그대로였다",
           "그리고 로컬에서 오래 돌수록",
           "기기별 모델이 서로 멀어진다",
           "(client drift — FedProx·SCAFFOLD 의 주제)"],
          bg="#fff7ed", ln="#fbbf24", hc="#92400e")
fig("w12_p1_comm_04", 880, yy + 192, b)

# ═════════ 05. 완전 복원 (핵심) ═════════
E = LK["exact"]
b = title(430, 30, "기울기 하나에서 원본이 그대로 나온다",
          "반복 최적화가 아니라 나눗셈 한 번 — MLP(784-32-10) 실측")
b += digit(96, 84, E["orig"], 5.4, "원본 이미지", INK)
b += digit(392, 84, E["rec"], 5.4, "기울기에서 복원", RED, RED)
b += (f'  <text x="330" y="{84+14*5.4+6}" text-anchor="middle" font-size="30" '
      f'font-weight="700" fill="{MUT}">→</text>\n')
b += (f'  <rect x="600" y="84" width="236" height="152" rx="10" fill="#ffffff" stroke="{LINE}"/>\n')
STAT = [("복원 PSNR", f"{E['psnr']:.2f} dB", RED),
        ("최대 화소 오차", f"{E['max_err']:.1e}", RED),
        ("라벨 복원 (300장)", f"{LK['label_acc']:.0%}", RED)]
for i, (k, v, c) in enumerate(STAT):
    b += f'  <text x="618" y="{116+i*46}" font-size="12.5" fill="{MUT}">{k}</text>\n'
    b += f'  <text x="618" y="{140+i*46}" font-size="19" font-weight="700" fill="{c}">{v}</text>\n'
b += (f'  <rect x="96" y="268" width="740" height="66" rx="10" fill="#0b2e2b"/>\n')
b += (f'  <text x="466" y="298" text-anchor="middle" font-size="17" font-weight="700" '
      f'fill="#cfe7e3">x = (∂L/∂W)[i, :] ÷ (∂L/∂b)[i]</text>\n')
b += (f'  <text x="466" y="320" text-anchor="middle" font-size="12.5" fill="#7fa9a3">'
      f'z = Wx + b 이면 ∂L/∂W = (∂L/∂z)·x^T 이고 ∂L/∂b = ∂L/∂z 이므로</text>\n')
b += note(96, 350, 740, 100, "최초 출처를 정확히",
          ["Phong 외, IEEE TIFS 13(5):1333–1345, 2018 — §3 Example 1, 식 (7)–(8), 관찰 (O1)",
           "임의 구조로의 일반화는 Geiping 외, NeurIPS 2020, 명제 3.1",
           "그 논문 스스로 \"Phong 외의 결과를 일반화한 것\" 이라고 밝힌다"],
          bg="#fef2f2", ln="#fca5a5", hc="#991b1b")
fig("w12_p2_invert_05", 880, 468, b)

# ═════════ 06. 배치 ═════════
b = title(470, 30, "배치는 누출을 없애지 않는다 — 대상을 바꾼다",
          "복원된 것은 개별 표본이 아니라 δ 가중 평균이다 (실측)")
px = 3.5
for i, r in enumerate(LK["batch"]):
    x = 60 + i * 152
    b += digit(x, 76, r["img"], px, f"B = {r['B']}", INK)
b += f'  <rect x="44" y="216" width="{908}" height="{28+len(LK["batch"])*0+56}" rx="10" fill="#ffffff" stroke="{LINE}"/>\n'
b += f'  <text x="60" y="238" font-size="12.5" font-weight="700" fill="{MUT}">배치</text>\n'
b += f'  <text x="60" y="262" font-size="12.5" fill="#334155">개별 표본 대비 PSNR</text>\n'
b += f'  <text x="60" y="288" font-size="12.5" font-weight="700" fill="{RED}">δ 가중평균 대비 PSNR</text>\n'
for i, r in enumerate(LK["batch"]):
    x = 300 + i * 108
    pf = "—" if r["psnr_vs_first"] is None else f"{r['psnr_vs_first']:.1f}"
    pm = "동일" if r["psnr_vs_mix"] is None else f"{r['psnr_vs_mix']:.1f}"
    b += f'  <text x="{x}" y="238" text-anchor="middle" font-size="12.5" font-weight="700" fill="{MUT}">{r["B"]}</text>\n'
    b += f'  <text x="{x}" y="262" text-anchor="middle" font-size="12.5" fill="#334155">{pf}</text>\n'
    b += f'  <text x="{x}" y="288" text-anchor="middle" font-size="13" font-weight="700" fill="{RED}">{pm}</text>\n'
b += note(44, 316, 908, 122, "오른쪽 줄을 보시라",
          ["배치 32에서도 δ 가중평균은 151.81 dB 로 완전히 복원된다",
           "누출의 양이 준 것이 아니라, 누출의 대상이 \"한 사람\"에서 \"32명의 평균\"으로 바뀐 것이다",
           "Geiping 외(NeurIPS 2020 §6)는 배치 100에서도 일부 이미지가 육안 식별 가능하며",
           "\"배치로 인한 왜곡은 균일하지 않다\" 고 보고한다 — 배치는 완화이지 보장이 아니다"],
          bg="#fef2f2", ln="#fca5a5", hc="#991b1b")
fig("w12_p2_batch_06", 996, 462, b)

# ═════════ 07. 클리핑 ═════════
b = title(430, 30, "기울기 클리핑의 방어력은 정확히 0이다",
          "배율이 분자와 분모에서 약분되기 때문 (실측)")
b += (f'  <rect x="140" y="72" width="580" height="62" rx="10" fill="#0b2e2b"/>\n')
b += (f'  <text x="430" y="112" text-anchor="middle" font-size="20" font-weight="700" '
      f'fill="#cfe7e3">(c · ∂L/∂W[i]) ÷ (c · ∂L/∂b[i])  =  ∂L/∂W[i] ÷ ∂L/∂b[i]</text>\n')
XL, PT, RH, BW = 250, 160, 44, 320
for i, r in enumerate(LK["clip_only"]):
    y = PT + i * RH
    b += (f'  <text x="{XL-12}" y="{y+22}" text-anchor="end" font-size="13" '
          f'font-weight="700" fill="#334155">C = {r["C"]:g}  (배율 {r["scale"]:.5f})</text>\n')
    w = max(r["psnr"] / 180 * BW, 3)
    b += f'  <rect x="{XL}" y="{y+5}" width="{w:.1f}" height="24" rx="3" fill="{RED}"/>\n'
    b += (f'  <text x="{XL+w+9:.1f}" y="{y+23}" font-size="13.5" font-weight="700" '
          f'fill="{RED}">{r["psnr"]:.2f} dB</text>\n')
b += f'  <line x1="{XL}" y1="{PT+2}" x2="{XL}" y2="{PT+len(LK["clip_only"])*RH-6}" stroke="#94a3b8" stroke-width="1.5"/>\n'
yy = PT + len(LK["clip_only"]) * RH + 10
b += (f'  <text x="430" y="{yy+18}" text-anchor="middle" font-size="15" font-weight="700" '
      f'fill="{RED}">기울기를 천 분의 일로 눌러도 복원은 그대로다</text>\n')
b += note(44, yy + 34, 792, 122, "그러면 DP-SGD 의 클리핑은 왜 있는가",
          ["클리핑은 프라이버시를 주는 장치가 아니라 민감도의 상한을 정하는 장치다",
           "프라이버시는 그 위에 얹는 노이즈에서 나온다 (Abadi 외, CCS 2016 §3.1)",
           "\"ℓ2 노름 클리핑은 기울기의 크기만 바꾸고 방향에는 영향을 주지 않는다\"",
           "— Li 외, CVPR 2022, §4.4"],
          bg="#fef2f2", ln="#fca5a5", hc="#991b1b")
fig("w12_p2_clip_07", 880, yy + 182, b)

# ═════════ 08. 노이즈 ═════════
b = title(470, 30, "노이즈만이 유일하게 듣는다",
          "DP-SGD 방식 · C=1.0 · 배치 1 (실측)")
show = [r for r in LK["noise"] if r["sigma"] in (0.0, 0.001, 0.01, 0.1, 1.0)]
px = 4.0
for i, r in enumerate(show):
    x = 66 + i * 182
    lab = f"σ = {r['sigma']:g}"
    col = RED if r["psnr"] > 100 else (AMB if r["psnr"] > 20 else GRN)
    b += digit(x, 76, r["img"], px, lab, INK)
    b += (f'  <text x="{x+14*px}" y="{76+28*px+38}" text-anchor="middle" font-size="13" '
          f'font-weight="700" fill="{col}">{r["psnr"]:.2f} dB</text>\n')
yy = 76 + 28 * px + 56
b += note(44, yy, 872, 106, "σ = 0.01 이면 공격은 이미 죽는다",
          ["σ=0.001 에서 35.5 dB, σ=0.01 에서 10.7 dB — 여기서 대부분 \"해결됐다\" 고 말하고 멈춘다",
           "그런데 그 σ 의 ε 은 얼마인가? 다음 그림이 그 답이다",
           "\"이 공격을 막았다\" 와 \"보장을 갖는다\" 는 전혀 다른 명제다"],
          bg="#fff7ed", ln="#fbbf24", hc="#92400e")
fig("w12_p2_noise_08", 960, yy + 146, b)

# ═════════ 09. 트레이드오프 (핵심 2) ═════════
b = title(470, 30, "공격을 막는 것은 공짜였다 — 보장을 얻는 것은 3.5%p였다",
          "같은 MLP · 클라이언트 10대 · 20라운드 · 표본 단위 클리핑 (실측)")
rowsD = [r for r in DP["rows"]]
XL, PT, RH = 96, 84, 52
COLS = [(XL + 0, "σ"), (XL + 90, "복원 PSNR"), (XL + 300, "ε  (δ = 1e−5)"),
        (XL + 500, "정확도"), (XL + 640, "손실")]
b += f'  <rect x="{XL-24}" y="{PT-30}" width="768" height="{28+len(rowsD)*RH}" rx="10" fill="#ffffff" stroke="{LINE}"/>\n'
for x, t in COLS:
    b += f'  <text x="{x}" y="{PT-8}" font-size="12.5" font-weight="700" fill="{MUT}">{t}</text>\n'
for i, r in enumerate(rowsD):
    y = PT + 22 + i * RH
    if i:
        b += f'  <line x1="{XL-10}" y1="{y-30}" x2="{XL+726}" y2="{y-30}" stroke="#eef2f7"/>\n'
    b += f'  <text x="{COLS[0][0]}" y="{y}" font-size="14" font-weight="700" fill="{INK}">{r["sigma"]:g}</text>\n'
    lp = r["leak_psnr"]
    lc = RED if lp > 100 else GRN
    lt = f"완전 복원 ({lp:.1f} dB)" if lp > 100 else f"파괴 ({lp:.1f} dB)"
    b += f'  <text x="{COLS[1][0]}" y="{y}" font-size="13" font-weight="700" fill="{lc}">{lt}</text>\n'
    if r["eps"] is None:
        b += f'  <text x="{COLS[2][0]}" y="{y}" font-size="13" fill="{MUT}">보장 없음</text>\n'
    else:
        ec = RED if r["eps"] > 100 else (AMB if r["eps"] > 10 else GRN)
        b += f'  <text x="{COLS[2][0]}" y="{y}" font-size="14" font-weight="700" fill="{ec}">{r["eps"]:,.2f}</text>\n'
    b += f'  <text x="{COLS[3][0]}" y="{y}" font-size="13.5" fill="#334155">{r["acc"]:.2%}</text>\n'
    d = r["acc_drop"] or 0
    dc = RED if d > 0.02 else (AMB if d > 0.005 else MUT)
    b += f'  <text x="{COLS[4][0]}" y="{y}" font-size="13.5" font-weight="700" fill="{dc}">{d*100:+.2f}%p</text>\n'
yy = PT + 22 + len(rowsD) * RH
b += note(44, yy + 6, 400, 146, "σ = 0.01 — 공격은 죽었다",
          ["정확도 손실 없음 · ε = 1,422,867",
           "e 의 142만 제곱은",
           "아무것도 제약하지 않는다",
           "→ 보장이 아니다"],
          bg="#fff7ed", ln="#fbbf24", hc="#92400e")
b += note(468, yy + 6, 400, 146, "σ = 2.0 — 보장을 얻었다",
          ["ε = 3.27 · 정확도 손실 3.50%p",
           "이것이 프라이버시의 실제 가격이다",
           "공짜였던 것은 방어이지",
           "→ 보장이 아니었다"],
          bg="#e6f4f2", ln="#9fd6cc", hc="#0b4a48")
b += (f'  <text x="456" y="{yy+172}" text-anchor="middle" font-size="11.5" fill="{MUT}">'
      f'손실이 음수인 줄은 실행 간 변동 폭(±0.2%p) 안이다 — 작은 노이즈는 정확도를 거의 건드리지 않는다</text>\n')
fig("w12_p2_tradeoff_09", 912, yy + 190, b)

# ═════════ 10. 보안 집계 ═════════
b = title(430, 30, "보안 집계가 막는 것과 못 막는 것",
          "Bonawitz 외, ACM CCS 2017 · 상쇄되는 마스크로 서버는 합계만 본다")
b += f'  <rect x="44" y="72" width="384" height="150" rx="12" fill="#e6f4f2" stroke="{TEAL}" stroke-width="2"/>\n'
b += f'  <text x="236" y="100" text-anchor="middle" font-size="16" font-weight="700" fill="{TEAL}">막는 것</text>\n'
for i, t in enumerate(["정직한 서버가 개별 업데이트를 보는 것",
                       "통신 확장률 1.73배 (m = 2^20, n = 2^10)",
                       "→ 실용적인 비용이다"]):
    b += f'  <text x="236" y="{128+i*26}" text-anchor="middle" font-size="13" fill="#334155">{t}</text>\n'
b += f'  <rect x="452" y="72" width="384" height="150" rx="12" fill="#fee2e2" stroke="{RED}" stroke-width="2"/>\n'
b += f'  <text x="644" y="100" text-anchor="middle" font-size="16" font-weight="700" fill="{RED}">못 막는 것</text>\n'
for i, t in enumerate(["① 합계 벡터 자체에 대한 역복원",
                       "② 라운드 반복 관찰 → 합계 분해",
                       "③ 악의적 서버의 우회 (참여자 수 무관)"]):
    b += f'  <text x="644" y="{128+i*26}" text-anchor="middle" font-size="13" fill="#334155">{t}</text>\n'
XL, PT, RH, BW = 250, 250, 42, 330
DROPS = [("탈락 0%", 2018, MUT), ("탈락 10%", 62239, AMB), ("탈락 30%", 143389, RED)]
mxd = 143389
b += f'  <text x="44" y="{PT-6}" font-size="14" font-weight="700" fill="{INK}">그리고 탈락에 취약하다 (500 클라이언트 · 100K 항목 · 서버 시간)</text>\n'
for i, (lab, v, c) in enumerate(DROPS):
    y = PT + 10 + i * RH
    b += f'  <text x="{XL-12}" y="{y+22}" text-anchor="end" font-size="13" font-weight="700" fill="#334155">{lab}</text>\n'
    w = max(v / mxd * BW, 3)
    b += f'  <rect x="{XL}" y="{y+5}" width="{w:.1f}" height="24" rx="3" fill="{c}"/>\n'
    b += f'  <text x="{XL+w+9:.1f}" y="{y+23}" font-size="13" font-weight="700" fill="{c}">{v:,} ms</text>\n'
b += (f'  <text x="{XL+BW+150}" y="{PT+10+2*RH+23}" font-size="13.5" font-weight="700" '
      f'fill="{RED}">71배</text>\n')
yy = PT + 10 + 3 * RH + 12
b += note(44, yy, 792, 100, "정직한 답",
          ["\"악의적 서버는 보안 집계가 마치 없는 것처럼 쉽게 우회할 수 있다. …",
           "보안 집계에 참여하는 사용자 수와 무관하게, 사용된 프로토콜의 종류와 무관하게 똑같이 효과적이다.\"",
           "— Pasquini 외, \"Eluding Secure Aggregation in Federated Learning via Model Inconsistency\", ACM CCS 2022"],
          bg="#fef2f2", ln="#fca5a5", hc="#991b1b")
fig("w12_p2_secagg_10", 880, yy + 140, b)

# ═════════ 11. 법 ═════════
b = title(430, 30, "가명정보와 익명정보는 다르다",
          "「개인정보 보호법」 제2조 · 제28조의2 · 제58조의2")
CARDS = [(44, "가명정보", "#fff7ed", "#fbbf24", "92400E",
          ["제2조 제1호 다목", "여전히 개인정보다",
           "통계작성·과학적 연구·공익적 기록보존", "목적에 한해 동의 없이 처리 가능",
           "(제28조의2)"]),
         (452, "익명정보", "#e6f4f2", "#9fd6cc", "0B4A48",
          ["제58조의2", "법 적용에서 제외된다",
           "\"시간ㆍ비용ㆍ기술 등을 합리적으로", "고려할 때 다른 정보를 사용하여도",
           "더 이상 개인을 알아볼 수 없는 정보\""])]
for x, hd, bg, ln, hc, items in CARDS:
    b += f'  <rect x="{x}" y="76" width="384" height="176" rx="13" fill="{bg}" stroke="{ln}" stroke-width="2"/>\n'
    b += f'  <text x="{x+192}" y="106" text-anchor="middle" font-size="19" font-weight="700" fill="#{hc}">{hd}</text>\n'
    for i, t in enumerate(items):
        b += f'  <text x="{x+192}" y="{134+i*24}" text-anchor="middle" font-size="12.5" fill="#334155">{t}</text>\n'
b += note(44, 272, 792, 146, "연합 학습의 모델 업데이트는 어느 쪽인가",
          ["개인정보보호위원회의 확정된 유권해석이 없다.",
           "그런데 오늘 2.1 에서 우리는 기울기에서 원본 입력을 해석적으로 복원했다.",
           "그러면 \"다른 정보를 사용하여도 더 이상 개인을 알아볼 수 없는\" 이라는 제58조의2 요건을",
           "연합 학습 그 자체만으로 충족한다고 보기는 어렵다.",
           "방어 가능한 유일한 서술 — \"사안별로 재식별 가능성을 평가해야 한다\""],
          bg="#fef2f2", ln="#fca5a5", hc="#991b1b")
b += (f'  <text x="430" y="440" text-anchor="middle" font-size="12" fill="{MUT}">'
      f'GDPR 도 같다 — EDPB 의견 28/2024 ¶31 은 학습 데이터가 모델 파라미터에 \'흡수\'되어 남을 수 있다고 판단했다 '
      f'(다만 연합 학습을 직접 다루지는 않는다)</text>\n')
fig("w12_p2_law_11", 880, 462, b)

# ═════════ 12. 판정표 ═════════
b = title(400, 30, "3교시 산출물 — 프라이버시 실사 한 장",
          "student.py 재실행값 · MLP(784-32-10) · MNIST")
LINES = [("①", "모델 · 첫 층 구조", "MLP · Linear(784, 32)", MUT),
         ("②", "복원 PSNR · 최대 화소 오차 (B=1)", "167.7 dB · 최대오차 1.2e−7", RED),
         ("③", "라벨 복원률", "50 / 50", RED),
         ("④", "배치 32 — 개별 표본 / 가중평균 대비", "11.7 dB / 151.8 dB", RED),
         ("⑤", "클리핑 C=0.01 (배율 0.00114)", "152.9 dB — 방어력 0", RED),
         ("⑥", "복원이 파괴되는 최소 σ", "0.01", GRN),
         ("⑦", "그 σ 의 ε · 정확도 손실", "1,422,867 · 손실 없음", AMB),
         ("⑧", "ε = 3.27 을 얻으려면", "σ = 2.0 · 정확도 3.50%p", INK)]
ty = 82
b += f'  <rect x="46" y="{ty}" width="708" height="{20+len(LINES)*36}" rx="10" fill="#ffffff" stroke="{LINE}"/>\n'
for i, (n, k, v, c) in enumerate(LINES):
    yy2 = ty + 34 + i * 36
    if i == len(LINES) - 1:
        b += f'  <line x1="60" y1="{yy2-22}" x2="740" y2="{yy2-22}" stroke="{LINE}"/>\n'
    b += f'  <text x="66" y="{yy2}" font-size="14" font-weight="700" fill="{MUT}">{n}</text>\n'
    b += f'  <text x="94" y="{yy2}" font-size="13" fill="#334155">{k}</text>\n'
    b += f'  <text x="740" y="{yy2}" text-anchor="end" font-size="13" font-weight="700" fill="{c}">{v}</text>\n'
by = ty + 36 + len(LINES) * 36
b += note(46, by, 708, 100, "결론 문단에 반드시 들어가야 하는 것",
          ["여러분의 응용에서 \"이 공격을 막았다\" 로 충분한지, 아니면 \"보장이 필요\" 한지",
           "그리고 그 판단의 근거. 두 문장이면 됩니다.",
           "주의: B=1 복원 PSNR 은 실행마다 153~168 dB 로 흔들립니다. 구조를 재현하십시오."])
fig("w12_p3_report_12", 800, by + 146, b)

for k, v in F.items():
    (OUT / f"{k}.svg").write_text(v, encoding="utf-8")
print(f"SVG {len(F)}장 → {OUT}")
