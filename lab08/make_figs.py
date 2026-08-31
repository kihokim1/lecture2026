# -*- coding: utf-8 -*-
"""8주차 그림 — 개념 도해 + mem.json / grid.json / energy.json 기반 실측 차트."""
import json, math, pathlib

M = json.load(open("/root/lab08/mem.json"))
G = json.load(open("/root/lab08/grid.json"))
E = json.load(open("/root/lab08/energy.json"))
OUT = pathlib.Path("/root/ondevice-ai/img/week08"); OUT.mkdir(parents=True, exist_ok=True)

HEAD = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        'font-family="\'Segoe UI\',Arial,sans-serif">\n'
        '  <rect width="{w}" height="{h}" fill="#f8fafc"/>\n')
F = {}
KB = lambda x: x / 1024.0


def fig(name, w, h, body):
    F[name] = HEAD.format(w=w, h=h) + body + "\n</svg>\n"


def title(x, y, t, sub=None):
    s = f'  <text x="{x}" y="{y}" text-anchor="middle" font-size="19" font-weight="700" fill="#0f172a">{t}</text>\n'
    if sub:
        s += f'  <text x="{x}" y="{y+22}" text-anchor="middle" font-size="13.5" fill="#64748b">{sub}</text>\n'
    return s


def note(x, y, w, h, head, body, bg="#edf6f4", ln="#9fd6cc", hc="#0b4a48"):
    s = f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{bg}" stroke="{ln}"/>\n'
    s += f'  <text x="{x+w/2}" y="{y+26}" text-anchor="middle" font-size="14.5" font-weight="700" fill="{hc}">{head}</text>\n'
    for i, ln2 in enumerate(body):
        s += f'  <text x="{x+w/2}" y="{y+50+i*20}" text-anchor="middle" font-size="12.5" fill="#334155">{ln2}</text>\n'
    return s


MK = ('<marker id="{i}" markerWidth="9" markerHeight="9" refX="7" refY="3.5" orient="auto">'
      '<path d="M0,0 L7,3.5 L0,7 Z" fill="{c}"/></marker>')

T = M["tiny"]
NAMES = ["FC-AutoEncoder (이상 감지)", "MobileNetV1 0.25 (사람 감지)",
         "ResNet-8 (이미지 분류)", "DS-CNN (키워드 인식)"]
SHORT = {"FC-AutoEncoder (이상 감지)": "FC-AutoEncoder",
         "MobileNetV1 0.25 (사람 감지)": "MobileNetV1-0.25",
         "ResNet-8 (이미지 분류)": "ResNet-8",
         "DS-CNN (키워드 인식)": "DS-CNN"}

# ═════════ 01. 두 개의 벽 ═════════
b = title(400, 30, "MCU에는 벽이 두 개 있다", "서로 다른 것이 채우므로, 서로 다른 손잡이로 넘어야 한다")
for x, w, c, bg, nm, cap, fill, items, foot in [
    (34, 350, "#7c3aed", "#ede9fe", "Flash (ROM)", "1 MB", "가중치 · 코드",
     ["읽기 전용 · 전원 꺼도 유지",
      "모델을 컴파일할 때 정해진다",
      "부족하면 빌드가 실패한다"],
     "4~7주차 기법이 겨냥한 곳"),
    (416, 350, "#e4711b", "#fff7ed", "SRAM (RAM)", "320 KB", "중간 결과 · 스택",
     ["읽고 쓰기 · 전원 끄면 사라짐",
      "모델을 실행하는 도중에 정해진다",
      "부족하면 동작 중에 터진다"],
     "이번 주가 겨냥하는 곳")]:
    b += f'  <rect x="{x}" y="72" width="{w}" height="236" rx="13" fill="{bg}" stroke="{c}" stroke-width="2.2"/>\n'
    b += f'  <text x="{x+w/2}" y="100" text-anchor="middle" font-size="16.5" font-weight="700" fill="{c}">{nm}</text>\n'
    b += f'  <rect x="{x+80}" y="112" width="{w-160}" height="46" rx="9" fill="#ffffff" stroke="{c}"/>\n'
    b += (f'  <text x="{x+w/2}" y="143" text-anchor="middle" font-size="24" font-weight="700" '
          f'fill="{c}" font-family="Courier New,monospace">{cap}</text>\n')
    b += f'  <text x="{x+w/2}" y="184" text-anchor="middle" font-size="15" font-weight="700" fill="#0f172a">무엇이 채우나 — {fill}</text>\n'
    for i, s2 in enumerate(items):
        b += f'  <text x="{x+w/2}" y="{210+i*24}" text-anchor="middle" font-size="12.8" fill="#334155">{s2}</text>\n'
    b += f'  <text x="{x+w/2}" y="292" text-anchor="middle" font-size="13" font-weight="700" fill="{c}">{foot}</text>\n'
b += note(34, 326, 732, 74, "STM32F746 에 int8 MobileNetV2 를 올리면",
          ["Flash 3,394 KB 필요 / 1,024 KB 예산  →  3.3배 초과      SRAM 1,470 KB 필요 / 320 KB 예산  →  4.6배 초과",
           "해상도를 32²까지 낮춰도 Flash 는 3,394 KB 그대로다 — 해상도는 SRAM 만 움직이는 손잡이다"])
fig("w08_p1_two_walls_01", 800, 414, b)

# ═════════ 02. 순위 역전 ═════════
b = title(400, 30, "파라미터 순위와 SRAM 순위는 같지 않다",
          "MLPerf Tiny 참조 모델 4종 · INT8 기준 실측")
LX, RX, TY, DY = 250, 550, 92, 58
b += '  <text x="250" y="76" text-anchor="middle" font-size="14.5" font-weight="700" fill="#7c3aed">파라미터 (많은 순)</text>\n'
b += '  <text x="550" y="76" text-anchor="middle" font-size="14.5" font-weight="700" fill="#e4711b">SRAM (많은 순)</text>\n'
left = sorted(NAMES, key=lambda n: -T[n]["params"])
right = sorted(NAMES, key=lambda n: -T[n]["sram_inplace"])
posL = {n: TY + i * DY for i, n in enumerate(left)}
posR = {n: TY + i * DY for i, n in enumerate(right)}
COL = {"FC-AutoEncoder (이상 감지)": "#dc2626", "MobileNetV1 0.25 (사람 감지)": "#2563eb",
       "ResNet-8 (이미지 분류)": "#64748b", "DS-CNN (키워드 인식)": "#94a3b8"}
for n in NAMES:
    y1, y2 = posL[n] + 18, posR[n] + 18
    c = COL[n]
    wdt = 3.4 if n in list(COL)[:2] else 1.6
    b += (f'  <path d="M366 {y1} C440 {y1}, 460 {y2}, 434 {y2}" stroke="{c}" '
          f'stroke-width="{wdt}" fill="none" opacity="0.85"/>\n')
for n in NAMES:
    c = COL[n]
    b += f'  <rect x="128" y="{posL[n]}" width="238" height="38" rx="8" fill="#ffffff" stroke="{c}" stroke-width="1.8"/>\n'
    b += f'  <text x="140" y="{posL[n]+24}" font-size="13" font-weight="700" fill="#0f172a">{SHORT[n]}</text>\n'
    b += f'  <text x="356" y="{posL[n]+24}" text-anchor="end" font-size="13" fill="{c}" font-family="Courier New,monospace">{T[n]["params"]:,}</text>\n'
    b += f'  <rect x="434" y="{posR[n]}" width="238" height="38" rx="8" fill="#ffffff" stroke="{c}" stroke-width="1.8"/>\n'
    b += f'  <text x="446" y="{posR[n]+24}" font-size="13" font-weight="700" fill="#0f172a">{SHORT[n]}</text>\n'
    b += (f'  <text x="662" y="{posR[n]+24}" text-anchor="end" font-size="13" fill="{c}" '
          f'font-family="Courier New,monospace">{KB(T[n]["sram_inplace"]):.1f} KB</text>\n')
b += note(84, 336, 632, 76, "파라미터 1위가 SRAM 은 꼴찌다",
          ["FC-AutoEncoder 는 MobileNetV1 보다 파라미터가 1.25배 많은데 SRAM 은 84분의 1만 쓴다",
           "완전연결층은 640개짜리 벡터를 넘기고, 합성곱층은 채널×높이×너비 덩어리를 넘기기 때문이다"],
          bg="#fff7ed", ln="#f59e0b", hc="#b45309")
fig("w08_p1_rank_flip_02", 800, 426, b)

# ═════════ 03. 컴퓨팅 스펙트럼 ═════════
b = title(400, 30, "온디바이스AI 의 오른쪽 끝", "메모리와 전력이 여섯 자릿수 넘게 벌어진다")
tiers = [
    ("클라우드 GPU", "수십 GB", "수 TB", "수백 W", "#64748b", "#f1f5f9", 232),
    ("스마트폰", "수 GB", "수백 GB", "수 W", "#2563eb", "#eff6ff", 178),
    ("MCU (TinyML)", "수백 KB", "수 MB", "수 mW", "#e4711b", "#fff7ed", 116),
]
for i, (nm, mem, sto, pw, c, bg, wd) in enumerate(tiers):
    y = 78 + i * 78
    x = 60
    b += f'  <rect x="{x}" y="{y}" width="{wd}" height="62" rx="10" fill="{bg}" stroke="{c}" stroke-width="2"/>\n'
    b += f'  <text x="{x+wd/2}" y="{y+26}" text-anchor="middle" font-size="14.5" font-weight="700" fill="{c}">{nm}</text>\n'
    b += f'  <text x="{x+wd/2}" y="{y+48}" text-anchor="middle" font-size="12" fill="#334155">메모리 {mem}</text>\n'
    for j, (lab, v) in enumerate([("저장", sto), ("전력", pw)]):
        bx = 330 + j * 200
        b += f'  <rect x="{bx}" y="{y}" width="176" height="62" rx="9" fill="#ffffff" stroke="{c}" stroke-width="1.2"/>\n'
        b += f'  <text x="{bx+88}" y="{y+26}" text-anchor="middle" font-size="12" fill="#64748b">{lab}</text>\n'
        b += f'  <text x="{bx+88}" y="{y+48}" text-anchor="middle" font-size="16" font-weight="700" fill="{c}">{v}</text>\n'
b += ('  <path d="M40 84 V296" stroke="#94a3b8" stroke-width="2" marker-end="url(#d)" fill="none"/>\n'
      '  <text x="22" y="196" text-anchor="middle" font-size="12.5" fill="#64748b" '
      'transform="rotate(-90 22 196)">자원이 줄어든다</text>\n')
b += note(60, 320, 646, 74, "MCUNet (NeurIPS 2020) 이 적은 격차",
          ["\"마이크로컨트롤러는 휴대폰에 비해 메모리와 저장 공간이 세 자릿수 적고,",
           "클라우드 GPU에 비하면 다섯~여섯 자릿수 적다\""])
b += '  <defs>' + MK.format(i="d", c="#94a3b8") + '</defs>\n'
fig("w08_p1_scale_03", 800, 408, b)

# ═════════ 04. 예산 지도 ═════════
b = title(400, 30, "STM32F746 예산 지도", "가로 Flash · 세로 SRAM · 회색 상자 안이 통과 (INT8 기준)")
XL, XR, PT, PB = 92, 720, 76, 320
fx = lambda v: XL + math.log10(max(v, 8) / 8) / math.log10(4200 / 8) * (XR - XL)
fy = lambda v: PB - 14 - math.log10(max(v, 0.4) / 0.4) / math.log10(2600 / 0.4) * (PB - PT - 14)
b += f'  <rect x="{XL}" y="{fy(320):.1f}" width="{fx(1024)-XL:.1f}" height="{PB-fy(320):.1f}" rx="5" fill="#dcfce7" stroke="#16a34a" stroke-width="2" stroke-dasharray="6 4"/>\n'
b += f'  <text x="{(XL+fx(1024))/2:.0f}" y="{PB-12}" text-anchor="middle" font-size="13.5" font-weight="700" fill="#15803d">통과 영역</text>\n'
b += f'  <line x1="{XL}" y1="{PB}" x2="{XR}" y2="{PB}" stroke="#94a3b8" stroke-width="1.5"/>\n'
b += f'  <line x1="{XL}" y1="{PT-6}" x2="{XL}" y2="{PB}" stroke="#94a3b8" stroke-width="1.5"/>\n'
for v in [10, 100, 1024, 4000]:
    b += f'  <text x="{fx(v):.1f}" y="{PB+20}" text-anchor="middle" font-size="11.5" fill="#94a3b8">{v if v<1024 else ("1 MB" if v==1024 else "4 MB")}</text>\n'
for v in [1, 10, 100, 320, 2000]:
    b += f'  <text x="{XL-8}" y="{fy(v)+4:.1f}" text-anchor="end" font-size="11.5" fill="#94a3b8">{v}</text>\n'
b += f'  <text x="{(XL+XR)/2:.0f}" y="{PB+40}" text-anchor="middle" font-size="13" font-weight="700" fill="#334155">Flash — 가중치 (KB, 로그)</text>\n'
b += f'  <text x="26" y="{(PT+PB)/2}" text-anchor="middle" font-size="13" font-weight="700" fill="#334155" transform="rotate(-90 26 {(PT+PB)/2})">SRAM — 중간 결과 (KB, 로그)</text>\n'
pts = [(SHORT[n], KB(T[n]["flash"]), KB(T[n]["sram_inplace"]), "#16a34a") for n in NAMES]
pts += [("MobileNetV2 @224²", KB(M["res_sweep"]["224"]["flash"]), KB(M["res_sweep"]["224"]["sram_inplace"]), "#dc2626"),
        ("MobileNetV2 @96²", KB(M["res_sweep"]["96"]["flash"]), KB(M["res_sweep"]["96"]["sram_inplace"]), "#dc2626")]
dy = {"FC-AutoEncoder": -14, "DS-CNN": -13, "ResNet-8": 21,
      "MobileNetV1-0.25": -13, "MobileNetV2 @224²": -13, "MobileNetV2 @96²": 21}
for nm, f_, s_, c in pts:
    X, Y = fx(f_), fy(s_)
    b += f'  <circle cx="{X:.1f}" cy="{Y:.1f}" r="7" fill="{c}" stroke="#ffffff" stroke-width="2"/>\n'
    anc = "end" if X > 560 else "middle"
    b += f'  <text x="{X:.1f}" y="{Y+dy[nm]:.1f}" text-anchor="{anc}" font-size="12" font-weight="700" fill="{c}">{nm}</text>\n'
b += note(92, 348, 628, 74, "해상도는 세로로만 움직인다",
          ["MobileNetV2 를 224² → 96² 로 낮추면 SRAM 은 1,470 → 270 KB (5.4배) 로 내려와 통과선을 넘지만,",
           "Flash 는 3,394 KB 에서 1 바이트도 안 움직인다 — 두 점의 가로 위치가 같다"],
          bg="#fff7ed", ln="#f59e0b", hc="#b45309")
fig("w08_p1_budget_map_04", 800, 436, b)

# ═════════ 05. 텐서 수명 ═════════
b = title(400, 30, "텐서에는 수명이 있다", "죽은 텐서의 자리는 다시 써도 된다")
NX, NY, NW = 96, 84, 96
ops = ["Conv", "ReLU", "Conv", "Add", "Conv"]
for i, op in enumerate(ops):
    x = NX + i * (NW + 22)
    b += f'  <rect x="{x}" y="{NY}" width="{NW}" height="34" rx="7" fill="#e0f2fe" stroke="#0284c7"/>\n'
    b += f'  <text x="{x+NW/2}" y="{NY+23}" text-anchor="middle" font-size="13" font-weight="700" fill="#075985">{op}</text>\n'
    if i < 4:
        b += f'  <path d="M{x+NW} {NY+17} H{x+NW+18}" stroke="#94a3b8" stroke-width="2" marker-end="url(#e)"/>\n'
tens = [("x0", 0, 1, "#2563eb"), ("x1", 1, 2, "#7c3aed"), ("x2", 2, 4, "#e4711b"),
        ("x3", 3, 4, "#16a34a"), ("x4", 4, 5, "#dc2626")]
b += '  <text x="60" y="150" text-anchor="end" font-size="12.5" font-weight="700" fill="#334155">수명</text>\n'
for j, (nm, s, e, c) in enumerate(tens):
    y = 138 + j * 26
    x1 = NX + s * (NW + 22)
    x2 = NX + e * (NW + 22) - 22
    b += f'  <rect x="{x1}" y="{y}" width="{max(x2-x1,40):.0f}" height="18" rx="5" fill="{c}" fill-opacity="0.82"/>\n'
    b += f'  <text x="{x1+8}" y="{y+14}" font-size="11.5" font-weight="700" fill="#ffffff" font-family="Courier New,monospace">{nm}</text>\n'
b += '  <text x="60" y="290" text-anchor="end" font-size="12.5" font-weight="700" fill="#334155">배치</text>\n'
b += '  <rect x="96" y="272" width="608" height="42" rx="7" fill="#ffffff" stroke="#94a3b8" stroke-dasharray="4 3"/>\n'
slots = [("x0", 96, 172, "#2563eb"), ("x1", 274, 172, "#7c3aed"),
         ("x2", 96, 172, "#e4711b"), ("x3", 274, 172, "#16a34a"), ("x4", 452, 172, "#dc2626")]
seen = {}
for nm, x, w, c in slots:
    lane = 0 if x == 96 else (1 if x == 274 else 2)
    yy = 276 + (0 if nm in ("x0", "x1", "x4") else 18)
    b += f'  <rect x="{x}" y="{yy}" width="{w}" height="16" rx="4" fill="{c}" fill-opacity="0.8"/>\n'
    b += f'  <text x="{x+7}" y="{yy+12.5}" font-size="10.5" font-weight="700" fill="#ffffff" font-family="Courier New,monospace">{nm}</text>\n'
b += '  <text x="400" y="336" text-anchor="middle" font-size="12.5" font-weight="700" fill="#64748b">↑ 수명이 겹치지 않는 텐서끼리 같은 자리를 나눠 쓴다</text>\n'
b += note(96, 352, 608, 100, "MobileNetV2 @224² · INT8 실측",
          ["① 재사용 없음 12,846 KB   →   ② 수명 기반 2,352 KB (5.46배 감소)",
           "→   ③ ② + in-place 1,470 KB (8.74배 감소)",
           "모델은 한 글자도 안 바꿨다 — 메모리를 어떻게 배치하느냐만 바꿨다"])
b += '  <defs>' + MK.format(i="e", c="#94a3b8") + '</defs>\n'
fig("w08_p2_lifetime_05", 800, 468, b)

# ═════════ 06. 메모리 프로파일 ═════════
prof = [v for k, op, v in M["res_sweep"]["224"]["per_node"] if v > 0]
b = title(400, 30, "100개 노드 중 세 개가 예산을 정한다",
          "MobileNetV2 @224² · INT8 · 노드별 필요 메모리 실측")
XL, XR, PT, PB = 84, 736, 78, 288
mx = max(prof)
sy = lambda v: PB - v / (mx * 1.06) * (PB - PT)
b += f'  <line x1="{XL}" y1="{PB}" x2="{XR}" y2="{PB}" stroke="#94a3b8" stroke-width="1.5"/>\n'
b += f'  <line x1="{XL}" y1="{PT-6}" x2="{XL}" y2="{PB}" stroke="#94a3b8" stroke-width="1.5"/>\n'
for gl in [500, 1000, 1500, 2000]:
    yy = sy(gl * 1024)
    b += f'  <line x1="{XL}" y1="{yy:.1f}" x2="{XR}" y2="{yy:.1f}" stroke="#e2e8f0"/>\n'
    b += f'  <text x="{XL-8}" y="{yy+4:.1f}" text-anchor="end" font-size="11" fill="#94a3b8">{gl:,}</text>\n'
bw = (XR - XL) / len(prof)
for i, v in enumerate(prof):
    c = "#dc2626" if v > mx * 0.5 else "#028090"
    x = XL + i * bw
    b += f'  <rect x="{x:.1f}" y="{sy(v):.1f}" width="{bw*0.78:.2f}" height="{PB-sy(v):.1f}" fill="{c}" fill-opacity="0.88"/>\n'
ybud = sy(320 * 1024)
b += f'  <line x1="{XL}" y1="{ybud:.1f}" x2="{XR}" y2="{ybud:.1f}" stroke="#16a34a" stroke-width="2.2" stroke-dasharray="7 4"/>\n'
b += f'  <text x="{XR-4}" y="{ybud-8:.1f}" text-anchor="end" font-size="12.5" font-weight="700" fill="#15803d">320 KB 예산선</text>\n'
b += f'  <text x="{XL+bw*7:.0f}" y="{sy(mx)-12:.1f}" text-anchor="middle" font-size="13.5" font-weight="700" fill="#b91c1c">2,352 KB</text>\n'
b += f'  <path d="M{XL+bw*10:.0f} {sy(mx)-6:.1f} L{XL+bw*22:.0f} {sy(mx*0.72):.1f}" stroke="#b91c1c" stroke-width="1.4" fill="none"/>\n'
b += f'  <text x="{XL+bw*24:.0f}" y="{sy(mx*0.70):.1f}" font-size="12" fill="#b91c1c">6·7·8번 노드 — 최댓값의 절반을 넘는 유일한 셋</text>\n'
b += f'  <text x="{(XL+XR)/2:.0f}" y="{PB+22}" text-anchor="middle" font-size="12.5" font-weight="700" fill="#334155">연산 노드 (실행 순서) — 왼쪽이 입력 쪽</text>\n'
b += f'  <text x="26" y="{(PT+PB)/2}" text-anchor="middle" font-size="12.5" font-weight="700" fill="#334155" transform="rotate(-90 26 {(PT+PB)/2})">필요 메모리 (KB)</text>\n'
b += note(84, 316, 652, 96, "MCUNetV2 (NeurIPS 2021) 가 보고한 것과 같은 현상",
          ["\"앞쪽 몇 개 블록이 나머지 네트워크보다 한 자릿수 큰 메모리를 쓴다\"  ·  \"세 번째 블록은 나머지보다 8배 크다\"",
           "최대 2,352 KB / 중앙값 147 KB = 16배     절반 초과 노드 3개 / 100개     마지막 10개 노드는 2~122 KB",
           "논문이 적은 1,372 kB 는 우리 6번 노드 값 1,372.0 KB 와 킬로바이트까지 같다"],
          bg="#fff7ed", ln="#f59e0b", hc="#b45309")
fig("w08_p2_profile_06", 800, 426, b)

# ═════════ 07. 세 개의 손잡이 ═════════
b = title(400, 30, "세 손잡이는 곱해진다 — 그러나 하나는 벽을 못 넘는다",
          "STM32F746 · 320 KB SRAM / 1 MB Flash 예산 판정 (실측)")
rows = [r for r in G["grid"] if (r["hw"], r["bpe"]) in
        [(224, 4), (224, 1), (160, 1), (96, 1), (64, 1)]]
cols = ["해상도", "정밀도", "Flash", "재사용없음", "수명기반", "+in-place", "판정"]
CW = [78, 74, 104, 114, 104, 104, 126]
X0, Y0, RH = 44, 78, 34
x = X0
for j, cnm in enumerate(cols):
    b += f'  <rect x="{x}" y="{Y0}" width="{CW[j]}" height="{RH}" fill="#0e4a44"/>\n'
    b += f'  <text x="{x+CW[j]/2}" y="{Y0+23}" text-anchor="middle" font-size="12.5" font-weight="700" fill="#ffffff">{cnm}</text>\n'
    x += CW[j]
for i, r in enumerate(rows):
    y = Y0 + RH + i * RH
    ok = r["verdict"] == "SRAM만 통과"
    bg = "#ecfdf5" if ok else ("#ffffff" if i % 2 == 0 else "#f8fafc")
    vals = [f'{r["hw"]}²', "FP32" if r["bpe"] == 4 else "INT8",
            f'{KB(r["flash"]):,.0f}', f'{KB(r["naive"]):,.0f}',
            f'{KB(r["life"]):,.0f}', f'{KB(r["inplace"]):,.0f}', r["verdict"]]
    x = X0
    for j, v in enumerate(vals):
        b += f'  <rect x="{x}" y="{y}" width="{CW[j]}" height="{RH}" fill="{bg}" stroke="#d3e5e2"/>\n'
        c = "#15803d" if (j == 6 and ok) else ("#b91c1c" if j == 6 else "#334155")
        fw = "700" if j in (5, 6) or j == 2 else "400"
        b += f'  <text x="{x+CW[j]/2}" y="{y+22}" text-anchor="middle" font-size="12.3" font-weight="{fw}" fill="{c}">{v}</text>\n'
        x += CW[j]
fy2 = Y0 + RH
b += f'  <rect x="{X0+152}" y="{fy2-RH}" width="104" height="{RH*(len(rows)+1)}" rx="4" fill="none" stroke="#7c3aed" stroke-width="2.4"/>\n'
b += f'  <text x="{X0+204}" y="{fy2+RH*len(rows)+22}" text-anchor="middle" font-size="12.5" font-weight="700" fill="#7c3aed">INT8 행에서 완전히 고정</text>\n'
b += note(44, 322, 712, 96, "읽어 낼 것 세 가지",
          ["① 51,384 KB → 270 KB, 190배. 정밀도 4배 · 해상도 5.4배 · 재사용 8.7배가 서로 곱해진다",
           "② 해상도는 Flash 를 1 바이트도 안 건드린다 — SRAM 전용 손잡이다",
           "③ 그래서 MobileNetV2 는 어떤 조합으로도 이 MCU 에 안 들어간다. Flash 벽은 다른 손잡이가 필요하다"])
fig("w08_p2_knobs_07", 800, 432, b)

# ═════════ 08. 손익분기 ═════════
b = title(400, 30, "손익분기 주기는 1.74분이다",
          "활성 전류가 수면 전류의 1,044배인데도")
XL, XR, PT, PB = 96, 720, 84, 270
per = [1, 5, 10, 60, 104.5, 300, 3600]
sweep = {r["period_s"]: r for r in E["sweep"]}
fx = lambda t: XL + math.log10(t / 0.8) / math.log10(5000 / 0.8) * (XR - XL)
b += f'  <line x1="{XL}" y1="{PB}" x2="{XR}" y2="{PB}" stroke="#94a3b8" stroke-width="1.5"/>\n'
for i, t in enumerate(per):
    r = sweep[t]
    x = fx(t)
    fa = r["active_frac"]
    h = 150
    hl = abs(t - 104.5) < 1
    b += f'  <rect x="{x-19:.1f}" y="{PB-h}" width="38" height="{h}" rx="4" fill="#e2e8f0" stroke="{"#dc2626" if hl else "none"}" stroke-width="{2 if hl else 0}"/>\n'
    b += f'  <rect x="{x-19:.1f}" y="{PB-h}" width="38" height="{h*fa:.1f}" rx="4" fill="#e4711b" fill-opacity="0.9"/>\n'
    lab = (f"{t:.0f}초" if t < 60 else (f"{t/60:.0f}분" if t < 3600 else "1시간"))
    if abs(t - 104.5) < 1:
        lab = "1.74분"
    b += f'  <text x="{x:.1f}" y="{PB+20}" text-anchor="middle" font-size="11.5" font-weight="700" fill="#334155">{lab}</text>\n'
    b += f'  <text x="{x:.1f}" y="{PB-h-8}" text-anchor="middle" font-size="11.5" font-weight="700" fill="#9a3412">{fa*100:.0f}%</text>\n'
xs = fx(104.5)
b += f'  <line x1="{xs:.1f}" y1="{PB+4}" x2="{xs:.1f}" y2="{PB+6}" stroke="#dc2626" stroke-width="2.4" stroke-dasharray="5 3"/>\n'
b += f'  <rect x="{xs-96:.1f}" y="{PB+32}" width="192" height="26" rx="7" fill="#fee2e2" stroke="#dc2626" stroke-width="1.4"/>\n'
b += f'  <text x="{xs:.1f}" y="{PB+50}" text-anchor="middle" font-size="13" font-weight="700" fill="#b91c1c">T* = 1.74분 (활성 = 수면)</text>\n'
b += f'  <text x="{XL+58}" y="{PB+50}" text-anchor="middle" font-size="12.5" font-weight="700" fill="#e4711b">← 모델이 지배</text>\n'
b += f'  <text x="{XR-58}" y="{PB+50}" text-anchor="middle" font-size="12.5" font-weight="700" fill="#64748b">잠이 지배 →</text>\n'
b += '  <rect x="292" y="70" width="216" height="30" rx="6" fill="#ffffff" stroke="#cbd5e1"/>\n'
b += '  <rect x="302" y="79" width="13" height="12" fill="#e4711b"/>\n'
b += '  <text x="321" y="89" font-size="11.5" fill="#334155">추론이 쓴 전하</text>\n'
b += '  <rect x="404" y="79" width="13" height="12" fill="#e2e8f0"/>\n'
b += '  <text x="423" y="89" font-size="11.5" fill="#334155">잠자며 쓴 전하</text>\n'
b += note(96, 366, 624, 96, "T* = t (1 + Ia / Is) = 0.1초 × 1045 = 104.5초",
          ["Ia = 3.3 mA (nRF52840 · CoreMark @64MHz · 플래시 실행 · DC/DC · 3V)",
           "Is = 3.16 uA (System ON · 256kB RAM 전체 유지 · RTC 기상)",
           "조건을 안 밝히면 Ia 는 LDO 기준 6.3 mA 로, Is 는 RAM 미유지 기준 1.50 uA 로 두 배씩 갈린다"])
fig("w08_p2_breakeven_08", 800, 480, b)

# ═════════ 09. 손잡이의 역전 ═════════
b = title(400, 30, "같은 최적화가 같은 값을 주지 않는다",
          "각 항목을 절반으로 줄였을 때 배터리 수명 배수 (실측 산술)")
knob = {k["period_s"]: k for k in E["knobs"]}
groups = [(1, "1초마다"), (60, "1분마다"), (3600, "1시간마다")]
GX, GW = 78, 224
for gi, (T_, glab) in enumerate(groups):
    x0 = GX + gi * (GW + 18)
    k = knob[T_]
    b += f'  <rect x="{x0}" y="76" width="{GW}" height="242" rx="11" fill="#ffffff" stroke="#cbd5e1"/>\n'
    b += f'  <text x="{x0+GW/2}" y="102" text-anchor="middle" font-size="15" font-weight="700" fill="#0f172a">{glab} 깨는 기기</text>\n'
    bars = [("추론 시간 ½", k["halve_time"], "#028090"),
            ("수면 전류 ½", k["halve_isleep"], "#7c3aed")]
    base, H = 292, 140
    for bi, (lab, v, c) in enumerate(bars):
        bx = x0 + 36 + bi * 84
        h = (v - 1.0) / 1.0 * H
        h = max(h, 2)
        win = v == max(bars[0][1], bars[1][1])
        b += f'  <rect x="{bx}" y="{base-h:.1f}" width="56" height="{h:.1f}" rx="4" fill="{c}" fill-opacity="{0.92 if win else 0.35}"/>\n'
        b += f'  <text x="{bx+28}" y="{base-h-8:.1f}" text-anchor="middle" font-size="14" font-weight="700" fill="{c}">{v:.2f}배</text>\n'
        b += f'  <text x="{bx+28}" y="{base+16}" text-anchor="middle" font-size="11.5" fill="#334155">{lab}</text>\n'
    b += f'  <line x1="{x0+24}" y1="{base}" x2="{x0+GW-24}" y2="{base}" stroke="#94a3b8" stroke-width="1.4"/>\n'
b += note(78, 344, 668, 74, "1초 주기에서는 모델이, 1시간 주기에서는 잠이 이긴다",
          ["1.98 대 1.004  →  1.013 대 1.82. 두 막대의 높이가 완전히 뒤바뀐다",
           "배터리 수명을 늘리려면 모델을 보기 전에 듀티 사이클을 먼저 재야 한다"],
          bg="#fff7ed", ln="#f59e0b", hc="#b45309")
fig("w08_p2_knob_flip_09", 800, 434, b)

# ═════════ 10. 캐스케이드 ═════════
b = title(400, 30, "상시 동작은 2단으로 푼다", "값싼 문지기가 자주, 비싼 본 모델은 드물게")
b += '  <rect x="52" y="80" width="200" height="86" rx="11" fill="#e0f2fe" stroke="#0284c7" stroke-width="2"/>\n'
b += '  <text x="152" y="108" text-anchor="middle" font-size="15" font-weight="700" fill="#075985">문지기 (gate)</text>\n'
b += '  <text x="152" y="130" text-anchor="middle" font-size="12.5" fill="#334155">1초에 한 번 · 10 ms</text>\n'
b += '  <text x="152" y="152" text-anchor="middle" font-size="12" fill="#64748b">"사람 목소리 비슷한 게 있었나?"</text>\n'
b += '  <path d="M252 123 H317" stroke="#e4711b" stroke-width="2.6" marker-end="url(#f)"/>\n'
b += '  <text x="287" y="107" text-anchor="middle" font-size="12.5" font-weight="700" fill="#9a3412">통과율 p</text>\n'
b += '  <rect x="322" y="80" width="200" height="86" rx="11" fill="#fff7ed" stroke="#e4711b" stroke-width="2"/>\n'
b += '  <text x="422" y="108" text-anchor="middle" font-size="15" font-weight="700" fill="#9a3412">본 모델</text>\n'
b += '  <text x="422" y="130" text-anchor="middle" font-size="12.5" fill="#334155">통과할 때만 · 100 ms</text>\n'
b += '  <text x="422" y="152" text-anchor="middle" font-size="12" fill="#64748b">"그게 어떤 단어인가?"</text>\n'
b += '  <rect x="556" y="80" width="192" height="86" rx="11" fill="#f1f5f9" stroke="#94a3b8" stroke-width="1.6"/>\n'
b += '  <text x="652" y="106" text-anchor="middle" font-size="13" fill="#64748b">손익분기 통과율</text>\n'
b += '  <text x="652" y="138" text-anchor="middle" font-size="24" font-weight="700" fill="#b91c1c">90 %</text>\n'
b += '  <text x="652" y="158" text-anchor="middle" font-size="11.5" fill="#64748b">넘으면 오히려 손해</text>\n'
XL, XR, PT, PB = 96, 720, 196, 320
rowsC = E["cascade"]["rows"]
mxg = max(r["gain"] for r in rowsC)
bw2 = (XR - XL) / len(rowsC) * 0.62
for i, r in enumerate(rowsC):
    x = XL + (i + 0.5) * (XR - XL) / len(rowsC)
    h = r["gain"] / mxg * (PB - PT)
    c = "#16a34a" if r["gain"] > 1 else "#dc2626"
    b += f'  <rect x="{x-bw2/2:.1f}" y="{PB-h:.1f}" width="{bw2:.1f}" height="{h:.1f}" rx="4" fill="{c}" fill-opacity="0.88"/>\n'
    b += f'  <text x="{x:.1f}" y="{PB-h-8:.1f}" text-anchor="middle" font-size="12.5" font-weight="700" fill="{c}">{r["gain"]:.2f}배</text>\n'
    b += f'  <text x="{x:.1f}" y="{PB+18}" text-anchor="middle" font-size="11.5" fill="#334155">p={r["p"]:.2f}</text>\n'
b += f'  <line x1="{XL}" y1="{PB}" x2="{XR}" y2="{PB}" stroke="#94a3b8" stroke-width="1.5"/>\n'
b += f'  <line x1="{XL}" y1="{PB-1/mxg*(PB-PT):.1f}" x2="{XR}" y2="{PB-1/mxg*(PB-PT):.1f}" stroke="#dc2626" stroke-width="1.6" stroke-dasharray="5 4"/>\n'
b += f'  <rect x="{XR-158}" y="{PT+2}" width="158" height="24" rx="5" fill="#ffffff" stroke="#dc2626" stroke-width="1"/>\n'
b += f'  <path d="M{XR-148} {PT+14} H{XR-124}" stroke="#dc2626" stroke-width="1.8" stroke-dasharray="5 4"/>\n'
b += f'  <text x="{XR-116}" y="{PT+18}" font-size="11.5" font-weight="700" fill="#b91c1c">1.00배 = 본전</text>\n'
b += f'  <text x="{(XL+XR)/2:.0f}" y="{PB+38}" text-anchor="middle" font-size="12.5" font-weight="700" fill="#334155">문지기 통과율 p 에 따른 배터리 수명 배수 (항상 본 모델만 돌릴 때 대비)</text>\n'
b += note(96, 382, 624, 74, "문지기의 오경보율은 정확도 문제가 아니라 에너지 예산 문제다",
          ["재현율을 높이려고 문턱을 낮추면 p 가 올라가고, 수명이 6.30배에서 4.80배로 떨어진다",
           "설계 지침 — 문지기는 재현율 우선, 정밀도는 에너지 예산이 허용하는 만큼만 희생"],
          bg="#fff7ed", ln="#f59e0b", hc="#b45309")
b += '  <defs>' + MK.format(i="f", c="#e4711b") + '</defs>\n'
fig("w08_p2_cascade_10", 800, 472, b)

# ═════════ 11. MACs 대리 지표의 반전 ═════════
b = title(400, 30, "7주차의 결론이 8주차에서 뒤집힌다",
          "대리 지표의 유효성은 지표의 성질이 아니라 기기와 조건의 성질이다")
for x, bg, ln, tc, hd, wk, items in [
    (36, "#fef2f2", "#dc2626", "#b91c1c", "데스크톱 CPU", "7주차 실측",
     ["연산량 ~ 지연 Spearman  0.301",
      "역전 쌍 19.5 % — 다섯 쌍 중 한 쌍은",
      "연산량이 적은 쪽이 더 느리다",
      "깊이별 분리 합성곱이 MAC 당 7.6배 비싸다"]),
    (414, "#ecfdf5", "#16a34a", "#15803d", "MCU (Cortex-M)", "MicroNets, MLSys 2021",
     ["모델 지연이 연산 수에 선형",
      "0.95 &lt; r² &lt; 0.99",
      "캐시 계층·경쟁 프로세스가 없고",
      "텐서 아레나를 TCM 에 고정할 수 있다"])]:
    b += f'  <rect x="{x}" y="72" width="350" height="164" rx="12" fill="{bg}" stroke="{ln}" stroke-width="2"/>\n'
    b += f'  <text x="{x+175}" y="100" text-anchor="middle" font-size="16" font-weight="700" fill="{tc}">{hd}</text>\n'
    b += f'  <text x="{x+175}" y="120" text-anchor="middle" font-size="12" fill="#64748b">{wk}</text>\n'
    for i, s2 in enumerate(items):
        b += f'  <text x="{x+175}" y="{146+i*23}" text-anchor="middle" font-size="12.5" fill="#334155">{s2}</text>\n'
b += '  <text x="400" y="163" text-anchor="middle" font-size="15" font-weight="700" fill="#64748b">↔</text>\n'
b += note(36, 252, 728, 118, "그러나 조건 네 개가 붙는다 — 빼고 옮기면 틀린 말이 된다",
          ["① 레이어 단위에서는 성립하지 않는다 — 채널을 138→140 으로 늘렸는데 37.5 ms → 21.5 ms 로 빨라진 사례",
           "② 백본마다 기울기가 다르다 — KWS 와 CIFAR-10 백본의 처리량이 약 40 % 차이",
           "③ 소프트웨어·하드웨어 스택에 묶여 있다 — TFLM + CMSIS-NN, Cortex-M4/M7 에서 측정한 한에서",
           "④ 논문의 \"op\" 은 MAC 이 아니다 — 곱셈-누산 1회 = 2 연산. 그대로 옮기면 2배가 어긋난다"],
          bg="#fff7ed", ln="#f59e0b", hc="#b45309")
fig("w08_p2_macs_ok_11", 800, 386, b)

# ═════════ 12. 실습 파이프라인 ═════════
b = title(400, 30, "실습 일곱 단계", "MCU 보드 없이, 노트북과 ONNX 파일 하나로")
steps = [
    ("①", "ONNX 내보내기", "torch.onnx.export", "노드 209개"),
    ("②", "모양 추론", "infer_shapes", "중간 텐서 모양"),
    ("③", "상수 접기", "고정점 반복", "상수 176개 제거"),
    ("④", "크기 재기", "원소 수 × 1 B", "활성 101개"),
    ("⑤", "수명 매기기", "birth / death", "12줄"),
    ("⑥", "최대 메모리", "살아 있는 것만 합산", "1,470 KB"),
    ("⑦", "예산 판정", "320 KB 대비", "4.6배 초과"),
]
CWd, GAPd = 98, 14
for i, (n, t1, code, res) in enumerate(steps):
    x = 15 + i * (CWd + GAPd)
    b += f'  <rect x="{x}" y="76" width="{CWd}" height="150" rx="10" fill="#ffffff" stroke="#028090" stroke-width="1.8"/>\n'
    b += f'  <circle cx="{x+CWd/2}" cy="102" r="16" fill="#028090"/>\n'
    b += f'  <text x="{x+CWd/2}" y="108" text-anchor="middle" font-size="15" font-weight="700" fill="#ffffff">{n}</text>\n'
    b += f'  <text x="{x+CWd/2}" y="140" text-anchor="middle" font-size="12.5" font-weight="700" fill="#0b4a48">{t1}</text>\n'
    mono = all(ord(ch) < 128 for ch in code)          # 한글은 Courier 에 글리프가 없다
    ff = ' font-family="Courier New,monospace"' if mono else ''
    fs = (8.6 if len(code) > 14 else 10) if mono else 10.5
    b += (f'  <text x="{x+CWd/2}" y="164" text-anchor="middle" font-size="{fs}"'
          f'{ff} fill="#64748b">{code}</text>\n')
    b += f'  <rect x="{x+8}" y="180" width="{CWd-16}" height="34" rx="6" fill="#edf6f4"/>\n'
    b += f'  <text x="{x+CWd/2}" y="201" text-anchor="middle" font-size="11" font-weight="700" fill="#028090">{res}</text>\n'
    if i < 6:
        b += f'  <path d="M{x+CWd+3} 102 H{x+CWd+11}" stroke="#94a3b8" stroke-width="2" marker-end="url(#g)"/>\n'
b += note(15, 244, 770, 96, "③ 을 빼먹으면 가중치를 SRAM 으로 잘못 센다",
          ["torch.onnx.export 는 가중치 일부를 Identity 노드로 흘려보낸다. 209개 노드 중 109개가 그것이다.",
           "3주차에서 배운 상수 접기(Constant Folding)를 그대로 쓴다 — 고정점 반복은 11주차 툴체인에서 다시 나온다.",
           "⑤ 의 death[i] = k 한 줄이 영리하다. 덮어쓰기 때문에 반복이 끝나면 자동으로 '마지막' 소비 노드가 남는다."])
b += '  <defs>' + MK.format(i="g", c="#94a3b8") + '</defs>\n'
fig("w08_p3_pipeline_12", 800, 356, b)

# ═════════ 13. 논문 대조 ═════════
b = title(400, 30, "40줄이 논문의 값을 재현했다",
          "일치하는 것 하나, 그리고 차이가 설명되는 것 하나")
for x, bg, ln, tc, hd, ours, theirs, verdict, vc in [
    (48, "#ecfdf5", "#16a34a", "#15803d", "MobileNetV2 앞쪽 최대 메모리",
     "우리 6번 노드\n1,372.0 KB", "MCUNetV2 보고\n1,372 kB", "킬로바이트까지 일치", "#15803d"),
    (416, "#fff7ed", "#e4711b", "#9a3412", "int8 MobileNetV2 SRAM 초과 배수",
     "우리 측정\n4.6배", "MCUNet 보고\n5.3배", "14 % 차이 — 설명 가능", "#9a3412")]:
    b += f'  <rect x="{x}" y="72" width="336" height="196" rx="12" fill="{bg}" stroke="{ln}" stroke-width="2"/>\n'
    b += f'  <text x="{x+168}" y="100" text-anchor="middle" font-size="14.5" font-weight="700" fill="{tc}">{hd}</text>\n'
    for j, txt in enumerate([ours, theirs]):
        bx = x + 20 + j * 158
        b += f'  <rect x="{bx}" y="116" width="138" height="72" rx="9" fill="#ffffff" stroke="{ln}" stroke-width="1.2"/>\n'
        for li, lnn in enumerate(txt.split("\n")):
            fs = 12 if li == 0 else 20
            fw = "400" if li == 0 else "700"
            cc = "#64748b" if li == 0 else tc
            b += (f'  <text x="{bx+69}" y="{140+li*30}" text-anchor="middle" font-size="{fs}" '
                  f'font-weight="{fw}" fill="{cc}">{lnn}</text>\n')
    b += f'  <rect x="{x+20}" y="204" width="296" height="44" rx="8" fill="#ffffff" stroke="{ln}" stroke-width="1.2"/>\n'
    b += f'  <text x="{x+168}" y="232" text-anchor="middle" font-size="14" font-weight="700" fill="{vc}">{verdict}</text>\n'
b += note(48, 284, 704, 118, "차이가 설명되는 것이 일치보다 중요하다",
          ["우리는 순수한 텐서 메모리만 셌다. 실제 런타임은 여기에 다음을 더한다 —",
           "연산자 임시 버퍼(합성곱의 im2col) · 메모리 정렬 여백 · 인터프리터 자료구조와 스택",
           "그래서 우리 값은 언제나 하한이다. \"우리 계산으로 300 KB니까 320 KB에 들어간다\" 는 위험한 판정이다"])
fig("w08_p3_verify_13", 800, 418, b)

for k, v in F.items():
    (OUT / f"{k}.svg").write_text(v, encoding="utf-8")
print(f"{len(F)}개 저장 → {OUT}")
