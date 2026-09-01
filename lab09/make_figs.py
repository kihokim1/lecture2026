# -*- coding: utf-8 -*-
"""9주차 그림 — 개념 도해 + pipeline/realtime/queue.json 기반 실측 차트."""
import json, math, pathlib

D = pathlib.Path("/root/lab09")
P = json.load(open(D / "pipeline.json"))
R = json.load(open(D / "realtime.json"))
Q = json.load(open(D / "queue.json"))
OUT = pathlib.Path("/root/ondevice-ai/img/week09"); OUT.mkdir(parents=True, exist_ok=True)

HEAD = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        'font-family="\'Segoe UI\',Arial,sans-serif">\n'
        '  <rect width="{w}" height="{h}" fill="#f8fafc"/>\n')
F = {}


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
    for i, l in enumerate(body):
        s += f'  <text x="{x+w/2}" y="{y+50+i*20}" text-anchor="middle" font-size="12.5" fill="#334155">{l}</text>\n'
    return s


MK = ('<marker id="{i}" markerWidth="9" markerHeight="9" refX="7" refY="3.5" orient="auto">'
      '<path d="M0,0 L7,3.5 L0,7 Z" fill="{c}"/></marker>')

ST = P["stages"]["640"]["stat"]
NAMES = ["① JPEG 디코드", "② 레터박스 리사이즈", "③ 정규화·축 변환", "④ 추론",
         "⑤ 상자 디코딩", "⑥ NMS", "⑦ 좌표 복원"]
TOT = P["stages"]["640"]["total"]

# ═════════ 01. 세 가지 지표 ═════════
b = title(400, 30, "\"30 FPS\" 는 무엇을 보장하는가", "세 가지 서로 다른 것이 한 이름으로 불린다")
for i, (nm, en, d1, d2, c, bg) in enumerate([
    ("처리량", "throughput", "초당 처리한 프레임 수", "서버가 하루에 몇 장을 볼 수 있나", "#64748b", "#f1f5f9"),
    ("지연", "latency", "한 프레임 처리에 걸린 시간", "모델 한 번 돌리는 데 얼마나", "#2563eb", "#eff6ff"),
    ("프레임 나이", "frame age", "빛이 센서에 닿은 순간부터 결과까지", "내 판단이 얼마나 오래된 세상에 대한 것인가", "#e4711b", "#fff7ed")]):
    x = 22 + i * 254
    hi = i == 2
    b += (f'  <rect x="{x}" y="76" width="238" height="150" rx="12" fill="{bg}" '
          f'stroke="{c}" stroke-width="{3 if hi else 1.8}"/>\n')
    b += f'  <text x="{x+119}" y="108" text-anchor="middle" font-size="18" font-weight="700" fill="{c}">{nm}</text>\n'
    b += f'  <text x="{x+119}" y="130" text-anchor="middle" font-size="12" fill="#64748b" font-style="italic">{en}</text>\n'
    b += f'  <text x="{x+119}" y="164" text-anchor="middle" font-size="12.5" fill="#0f172a">{d1}</text>\n'
    for j, ln in enumerate([d2]):
        b += f'  <text x="{x+119}" y="{194+j*18}" text-anchor="middle" font-size="11.5" fill="#334155">{ln}</text>\n'
    if hi:
        b += f'  <text x="{x+119}" y="248" text-anchor="middle" font-size="13" font-weight="700" fill="{c}">← 실시간 지각에서 중요한 것</text>\n'
b += note(22, 262, 756, 96, "지연을 평가에 넣으면 같은 모델의 AP 가 6분의 1이 된다",
          ["Li, Wang, Ramanan, 「Towards Streaming Perception」 (ECCV 2020 최우수 논문 가작)",
           "\"알고리즘이 프레임 처리를 끝냈을 때, 주변 세상은 이미 변해 있다\"    AP  38.0  →  6.2",
           "GPU 를 무한히 준다고 가정해도 20.3 — 하드웨어가 아니라 구조의 문제다"],
          bg="#fff7ed", ln="#f59e0b", hc="#b45309")
fig("w09_p1_metrics_01", 800, 372, b)

# ═════════ 02. 일곱 단계 ═════════
b = title(400, 30, "실시간 비전 파이프라인의 일곱 단계", "모델은 일곱 조각 중 하나다")
steps = [("①", "JPEG\n디코드", "바이트 → RGB"), ("②", "레터박스\n리사이즈", "1080×810 → 640²"),
         ("③", "정규화\n축 변환", "uint8 → float32"), ("④", "추론", "→ (1,84,8400)"),
         ("⑤", "상자\n디코딩", "8400 → 46"), ("⑥", "NMS", "46 → 5"),
         ("⑦", "좌표\n복원", "640² → 원본")]
CW, GP = 96, 12
for i, (n, t1, d) in enumerate(steps):
    x = 22 + i * (CW + GP)
    hot = i == 3
    b += (f'  <rect x="{x}" y="76" width="{CW}" height="130" rx="11" '
          f'fill="{"#fff7ed" if hot else "#ffffff"}" stroke="{"#e4711b" if hot else "#028090"}" '
          f'stroke-width="{2.4 if hot else 1.6}"/>\n')
    b += f'  <circle cx="{x+CW/2}" cy="102" r="15" fill="{"#e4711b" if hot else "#028090"}"/>\n'
    b += f'  <text x="{x+CW/2}" y="108" text-anchor="middle" font-size="14" font-weight="700" fill="#ffffff">{n}</text>\n'
    for j, ln in enumerate(t1.split("\n")):
        b += (f'  <text x="{x+CW/2}" y="{140+j*17}" text-anchor="middle" font-size="12.5" '
              f'font-weight="700" fill="{"#9a3412" if hot else "#0b4a48"}">{ln}</text>\n')
    b += f'  <text x="{x+CW/2}" y="192" text-anchor="middle" font-size="9.6" fill="#64748b">{d}</text>\n'
    if i < 6:
        b += f'  <path d="M{x+CW+2} 102 H{x+CW+9}" stroke="#94a3b8" stroke-width="2" marker-end="url(#a)"/>\n'
b += ('  <text x="22" y="234" font-size="12.5" fill="#64748b">카메라</text>\n'
      '  <text x="778" y="234" text-anchor="end" font-size="12.5" fill="#64748b">화면·판단</text>\n'
      '  <path d="M70 230 H672" stroke="#cbd5e1" stroke-width="1.4" stroke-dasharray="4 4" marker-end="url(#a)"/>\n'
      '  <text x="370" y="224" text-anchor="middle" font-size="12.5" font-weight="700" fill="#e4711b">프레임 나이</text>\n')
b += note(22, 252, 756, 96, "라이브러리에 맡기면 편하지만, 그 순간 관측 불가능해진다",
          ["⑤⑥⑦ 을 라이브러리가 대신 해 주면 그 비용을 따로 잴 수 없다 — 그래서 직접 짰다",
           "내보낼 때 nms=False 를 주지 않으면 NMS 가 그래프 안으로 들어가 ④ 에 섞여 버린다",
           "편리함은 관측 불가능성과 맞바꾸는 것이다"])
b += '  <defs>' + MK.format(i="a", c="#94a3b8") + '</defs>\n'
fig("w09_p1_pipeline_02", 800, 364, b)

# ═════════ 03. 단계별 시간 ═════════
b = title(400, 30, "시간은 어디로 갔나", "YOLO11n · 640² · CPU 2스레드 · 30프레임 p50 (실측)")
XL, XR, PT = 176, 748, 78
RH, GAP = 30, 8
mx = max(ST[n]["p50"] for n in NAMES)
for i, n in enumerate(NAMES):
    y = PT + i * (RH + GAP)
    v = ST[n]["p50"]
    w = max(v / mx * (XR - XL), 2)
    hot = n == "④ 추론"
    c = "#e4711b" if hot else ("#dc2626" if n == "⑥ NMS" else "#028090")
    b += f'  <text x="{XL-10}" y="{y+20}" text-anchor="end" font-size="12.5" font-weight="{700 if hot else 400}" fill="#334155">{n}</text>\n'
    b += f'  <rect x="{XL}" y="{y}" width="{w:.1f}" height="{RH}" rx="4" fill="{c}" fill-opacity="0.9"/>\n'
    lx = XL + w + 8 if w < 460 else XL + w - 8
    anc = "start" if w < 460 else "end"
    fc = c if w < 460 else "#ffffff"
    b += (f'  <text x="{lx:.1f}" y="{y+20}" text-anchor="{anc}" font-size="12.5" font-weight="700" '
          f'fill="{fc}">{v:.2f} ms · {v/TOT*100:.1f} %</text>\n')
b += f'  <line x1="{XL}" y1="{PT-6}" x2="{XL}" y2="{PT+7*(RH+GAP)-4}" stroke="#94a3b8" stroke-width="1.4"/>\n'
b += note(64, 348, 672, 96, f"합계 {TOT:.1f} ms = {1000/TOT:.1f} FPS",
          ["추론 73.0 %   ·   나머지 여섯 단계 27.0 % (13.2 ms)   ·   NMS 0.2 %",
           "\"탐지의 병목\" 이라 배워 온 NMS 가 0.10 ms 다 — 2교시에서 정면으로 다룬다",
           "그리고 13.2 ms 는 무시할 수 없다. 이 값이 다음 그림의 벽을 만든다"])
fig("w09_p1_stages_03", 800, 460, b)

# ═════════ 04. 해상도별 비중 ═════════
b = title(400, 30, "모델이 빨라질수록 나머지가 지배한다", "해상도만 낮췄을 때 추론 비중의 붕괴 (실측)")
XL, PB, PT = 96, 288, 84
SZ = ["640", "512", "416", "320", "256"]
mxt = P["stages"]["640"]["total"]
BW = 84
for i, s in enumerate(SZ):
    x = XL + i * 128
    st = P["stages"][s]
    hi = (PB - PT) * st["infer"] / mxt
    ho = (PB - PT) * st["other"] / mxt
    b += f'  <rect x="{x}" y="{PB-hi-ho:.1f}" width="{BW}" height="{ho:.1f}" rx="3" fill="#94a3b8"/>\n'
    b += f'  <rect x="{x}" y="{PB-hi:.1f}" width="{BW}" height="{hi:.1f}" rx="3" fill="#e4711b" fill-opacity="0.92"/>\n'
    b += f'  <text x="{x+BW/2}" y="{PB-hi/2+5:.1f}" text-anchor="middle" font-size="13" font-weight="700" fill="#ffffff">{st["infer"]/st["total"]*100:.0f}%</text>\n'
    b += f'  <text x="{x+BW/2}" y="{PB-hi-ho-10:.1f}" text-anchor="middle" font-size="12.5" font-weight="700" fill="#334155">{st["total"]:.1f} ms</text>\n'
    b += f'  <text x="{x+BW/2}" y="{PB+20}" text-anchor="middle" font-size="13" font-weight="700" fill="#0f172a">{s}²</text>\n'
b += f'  <line x1="{XL-8}" y1="{PB}" x2="{XL+5*128-32}" y2="{PB}" stroke="#94a3b8" stroke-width="1.5"/>\n'
b += '  <rect x="612" y="92" width="150" height="52" rx="6" fill="#ffffff" stroke="#cbd5e1"/>\n'
b += '  <rect x="622" y="102" width="14" height="13" fill="#e4711b"/>\n'
b += '  <text x="643" y="113" font-size="11.5" fill="#334155">추론</text>\n'
b += '  <rect x="622" y="122" width="14" height="13" fill="#94a3b8"/>\n'
b += '  <text x="643" y="133" font-size="11.5" fill="#334155">나머지 여섯 단계</text>\n'
b += note(96, 328, 608, 96, "추론은 5.6배 빨라졌는데 나머지는 1.6배밖에 안 줄었다",
          ["JPEG 디코드는 원본 이미지 크기에 달려 있어 출력 해상도와 거의 무관하다",
           "그래서 256² 에서는 모델이 소수파가 된다 — 43 % 대 57 %",
           "8주차의 해상도 손잡이가 여기서는 다른 이유로 중요해진다"],
          bg="#fff7ed", ln="#f59e0b", hc="#b45309")
fig("w09_p1_share_04", 800, 442, b)

# ═════════ 05. 암달의 벽 ═════════
b = title(400, 30, "추론을 0초로 만들어도 3.71배가 상한이다",
          "640² · 49.0 ms 기준, 추론만 빨라질 때 파이프라인 전체 배수 (실측)")
XL, XR, PT, PB = 108, 700, 84, 292
rows = [(1, "그대로"), (2, "2배 빨라지면"), (4, "4배"), (8, "8배"), (None, "0초가 되면")]
amd = {a["k"]: a for a in P["amdahl"]}
mxsp = 4.0
for i, (k, lab) in enumerate(rows):
    y = PT + i * 42
    a = amd[k if k != 1 else 1]
    sp = a["speedup"]
    w = sp / mxsp * (XR - XL)
    last = k is None
    c = "#dc2626" if last else "#028090"
    b += f'  <text x="{XL-10}" y="{y+21}" text-anchor="end" font-size="12.5" font-weight="{700 if last else 400}" fill="#334155">{lab}</text>\n'
    b += f'  <rect x="{XL}" y="{y}" width="{w:.1f}" height="30" rx="4" fill="{c}" fill-opacity="{0.95 if last else 0.85}"/>\n'
    b += f'  <text x="{XL+w+8:.1f}" y="{y+21}" font-size="13" font-weight="700" fill="{c}">{sp:.2f}배 · {a["total"]:.1f} ms</text>\n'
wall = 3.71 / mxsp * (XR - XL) + XL
b += f'  <line x1="{wall:.1f}" y1="{PT-12}" x2="{wall:.1f}" y2="{PB}" stroke="#dc2626" stroke-width="2.4" stroke-dasharray="6 4"/>\n'
b += f'  <text x="{wall+8:.1f}" y="{PT-18}" font-size="13" font-weight="700" fill="#b91c1c">넘을 수 없는 벽</text>\n'
b += note(108, 316, 588, 96, "\"병렬 처리 속도를 높이는 데 들인 노력은",
          ["순차 처리 속도에서도 거의 같은 크기의 성취가 동반되지 않으면 낭비된다\"",
           "— Gene Amdahl, AFIPS 1967 (원논문에는 수식이 하나도 없다)",
           "4~8주차 다섯 주 동안 배운 모든 모델 최적화의 상한이 이 파이프라인에서 3.71배다"])
fig("w09_p1_amdahl_05", 800, 442, b)

# ═════════ 06. 문턱과 NMS 비용 ═════════
b = title(400, 30, "NMS 는 16.6배 늘었는데 전체는 1.00배다",
          "신뢰도 문턱을 0.25 에서 0.001 까지 · 두 계열은 같은 축이다 (실측)")
XL, XR, PB, PT = 132, 748, 288, 88
CF = R["conf"]
MXY = 60.0
sy = lambda v: PB - v / MXY * (PB - PT)
for g in [0, 10, 20, 30, 40, 50, 60]:
    b += f'  <line x1="{XL}" y1="{sy(g):.1f}" x2="{XR}" y2="{sy(g):.1f}" stroke="#e2e8f0"/>\n'
    b += f'  <text x="{XL-8}" y="{sy(g)+4:.1f}" text-anchor="end" font-size="11" fill="#94a3b8">{g}</text>\n'
b += f'  <text x="34" y="{(PT+PB)/2}" text-anchor="middle" font-size="12.5" font-weight="700" fill="#334155" transform="rotate(-90 34 {(PT+PB)/2})">밀리초 (ms)</text>\n'
for i, c in enumerate(CF):
    x = XL + 30 + i * 122
    hn = PB - sy(c["nms"])
    ht = PB - sy(c["total"])
    b += f'  <rect x="{x}" y="{sy(c["total"]):.1f}" width="44" height="{ht:.1f}" rx="3" fill="#94a3b8"/>\n'
    b += f'  <text x="{x+22}" y="{sy(c["total"])-8:.1f}" text-anchor="middle" font-size="11.5" font-weight="700" fill="#475569">{c["total"]:.1f}</text>\n'
    b += f'  <rect x="{x+54}" y="{sy(c["nms"]):.1f}" width="44" height="{max(hn,2.5):.1f}" rx="2" fill="#dc2626"/>\n'
    b += f'  <text x="{x+76}" y="{sy(c["nms"])-8:.1f}" text-anchor="middle" font-size="11.5" font-weight="700" fill="#b91c1c">{c["nms"]:.3f}</text>\n'
    b += f'  <text x="{x+49}" y="{PB+20}" text-anchor="middle" font-size="12" font-weight="700" fill="#0f172a">conf {c["conf"]}</text>\n'
    b += f'  <text x="{x+49}" y="{PB+38}" text-anchor="middle" font-size="11.5" fill="#64748b">후보 {c["n_cand"]}개</text>\n'
b += f'  <line x1="{XL}" y1="{PB}" x2="{XR}" y2="{PB}" stroke="#94a3b8" stroke-width="1.5"/>\n'
b += '  <rect x="248" y="352" width="304" height="28" rx="6" fill="#ffffff" stroke="#cbd5e1"/>\n'
b += '  <rect x="260" y="359" width="13" height="13" fill="#94a3b8"/>\n'
b += '  <text x="280" y="370" font-size="11.5" fill="#334155">파이프라인 전체</text>\n'
b += '  <rect x="392" y="359" width="13" height="13" fill="#dc2626"/>\n'
b += '  <text x="412" y="370" font-size="11.5" fill="#334155">NMS</text>\n'
b += note(96, 396, 608, 96, "빨간 막대는 거의 보이지 않는다 — 그것이 이 그림의 전부다",
          ["NMS 0.119 → 1.970 ms (16.6배)   ·   파이프라인 55.8 → 55.7 ms (1.00배)",
           "붐비는 장면(3×3 타일)으로도 확인 — NMS 4.8배, 전체 1.04배",
           "우리 파이프라인에서 NMS 는 손댈 대상이 아니다. 그런데 논문은 병목이라고 말한다"],
          bg="#fff7ed", ln="#f59e0b", hc="#b45309")
fig("w09_p2_nms_cost_06", 800, 510, b)

# ═════════ 07. 분모 ═════════
b = title(400, 30, "같은 NMS, 반대 결론", "바뀐 것은 NMS 가 아니라 분모다")
cases = [("우리 (CPU · YOLO11n · 640²)", 35.8, 0.10, "0.2 %", "손댈 필요 없다", "#028090"),
         ("SSD300 (논문 §3.7 · GPU)", 20.0, 1.7, "약 8 %", "미미하다", "#64748b"),
         ("YOLOv10-N (T4 + TensorRT)", 1.84, 4.35, "70 %", "최우선 과제다", "#dc2626")]
XL, XR = 250, 700
mxbar = 36.0
for i, (nm, back, nmsv, pct, verdict, c) in enumerate(cases):
    y = 86 + i * 74
    tot = back + nmsv
    wb = back / mxbar * (XR - XL)
    wn = max(nmsv / mxbar * (XR - XL), 3)
    b += f'  <text x="{XL-10}" y="{y+20}" text-anchor="end" font-size="12.5" font-weight="700" fill="#334155">{nm}</text>\n'
    b += f'  <rect x="{XL}" y="{y}" width="{wb:.1f}" height="30" rx="3" fill="#94a3b8"/>\n'
    b += f'  <rect x="{XL+wb:.1f}" y="{y}" width="{wn:.1f}" height="30" rx="3" fill="#dc2626"/>\n'
    b += f'  <text x="{XL+wb+wn+10:.1f}" y="{y+20}" font-size="13" font-weight="700" fill="{c}">NMS {pct}</text>\n'
    b += f'  <text x="{XL-10}" y="{y+38}" text-anchor="end" font-size="11.5" fill="#64748b">백본 {back} ms · NMS {nmsv} ms</text>\n'
    b += f'  <text x="{XL+wb+wn+10:.1f}" y="{y+38}" font-size="11.5" fill="#64748b">{verdict}</text>\n'
b += '  <rect x="560" y="52" width="200" height="26" rx="5" fill="#ffffff" stroke="#cbd5e1"/>\n'
b += '  <rect x="570" y="59" width="13" height="12" fill="#94a3b8"/>\n'
b += '  <text x="589" y="69" font-size="11.5" fill="#334155">백본</text>\n'
b += '  <rect x="640" y="59" width="13" height="12" fill="#dc2626"/>\n'
b += '  <text x="659" y="69" font-size="11.5" fill="#334155">NMS</text>\n'
b += note(60, 316, 680, 96, "NMS 의 절대 시간은 0.1 / 1.7 / 4.35 ms — 같은 자릿수 안에 있다",
          ["TensorRT 로 순전파를 1.84 ms 로 줄여 놓으면, 그 순간 밀리초짜리 NMS 가 지배자가 된다",
           "\"NMS 는 병목이다\" 도 \"NMS 는 병목이 아니다\" 도 그 자체로는 틀린 문장이다",
           "DETR 은 NMS 가 느리다고 말한 적이 없다 — 손수 설계된 구성 요소와 참 양성 삭제가 논거다"])
fig("w09_p2_denominator_07", 800, 434, b)

# ═════════ 08. 구현 ═════════
b = title(400, 30, "같은 알고리즘, 두 가지 구현", "결과는 한 개도 안 틀리고 속도만 11.3배 갈린다 · 세로축 로그 (실측)")
IM = Q["nms_impl"]
XL, XR, PB, PT = 150, 700, 288, 96
LO, HI = 0.03, 30.0
sy = lambda v: PB - (math.log10(max(v, LO) / LO) / math.log10(HI / LO)) * (PB - PT)
for g in [0.03, 0.1, 0.3, 1, 3, 10, 30]:
    b += f'  <line x1="{XL}" y1="{sy(g):.1f}" x2="{XR}" y2="{sy(g):.1f}" stroke="#e2e8f0"/>\n'
    b += f'  <text x="{XL-8}" y="{sy(g)+4:.1f}" text-anchor="end" font-size="11" fill="#94a3b8">{g:g}</text>\n'
b += f'  <text x="52" y="{(PT+PB)/2}" text-anchor="middle" font-size="12.5" font-weight="700" fill="#334155" transform="rotate(-90 52 {(PT+PB)/2})">밀리초 (로그)</text>\n'
for i, r in enumerate(IM):
    x = XL + 44 + i * 176
    for j, (v, c, lab) in enumerate([(r["vec"], "#028090", f'{r["vec"]:.3f}'),
                                     (r["naive"], "#dc2626", f'{r["naive"]:.2f}')]):
        bx = x + j * 62
        b += f'  <rect x="{bx}" y="{sy(v):.1f}" width="52" height="{PB-sy(v):.1f}" rx="3" fill="{c}" fill-opacity="0.9"/>\n'
        b += f'  <text x="{bx+26}" y="{sy(v)-8:.1f}" text-anchor="middle" font-size="12" font-weight="700" fill="{c}">{lab}</text>\n'
    b += f'  <text x="{x+57}" y="{PB+20}" text-anchor="middle" font-size="12.5" font-weight="700" fill="#0f172a">후보 {r["n"]}개</text>\n'
    b += f'  <text x="{x+57}" y="{PB+40}" text-anchor="middle" font-size="13" font-weight="700" fill="#b45309">{r["ratio"]:.1f}배</text>\n'
    b += f'  <text x="{x+57}" y="{PB+58}" text-anchor="middle" font-size="11" fill="#64748b">남은 상자 {r["keep_vec"]} = {r["keep_naive"]}</text>\n'
b += f'  <line x1="{XL}" y1="{PB}" x2="{XR}" y2="{PB}" stroke="#94a3b8" stroke-width="1.5"/>\n'
b += '  <rect x="228" y="356" width="344" height="28" rx="6" fill="#ffffff" stroke="#cbd5e1"/>\n'
b += '  <rect x="240" y="363" width="13" height="13" fill="#028090"/>\n'
b += '  <text x="260" y="374" font-size="11.5" fill="#334155">넘파이 벡터화</text>\n'
b += '  <rect x="392" y="363" width="13" height="13" fill="#dc2626"/>\n'
b += '  <text x="412" y="374" font-size="11.5" fill="#334155">파이썬 이중 루프</text>\n'
b += note(80, 400, 640, 96, "798개에서 19.9 ms — 이제 전체의 29 % 다",
          ["알고리즘을 안 바꾸고 구현만 바꿔서 병목이 되거나 안 되거나 한다",
           "전처리도 같다 — PIL 13.12 ms → OpenCV 6.69 ms (단계 1.96배, 파이프라인 1.13배)",
           "그리고 동적 INT8 양자화는 크기를 3.5배 줄이면서 속도를 1.5배 나쁘게 만들었다"])
fig("w09_p2_impl_08", 800, 514, b)

# ═════════ 09. 배치 ═════════
b = title(400, 30, "처리량 1.24배를 사는 데 지연 6.5배를 지불한다", "320² · 배치를 키웠을 때 (실측)")
BT = R["batch"]
XL, PB, PT = 128, 272, 92
mxl = max(x["batch_ms"] for x in BT)
mxf = max(x["fps"] for x in BT)
for i, r in enumerate(BT):
    x = XL + i * 152
    hl = r["batch_ms"] / mxl * (PB - PT)
    hf = r["fps"] / mxf * (PB - PT)
    b += f'  <rect x="{x}" y="{PB-hf:.1f}" width="52" height="{hf:.1f}" rx="3" fill="#028090" fill-opacity="0.9"/>\n'
    b += f'  <text x="{x+26}" y="{PB-hf-8:.1f}" text-anchor="middle" font-size="12" font-weight="700" fill="#028090">{r["fps"]:.0f}</text>\n'
    b += f'  <rect x="{x+66}" y="{PB-hl:.1f}" width="52" height="{hl:.1f}" rx="3" fill="#dc2626" fill-opacity="0.9"/>\n'
    b += f'  <text x="{x+92}" y="{PB-hl-8:.1f}" text-anchor="middle" font-size="12" font-weight="700" fill="#b91c1c">{r["batch_ms"]:.1f}</text>\n'
    b += f'  <text x="{x+59}" y="{PB+20}" text-anchor="middle" font-size="13" font-weight="700" fill="#0f172a">배치 {r["B"]}</text>\n'
b += f'  <line x1="{XL-14}" y1="{PB}" x2="{XL+4*152-30}" y2="{PB}" stroke="#94a3b8" stroke-width="1.5"/>\n'
b += '  <rect x="196" y="312" width="408" height="28" rx="6" fill="#ffffff" stroke="#cbd5e1"/>\n'
b += '  <rect x="208" y="319" width="13" height="13" fill="#028090"/>\n'
b += '  <text x="228" y="330" font-size="11.5" fill="#334155">처리량 (FPS) — 오른다</text>\n'
b += '  <rect x="404" y="319" width="13" height="13" fill="#dc2626"/>\n'
b += '  <text x="424" y="330" font-size="11.5" fill="#334155">마지막 프레임 대기 (ms) — 나빠진다</text>\n'
b += note(80, 356, 640, 96, "서버에서는 남는 장사, 카메라 한 대에서는 살 것이 없다",
          ["배치 8 을 채우려면 프레임 여덟 장이 모여야 하고, 30 fps 카메라에서 그것은 233 ms 다",
           "처리량을 24 % 올리려고 프레임 나이에 233 ms 를 얹는 셈이다",
           "Clipper (NSDI 2017) 는 서버에서 배치로 처리량을 최대 26배까지 올린 사례를 보고한다"],
          bg="#fff7ed", ln="#f59e0b", hc="#b45309")
fig("w09_p2_batch_09", 800, 470, b)

# ═════════ 10. 큐 ═════════
b = title(400, 30, "「드롭 0」 이 최악의 지표다",
          "30 fps 카메라 · 10초 · 실측 지연 분포 기반 시뮬레이션")
QQ = Q["queue"]
rowsQ = [("640²", "다 쌓는다", QQ["640_queue"], "#dc2626"),
         ("640²", "최신만 남긴다", QQ["640_latest"], "#16a34a"),
         ("320²", "다 쌓는다", QQ["320_queue"], "#94a3b8"),
         ("320²", "최신만 남긴다", QQ["320_latest"], "#94a3b8")]
X0, Y0, RH2 = 40, 84, 46
CWs = [72, 116, 106, 92, 148, 148]
hdr = ["해상도", "정책", "처리한 장수", "버린 장수", "프레임 나이 p50", "마지막 프레임 나이"]
x = X0
for j, hh in enumerate(hdr):
    b += f'  <rect x="{x}" y="{Y0}" width="{CWs[j]}" height="34" fill="#0e4a44"/>\n'
    b += f'  <text x="{x+CWs[j]/2}" y="{Y0+22}" text-anchor="middle" font-size="12" font-weight="700" fill="#ffffff">{hh}</text>\n'
    x += CWs[j]
for i, (sz, pol, r, c) in enumerate(rowsQ):
    y = Y0 + 34 + i * RH2
    bg = "#fef2f2" if i == 0 else ("#ecfdf5" if i == 1 else "#f8fafc")
    vals = [sz, pol, f'{r["processed"]}장 ({r["fps_out"]:.1f} fps)', f'{r["dropped"]}장',
            f'{r["age_p50"]:,.0f} ms', f'{r["age_last"]:,.0f} ms']
    x = X0
    for j, v in enumerate(vals):
        b += f'  <rect x="{x}" y="{y}" width="{CWs[j]}" height="{RH2}" fill="{bg}" stroke="#d3e5e2"/>\n'
        big = j >= 4 and i < 2
        b += (f'  <text x="{x+CWs[j]/2}" y="{y+RH2/2+6}" text-anchor="middle" '
              f'font-size="{14.5 if big else 12}" font-weight="{700 if (big or j<2) else 400}" '
              f'fill="{c if big else "#334155"}">{v}</text>\n')
        x += CWs[j]
b += ('  <text x="704" y="146" font-size="12.5" font-weight="700" fill="#b91c1c">← 지표는 완벽</text>\n'
      '  <text x="704" y="192" font-size="12.5" font-weight="700" fill="#15803d">← 지표는 나쁨</text>\n')
b += note(40, 322, 720, 116, "처리 능력 20.4 fps  ·  입력 30 fps  →  매초 9.6장이 남는다",
          ["첫 줄 — 300장 다 처리, 한 장도 안 버림, 30.0 fps. 그런데 4.7초 전 장면을 보고 판단한다",
           "둘째 줄 — 96장을 버렸다. 그런데 프레임 나이가 66 ms 로 일정하다",
           "셋째·넷째 줄 — 처리 능력이 입력률을 넘으면 정책 논쟁 자체가 사라진다",
           "「Towards Streaming Perception」의 AP 38.0 → 6.2 와 같은 현상이다"],
          bg="#fff7ed", ln="#f59e0b", hc="#b45309")
fig("w09_p2_queue_10", 800, 458, b)

# ═════════ 11. 세그멘테이션 34% 분해 ═════════
b = title(400, 30, "\"34% 빨라졌다\" 를 분해하면", "MobileNetV3 논문 Table 7 · Cityscapes · Pixel 3 단일 코어")
XL, XR = 306, 556
bars = [("MobileNetV2 + R-ASPP", 3.90, "#dc2626", "72.84"),
        ("V2 + LR-ASPP (헤드만 교체)", 2.98, "#e4711b", "72.97"),
        ("V3 + R-ASPP", 2.60, "#0284c7", "71.91"),
        ("MobileNetV3 + LR-ASPP", 2.55, "#16a34a", "72.37")]
for i, (nm, v, c, miou) in enumerate(bars):
    y = 88 + i * 52
    w = v / 4.0 * (XR - XL)
    b += f'  <text x="{XL-10}" y="{y+22}" text-anchor="end" font-size="12.5" font-weight="700" fill="#334155">{nm}</text>\n'
    b += f'  <rect x="{XL}" y="{y}" width="{w:.1f}" height="32" rx="4" fill="{c}" fill-opacity="0.9"/>\n'
    b += f'  <text x="{XL+w+10:.1f}" y="{y+22}" font-size="13" font-weight="700" fill="{c}">{v} s · mIOU {miou}</text>\n'
    if i in (0, 3):
        b += f'  <rect x="736" y="{y+4}" width="44" height="24" rx="6" fill="#fef3c7" stroke="#b45309"/>\n'
        b += f'  <text x="758" y="{y+21}" text-anchor="middle" font-size="12.5" font-weight="700" fill="#b45309">{"①" if i==0 else "②"}</text>\n'
b += ('  <path d="M758 120 V248" stroke="#b45309" stroke-width="1.6" stroke-dasharray="5 3" fill="none"/>\n'
      '  <text x="620" y="308" text-anchor="middle" font-size="13" font-weight="700" fill="#b45309">① → ② 가 초록의 34%</text>\n')
b += note(48, 324, 704, 116, "34% 안에는 세 가지 변경이 섞여 있다",
          ["① 백본 MobileNetV2 → V3     ② 마지막 블록 채널 반감(RF2)     ③ 헤드 R-ASPP → LR-ASPP",
           "헤드만 바꾼 두 쌍(1↔2, 3↔4)을 보면 약 2 % 빨라지고 mIOU 가 0.4 오른다",
           "논문 자신도 그렇게 적었다 — \"LR-ASPP 는 R-ASPP 보다 약간 더 빠르면서 성능은 개선된다\"",
           "6주차 이득 분해 · 8주차 MCUNetV2 네 수치에 이은 세 번째 사례다"],
          bg="#fff7ed", ln="#f59e0b", hc="#b45309")
fig("w09_p2_seg_11", 800, 460, b)

# ═════════ 12. 시뮬레이터 구조 ═════════
b = title(400, 30, "큐 시뮬레이터의 네 부분", "실제 카메라 없이 프레임 나이를 재는 법")
parts = [("①", "도착", "arr = np.arange(0, dur, 1/fps)", "카메라가 33.3 ms 마다 민다", "#2563eb", "#eff6ff"),
         ("②", "큐 정책", "if policy=='latest': q = []", "다 쌓거나, 오래된 것을 버리거나", "#7c3aed", "#ede9fe"),
         ("③", "서비스", "now += svc_ms * lognormal()", "실측 p50 과 흔들림을 넣는다", "#028090", "#edf6f4"),
         ("④", "나이 측정", "ages.append(now - born)", "처리 시간이 아니라 나이다", "#e4711b", "#fff7ed")]
for i, (n, t1, code, d, c, bg) in enumerate(parts):
    x = 22 + i * 192
    b += f'  <rect x="{x}" y="76" width="176" height="176" rx="12" fill="{bg}" stroke="{c}" stroke-width="2"/>\n'
    b += f'  <circle cx="{x+88}" cy="108" r="18" fill="{c}"/>\n'
    b += f'  <text x="{x+88}" y="115" text-anchor="middle" font-size="16" font-weight="700" fill="#ffffff">{n}</text>\n'
    b += f'  <text x="{x+88}" y="150" text-anchor="middle" font-size="15" font-weight="700" fill="{c}">{t1}</text>\n'
    b += f'  <rect x="{x+8}" y="164" width="160" height="30" rx="6" fill="#ffffff" stroke="{c}" stroke-width="0.8"/>\n'
    b += (f'  <text x="{x+88}" y="184" text-anchor="middle" font-size="8.4" '
          f'font-family="Courier New,monospace" fill="#475569">{code}</text>\n')
    b += f'  <text x="{x+88}" y="216" text-anchor="middle" font-size="11.5" fill="#334155">{d}</text>\n'
    if i < 3:
        b += f'  <path d="M{x+180} 108 H{x+190}" stroke="#94a3b8" stroke-width="2" marker-end="url(#b)"/>\n'
b += note(22, 272, 756, 116, "④ 의 now - born 한 줄이 이번 주의 전부다",
          ["결과가 나온 시각에서 그 프레임이 태어난 시각을 뺀다 — 처리 시간이 아니라 나이다",
           "FPS 와 드롭 수만 보는 대시보드는 「4.7초 지연」을 정상으로 보고한다",
           "한계 — 로그정규 흔들림 · 카메라 노출과 전송 제외 · 스레드 경합 없음",
           "그래도 「큐가 발산하면 나이가 발산한다」 는 결론은 모형에 의존하지 않는다"])
b += '  <defs>' + MK.format(i="b", c="#94a3b8") + '</defs>\n'
fig("w09_p3_sim_12", 800, 406, b)

for k, v in F.items():
    (OUT / f"{k}.svg").write_text(v, encoding="utf-8")
print(f"{len(F)}개 저장 → {OUT}")
