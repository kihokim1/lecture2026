# -*- coding: utf-8 -*-
"""11주차 그림 — 개념 도해 + cover/compile/boundary.json 기반 실측 차트."""
import json, math, pathlib

D = pathlib.Path("/root/lab11")
CV = json.load(open(D / "cover.json"))
CP = json.load(open(D / "compile.json"))
BD = json.load(open(D / "boundary.json"))
OUT = pathlib.Path("/root/ondevice-ai/img/week11"); OUT.mkdir(parents=True, exist_ok=True)

HEAD = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        'font-family="\'Segoe UI\',Arial,sans-serif">\n'
        '  <rect width="{w}" height="{h}" fill="#f8fafc"/>\n')
F = {}

TEAL, AMB, RED, BLUE = "#028090", "#e4711b", "#dc2626", "#2563eb"
INK, MUT, LINE = "#0f172a", "#64748b", "#cbd5e1"


def fig(name, w, h, body):
    F[name] = HEAD.format(w=w, h=h) + body + "\n</svg>\n"


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


def tw(s, size):
    """대략적인 텍스트 폭 — 한글/기호는 전각, 라틴·숫자는 반각."""
    return sum(size * (1.0 if ord(ch) > 0x2000 else 0.56) for ch in s)


def legend(x, y, items, size=12.5):
    s = ""
    cx = x
    for c, lab in items:
        s += f'  <rect x="{cx}" y="{y-9}" width="13" height="13" rx="3" fill="{c}"/>\n'
        s += f'  <text x="{cx+19}" y="{y+2}" font-size="{size}" fill="#334155">{lab}</text>\n'
        cx += 19 + tw(lab, size) + 22
    return s


MK = ('<marker id="{i}" markerWidth="9" markerHeight="9" refX="7" refY="3.5" '
      'orient="auto" markerUnits="userSpaceOnUse">'
      '<path d="M0,0 L7,3.5 L0,7 Z" fill="{c}"/></marker>')

M = {m["name"]: m for m in CV["models"]}
NAMES = ["ResNet-18", "MobileNetV2", "YOLO11n @320", "Transformer 인코더"]
SHORT = {"ResNet-18": "ResNet-18 (2015)", "MobileNetV2": "MobileNetV2 (2018)",
         "YOLO11n @320": "YOLO11n (2024)", "Transformer 인코더": "Transformer 인코더"}

# ═════════ 01. 네 모델의 커버리지 ═════════
b = title(470, 30, "같은 가속기, 다른 모델 — 무엇을 받아 주는가",
          "A형(합성곱 코어) 허용목록 기준 · ONNX Runtime CPU 실측")
XL, XR, PT, RH = 220, 700, 78, 74
for i, nm in enumerate(NAMES):
    p = M[nm]["profiles"]["A"]
    y = PT + i * RH
    b += (f'  <text x="{XL-12}" y="{y+20}" text-anchor="end" font-size="13.5" '
          f'font-weight="700" fill="#334155">{SHORT[nm]}</text>\n')
    for j, (v, c) in enumerate([(p["node_cov"], "#94a3b8"), (p["time_cov"], TEAL)]):
        yy = y + j * 22
        w = max(v * (XR - XL), 2)
        b += (f'  <rect x="{XL}" y="{yy}" width="{w:.1f}" height="18" rx="3" '
              f'fill="{c}"/>\n')
        b += (f'  <text x="{XL+w+8:.1f}" y="{yy+14}" font-size="12.5" '
              f'font-weight="700" fill="{c}">{v:.1%}</text>\n')
    sw = p["switches"]
    sc = RED if sw > 60 else (AMB if sw > 0 else "#16a34a")
    b += (f'  <text x="{XR+96}" y="{y+26}" font-size="14" font-weight="700" '
          f'fill="{sc}">왕복 {sw}회</text>\n')
b += f'  <line x1="{XL}" y1="{PT-10}" x2="{XL}" y2="{PT+4*RH-14}" stroke="#94a3b8" stroke-width="1.5"/>\n'
b += legend(XL, PT + 4 * RH + 6, [("#94a3b8", "노드 커버리지"), (TEAL, "시간 커버리지")])
b += note(30, PT + 4 * RH + 26, 880, 100,
          "차이는 성능이 아니라 연대다",
          ["가속기 회로는 설계 시점에 존재하던 연산자를 받도록 굳는다",
           "모델은 매년 새 연산자를 들고 나온다 — 격차는 시간이 지날수록 벌어진다",
           "ResNet-18 은 통째로 올라가고(왕복 0회), YOLO11n 은 3분의 1만 올라간다(왕복 165회)"])
fig("w11_p1_coverage_01", 940, PT + 4 * RH + 146, b)

# ═════════ 02. 그래프 분할 개념 ═════════
b = title(460, 30, "미지원 연산자 하나가 그래프를 자른다",
          "지원되지 않는 노드를 만나면 실행이 호스트로 돌아온다")
CH = [("Conv", 1), ("Add", 1), ("Clip", 0), ("Conv", 1), ("Add", 1),
      ("Clip", 0), ("Conv", 1), ("Gemm", 1)]
x0, y0, bw, gap = 62, 96, 78, 22
for i, (nm, ok) in enumerate(CH):
    x = x0 + i * (bw + gap)
    c = TEAL if ok else RED
    bg = "#e6f4f2" if ok else "#fee2e2"
    b += (f'  <rect x="{x}" y="{y0}" width="{bw}" height="46" rx="8" fill="{bg}" '
          f'stroke="{c}" stroke-width="2"/>\n')
    b += (f'  <text x="{x+bw/2}" y="{y0+29}" text-anchor="middle" font-size="14" '
          f'font-weight="700" fill="{c}">{nm}</text>\n')
    if i < len(CH) - 1:
        nxt = CH[i + 1][1]
        cross = nxt != ok
        col = AMB if cross else "#94a3b8"
        b += (f'  <path d="M{x+bw+2} {y0+23} H{x+bw+gap-4}" stroke="{col}" '
              f'stroke-width="2" marker-end="url(#{"o" if cross else "g"})"/>\n')
# 장치 띠
b += f'  <text x="18" y="{y0-24}" font-size="12.5" font-weight="700" fill="{TEAL}">가속기</text>\n'
b += f'  <text x="18" y="{y0+92}" font-size="12.5" font-weight="700" fill="{RED}">호스트 CPU</text>\n'
blocks, st = [], 0
for i in range(1, len(CH) + 1):
    if i == len(CH) or CH[i][1] != CH[st][1]:
        blocks.append((st, i - 1, CH[st][1]))
        st = i
for a, z, ok in blocks:
    xa = x0 + a * (bw + gap)
    xz = x0 + z * (bw + gap) + bw
    yy = y0 - 27 if ok else y0 + 60
    b += (f'  <rect x="{xa}" y="{yy}" width="{xz-xa}" height="9" rx="4" '
          f'fill="{TEAL if ok else RED}" fill-opacity="0.6"/>\n')
    b += (f'  <text x="{(xa+xz)/2}" y="{yy-6 if ok else yy+22}" text-anchor="middle" '
          f'font-size="11" font-weight="700" fill="{TEAL if ok else RED}">블록 {blocks.index((a,z,ok))+1}</text>\n')
# 왕복 표시
for i in range(len(CH) - 1):
    if CH[i][1] != CH[i + 1][1]:
        x = x0 + i * (bw + gap) + bw + gap / 2
        b += (f'  <text x="{x}" y="{y0+92}" text-anchor="middle" font-size="20" '
              f'font-weight="700" fill="{AMB}">↕</text>\n')
b += (f'  <text x="460" y="{y0+126}" text-anchor="middle" font-size="14" '
      f'font-weight="700" fill="{AMB}">이 그래프에서 왕복 4회 · 블록 5개</text>\n')
b += note(30, y0 + 142, 860, 122, "경계마다 무슨 일이 일어나는가",
          ["① 텐서를 호스트 메모리로 되돌린다 (또는 반대로) — 바이트 × 대역폭",
           "② 레이아웃을 바꾼다 — 대부분의 NPU 는 NHWC, PyTorch 내보내기는 NCHW",
           "③ 동기화하고 커널을 다시 띄운다 — 왕복마다 붙는 고정비",
           "TFLite 공식 문서: \"파티션이 여러 개가 되는 것은 바람직하지 않다\""])
b += '  <defs>' + MK.format(i="g", c="#94a3b8") + MK.format(i="o", c=AMB) + '</defs>\n'
fig("w11_p1_partition_02", 920, y0 + 288, b)

# ═════════ 03. 노드 대 시간 ═════════
b = title(440, 30, "노드로 세는 것과 시간으로 세는 것",
          "벤더는 노드로 자랑하고 사용자는 시간으로 실망한다")
rows = [
    ("MobileNetV2 · A형", M["MobileNetV2"]["profiles"]["A"]["node_cov"],
     M["MobileNetV2"]["profiles"]["A"]["time_cov"]),
    ("YOLO11n · A형", M["YOLO11n @320"]["profiles"]["A"]["node_cov"],
     M["YOLO11n @320"]["profiles"]["A"]["time_cov"]),
    ("Transformer · A형", M["Transformer 인코더"]["profiles"]["A"]["node_cov"],
     M["Transformer 인코더"]["profiles"]["A"]["time_cov"]),
    ("Transformer · MatMul 만", M["Transformer 인코더"]["curve"][0]["node_cov"],
     M["Transformer 인코더"]["curve"][0]["time_cov"]),
]
XL, XR, PT, RH = 200, 660, 84, 68
for i, (nm, nc, tc) in enumerate(rows):
    y = PT + i * RH
    b += (f'  <text x="{XL-12}" y="{y+22}" text-anchor="end" font-size="13.5" '
          f'font-weight="700" fill="#334155">{nm}</text>\n')
    for j, (v, c) in enumerate([(nc, "#94a3b8"), (tc, TEAL)]):
        yy = y + j * 21
        w = max(v * (XR - XL), 2)
        b += f'  <rect x="{XL}" y="{yy}" width="{w:.1f}" height="17" rx="3" fill="{c}"/>\n'
        b += (f'  <text x="{XL+w+8:.1f}" y="{yy+13}" font-size="12.5" '
              f'font-weight="700" fill="{c}">{v:.1%}</text>\n')
    gap = round(tc * 100, 1) - round(nc * 100, 1)
    b += (f'  <text x="{XR+92}" y="{y+26}" font-size="14" font-weight="700" '
          f'fill="{AMB}">{gap:+.1f}%p</text>\n')
b += f'  <line x1="{XL}" y1="{PT-10}" x2="{XL}" y2="{PT+4*RH-16}" stroke="#94a3b8" stroke-width="1.5"/>\n'
b += legend(XL, PT + 4 * RH + 2, [("#94a3b8", "노드 커버리지"), (TEAL, "시간 커버리지")])
b += note(30, PT + 4 * RH + 22, 820, 100,
          "같은 격차가 두 방향으로 벌어진다",
          ["MobileNetV2 — Clip 35개는 노드의 35%인데 시간의 10%다 (제일 싼 연산)",
           "Transformer — MatMul 만 지원해도 노드 8.8%에 시간 57.2%다 (제일 비싼 연산)",
           "벤더에게 물어야 할 것: \"몇 개 지원합니까\"가 아니라 \"제 시간의 몇 %를 받습니까\""],
          bg="#fff7ed", ln="#fbbf24", hc="#92400e")
fig("w11_p1_nodetime_03", 880, PT + 4 * RH + 142, b)

# ═════════ 04. 융합 ═════════
b = title(470, 30, "컴파일러가 하는 일 — 그리고 그 대가",
          "ONNX Runtime 그래프 최적화 끔 ↔ 전부 (실측)")
CM = {m["name"]: m for m in CP["models"]}
XL, PT, RH = 208, 82, 76
maxn = max(CM[n]["levels"]["none"]["nodes"] for n in NAMES)
BW = 250
for i, nm in enumerate(NAMES):
    r = CM[nm]
    n0, n1 = r["levels"]["none"]["nodes"], r["levels"]["all"]["nodes"]
    y = PT + i * RH
    b += (f'  <text x="{XL-12}" y="{y+26}" text-anchor="end" font-size="13.5" '
          f'font-weight="700" fill="#334155">{SHORT[nm]}</text>\n')
    w0 = n0 / maxn * BW
    w1 = n1 / maxn * BW
    b += f'  <rect x="{XL}" y="{y}" width="{w0:.1f}" height="17" rx="3" fill="#94a3b8"/>\n'
    b += (f'  <text x="{XL+w0+7:.1f}" y="{y+13}" font-size="12" fill="#64748b">{n0}</text>\n')
    b += f'  <rect x="{XL}" y="{y+21}" width="{w1:.1f}" height="17" rx="3" fill="{TEAL}"/>\n'
    b += (f'  <text x="{XL+w1+7:.1f}" y="{y+34}" font-size="12" font-weight="700" '
          f'fill="{TEAL}">{n1}</text>\n')
    sp, am = r["speedup"], r["amortize_runs"]
    b += (f'  <text x="{XL+BW+128}" y="{y+18}" font-size="15" font-weight="700" '
          f'fill="{AMB}">{sp:.2f}배 빨라짐</text>\n')
    b += (f'  <text x="{XL+BW+128}" y="{y+38}" font-size="12.5" '
          f'fill="{MUT}">빌드 {r["build_ratio"]:.2f}배 · {round(am)}회면 회수</text>\n')
b += f'  <line x1="{XL}" y1="{PT-10}" x2="{XL}" y2="{PT+4*RH-18}" stroke="#94a3b8" stroke-width="1.5"/>\n'
b += legend(XL, PT + 4 * RH + 2, [("#94a3b8", "최적화 끔 (노드 수)"), (TEAL, "최적화 전부")])
b += note(30, PT + 4 * RH + 22, 880, 100,
          "무엇이 사라졌나 — 융합의 직접 증거",
          ["ResNet-18 : Relu 17개 · Add 8개가 한 개도 남지 않고 Conv 안으로 접혔다",
           "MobileNetV2 : Clip 35개 소멸 → 2.25배. 컴파일만으로, 같은 CPU 에서",
           "YOLO11n : Mul 79 + Sigmoid 78 → QuickGelu 77. 그런데 ReorderInput 22 + ReorderOutput 29 가 새로 생겼다"])
fig("w11_p1_fusion_04", 940, PT + 4 * RH + 142, b)

# ═════════ 05. 커버리지 곡선 ═════════
b = title(470, 30, "연산자를 하나씩 늘리면 — 커버리지와 왕복은 같이 안 움직인다",
          "시간이 큰 연산자부터 허용목록에 추가 (실측)")
PX, PY, PW, PH = 90, 78, 560, 232
b += f'  <rect x="{PX}" y="{PY}" width="{PW}" height="{PH}" fill="#ffffff" stroke="{LINE}"/>\n'
for g in range(0, 101, 25):
    yy = PY + PH - g / 100 * PH
    b += f'  <line x1="{PX}" y1="{yy:.1f}" x2="{PX+PW}" y2="{yy:.1f}" stroke="#eef2f7"/>\n'
    b += (f'  <text x="{PX-9}" y="{yy+4:.1f}" text-anchor="end" font-size="11.5" '
          f'fill="{MUT}">{g}%</text>\n')
CURVES = [("YOLO11n @320", TEAL), ("Transformer 인코더", BLUE)]
for nm, c in CURVES:
    cur = M[nm]["curve"]
    n = len(cur)
    pts = " ".join(f"{PX+k/(n-1)*PW:.1f},{PY+PH-cur[k]['time_cov']*PH:.1f}" for k in range(n))
    b += f'  <polyline points="{pts}" fill="none" stroke="{c}" stroke-width="2.6"/>\n'
    mx = max(x["switches"] for x in cur)
    pts2 = " ".join(f"{PX+k/(n-1)*PW:.1f},{PY+PH-cur[k]['switches']/mx*PH:.1f}"
                    for k in range(n))
    b += (f'  <polyline points="{pts2}" fill="none" stroke="{c}" stroke-width="2.2" '
          f'stroke-dasharray="6 4" stroke-opacity="0.75"/>\n')
    # 비단조 구간 표시
    for k in range(1, n):
        if cur[k]["switches"] > cur[k - 1]["switches"]:
            xx = PX + k / (n - 1) * PW
            yy = PY + PH - cur[k]["switches"] / mx * PH
            b += f'  <circle cx="{xx:.1f}" cy="{yy:.1f}" r="6" fill="none" stroke="{RED}" stroke-width="2.4"/>\n'
b += (f'  <text x="{PX+PW/2}" y="{PY+PH+24}" text-anchor="middle" font-size="12.5" '
      f'fill="{MUT}">← 허용목록에 넣은 연산자 종류 수 (시간이 큰 순) →</text>\n')
b += legend(PX, PY + PH + 46, [(TEAL, "YOLO11n"), (BLUE, "Transformer")])
b += (f'  <text x="{PX+300}" y="{PY+PH+48}" font-size="12" fill="{MUT}">'
      f'실선 = 시간 커버리지 · 점선 = 왕복(각 모델 최대값 대비)</text>\n')
b += (f'  <circle cx="{PX+308}" cy="{PY+PH+66}" r="6" fill="none" stroke="{RED}" stroke-width="2.4"/>\n')
b += (f'  <text x="{PX+322}" y="{PY+PH+70}" font-size="12" font-weight="700" fill="{RED}">'
      f'= 연산자를 더 지원했는데 왕복이 늘어난 지점</text>\n')
b += note(672, PY, 258, 232, "숫자로 보면",
          ["YOLO11n", "Conv 만 → 왕복 141",
           "+ Sigmoid → 147 (늘었다)",
           "+ Mul → 83",
           " ",
           "Transformer",
           "MatMul → 42",
           "+ Add → 63",
           "+ Transpose → 103 (2.5배)"],
          bg="#fff7ed", ln="#fbbf24", hc="#92400e")
b += note(30, PY + PH + 84, 900, 76, "왜 늘어나는가",
          ["Transpose 처럼 그래프 곳곳에 흩어진 연산자를 지원하면",
           "기존 섬이 커지는 게 아니라 새 섬이 여러 개 생긴다 — 커버리지와 조각 수는 같은 방향이 아니다"])
fig("w11_p1_curve_05", 960, PY + PH + 180, b)

# ═════════ 06. 왕복세 ═════════
b = title(470, 30, "암달 천장 위에 왕복세가 얹힌다",
          "실측 분할 결과 + 실측 대역폭 · 가속기 배속 S 를 무한대로 두어도")
BE = {(r["model"], r["profile"]): r for r in BD["breakeven"]}
ROWS = [("MobileNetV2", "A", "MobileNetV2 · A형"),
        ("MobileNetV2", "B", "MobileNetV2 · B형 (Clip 추가)"),
        ("YOLO11n @320", "A", "YOLO11n · A형"),
        ("YOLO11n @320", "B", "YOLO11n · B형"),
        ("Transformer 인코더", "A", "Transformer · A형"),
        ("Transformer 인코더", "C", "Transformer · C형")]
XL, PT, RH, BW = 250, 84, 52, 400
mx = 24.0


def lg(v):
    return math.log10(max(v, 1.0)) / math.log10(mx)


for i, (mn, pf, lab) in enumerate(ROWS):
    r = BE[(mn, pf)]
    y = PT + i * RH
    am = r["amdahl_only"]
    re = r["spinf"]
    b += (f'  <text x="{XL-12}" y="{y+22}" text-anchor="end" font-size="13" '
          f'font-weight="700" fill="#334155">{lab}</text>\n')
    if am is None:
        b += (f'  <text x="{XL+6}" y="{y+24}" font-size="13.5" font-weight="700" '
              f'fill="#16a34a">왕복 0회 — 천장 없음</text>\n')
        continue
    w1 = lg(am) * BW
    w2 = lg(re) * BW
    b += f'  <rect x="{XL}" y="{y}" width="{w1:.1f}" height="16" rx="3" fill="#cbd5e1"/>\n'
    b += (f'  <text x="{XL+w1+7:.1f}" y="{y+13}" font-size="12" fill="{MUT}">{am:.2f}배</text>\n')
    b += f'  <rect x="{XL}" y="{y+19}" width="{w2:.1f}" height="16" rx="3" fill="{RED}"/>\n'
    b += (f'  <text x="{XL+w2+7:.1f}" y="{y+32}" font-size="12.5" font-weight="700" '
          f'fill="{RED}">{re:.2f}배</text>\n')
    b += (f'  <text x="{XL+BW+118}" y="{y+26}" font-size="12" fill="{MUT}">'
          f'왕복 {r["switches"]}회 · {r["boundary_ms"]:.2f} ms</text>\n')
b += f'  <line x1="{XL}" y1="{PT-10}" x2="{XL}" y2="{PT+6*RH-20}" stroke="#94a3b8" stroke-width="1.5"/>\n'
b += legend(XL, PT + 6 * RH + 14, [("#cbd5e1", "암달 천장 (왕복 무시)"), (RED, "왕복세 포함 실제 천장")])
b += (f'  <text x="{XL+BW+118}" y="{PT+6*RH+16}" font-size="11.5" fill="{MUT}">'
      f'가로축 로그 눈금</text>\n')
b += note(30, PT + 6 * RH + 34, 900, 100,
          "연산자 하나의 값어치",
          ["MobileNetV2 에 2배 빠른 가속기를 넣으면 → 1.03배 (사실상 이득 없음)",
           "같은 가속기가 Clip 하나만 더 받아 주면 → 2.00배",
           "칩을 두 배 빠르게 만드는 것보다, 연산자 하나를 지원하는 것이 압도적으로 싸다"],
          bg="#fef2f2", ln="#fca5a5", hc="#991b1b")
fig("w11_p2_switch_06", 960, PT + 6 * RH + 180, b)

# ═════════ 07. 실행 순서 ═════════
b = title(420, 30, "같은 그래프, 같은 허용목록 — 순서만 바꿨다",
          "위상 정렬의 타이브레이크 정책 한 줄 (실측)")
ORD = [("MobileNetV2", "A"), ("YOLO11n @320", "A"), ("YOLO11n @320", "B"),
       ("Transformer 인코더", "A"), ("Transformer 인코더", "B"), ("Transformer 인코더", "C")]
LB = {"MobileNetV2": "MobileNetV2", "YOLO11n @320": "YOLO11n",
      "Transformer 인코더": "Transformer"}
XL, PT, RH, BW = 210, 82, 50, 330
mxs = max(M[a]["profiles"][p]["switches"] for a, p in ORD)
for i, (mn, pf) in enumerate(ORD):
    p = M[mn]["profiles"][pf]
    d, s = p["switches"], p["switches_sticky"]
    y = PT + i * RH
    b += (f'  <text x="{XL-12}" y="{y+22}" text-anchor="end" font-size="13" '
          f'font-weight="700" fill="#334155">{LB[mn]} · {pf}형</text>\n')
    for j, (v, c) in enumerate([(d, "#94a3b8"), (s, TEAL)]):
        yy = y + j * 17
        w = max(v / mxs * BW, 2)
        b += f'  <rect x="{XL}" y="{yy}" width="{w:.1f}" height="14" rx="3" fill="{c}"/>\n'
        b += (f'  <text x="{XL+w+7:.1f}" y="{yy+11}" font-size="11.5" '
              f'font-weight="700" fill="{c}">{v}</text>\n')
    red = (d - s) / d if d else 0
    cc = "#16a34a" if red > 0.15 else (MUT if red > 0 else "#94a3b8")
    b += (f'  <text x="{XL+BW+72}" y="{y+22}" font-size="14" font-weight="700" '
          f'fill="{cc}">{red:.0%} 감소</text>\n')
b += f'  <line x1="{XL}" y1="{PT-10}" x2="{XL}" y2="{PT+6*RH-18}" stroke="#94a3b8" stroke-width="1.5"/>\n'
b += legend(XL, PT + 6 * RH - 2, [("#94a3b8", "파일 순서"), (TEAL, "장치 유지 순서")])
b += note(30, PT + 6 * RH + 16, 780, 100,
          "코드도 모델도 하드웨어도 그대로다",
          ["MobileNetV2 가 0% 인 것도 의미가 있다 — 사슬 모양은 위상 정렬이 사실상 하나뿐이다",
           "가지가 많은 그래프일수록 스케줄러가 할 일이 많다",
           "\"SDK 판을 올렸더니 같은 모델이 빨라졌다/느려졌다\" 의 상당수가 여기서 온다"])
fig("w11_p2_order_07", 840, PT + 6 * RH + 162, b)

# ═════════ 08. 레이아웃 세금 ═════════
b = title(400, 30, "계산을 하나도 안 하는 노드가 시간의 9.7%",
          "ONNX Runtime 이 그래프에 끼워 넣은 레이아웃 변환 노드 (실측)")
RE = {r["name"]: r for r in BD["reorder"]}
XL, PT, RH, BW = 190, 84, 62, 420
mxt = max(r["total_ms"] for r in BD["reorder"])
for i, nm in enumerate(["ResNet-18", "MobileNetV2", "YOLO11n @320"]):
    r = RE[nm]
    y = PT + i * RH
    tot, rms = r["total_ms"], r["reorder_ms"]
    wt = tot / mxt * BW
    wr = rms / mxt * BW if r["share"] >= 0.005 else 0.0
    b += (f'  <text x="{XL-12}" y="{y+26}" text-anchor="end" font-size="13.5" '
          f'font-weight="700" fill="#334155">{nm}</text>\n')
    b += f'  <rect x="{XL}" y="{y}" width="{wt:.1f}" height="30" rx="4" fill="#cbd5e1"/>\n'
    if wr > 0:
        b += f'  <rect x="{XL}" y="{y}" width="{max(wr,3):.1f}" height="30" rx="4" fill="{RED}"/>\n'
    b += (f'  <text x="{XL+wt+10:.1f}" y="{y+20}" font-size="12.5" fill="{MUT}">'
          f'{tot:.2f} ms</text>\n')
    sc = RED if r["share"] > 0.02 else MUT
    b += (f'  <text x="{XL+BW+120}" y="{y+20}" font-size="15" font-weight="700" '
          f'fill="{sc}">{r["share"]:.1%}</text>\n')
b += legend(XL, PT + 3 * RH + 4, [("#cbd5e1", "전체 추론 시간"), (RED, "레이아웃 변환 노드")])
b += note(30, PT + 3 * RH + 24, 740, 122, "왜 YOLO11n 만 비싼가",
          ["ResNet-18 은 사슬 모양 — 레이아웃을 한 번 바꾸면 끝까지 유지된다 (0.0%)",
           "YOLO11n 은 가지가 많다 — ReorderInput 22개 + ReorderOutput 29개가 삽입됐다",
           "가지가 많은 그래프 = 레이아웃 경계가 많은 그래프",
           "실측 변환 대역폭 9.87 GB/s, 그중 (1,3,320,320) 입력은 2.92 GB/s 로 가장 느리다"],
          bg="#fef2f2", ln="#fca5a5", hc="#991b1b")
fig("w11_p2_layout_08", 800, PT + 3 * RH + 166, b)

# ═════════ 09. INT8 그래프 폭발 ═════════
Q = BD["qdq"]
b = title(400, 30, "INT8 을 요구하면 그래프가 2.62배가 된다",
          "YOLO11n @320 · ONNX Runtime 동적 양자화 (실측)")
XL, PT, BW = 150, 90, 460
mxn = Q["int8_nodes"]
for i, (lab, v, c) in enumerate([("FP32", Q["fp32_nodes"], TEAL),
                                 ("동적 INT8", Q["int8_nodes"], RED)]):
    y = PT + i * 56
    w = v / mxn * BW
    b += (f'  <text x="{XL-12}" y="{y+27}" text-anchor="end" font-size="14" '
          f'font-weight="700" fill="#334155">{lab}</text>\n')
    b += f'  <rect x="{XL}" y="{y}" width="{w:.1f}" height="38" rx="4" fill="{c}"/>\n'
    b += (f'  <text x="{XL+w+10:.1f}" y="{y+26}" font-size="17" font-weight="700" '
          f'fill="{c}">{v}노드</text>\n')
b += (f'  <text x="{XL+BW/2}" y="{PT+124}" text-anchor="middle" font-size="15" '
      f'font-weight="700" fill="{RED}">늘어난 518개는 새 계산이 아니라 양자화·역양자화·재조정이다</text>\n')
TBL = [("Conv → ConvInteger", "88", "88"), ("Mul", "79", "255"),
       ("Add", "16", "103"), ("DynamicQuantizeLinear", "0", "80"), ("Cast", "0", "88")]
ty = PT + 144
b += f'  <rect x="{XL-40}" y="{ty}" width="540" height="{28+len(TBL)*26}" rx="8" fill="#ffffff" stroke="{LINE}"/>\n'
b += f'  <text x="{XL-24}" y="{ty+19}" font-size="12.5" font-weight="700" fill="{MUT}">연산자</text>\n'
b += f'  <text x="{XL+270}" y="{ty+19}" text-anchor="middle" font-size="12.5" font-weight="700" fill="{TEAL}">FP32</text>\n'
b += f'  <text x="{XL+400}" y="{ty+19}" text-anchor="middle" font-size="12.5" font-weight="700" fill="{RED}">INT8</text>\n'
for i, (a, c1, c2) in enumerate(TBL):
    yy = ty + 44 + i * 26
    b += f'  <text x="{XL-24}" y="{yy}" font-size="12.5" fill="#334155">{a}</text>\n'
    b += f'  <text x="{XL+270}" y="{yy}" text-anchor="middle" font-size="12.5" fill="{MUT}">{c1}</text>\n'
    bold = ' font-weight="700"' if c2 != c1 else ''
    b += f'  <text x="{XL+400}" y="{yy}" text-anchor="middle" font-size="12.5"{bold} fill="{RED}">{c2}</text>\n'
b += note(30, ty + 40 + len(TBL) * 26, 740, 120,
          "9주차의 미결 항목이 여기서 풀린다",
          ["9주차에서 동적 INT8 은 YOLO11n 을 1.52배 느리게 만들었다",
           f"이제 두 번째 이유가 보인다 — 새로 생긴 DynamicQuantizeLinear 80 · Cast 88 과 재조정 Mul +176 · Add +87 이",
           f"계산을 하나도 늘리지 않으면서 전체 시간의 {Q['qdq_share_time']:.0%}를 쓴다",
           "INT8 전용 NPU 가 DynamicQuantizeLinear 를 안 받으면, 그 80개가 전부 왕복이 된다"],
          bg="#fff7ed", ln="#fbbf24", hc="#92400e")
fig("w11_p2_int8_09", 800, ty + 206 + len(TBL) * 26, b)

# ═════════ 10. 벤더 수치 ═════════
b = title(414, 30, "사양서의 숫자는 상한이지 예측이 아니다",
          "Google Edge TPU 실측 — Boroumand 외, PACT 2021 §III-A")
PX, PY, PW, PH = 110, 92, 410, 190
b += f'  <rect x="{PX}" y="{PY}" width="{PW}" height="{PH}" rx="8" fill="#ffffff" stroke="{LINE}"/>\n'
b += f'  <rect x="{PX+40}" y="{PY+16}" width="90" height="{PH-40}" rx="5" fill="#cbd5e1"/>\n'
b += (f'  <text x="{PX+85}" y="{PY+8}" text-anchor="middle" font-size="13" '
      f'font-weight="700" fill="{MUT}">이론 최대</text>\n')
b += (f'  <text x="{PX+85}" y="{PY+PH+20}" text-anchor="middle" font-size="13.5" '
      f'font-weight="700" fill="{MUT}">2 TFLOP/s</text>\n')
h2 = (PH - 40) * 0.244
b += f'  <rect x="{PX+170}" y="{PY+PH-24-h2:.1f}" width="90" height="{h2:.1f}" rx="5" fill="{AMB}"/>\n'
b += (f'  <text x="{PX+215}" y="{PY+PH-32-h2:.1f}" text-anchor="middle" font-size="13" '
      f'font-weight="700" fill="{AMB}">평균 실측</text>\n')
b += (f'  <text x="{PX+215}" y="{PY+PH+20}" text-anchor="middle" font-size="13.5" '
      f'font-weight="700" fill="{AMB}">75.6% 낮음</text>\n')
h3 = max((PH - 40) * 0.01, 4)
b += f'  <rect x="{PX+276}" y="{PY+PH-24-h3:.1f}" width="80" height="{h3:.1f}" fill="{RED}"/>\n'
b += (f'  <path d="M{PX+316} {PY+PH-48:.1f} V{PY+PH-26-h3:.1f}" stroke="{RED}" '
      f'stroke-width="1.6" marker-end="url(#v)"/>\n')
b += (f'  <text x="{PX+316}" y="{PY+PH-56:.1f}" text-anchor="middle" font-size="12" '
      f'font-weight="700" fill="{RED}">LSTM·Transducer</text>\n')
b += (f'  <text x="{PX+316}" y="{PY+PH+20}" text-anchor="middle" font-size="13.5" '
      f'font-weight="700" fill="{RED}">1% 미만</text>\n')
b += '  <defs>' + MK.format(i="v", c=RED) + '</defs>\n'
b += note(548, PY, 250, 190, "논문이 함께 말하는 것",
          ["PE 활용률이 모든 모델에서",
           "일관되게 낮다",
           " ",
           "전체 에너지의 50.3%가",
           "오프칩 메모리 접근에 쓰인다",
           " ",
           "→ 10주차의 그림 그대로다"],
          bg="#fff7ed", ln="#fbbf24", hc="#92400e")
b += note(30, PY + PH + 40, 768, 100, "그래서 무엇을 물어야 하나",
          ["\"몇 TOPS 입니까\" 가 아니라 \"제 모델 시간의 몇 %를 받습니까\"",
           "모바일 이종 프로세서 실측에서는 미지원 연산자 비율이 약 48%에 달했고,",
           "그 경우 이종 병렬이 단일 프로세서 단독보다 1.9~3.1배 느렸다 (Liu 외, MobiSys 2024)"])
fig("w11_p2_vendor_10", 828, PY + PH + 186, b)

# ═════════ 11. 컴파일러 두 층 ═════════
b = title(400, 30, "딥러닝 컴파일러의 두 층",
          "어느 최적화가 이식되고 어느 것이 이 칩에만 통하는가")
LAY = [("고수준 IR (그래프 IR)", "하드웨어 독립", TEAL, "#e6f4f2",
        ["연산자 융합 · 죽은 코드 제거", "정적 메모리 계획 · 레이아웃 변환",
         "→ 어느 가속기에나 통한다"]),
       ("저수준 IR", "하드웨어 종속", AMB, "#fff7ed",
        ["하드웨어 내장 명령 매핑", "메모리 할당·프리페치 · 지연 은닉",
         "병렬화 · 루프 최적화", "→ 이 칩에서만 통한다"])]
y = 78
for nm, kind, c, bg, items in LAY:
    h = 44 + len(items) * 22
    b += f'  <rect x="60" y="{y}" width="680" height="{h}" rx="12" fill="{bg}" stroke="{c}" stroke-width="2"/>\n'
    b += f'  <text x="88" y="{y+30}" font-size="17" font-weight="700" fill="{c}">{nm}</text>\n'
    b += (f'  <rect x="580" y="{y+12}" width="136" height="26" rx="13" fill="#ffffff" stroke="{c}"/>\n')
    b += (f'  <text x="648" y="{y+30}" text-anchor="middle" font-size="12.5" '
          f'font-weight="700" fill="{c}">{kind}</text>\n')
    for i, it in enumerate(items):
        b += f'  <text x="92" y="{y+56+i*22}" font-size="13" fill="#334155">· {it}</text>\n'
    y += h + 26
b += f'  <path d="M400 {y-22} V{y-6}" stroke="#94a3b8" stroke-width="2" marker-end="url(#t)"/>\n'
b += note(30, y + 4, 740, 122, "ONNX Runtime 이 이 경계를 그대로 보여 준다",
          ["Basic 최적화 — 파티셔닝 이전에 실행, 모든 실행 공급자에 적용 (하드웨어 독립)",
           "Extended · Layout — 파티셔닝 이후, CPU/CUDA/ROCm 노드에만 적용 (하드웨어 종속)",
           "즉 가속기로 보낸 노드는 ORT 의 확장 융합을 받지 못한다",
           "그래서 ENABLE_ALL 로 저장한 모델은 저장 당시 환경에서만 쓸 수 있다"])
b += '  <defs>' + MK.format(i="t", c="#94a3b8") + '</defs>\n'
fig("w11_p2_toolchain_11", 800, y + 150, b)

# ═════════ 12. 판정표 ═════════
b = title(400, 30, "여든 줄이면 이식 판정이 나온다",
          "3교시 student.py 재실행값 · MobileNetV2 · A형 기준 (1교시 본실험은 89.9% — 재현 폭 안이다)")
LINES = [("①", "전체 노드 / 실계산 노드", "209 / 100", MUT),
         ("②", "총 추론 시간", "15.36 ms", MUT),
         ("③", "노드 커버리지 / 시간 커버리지", "65.0% / 90.9%", TEAL),
         ("④", "가장 비싼 미지원 연산자", "Clip 35개 · 시간의 9.1%", AMB),
         ("⑤", "호스트↔가속기 왕복", "70회", RED),
         ("⑥", "경계를 넘는 텐서", "70개 · 46.6 MB", RED),
         ("⑦", "암달 천장 → 왕복세 포함 실제 천장", "10.96배 → 2.56배", RED),
         ("⑧", "판정", "조각이 문제 — Clip 지원 여부부터 확인", INK)]
ty = 82
b += f'  <rect x="46" y="{ty}" width="708" height="{20+len(LINES)*36}" rx="10" fill="#ffffff" stroke="{LINE}"/>\n'
for i, (n, k, v, c) in enumerate(LINES):
    yy = ty + 34 + i * 36
    if i == len(LINES) - 1:
        b += f'  <line x1="60" y1="{yy-22}" x2="740" y2="{yy-22}" stroke="{LINE}"/>\n'
    b += f'  <text x="66" y="{yy}" font-size="14" font-weight="700" fill="{MUT}">{n}</text>\n'
    b += f'  <text x="94" y="{yy}" font-size="13.5" fill="#334155">{k}</text>\n'
    b += (f'  <text x="740" y="{yy}" text-anchor="end" font-size="13.5" '
          f'font-weight="700" fill="{c}">{v}</text>\n')
by = ty + 36 + len(LINES) * 36
b += note(46, by, 708, 100, "이 표가 캡스톤에서 하는 일",
          ["가속기 보드를 주문하기 전에 이 여덟 줄을 채운다",
           "결론 한 문장은 늘 같은 형태다 — \"가속기를 사는 것\"과 \"모델을 고치는 것\" 중 무엇이 먼저인가",
           "주의: 이 모형은 데이터 이동만 센다. 커널 실행 고정비를 넣으면 판정은 더 나빠진다"])
fig("w11_p3_report_12", 800, by + 146, b)

for k, v in F.items():
    (OUT / f"{k}.svg").write_text(v, encoding="utf-8")
print(f"SVG {len(F)}장 → {OUT}")
