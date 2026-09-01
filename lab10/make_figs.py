# -*- coding: utf-8 -*-
"""10주차 그림 — 개념 도해 + llm.json / budget.json 기반 실측 차트."""
import json, math, pathlib

D = pathlib.Path("/root/lab10")
L = json.load(open(D / "llm.json"))
B = json.load(open(D / "budget.json"))
OUT = pathlib.Path("/root/ondevice-ai/img/week10"); OUT.mkdir(parents=True, exist_ok=True)

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

SP = L["split"]
PRE, DEC = SP["prefill_tps"], SP["decode_tps"]

# ═════════ 01. 두 단계 ═════════
b = title(400, 30, "한 숫자에 감춰진 두 개의 연산", "\"초당 20토큰\" 은 무엇을 보장하는가")
for x, c, bg, nm, en, what, metric, shape in [
    (30, "#2563eb", "#eff6ff", "프리필", "prefill", "프롬프트 N개 토큰을 한 번에",
     "TTFT — 첫 토큰까지의 시간", "행렬 × 행렬"),
    (410, "#e4711b", "#fff7ed", "디코드", "decode", "그다음부터 토큰을 하나씩",
     "TPOT — 토큰당 시간", "행렬 × 벡터")]:
    b += f'  <rect x="{x}" y="74" width="360" height="176" rx="13" fill="{bg}" stroke="{c}" stroke-width="2.2"/>\n'
    b += f'  <text x="{x+180}" y="104" text-anchor="middle" font-size="20" font-weight="700" fill="{c}">{nm}</text>\n'
    b += f'  <text x="{x+180}" y="124" text-anchor="middle" font-size="12" font-style="italic" fill="#64748b">{en}</text>\n'
    b += f'  <text x="{x+180}" y="152" text-anchor="middle" font-size="13.5" fill="#0f172a">{what}</text>\n'
    b += f'  <rect x="{x+40}" y="166" width="280" height="30" rx="7" fill="#ffffff" stroke="{c}"/>\n'
    b += f'  <text x="{x+180}" y="186" text-anchor="middle" font-size="13" font-weight="700" fill="{c}">{metric}</text>\n'
    b += f'  <text x="{x+180}" y="222" text-anchor="middle" font-size="15" font-weight="700" fill="#0f172a">{shape}</text>\n'
# 토큰 그림
for i in range(6):
    b += f'  <rect x="{56+i*40}" y="262" width="30" height="22" rx="4" fill="#2563eb" fill-opacity="0.85"/>\n'
b += '  <text x="316" y="278" font-size="12.5" fill="#334155">한 번에 통과</text>\n'
for i in range(3):
    b += f'  <rect x="{436+i*76}" y="262" width="30" height="22" rx="4" fill="#e4711b" fill-opacity="0.85"/>\n'
    if i < 2:
        b += f'  <path d="M{470+i*76} 273 H{506+i*76}" stroke="#94a3b8" stroke-width="1.8" marker-end="url(#a)"/>\n'
b += '  <text x="672" y="278" font-size="12.5" fill="#334155">하나씩, 순서대로</text>\n'
b += note(30, 300, 740, 96, "디코드는 원리적으로 병렬화가 불가능하다",
          ["100번째 토큰은 99번째가 나오기 전에는 계산을 시작조차 할 수 없다 — 자기회귀의 정의다",
           "MLPerf Inference 는 LLM 항목에서 TTFT 와 TPOT 를 따로 잰다",
           "사용자가 겪는 것은 TTFT 한 번 + TPOT × 생성 길이 다"])
b += '  <defs>' + MK.format(i="a", c="#94a3b8") + '</defs>\n'
fig("w10_p1_two_phases_01", 800, 412, b)

# ═════════ 02. 21배 격차 ═════════
b = title(400, 30, f"같은 모델, 같은 CPU — 처리량 {PRE/DEC:.0f}배",
          "SmolLM2-135M · FP32 · CPU 2스레드 (실측)")
XL, XR, PT = 132, 736, 92
mx = PRE * 1.12
for i, (nm, v, c) in enumerate([("프리필", PRE, "#2563eb"), ("디코드", DEC, "#e4711b")]):
    y = PT + i * 76
    w = max(v / mx * (XR - XL), 4)
    b += f'  <text x="{XL-10}" y="{y+34}" text-anchor="end" font-size="15" font-weight="700" fill="#334155">{nm}</text>\n'
    b += f'  <rect x="{XL}" y="{y}" width="{w:.1f}" height="52" rx="5" fill="{c}" fill-opacity="0.9"/>\n'
    b += f'  <text x="{XL+w+12:.1f}" y="{y+34}" font-size="17" font-weight="700" fill="{c}">{v:.1f} tok/s</text>\n'
b += f'  <line x1="{XL}" y1="{PT-8}" x2="{XL}" y2="{PT+128}" stroke="#94a3b8" stroke-width="1.5"/>\n'
for x, c, bg, hd, lines in [
    (30, "#2563eb", "#eff6ff", "프리필 — 행렬 × 행렬",
     ["128개 토큰이 한꺼번에 들어온다", "가중치를 한 번 읽어 128개 토큰분 계산",
      "연산 강도 높음 → 연산에 묶인다"]),
    (410, "#e4711b", "#fff7ed", "디코드 — 행렬 × 벡터",
     ["토큰이 하나씩 들어온다", "가중치를 한 번 읽어 1개 토큰분 계산",
      "연산 강도 ≈ 1 → 대역폭에 묶인다"])]:
    b += f'  <rect x="{x}" y="252" width="360" height="112" rx="12" fill="{bg}" stroke="{c}" stroke-width="1.8"/>\n'
    b += f'  <text x="{x+180}" y="278" text-anchor="middle" font-size="15" font-weight="700" fill="{c}">{hd}</text>\n'
    for j, l in enumerate(lines):
        b += f'  <text x="{x+180}" y="{302+j*20}" text-anchor="middle" font-size="12.3" fill="#334155">{l}</text>\n'
b += note(30, 378, 740, 96, "대역폭으로 환산하면",
          [f"{DEC:.1f} tok/s × {L['cfg']['bytes']/1e6:.1f} MB ≈ {DEC*L['cfg']['bytes']/1e9:.1f} GB/s",
           "이 식이 맞다면 모델을 절반으로 줄이면 디코드가 두 배 빨라져야 한다 — 2교시에서 검증한다",
           "주의: 이 식은 논문에 없다. 루프라인에 \"강도 ≈ 1\" 을 대입해 우리가 유도한 것이다"])
fig("w10_p1_gap_02", 800, 490, b)

# ═════════ 03. 루프라인 ═════════
b = title(400, 30, "루프라인 — 강도가 낮으면 대역폭이 정한다",
          "도달 가능 성능 = min(최대 연산 성능, 대역폭 × 연산 강도)")
XL, XR, PB, PT = 100, 730, 300, 86
b += f'  <line x1="{XL}" y1="{PB}" x2="{XR}" y2="{PB}" stroke="#94a3b8" stroke-width="1.5"/>\n'
b += f'  <line x1="{XL}" y1="{PT-8}" x2="{XL}" y2="{PB}" stroke="#94a3b8" stroke-width="1.5"/>\n'
# 지붕: 기울기 구간 + 평평한 구간, x 는 로그
fx = lambda I: XL + math.log10(max(I, 0.25) / 0.25) / math.log10(4000 / 0.25) * (XR - XL)
KNEE = 165.0
b += f'  <path d="M{fx(0.25):.1f} {PB-10} L{fx(KNEE):.1f} {PT+16} H{XR}" stroke="#028090" stroke-width="3" fill="none"/>\n'
b += f'  <text x="{fx(KNEE)+90:.1f}" y="{PT+6}" font-size="13" font-weight="700" fill="#028090">최대 연산 성능</text>\n'
b += f'  <text x="{fx(30):.1f}" y="{PB-162}" font-size="13" font-weight="700" fill="#028090" transform="rotate(-31 {fx(30):.1f} {PB-162})">대역폭 × 강도</text>\n'
b += f'  <line x1="{fx(KNEE):.1f}" y1="{PT+16}" x2="{fx(KNEE):.1f}" y2="{PB}" stroke="#94a3b8" stroke-width="1.2" stroke-dasharray="4 3"/>\n'
b += f'  <text x="{fx(KNEE):.1f}" y="{PB+20}" text-anchor="middle" font-size="12" fill="#64748b">165</text>\n'
for I, lab, c in [(1.0, "배치 1 디코드\n강도 ≈ 1", "#e4711b"), (4.0, "W4A16 양자화 후\n강도 ≈ 4", "#7c3aed")]:
    X = fx(I)
    Y = PB - 10 - (PT + 16 - (PB - 10)) * 0  # 점은 지붕 위
    yy = PB - 10 + (PT + 16 - (PB - 10)) * (math.log10(I / 0.25) / math.log10(KNEE / 0.25))
    b += f'  <circle cx="{X:.1f}" cy="{yy:.1f}" r="7" fill="{c}" stroke="#ffffff" stroke-width="2"/>\n'
    for j, l in enumerate(lab.split("\n")):
        b += f'  <text x="{X:.1f}" y="{yy-30+j*17:.1f}" text-anchor="middle" font-size="12" font-weight="700" fill="{c}">{l}</text>\n'
b += f'  <text x="{(XL+XR)/2:.0f}" y="{PB+44}" text-anchor="middle" font-size="13" font-weight="700" fill="#334155">연산 강도 — 읽어 온 바이트당 연산 수 (로그)</text>\n'
b += f'  <text x="34" y="{(PT+PB)/2}" text-anchor="middle" font-size="13" font-weight="700" fill="#334155" transform="rotate(-90 34 {(PT+PB)/2})">도달 가능 성능</text>\n'
b += note(60, 356, 680, 96, "\"강도 165 미만인 작업은 모두 메모리에 묶인다\"",
          ["Lin 외, AWQ (MLSys 2024 최우수 논문) §4.1 — 4090 GPU 기준 165 TFLOPS / 1 TB/s",
           "그리고 \"FP16 으로 실행할 때 온디바이스 LLM 의 생성 단계는 연산 강도가 약 1\" 이다",
           "원논문(Williams 외 2009)의 용어는 operational intensity 다 — 인용할 때 바꾸지 말 것"],
          bg="#fff7ed", ln="#f59e0b", hc="#b45309")
fig("w10_p1_roofline_03", 800, 472, b)

# ═════════ 04. 토크나이저 ═════════
b = title(400, 30, "같은 내용, 다른 조각 수", "SmolLM2 토크나이저로 직접 셌다 (실측)")
EN, KO = B["영어"], B["한국어"]
XL, XR = 200, 700
mxT = max(EN["tokens"], KO["tokens"]) * 1.1
for i, (nm, d, c) in enumerate([("영어", EN, "#2563eb"), ("한국어", KO, "#dc2626")]):
    y = 88 + i * 92
    ww = d["words"] / mxT * (XR - XL)
    wt = d["tokens"] / mxT * (XR - XL)
    b += f'  <text x="{XL-12}" y="{y+22}" text-anchor="end" font-size="15" font-weight="700" fill="#334155">{nm}</text>\n'
    b += f'  <rect x="{XL}" y="{y}" width="{ww:.1f}" height="30" rx="4" fill="#94a3b8"/>\n'
    b += f'  <text x="{XL+ww+8:.1f}" y="{y+21}" font-size="12.5" fill="#475569">단어 {d["words"]}</text>\n'
    b += f'  <rect x="{XL}" y="{y+36}" width="{wt:.1f}" height="30" rx="4" fill="#dc2626" fill-opacity="0.9"/>\n'
    b += f'  <text x="{XL+wt+8:.1f}" y="{y+57}" font-size="13" font-weight="700" fill="#b91c1c">토큰 {d["tokens"]}</text>\n'
    b += f'  <text x="{XL-12}" y="{y+46}" text-anchor="end" font-size="12.5" font-weight="700" fill="{c}">단어당 {d["tpw"]:.2f}</text>\n'
b += f'  <line x1="{XL}" y1="80" x2="{XL}" y2="264" stroke="#94a3b8" stroke-width="1.5"/>\n'
b += ('  <rect x="200" y="278" width="500" height="30" rx="6" fill="#ffffff" stroke="#cbd5e1"/>\n'
      '  <rect x="212" y="285" width="13" height="13" fill="#94a3b8"/>\n'
      '  <text x="232" y="296" font-size="11.5" fill="#334155">단어 수</text>\n'
      '  <rect x="330" y="285" width="13" height="13" fill="#dc2626"/>\n'
      '  <text x="350" y="296" font-size="11.5" fill="#334155">토큰 수</text>\n')
need_en = [r for r in B["budget"] if r["mode"] == "묵독 · 비소설" and r["lang"] == "영어"][0]
need_ko = [r for r in B["budget"] if r["mode"] == "묵독 · 비소설" and r["lang"] == "한국어"][0]
got = B["measured_decode_tps"]
b += note(40, 324, 720, 116, f"같은 내용을 한국어로 쓰면 토큰이 {KO['tpw']/EN['tpw']:.1f}배 든다",
          [f"묵독 238 wpm (Brysbaert 2019) 을 따라가려면 — 영어 {need_en['need_tps']:.1f} tok/s · 한국어 {need_ko['need_tps']:.1f} tok/s",
           f"우리 실측 디코드 {got:.1f} tok/s 를 대면 — 영어 {got/need_en['need_tps']:.2f}배 여유 · 한국어 {got/need_ko['need_tps']:.2f}배 미달",
           "모델은 한 글자도 안 바뀌었다. 바뀐 것은 토크나이저가 우리 언어를 몇 조각으로 쪼개는가뿐이다",
           "영어권 블로그의 \"10 tok/s 면 충분하다\" 를 옮기면 7배 이상 틀린 예산을 세우게 된다"],
          bg="#fff7ed", ln="#f59e0b", hc="#b45309")
fig("w10_p1_tokenizer_04", 800, 456, b)

# ═════════ 05. TTFT vs 프롬프트 길이 ═════════
b = title(400, 30, "프롬프트가 길어지면 첫 글자가 늦어진다",
          "TTFT 는 크게 늘고, TPOT 도 조금씩 늘어난다 (실측)")
CX = L["ctx"]
XL, XR, PB, PT = 104, 700, 274, 90
mxt = max(c["ttft"] for c in CX) * 1.08
BW = 56
for i, c in enumerate(CX):
    x = XL + 22 + i * 98
    h = c["ttft"] / mxt * (PB - PT)
    col = "#dc2626" if c["ttft"] > 3000 else "#028090"
    b += f'  <rect x="{x}" y="{PB-h:.1f}" width="{BW}" height="{h:.1f}" rx="4" fill="{col}" fill-opacity="0.9"/>\n'
    lab = f'{c["ttft"]/1000:.1f} s' if c["ttft"] >= 1000 else f'{c["ttft"]:.0f} ms'
    b += f'  <text x="{x+BW/2}" y="{PB-h-9:.1f}" text-anchor="middle" font-size="12.5" font-weight="700" fill="{col}">{lab}</text>\n'
    b += f'  <text x="{x+BW/2}" y="{PB+20}" text-anchor="middle" font-size="12" font-weight="700" fill="#0f172a">{c["prompt"]}</text>\n'
    b += f'  <text x="{x+BW/2}" y="{PB+38}" text-anchor="middle" font-size="11" fill="#64748b">{c["tpot"]:.0f} ms</text>\n'
b += f'  <line x1="{XL}" y1="{PB}" x2="{XR}" y2="{PB}" stroke="#94a3b8" stroke-width="1.5"/>\n'
b += f'  <text x="{XL-8}" y="{PB+20}" text-anchor="end" font-size="11.5" fill="#64748b">프롬프트</text>\n'
b += f'  <text x="{XL-8}" y="{PB+38}" text-anchor="end" font-size="11.5" fill="#64748b">TPOT</text>\n'
b += f'  <text x="{(XL+XR)/2:.0f}" y="{PT-14}" text-anchor="middle" font-size="13" font-weight="700" fill="#334155">막대 = TTFT (첫 토큰까지의 시간)</text>\n'
b += note(60, 330, 680, 116, "프롬프트 32배 → TTFT 27.4배 · TPOT 1.40배",
          ["어텐션이 길이의 제곱이라면 TTFT 가 훨씬 더 나빠져야 한다 — 이 크기에서는 아직 선형층이 지배한다",
           "TPOT 가 47.9 → 67.0 ms 로 커진 것이 KV 캐시의 흔적이다 (2교시 2.2)",
           "디코드는 영어라면 이미 충분히 빠르다. 사용자가 답답해하는 것은 이 4.4초다",
           "그동안 화면에는 아무것도 안 나온다"])
fig("w10_p1_ttft_05", 800, 462, b)

# ═════════ 06. 양자화 반전 ═════════
b = title(400, 30, "같은 기법, 반대 결과", "동적 INT8 양자화 — 9주차 탐지 모델 대 10주차 LLM 디코드")
Q = L["quant"]
for x, c, bg, wk, model, size, speed, verdict, why in [
    (30, "#dc2626", "#fef2f2", "9주차", "YOLO11n 탐지 (합성곱)", "3.52배 ↓",
     "1.52배 느려짐", "손해", "합성곱은 동적 양자화 대상이 아니고, 역양자화만 얹혔다"),
    (410, "#16a34a", "#ecfdf5", "10주차", "SmolLM2 디코드 (선형)",
     f'{Q["fp32"]["bytes"]/Q["int8"]["bytes"]:.2f}배 ↓',
     f'{Q["fp32"]["tpot"]/Q["int8"]["tpot"]:.2f}배 빨라짐', "이득",
     "디코드는 읽어 오는 바이트가 전부인 작업이다")]:
    b += f'  <rect x="{x}" y="76" width="360" height="196" rx="13" fill="{bg}" stroke="{c}" stroke-width="2.2"/>\n'
    b += f'  <text x="{x+180}" y="102" text-anchor="middle" font-size="12" font-weight="700" fill="#64748b">{wk}</text>\n'
    b += f'  <text x="{x+180}" y="126" text-anchor="middle" font-size="15.5" font-weight="700" fill="#0f172a">{model}</text>\n'
    b += f'  <rect x="{x+30}" y="142" width="140" height="52" rx="8" fill="#ffffff" stroke="{c}"/>\n'
    b += f'  <text x="{x+100}" y="162" text-anchor="middle" font-size="11.5" fill="#64748b">파일 크기</text>\n'
    b += f'  <text x="{x+100}" y="184" text-anchor="middle" font-size="16" font-weight="700" fill="#475569">{size}</text>\n'
    b += f'  <rect x="{x+190}" y="142" width="140" height="52" rx="8" fill="#ffffff" stroke="{c}"/>\n'
    b += f'  <text x="{x+260}" y="162" text-anchor="middle" font-size="11.5" fill="#64748b">속도</text>\n'
    b += f'  <text x="{x+260}" y="184" text-anchor="middle" font-size="14.5" font-weight="700" fill="{c}">{speed}</text>\n'
    b += f'  <text x="{x+180}" y="220" text-anchor="middle" font-size="19" font-weight="700" fill="{c}">{verdict}</text>\n'
    b += f'  <text x="{x+180}" y="248" text-anchor="middle" font-size="11.8" fill="#334155">{why}</text>\n'
b += '  <text x="400" y="184" text-anchor="middle" font-size="18" font-weight="700" fill="#64748b">↔</text>\n'
b += note(30, 288, 740, 116, "양자화는 \"모델을 빠르게 하는 기법\" 이 아니라 \"메모리 트래픽을 줄이는 기법\" 이다",
          ["\"주어진 모델의 FLOPs 는 고정되어 있으므로, 최대 성능을 올리는 유일한 방법은 총 메모리 트래픽을 줄이는 것\" — AWQ §4.1",
           "\"최신 INT4 양자화 기법들은 낮은 배치의 엣지 LLM 추론만 가속한다\" — QServe (프리프린트)",
           f"다만 정확히 비례하지는 않는다 — 바이트 {Q['fp32']['bytes']/Q['int8']['bytes']:.2f}배 감소에 속도 {Q['fp32']['tpot']/Q['int8']['tpot']:.2f}배 향상",
           "모형은 방향을 맞히고 배수는 못 맞힌다. 그 27% 를 설명하는 것이 3교시 선택 과제다"],
          bg="#fff7ed", ln="#f59e0b", hc="#b45309")
fig("w10_p2_quant_flip_06", 800, 424, b)

# ═════════ 07. KV 캐시 ═════════
b = title(400, 30, "KV 캐시가 가중치를 넘어서는 지점",
          f"토큰당 {L['kv_per_token']/1024:.1f} KB (GQA 3:1 · FP32) · 가중치 {L['cfg']['bytes']/1e6:.0f} MB")
rows = [(2048, 1), (2048, 8), (2048, 32), (8192, 1), (8192, 8), (8192, 32)]
kvmap = {(r["seq"], r["batch"]): r for r in L["kv"]}
XL, XR, PB, PT = 190, 720, 288, 88
mxk = max(kvmap[r]["kv"] for r in rows)
for i, r in enumerate(rows):
    y = PT + i * 34
    d = kvmap[r]
    w = max(d["kv"] / mxk * (XR - XL), 3)
    over = d["ratio"] >= 1
    c = "#dc2626" if over else "#028090"
    b += f'  <text x="{XL-10}" y="{y+18}" text-anchor="end" font-size="12.5" fill="#334155">문맥 {r[0]:,} · 배치 {r[1]}</text>\n'
    b += f'  <rect x="{XL}" y="{y}" width="{w:.1f}" height="24" rx="3" fill="{c}" fill-opacity="0.9"/>\n'
    lx = XL + w + 8 if w < 380 else XL + w - 8
    anc = "start" if w < 380 else "end"
    fc = c if w < 380 else "#ffffff"
    if anc == "start":
        lx = max(lx, XL + (L["cfg"]["bytes"] / mxk) * (XR - XL) + 16)
    b += (f'  <text x="{lx:.1f}" y="{y+17}" text-anchor="{anc}" font-size="12" font-weight="700" '
          f'fill="{fc}">{d["kv"]/1e6:,.0f} MB · {d["ratio"]:.2f}× 가중치</text>\n')
wline = XL + (L["cfg"]["bytes"] / mxk) * (XR - XL)
b += f'  <line x1="{wline:.1f}" y1="{PT-10}" x2="{wline:.1f}" y2="{PB}" stroke="#b45309" stroke-width="2" stroke-dasharray="6 4"/>\n'
b += f'  <text x="{wline+8:.1f}" y="{PT-16}" font-size="12.5" font-weight="700" fill="#b45309">가중치 538 MB</text>\n'
b += f'  <line x1="{XL}" y1="{PT-10}" x2="{XL}" y2="{PB}" stroke="#94a3b8" stroke-width="1.4"/>\n'
b += note(50, 312, 700, 116, "배치 1 에서는 KV 가 주인공이 아니다 — 그런데 배치 8 이면 뒤집힌다",
          [f"교차점은 {L['kv_crossover_tokens']:,.0f} 토큰인데 이 모델의 최대 문맥은 8,192 다",
           "한 사람만 쓰는 온디바이스에서는 가중치가, 서버에서는 KV 가 지배한다",
           f"GQA 가 아니라 MHA 였다면 3배인 {L['kv_mha_would_be']/1e6:,.0f} MB 였을 것이다",
           "\"작은 배치·짧은 길이에서는 가중치가, 큰 배치·긴 길이에서는 KV 가 지배한다\" — Pope 외 §2"])
fig("w10_p2_kv_07", 800, 448, b)

# ═════════ 08. GQA ═════════
b = title(400, 30, "GQA — 헤드를 나눠 쓰면 캐시가 줄어든다", "SmolLM2 는 질의 헤드 9개에 KV 헤드 3개")
for x, c, bg, nm, nq, nkv, kvsz in [
    (40, "#94a3b8", "#f1f5f9", "MHA (헤드마다 K/V)", 9, 9, "1,132.5 MB"),
    (420, "#028090", "#edf6f4", "GQA 3:1 (셋이 하나를 공유)", 9, 3, "377.5 MB")]:
    b += f'  <rect x="{x}" y="74" width="340" height="212" rx="13" fill="{bg}" stroke="{c}" stroke-width="2"/>\n'
    b += f'  <text x="{x+170}" y="98" text-anchor="middle" font-size="15" font-weight="700" fill="{c}">{nm}</text>\n'
    for i in range(9):
        qx = x + 24 + i * 34
        b += f'  <rect x="{qx}" y="124" width="26" height="26" rx="5" fill="#7c3aed" fill-opacity="0.85"/>\n'
        gi = i if nkv == 9 else i // 3
        kx = x + 24 + (gi * 34 if nkv == 9 else 24 + gi * 100)
        end = (kx + 13) if nkv == 9 else (x + 85 + gi * 100)
        b += f'  <path d="M{qx+13} 152 L{end} 190" stroke="{c}" stroke-width="1.2" opacity="0.6" fill="none"/>\n'
    b += f'  <text x="{x+170}" y="114" text-anchor="middle" font-size="11.5" fill="#64748b">질의 헤드 {nq}개</text>\n'
    for g in range(nkv):
        kx = x + 24 + (g * 34 if nkv == 9 else 24 + g * 100)
        wd = 26 if nkv == 9 else 74
        b += f'  <rect x="{kx}" y="192" width="{wd}" height="26" rx="5" fill="{c}"/>\n'
    b += f'  <text x="{x+170}" y="238" text-anchor="middle" font-size="11.5" fill="#64748b">KV 헤드 {nkv}개</text>\n'
    b += f'  <rect x="{x+70}" y="248" width="200" height="28" rx="7" fill="#ffffff" stroke="{c}"/>\n'
    b += f'  <text x="{x+170}" y="267" text-anchor="middle" font-size="13.5" font-weight="700" fill="{c}">문맥 8,192 → {kvsz}</text>\n'
b += note(40, 302, 720, 96, "\"업트레이닝한 GQA 가 MQA 에 필적하는 속도로 MHA 에 가까운 품질을 달성한다\"",
          ["— Ainslie 외, GQA (EMNLP 2023). 다만 공짜가 아니다 — 원 사전학습 연산량의 5% 를 업트레이닝에 쓴다",
           "그리고 \"속도가 3배 빨라진다\" 로 옮기면 틀린다 — 그것은 KV 가 지배하는 구간의 이야기다",
           "온디바이스 배치 1 에서 KV 는 가중치의 0.70배에 그친다"],
          bg="#fff7ed", ln="#f59e0b", hc="#b45309")
fig("w10_p2_gqa_08", 800, 414, b)

# ═════════ 09. 배치 반전 ═════════
b = title(400, 30, "배치의 교환비가 9주차와 정반대다", "배치 1 → 8 (실측)")
BA = L["batch"]
b1, b8 = BA[0], BA[-1]
XL, XR, PB, PT = 120, 700, 260, 92
mxf = max(r["tps"] for r in BA) * 1.15
for i, r in enumerate(BA):
    x = XL + 30 + i * 140
    h = r["tps"] / mxf * (PB - PT)
    b += f'  <rect x="{x}" y="{PB-h:.1f}" width="52" height="{h:.1f}" rx="4" fill="#028090" fill-opacity="0.9"/>\n'
    b += f'  <text x="{x+26}" y="{PB-h-9:.1f}" text-anchor="middle" font-size="12.5" font-weight="700" fill="#028090">{r["tps"]:.0f}</text>\n'
    hl = r["step_ms"] / (max(x2["step_ms"] for x2 in BA) * 1.15) * (PB - PT)
    b += f'  <rect x="{x+62}" y="{PB-hl:.1f}" width="52" height="{hl:.1f}" rx="4" fill="#dc2626" fill-opacity="0.9"/>\n'
    b += f'  <text x="{x+88}" y="{PB-hl-9:.1f}" text-anchor="middle" font-size="12.5" font-weight="700" fill="#b91c1c">{r["step_ms"]:.0f}</text>\n'
    b += f'  <text x="{x+57}" y="{PB+20}" text-anchor="middle" font-size="13" font-weight="700" fill="#0f172a">배치 {r["B"]}</text>\n'
b += f'  <line x1="{XL}" y1="{PB}" x2="{XR}" y2="{PB}" stroke="#94a3b8" stroke-width="1.5"/>\n'
b += ('  <rect x="220" y="286" width="380" height="28" rx="6" fill="#ffffff" stroke="#cbd5e1"/>\n'
      '  <rect x="232" y="293" width="13" height="13" fill="#028090"/>\n'
      '  <text x="252" y="304" font-size="11.5" fill="#334155">처리량 (tok/s)</text>\n'
      '  <rect x="392" y="293" width="13" height="13" fill="#dc2626"/>\n'
      '  <text x="412" y="304" font-size="11.5" fill="#334155">사용자가 느끼는 TPOT (ms)</text>\n')
cmp_rows = [("9주차 비전 (배치 1→8)", "1.24배", "6.5배", "나쁜 거래", "#dc2626"),
            ("10주차 LLM 디코드 (배치 1→8)",
             f'{b8["tps"]/b1["tps"]:.1f}배', f'{b8["step_ms"]/b1["step_ms"]:.2f}배', "좋은 거래", "#16a34a")]
for i, (nm, g, c2, v, col) in enumerate(cmp_rows):
    y = 330 + i * 40
    b += f'  <rect x="60" y="{y}" width="680" height="34" rx="6" fill="{"#ecfdf5" if i else "#fef2f2"}" stroke="{col}"/>\n'
    b += f'  <text x="80" y="{y+22}" font-size="12.5" font-weight="700" fill="#334155">{nm}</text>\n'
    b += f'  <text x="440" y="{y+22}" text-anchor="middle" font-size="12.5" fill="#475569">처리량 {g}</text>\n'
    b += f'  <text x="580" y="{y+22}" text-anchor="middle" font-size="12.5" fill="#475569">지연 {c2}</text>\n'
    b += f'  <text x="700" y="{y+22}" text-anchor="middle" font-size="13" font-weight="700" fill="{col}">{v}</text>\n'
b += note(60, 418, 680, 76, "왜 뒤집히는가 — 가중치 읽기를 나눠 낸다",
          ["\"요청들이 같은 모델 가중치를 공유하므로, 가중치를 옮기는 오버헤드가 배치 안의 요청들에 걸쳐 분할 상환된다\"",
           "— Kwon 외, vLLM (SOSP 2023) §2.3.   9주차의 합성곱 모델은 나눠 낼 것이 애초에 없었다"])
fig("w10_p2_batch_09", 800, 510, b)

# ═════════ 10. 서빙 ═════════
b = title(400, 30, "서빙 시스템이 푸는 두 문제", "배치는 저절로 채워지지 않는다")
for x, c, bg, hd, prob, sol, quote in [
    (30, "#2563eb", "#eff6ff", "반복 수준 스케줄링 (Orca)",
     "요청마다 생성 길이가 다르다 —\n먼저 끝난 자리가 놀면서 기다린다",
     "토큰 하나를 만들 때마다\n배치 구성을 다시 짠다",
     "논문의 용어는 iteration-level scheduling.\n\"연속 배칭\" 은 나중에 붙은 이름이다"),
    (410, "#7c3aed", "#ede9fe", "PagedAttention (vLLM)",
     "KV 캐시를 미리 얼마나 잡나 —\n크게 잡으면 낭비, 작게 잡으면 부족",
     "운영체제의 가상 메모리를 흉내 낸다.\n고정 크기 블록 + 논리/물리 분리",
     "\"기존 시스템에서는 KV 캐시 메모리의\n20.4~38.2% 만이 실제로 쓰인다\"")]:
    b += f'  <rect x="{x}" y="74" width="360" height="230" rx="13" fill="{bg}" stroke="{c}" stroke-width="2"/>\n'
    b += f'  <text x="{x+180}" y="102" text-anchor="middle" font-size="15.5" font-weight="700" fill="{c}">{hd}</text>\n'
    b += f'  <text x="{x+24}" y="128" font-size="11.5" font-weight="700" fill="#64748b">문제</text>\n'
    for j, l in enumerate(prob.split("\n")):
        b += f'  <text x="{x+24}" y="{148+j*19}" font-size="12.3" fill="#334155">{l}</text>\n'
    b += f'  <text x="{x+24}" y="{200}" font-size="11.5" font-weight="700" fill="#64748b">해법</text>\n'
    for j, l in enumerate(sol.split("\n")):
        b += f'  <text x="{x+24}" y="{220+j*19}" font-size="12.3" font-weight="700" fill="{c}">{l}</text>\n'
    b += f'  <rect x="{x+18}" y="256" width="324" height="38" rx="7" fill="#ffffff" stroke="{c}" stroke-width="0.9"/>\n'
    for j, l in enumerate(quote.split("\n")):
        b += f'  <text x="{x+180}" y="{272+j*16}" text-anchor="middle" font-size="10.6" fill="#475569">{l}</text>\n'
b += note(30, 320, 740, 96, "인용 주의 — 이 분야는 블로그 수치가 논문 수치로 둔갑하는 일이 특히 잦다",
          ["Orca 의 \"36.9배\" 는 GPT-3 175B · FasterTransformer 대비 · 정규화 중앙값 190 ms 목표 아래의 값이다",
           "vLLM 의 향상은 Orca(Oracle) 대비 2~4배다. 도는 \"24배\" 는 블로그의 HuggingFace 대비 수치다",
           "그리고 Orca(Oracle) 은 Orca 가 비공개라 저자들이 만든 이상적 상한이다"],
          bg="#fff7ed", ln="#f59e0b", hc="#b45309")
fig("w10_p2_spec_placeholder", 800, 440, b)
F["w10_p2_serving_10"] = F.pop("w10_p2_spec_placeholder")

# ═════════ 11. 추측 디코딩 ═════════
b = title(400, 30, "추측 디코딩 — 순차성을 우회한다", "작은 모델이 찍고, 큰 모델이 한 번에 검증한다")
b += '  <text x="60" y="96" font-size="13" font-weight="700" fill="#64748b">보통의 디코드</text>\n'
for i in range(5):
    x = 200 + i * 108
    b += f'  <rect x="{x}" y="80" width="86" height="30" rx="5" fill="#e4711b" fill-opacity="0.85"/>\n'
    b += f'  <text x="{x+43}" y="100" text-anchor="middle" font-size="11.5" fill="#ffffff">큰 모델 1회</text>\n'
    if i < 4:
        b += f'  <path d="M{x+88} 95 H{x+104}" stroke="#94a3b8" stroke-width="1.8" marker-end="url(#b)"/>\n'
b += '  <text x="60" y="168" font-size="13" font-weight="700" fill="#64748b">추측 디코딩</text>\n'
for i in range(4):
    x = 200 + i * 62
    b += f'  <rect x="{x}" y="152" width="52" height="30" rx="5" fill="#2563eb" fill-opacity="0.85"/>\n'
    b += f'  <text x="{x+26}" y="172" text-anchor="middle" font-size="10.5" fill="#ffffff">초안</text>\n'
b += '  <path d="M452 167 H472" stroke="#94a3b8" stroke-width="1.8" marker-end="url(#b)"/>\n'
b += '  <rect x="478" y="152" width="250" height="30" rx="5" fill="#16a34a" fill-opacity="0.9"/>\n'
b += '  <text x="603" y="172" text-anchor="middle" font-size="12" font-weight="700" fill="#ffffff">큰 모델이 네 개를 한 번에 검증</text>\n'
b += '  <text x="200" y="206" font-size="11.5" fill="#64748b">작고 빠른 모델이 앞질러 찍는다</text>\n'
b += '  <text x="478" y="206" font-size="11.5" fill="#64748b">검증은 프리필처럼 병렬이 된다</text>\n'
b += note(60, 226, 680, 116, "핵심은 속도가 아니라 보증이다",
          ["\"큰 모델에서의 정확한 디코딩을 … 여러 토큰을 동시에 생성하면서도 분포를 바꾸지 않고 더 빠르게\"",
           "— Leviathan, Kalman, Matias (ICML 2023). T5-XXL 에서 2~3배, 동일한 출력",
           "DeepMind 판(Chen 외, arXiv 2023)은 \"하드웨어 수치 범위 안에서 분포를 보존\" 이라고 더 조심스럽다",
           "2~3배는 T5-XXL·TPU·T5X 대비 값이다 — 온디바이스 이득은 초안 모델의 수용률에 달려 있다"])
b += '  <defs>' + MK.format(i="b", c="#94a3b8") + '</defs>\n'
fig("w10_p2_spec_11", 800, 358, b)

# ═════════ 12. 토큰 예산 ═════════
b = title(400, 30, "내 토크나이저로 예산을 세운다", "여덟 줄이 응용의 성립 여부를 알려 준다")
XL, PB, PT = 150, 264, 90
rows12 = [("영어", need_en["need_tps"], "#2563eb"), ("한국어", need_ko["need_tps"], "#dc2626")]
mxb = max(max(r[1] for r in rows12), got) * 1.2
for i, (nm, need, c) in enumerate(rows12):
    x = XL + 60 + i * 260
    hn = need / mxb * (PB - PT)
    hg = got / mxb * (PB - PT)
    b += f'  <rect x="{x}" y="{PB-hn:.1f}" width="66" height="{hn:.1f}" rx="4" fill="#dc2626" fill-opacity="0.9"/>\n'
    b += f'  <text x="{x+33}" y="{PB-hn-9:.1f}" text-anchor="middle" font-size="12.5" font-weight="700" fill="#b91c1c">{need:.1f}</text>\n'
    b += f'  <rect x="{x+80}" y="{PB-hg:.1f}" width="66" height="{hg:.1f}" rx="4" fill="#028090" fill-opacity="0.9"/>\n'
    b += f'  <text x="{x+113}" y="{PB-hg-9:.1f}" text-anchor="middle" font-size="12.5" font-weight="700" fill="#028090">{got:.1f}</text>\n'
    ok = got >= need
    b += f'  <text x="{x+73}" y="{PB+20}" text-anchor="middle" font-size="13.5" font-weight="700" fill="#0f172a">{nm}</text>\n'
    b += (f'  <text x="{x+73}" y="{PB+42}" text-anchor="middle" font-size="13" font-weight="700" '
          f'fill="{"#16a34a" if ok else "#dc2626"}">{got/need:.2f}배 · {"여유 있다" if ok else "못 따라간다"}</text>\n')
b += f'  <line x1="{XL}" y1="{PB}" x2="{XL+520}" y2="{PB}" stroke="#94a3b8" stroke-width="1.5"/>\n'
b += ('  <rect x="230" y="312" width="360" height="28" rx="6" fill="#ffffff" stroke="#cbd5e1"/>\n'
      '  <rect x="242" y="319" width="13" height="13" fill="#dc2626"/>\n'
      '  <text x="262" y="330" font-size="11.5" fill="#334155">필요 (묵독 238 wpm)</text>\n'
      '  <rect x="420" y="319" width="13" height="13" fill="#028090"/>\n'
      '  <text x="440" y="330" font-size="11.5" fill="#334155">실측 디코드</text>\n')
b += note(60, 356, 680, 96, "같은 모델이 영어에서는 통과하고 한국어에서는 미달이다",
          ["필요 속도 = 읽기 속도(wpm) ÷ 60 ÷ 토큰당 단어 수 — 마지막 항이 언어마다 다르다",
           "\"1 토큰 ≈ 0.75 단어\" 는 OpenAI 토크나이저·영어 기준이다. 그대로 쓰면 안 된다",
           "여러분이 쓸 실제 모델의 토크나이저로, 실제 문장으로, 직접 세라"],
          bg="#fff7ed", ln="#f59e0b", hc="#b45309")
fig("w10_p3_budget_12", 800, 472, b)

for k, v in F.items():
    (OUT / f"{k}.svg").write_text(v, encoding="utf-8")
print(f"{len(F)}개 저장 → {OUT}")
