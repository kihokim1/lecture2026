"""실험 1 — 우리가 만든 교재를 캡스톤 심사 기준으로 채점한다.

14주차는 남을 채점하는 주차다. 그러니 먼저 우리 자신을 채점한다.
13주차에서 "인용된 숫자에는 조건이 붙어야 한다"고 열두 주 내내 말했다.
**그 말을 우리 교재 54편에 그대로 적용하면 몇 점이 나오는가?**

측정 항목
  ① 수치 주장 중 같은 문장 안에 조건 표지가 있는 비율
  ② 「더 읽어보기」 인용 중 게재처·연도가 명시된 비율
  ③ 그림 캡션 중 실측 표시가 붙은 비율
  ④ 주차별 「재현 정보」 블록 존재 여부
  ⑤ 주차별 실험 코드 존재 여부

주의: 이것은 **정규식 근사**다. 13주차의 자동 서지 조회가 15.6% 틀렸듯
      이 감사도 틀린다. 그래서 표본을 손으로 세어 오차를 함께 보고한다.

출력: audit.json
"""
import json, pathlib, re, collections

PAGES = pathlib.Path("/root/ondevice-ai/wikidocs-repo/pages")
LABS = pathlib.Path("/root")

# ── 수치 주장 패턴 ────────────────────────────────────────────────
UNITS = (r"ms|s|분|시간|%p|%|배|dB|MB|KB|GB|GiB|MiB|B/s|FPS|fps|tok/s|"
         r"mA|µA|uA|mW|W|mJ|J|Hz|kHz|MHz|GHz|개|회|편|건|장|층|비트|bit|"
         r"GFLOPs|MFLOPs|FLOPs|MACs|파라미터")
NUM = re.compile(r"(?<![A-Za-z0-9_.])(\d[\d,]*(?:\.\d+)?)\s*(" + UNITS + r")(?![A-Za-z0-9])")

# ── 조건 표지 ─────────────────────────────────────────────────────
COND = re.compile(
    r"배치|스레드|해상도|시드|문맥|버전|대비|기준|조건|온도|전원|워밍업|"
    r"CPU|GPU|MCU|Jetson|Raspberry|Cortex|ARM|x86|nRF|ESP32|RTX|Titan|Mali|"
    r"MNIST|CIFAR|ImageNet|COCO|KWS|GSC|Speech Commands|WikiText|"
    r"ONNX|TFLite|PyTorch|TensorRT|ORT|onnxruntime|opacus|"
    r"FP32|FP16|INT8|INT4|W8A8|양자화|희소|프루닝|"
    r"에서|기준으로|일 때|경우|설정|\bB=|\bα=|\bσ=|\bε=|§")

# 조건이 필요 없는 수치 — 구조량·목차·시간 배분 등
EXEMPT = re.compile(r"^\s*(\||#|>|\d+\.\s|[-*]\s*\[)")
META = re.compile(r"분\)|배정|교시|주차|쪽|페이지|절|문항|점|명|팀|년|월|일")

CITE_VENUE = re.compile(
    r"Proc\.|Conf\.|Conference|Symposium|Trans\.|Journal|arXiv|"
    r"NeurIPS|NIPS|ICML|ICLR|CVPR|ECCV|ICCV|MLSys|OSDI|SOSP|CCS|PLDI|"
    r"EMNLP|AISTATS|AFIPS|Commun\. ACM|Found\. Trends|IEEE|USENIX|ACM")
CITE_YEAR = re.compile(r"\b(19|20)\d{2}\b")

out = {}
rows = []
tot_claims = tot_cond = 0
tot_cite = tot_cite_ok = 0
tot_fig = tot_fig_real = 0
per_week = collections.defaultdict(lambda: {"claims": 0, "cond": 0, "figs": 0,
                                            "figs_real": 0, "cites": 0, "cites_ok": 0})

samples = []
raw_claims = 0
# 14주차 자신은 감사 대상에서 뺀다 — 감사 결과를 보고하는 문서를
# 그 감사에 넣으면 순환이 된다.
SRC_FILES = [f for f in sorted(PAGES.glob("*.md")) if not f.name.startswith("w14-")]
for f in SRC_FILES:
    m = re.match(r"w(\d+)-", f.name)
    wk = int(m.group(1)) if m else 0
    txt = f.read_text(encoding="utf-8")
    in_read = in_code = False
    for ln in txt.splitlines():
        # 필터 이전의 원시 집계 (필터 효과를 보고하기 위해)
        if not re.match(r"^\s*(#|>)", ln):
            for _s in re.split(r"(?<=다)\.|(?<=\.)\s|\. ", ln):
                _h = [h for h in NUM.findall(_s) if not META.search(h[1])]
                if _h:
                    raw_claims += 1
        # 코드 블록은 산문이 아니다 — 통째로 제외
        if ln.lstrip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        # 콘솔 출력·정렬된 수치 열도 문장이 아니다
        if len(re.findall(r"\s{2,}", ln)) >= 2 and len(re.findall(r"\d", ln)) >= 4:
            continue
        # 「더 읽어보기」 절의 인용
        if re.match(r"^#{2,4}\s*(\d+\.\s*)?더 읽어보기", ln.strip()):
            in_read = True
            continue
        if in_read and ln.startswith("#"):
            in_read = False
        if in_read and re.match(r"^\s*[-*]\s", ln):
            tot_cite += 1
            per_week[wk]["cites"] += 1
            ok = bool(CITE_VENUE.search(ln)) and bool(CITE_YEAR.search(ln))
            if ok:
                tot_cite_ok += 1
                per_week[wk]["cites_ok"] += 1
            continue
        # 그림 캡션
        if ln.strip().startswith("!["):
            alt = ln[ln.find("[") + 1:ln.find("]")]
            tot_fig += 1
            per_week[wk]["figs"] += 1
            if "실측" in alt:
                tot_fig_real += 1
                per_week[wk]["figs_real"] += 1
            continue
        # 수치 주장 — 표(|)와 목록 머리는 문장이 아니므로 제외
        if EXEMPT.match(ln):
            continue
        for sent in re.split(r"(?<=다)\.|(?<=\.)\s|\. ", ln):
            hits = NUM.findall(sent)
            if not hits:
                continue
            real = [h for h in hits if not META.search(h[1])]
            if not real:
                continue
            tot_claims += 1
            per_week[wk]["claims"] += 1
            has = bool(COND.search(sent))
            if has:
                tot_cond += 1
                per_week[wk]["cond"] += 1
            if len(samples) < 400:
                samples.append({"week": wk, "file": f.name, "cond": has,
                                "sent": sent.strip()[:180]})

# ── 주차별 「재현 정보」와 실험 코드 ───────────────────────────────
repro_block, lab_code = {}, {}
for wk in range(1, 14):
    ov = PAGES / f"w{wk:02d}-0-overview.md"
    repro_block[wk] = bool(ov.exists() and "재현 정보" in ov.read_text(encoding="utf-8"))
    d = LABS / f"lab{wk:02d}"
    d2 = LABS / f"lab{wk}"
    lab_code[wk] = d.exists() or d2.exists()

weeks = sorted(w for w in per_week if w)
for w in weeks:
    d = per_week[w]
    rows.append({
        "week": w, "claims": d["claims"], "cond": d["cond"],
        "pct": round(100 * d["cond"] / d["claims"], 1) if d["claims"] else None,
        "figs": d["figs"], "figs_real": d["figs_real"],
        "cites": d["cites"], "cites_ok": d["cites_ok"],
        "repro": repro_block.get(w, False), "code": lab_code.get(w, False),
    })

ap = per_week.get(0, {"claims": 0, "cond": 0, "figs": 0, "figs_real": 0,
                      "cites": 0, "cites_ok": 0})
out["cfg"] = {"n_pages": len(SRC_FILES), "scope": "1~13주차 + 부록 (14주차 자신은 제외)"}
out["rows"] = rows
out["appendix"] = dict(ap)
out["summary"] = {
    "claims": tot_claims, "raw_claims": raw_claims, "cond": tot_cond,
    "cond_pct": round(100 * tot_cond / tot_claims, 1) if tot_claims else 0,
    "cites": tot_cite, "cites_ok": tot_cite_ok,
    "cites_pct": round(100 * tot_cite_ok / tot_cite, 1) if tot_cite else 0,
    "figs": tot_fig, "figs_real": tot_fig_real,
    "figs_pct": round(100 * tot_fig_real / tot_fig, 1) if tot_fig else 0,
    "repro_weeks": sum(1 for w in range(1, 14) if repro_block[w]),
    "code_weeks": sum(1 for w in range(1, 14) if lab_code[w]),
    "n_weeks": 13,
}
out["samples"] = samples

print("=" * 74)
print("교재 자가 감사 — 우리 기준을 우리에게 적용하면")
print("=" * 74)
print(f"  페이지            {out['cfg']['n_pages']}편")
s = out["summary"]
print(f"  필터 전 원시 집계  {s['raw_claims']}개 → 코드/표/콘솔 제외 후 {s['claims']}개")
print(f"  수치 주장 문장    {s['claims']}개 중 조건 표지 있음 {s['cond']}개 "
      f"= {s['cond_pct']}%")
print(f"  인용              {s['cites']}건 중 게재처+연도 명시 {s['cites_ok']}건 "
      f"= {s['cites_pct']}%")
print(f"  그림 캡션         {s['figs']}개 중 실측 표시 {s['figs_real']}개 "
      f"= {s['figs_pct']}%")
print(f"  재현 정보 블록    13주 중 {s['repro_weeks']}주")
print(f"  실험 코드         13주 중 {s['code_weeks']}주")
print("-" * 74)
print(f"{'주차':>4}{'수치문장':>9}{'조건있음':>9}{'비율':>8}{'그림':>6}"
      f"{'실측표시':>9}{'인용':>6}{'서지OK':>8}")
for r in rows:
    print(f"{r['week']:>4}{r['claims']:>9}{r['cond']:>9}"
          f"{(str(r['pct']) + '%') if r['pct'] is not None else '-':>8}"
          f"{r['figs']:>6}{r['figs_real']:>9}{r['cites']:>6}{r['cites_ok']:>8}")

# ══════════════════════════════════════════════════════════════════
# 손 라벨 28문장과 대조 — 이 감사 자체는 얼마나 틀리는가
# ══════════════════════════════════════════════════════════════════
from handcheck import HAND  # noqa: E402

def norm(t):
    return re.sub(r"\s+", "", t)

hand_rows, miss = [], []
for prefix, label in HAND:
    hit = next((x for x in samples if norm(prefix) in norm(x["sent"])), None)
    if hit is None:
        miss.append(prefix)
        continue
    hand_rows.append({"sent": hit["sent"], "auto": hit["cond"], "hand": label})

claims = [r for r in hand_rows if r["hand"] != "NOTCLAIM"]
notclaim = [r for r in hand_rows if r["hand"] == "NOTCLAIM"]
agree = [r for r in claims if r["auto"] == (r["hand"] == "CLAIM_O")]
fp = [r for r in claims if r["auto"] and r["hand"] == "CLAIM_X"]
fn = [r for r in claims if (not r["auto"]) and r["hand"] == "CLAIM_O"]
hand_o = [r for r in claims if r["hand"] == "CLAIM_O"]

out["hand"] = {
    "n": len(hand_rows), "unmatched": miss,
    "n_claims": len(claims), "n_notclaim": len(notclaim),
    "notclaim_pct": round(100 * len(notclaim) / len(hand_rows), 1) if hand_rows else 0,
    "agree": len(agree), "fp": len(fp), "fn": len(fn),
    "accuracy": round(100 * len(agree) / len(claims), 1) if claims else 0,
    "error_rate": round(100 * (len(claims) - len(agree)) / len(claims), 1) if claims else 0,
    "hand_cond_pct": round(100 * len(hand_o) / len(claims), 1) if claims else 0,
    "rows": hand_rows,
}
h = out["hand"]
print("\n" + "=" * 74)
print("이 감사 자체는 얼마나 틀리는가 — 손 라벨 28문장과 대조")
print("=" * 74)
if miss:
    print(f"  ⚠ 매칭 실패 {len(miss)}건: {miss[:3]}")
print(f"  표본 {h['n']}문장 중 애초에 수치 주장이 아닌 것  {h['n_notclaim']}개 = {h['notclaim_pct']}%")
print(f"  → 자동 감사는 분모부터 부풀린다")
print(f"  실제 수치 주장 {h['n_claims']}개에 대해")
print(f"      일치 {h['agree']} · 거짓양성 {h['fp']} · 거짓음성 {h['fn']}")
print(f"      정확도 {h['accuracy']}% · 오류율 {h['error_rate']}%")
print(f"  사람이 센 조건 명시율 {h['hand_cond_pct']}% "
      f" (자동 추정 {out['summary']['cond_pct']}%)")

json.dump(out, open("/root/lab14/audit.json", "w"), ensure_ascii=False, indent=1)
print("\n→ audit.json")
