"""실험 2 — 루브릭으로 매긴 점수는 얼마나 흔들리는가.

13주차에서 "구조량은 안 흔들리고 규모량은 흔들린다"를 실측했다.
채점에도 같은 구분이 있다.

    규모량 = 총점 (몇 점인가)
    구조량 = 순위 · 상위 집합 (누가 잘했는가)

채점자가 항목당 확률 q 로 한 단계 오판할 때
  A. 총점이 얼마나 흔들리는가
  B. 채점자를 두 명으로 늘리면 얼마나 줄어드는가
  C. 순위·상위 집합은 얼마나 흔들리는가
  D. 어느 항목이 총점 변동을 지배하는가
  E. 두 팀의 참값 격차가 얼마나 벌어져야 순위 뒤바뀜이 5% 아래로 내려가는가

출력: rubric.json
"""
import json, itertools, statistics as st
import numpy as np

rng = np.random.default_rng(14)

# ── 루브릭 (14주차 교재 §2) ────────────────────────────────────────
ITEMS = [("문제 정의·타깃 적절성", 15),
         ("경량화·가속 기법의 타당성", 20),
         ("정확도 방어율", 20),
         ("타깃 기기 지연·메모리 개선율", 20),
         ("프로파일링 증거 (필수)", 15),
         ("발표·문서화 (재현성)", 10)]
W = np.array([w for _, w in ITEMS], dtype=float)
LEVEL = np.array([0.4, 0.7, 1.0])          # 미흡 · 보통 · 우수
N_TEAM, N_TRIAL, Q = 7, 20000, 0.20        # 7팀 · 시행 2만 · 오판 확률 0.20

out = {"cfg": {"items": [{"name": n, "w": w} for n, w in ITEMS],
               "levels": LEVEL.tolist(), "n_team": N_TEAM,
               "n_trial": N_TRIAL, "q": Q}}


def grade(true_idx, n_grader, rng):
    """참 수준 인덱스 배열(팀×항목) → 채점자 n명 평균 총점."""
    tot = np.zeros(true_idx.shape[0])
    for _ in range(n_grader):
        idx = true_idx.copy()
        flip = rng.random(idx.shape) < Q
        step = rng.choice([-1, 1], size=idx.shape)
        idx = np.clip(idx + flip * step, 0, len(LEVEL) - 1)
        tot += (LEVEL[idx] * W).sum(1)
    return tot / n_grader


# ══════════════════════════════════════════════════════════════════
# A·B. 총점 오차 — 채점자 1명 vs 2명 vs 3명
# ══════════════════════════════════════════════════════════════════
print("A·B. 총점 오차 (채점자 수별)")
res = {}
for ng in (1, 2, 3):
    errs, r = [], np.random.default_rng(100 + ng)
    for _ in range(N_TRIAL // 20):
        ti = r.integers(0, len(LEVEL), size=(N_TEAM, len(ITEMS)))
        truth = (LEVEL[ti] * W).sum(1)
        errs.extend(np.abs(grade(ti, ng, r) - truth))
    e = np.array(errs)
    res[ng] = {"mae": round(float(e.mean()), 3), "sd": round(float(e.std()), 3),
               "p95": round(float(np.percentile(e, 95)), 3),
               "max": round(float(e.max()), 3)}
    print(f"   채점자 {ng}명  평균 절대 오차 {res[ng]['mae']:5.2f}점 · "
          f"95분위 {res[ng]['p95']:5.2f}점 · 최대 {res[ng]['max']:5.2f}점")
out["graders"] = res
out["gain_1_to_2"] = round(res[1]["mae"] / res[2]["mae"], 3)

# ══════════════════════════════════════════════════════════════════
# C. 순위와 상위 집합은 얼마나 흔들리는가 (구조량)
# ══════════════════════════════════════════════════════════════════
print("\nC. 규모량(총점) vs 구조량(순위·상위 집합)")


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(-a)), np.argsort(np.argsort(-b))
    n = len(a)
    return 1 - 6 * ((ra - rb) ** 2).sum() / (n * (n * n - 1))


r = np.random.default_rng(7)
sp, top3, top1, score_cv = [], [], [], []
for _ in range(N_TRIAL // 10):
    ti = r.integers(0, len(LEVEL), size=(N_TEAM, len(ITEMS)))
    truth = (LEVEL[ti] * W).sum(1)
    got = grade(ti, 2, r)
    sp.append(spearman(truth, got))
    t_true = set(np.argsort(-truth)[:3])
    t_got = set(np.argsort(-got)[:3])
    top3.append(len(t_true & t_got) / 3)
    top1.append(int(np.argmax(truth) == np.argmax(got)))
    score_cv.append(100 * np.abs(got - truth).mean() / truth.mean())
out["structure"] = {
    "spearman_mean": round(float(np.mean(sp)), 4),
    "top3_overlap": round(100 * float(np.mean(top3)), 1),
    "top1_hit": round(100 * float(np.mean(top1)), 1),
    "score_err_pct": round(float(np.mean(score_cv)), 2),
}
s = out["structure"]
print(f"   총점 상대 오차   {s['score_err_pct']}%   ← 규모량")
print(f"   순위 상관(스피어만) {s['spearman_mean']}   ← 구조량")
print(f"   상위 3팀 집합 일치 {s['top3_overlap']}% · 1등 적중 {s['top1_hit']}%")

# ══════════════════════════════════════════════════════════════════
# D. 어느 항목이 총점 변동을 지배하는가
# ══════════════════════════════════════════════════════════════════
print("\nD. 항목별 총점 변동 기여")
r = np.random.default_rng(21)
contrib = []
for j in range(len(ITEMS)):
    d = []
    for _ in range(4000):
        ti = r.integers(0, len(LEVEL), size=(1, len(ITEMS)))
        base = (LEVEL[ti] * W).sum()
        idx = ti.copy()
        step = r.choice([-1, 1])
        idx[0, j] = np.clip(idx[0, j] + step, 0, len(LEVEL) - 1)
        d.append(abs((LEVEL[idx] * W).sum() - base))
    contrib.append(float(np.mean(d)))
tot_c = sum(contrib)
out["contrib"] = [{"name": ITEMS[j][0], "w": ITEMS[j][1],
                   "shift": round(contrib[j], 3),
                   "share": round(100 * contrib[j] / tot_c, 1)}
                  for j in range(len(ITEMS))]
for c in sorted(out["contrib"], key=lambda x: -x["shift"]):
    print(f"   {c['name']:<26}{c['w']:>3}점   한 단계 오판 시 총점 {c['shift']:5.2f}점 "
          f"이동 · 기여 {c['share']:4.1f}%")

# ══════════════════════════════════════════════════════════════════
# E. 두 팀의 참값 격차 대비 순위 뒤바뀜 확률
# ══════════════════════════════════════════════════════════════════
print("\nE. 참값 격차와 순위 뒤바뀜 (채점자 2명)")
r = np.random.default_rng(33)
flip_rows = []
for gap_items in range(0, 7):
    # A팀: 앞 gap_items 개 항목이 한 단계 높다
    n, flips, gaps = 6000, 0, []
    for _ in range(n):
        base = r.integers(0, len(LEVEL) - 1, size=(len(ITEMS),))
        a = base.copy()
        a[:gap_items] = np.clip(a[:gap_items] + 1, 0, len(LEVEL) - 1)
        ti = np.stack([a, base])
        truth = (LEVEL[ti] * W).sum(1)
        if truth[0] <= truth[1]:
            continue
        gaps.append(truth[0] - truth[1])
        got = grade(ti, 2, r)
        flips += int(got[0] <= got[1])
    if not gaps:
        continue
    flip_rows.append({"gap_items": gap_items,
                      "gap_mean": round(float(np.mean(gaps)), 2),
                      "flip_pct": round(100 * flips / len(gaps), 1),
                      "n": len(gaps)})
out["flip"] = flip_rows
for f in flip_rows:
    print(f"   상위 팀이 {f['gap_items']}개 항목 우세 → 참값 격차 평균 {f['gap_mean']:5.2f}점 · "
          f"순위 뒤바뀜 {f['flip_pct']:5.1f}%")
safe = next((f for f in flip_rows if f["flip_pct"] < 5.0), None)
out["safe_gap"] = safe
if safe:
    print(f"\n   → 뒤바뀜 5% 아래로 내려가려면 참값 격차가 {safe['gap_mean']}점 이상이어야 한다")

json.dump(out, open("/root/lab14/rubric.json", "w"), ensure_ascii=False, indent=1)
print("\n→ rubric.json")
