"""6주차 지식 증류 — 최종 실측 실험.

데이터 분할 (Fashion-MNIST, 전부 서로소):
  교사 학습 20,000 | 학생 라벨 10,000 | 라벨 없는 전이 집합 30,000 | 테스트 10,000

핵심: 교사를 학생과 **다른 데이터**로 학습시켜야 교사의 소프트 타깃이
학생에게 새로운 정보가 된다. 같은 데이터로 학습한 교사는 그 데이터를 외워
소프트 타깃이 사실상 원-핫이 되고, 증류가 아무것도 전달하지 못한다.
"""
import json, os, time
import numpy as np
import torch
import torch.nn.functional as F
from kd_common import _idx, _prep, Net, train, evaluate, logits_of, n_params, bench_latency

OUT = "/root/ondevice-ai/lab06/final.json"
CK = "/root/ondevice-ai/lab06/ck"
os.makedirs(CK, exist_ok=True)
R = json.load(open(OUT)) if os.path.exists(OUT) else {}


def save():
    json.dump(R, open(OUT, "w"), indent=1, ensure_ascii=False)


def stamp(k, v):
    R[k] = v; save(); print(f"  ▸ {k}: {v}", flush=True)


# ───────────────────── 데이터 ─────────────────────
d = "/root/data/fmnist"
xtr = _idx(f"{d}/train-images-idx3-ubyte.gz"); ytr = _idx(f"{d}/train-labels-idx1-ubyte.gz")
xte = _idx(f"{d}/t10k-images-idx3-ubyte.gz");  yte = _idx(f"{d}/t10k-labels-idx1-ubyte.gz")
rng = np.random.RandomState(0)
ti, si, ui = [], [], []
for c in range(10):
    pool = rng.permutation(np.where(ytr == c)[0])
    ti.append(pool[:2000]); si.append(pool[2000:3000]); ui.append(pool[3000:6000])
ti, si, ui = [rng.permutation(np.concatenate(a)) for a in (ti, si, ui)]
XT, YT = _prep(xtr[ti]), torch.from_numpy(ytr[ti].astype(np.int64))
XS, YS = _prep(xtr[si]), torch.from_numpy(ytr[si].astype(np.int64))
XU, YU = _prep(xtr[ui]), torch.from_numpy(ytr[ui].astype(np.int64))
XE, YE = _prep(xte), torch.from_numpy(yte.astype(np.int64))
XA = torch.cat([XS, XU])
print(f"교사 {len(XT)} | 학생라벨 {len(XS)} | 무라벨 {len(XU)} | 테스트 {len(XE)}", flush=True)

EP_SMALL, EP_BIG = 80, 60   # 10k 전용 / 40k 전이집합
SEEDS3, SEEDS2 = [0, 1, 2], [0, 1]

# ───────────────────── 1. 교사 4종 ─────────────────────
TW = [8, 16, 32, 64]
TM, TLA, TLS = {}, {}, {}
for w in TW:
    p = f"{CK}/t{w}.pt"; m = Net(w)
    if os.path.exists(p):
        m.load_state_dict(torch.load(p))
    else:
        t0 = time.time(); train(m, XT, YT, XE, YE, epochs=25, seed=300 + w)
        torch.save(m.state_dict(), p); print(f"교사 w={w} {time.time()-t0:.0f}s", flush=True)
    TM[w] = m
    TLA[w] = logits_of(m, XA); TLS[w] = logits_of(m, XS)
    ltr = TLA[w]
    stamp(f"teacher_w{w}", {
        "params": n_params(m), "acc": round(evaluate(m, XE, YE), 2),
        "file_B": os.path.getsize(p), "latency_ms": round(bench_latency(m, XE), 4),
        "전이집합_정확도": round(100 * (ltr.argmax(1) == torch.cat([YS, YU])).float().mean().item(), 2),
        "평균_최대확률": round(F.softmax(ltr, 1).max(1).values.mean().item(), 4)})

PSEUDO = torch.cat([YS, TLA[16][len(XS):].argmax(1)])
YA_TRUE = torch.cat([YS, YU])


def run(tag, x, y, ep, seeds=SEEDS3, keep=False, **kw):
    if tag in R:
        return R[tag]
    accs, curves = [], []
    for s in seeds:
        m = Net(4)
        acc, _, hist = train(m, x, y, XE, YE, epochs=ep, seed=s, curve=True, **kw)
        accs.append(acc); curves.append(hist)
        if keep and s == 0:
            torch.save(m.state_dict(), f"{CK}/s_{tag}.pt")
    v = {"accs": [round(a, 2) for a in accs],
         "mean": round(float(np.mean(accs)), 2),
         "std": round(float(np.std(accs)), 2),
         "curve": curves[0]}
    stamp(tag, v); return v


# ───────────────────── 2. 핵심 비교 ─────────────────────
print("\n── 핵심 비교 ──", flush=True)
run("A_ce_10k", XS, YS, EP_SMALL, keep=True)
run("B_kd_10k", XS, YS, EP_SMALL, teacher_logits=TLS[16], T=4., alpha=0.7)
run("C_kd_transfer", XA, PSEUDO, EP_BIG, keep=True, teacher_logits=TLA[16], T=4., alpha=0.7)
run("D_pseudo_ce", XA, PSEUDO, EP_BIG)
run("E_ce_all_true_labels", XA, YA_TRUE, EP_BIG)
run("F_ls0.1_10k", XS, YS, EP_SMALL, label_smooth=0.1)

# ───────────────────── 3. 교사 용량 격차 ─────────────────────
print("\n── 교사 용량 격차 ──", flush=True)
for w in TW:
    tag = f"G_kd_transfer_w{w}"
    if w == 16:
        R[tag] = R["C_kd_transfer"]; save(); continue
    run(tag, XA, torch.cat([YS, TLA[w][len(XS):].argmax(1)]), EP_BIG,
        seeds=SEEDS2, teacher_logits=TLA[w], T=4., alpha=0.7)

# ───────────────────── 4. Temperature ─────────────────────
print("\n── Temperature ──", flush=True)
for T in [1., 2., 4., 8., 16.]:  # T=4 는 C 재사용
    tag = f"H_T{T:g}"
    if T == 4.:
        R[tag] = R["C_kd_transfer"]; save(); continue
    run(tag, XA, PSEUDO, EP_BIG, seeds=SEEDS2, teacher_logits=TLA[16], T=T, alpha=0.7)

# ───────────────────── 5. alpha ─────────────────────
print("\n── alpha ──", flush=True)
for a in [0.3, 0.9, 1.0]:
    run(f"I_a{a:g}", XA, PSEUDO, EP_BIG, seeds=SEEDS2,
        teacher_logits=TLA[16], T=4., alpha=a)

# ───────────────────── 6. Dark Knowledge 절제 (DKPP) ─────────────────────
print("\n── DKPP (비정답 로짓 섞기) ──", flush=True)
g = torch.Generator().manual_seed(777)
PL = TLA[16].clone()
am = PL.argmax(1)
for i in range(len(PL)):
    others = torch.tensor([c for c in range(10) if c != am[i].item()])
    PL[i, others] = PL[i, others][torch.randperm(9, generator=g)]
run("J_dkpp", XA, PSEUDO, EP_BIG, teacher_logits=PL, T=4., alpha=0.7)

# ───────────────────── 7. 크기·속도 (KD는 아무것도 안 바꾼다) ─────────────────────
print("\n── 크기·속도 ──", flush=True)
cost = {}
for tag in ["A_ce_10k", "C_kd_transfer"]:
    m = Net(4); m.load_state_dict(torch.load(f"{CK}/s_{tag}.pt"))
    cost[tag] = {"params": n_params(m), "file_B": os.path.getsize(f"{CK}/s_{tag}.pt"),
                 "latency_ms": round(bench_latency(m, XE), 4),
                 "acc": round(evaluate(m, XE, YE), 2)}
cost["teacher_w16"] = R["teacher_w16"]
stamp("cost", cost)

# ───────────────────── 8. 5주차 연결 — INT8/INT4 양자화 후 ─────────────────────
print("\n── 양자화 후 ──", flush=True)


def quant(model, bits):
    qmax = 2 ** bits - 1
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() < 2:
                continue
            lo, hi = p.min().item(), p.max().item()
            S = (hi - lo) / qmax
            if S == 0:
                continue
            Z = round(-lo / S)
            p.copy_((torch.clamp(torch.round(p / S + Z), 0, qmax) - Z) * S)
    return model


qq = {}
for tag in ["A_ce_10k", "C_kd_transfer"]:
    for bits in [8, 4, 3]:
        m = Net(4); m.load_state_dict(torch.load(f"{CK}/s_{tag}.pt"))
        qq[f"{tag}_int{bits}"] = round(evaluate(quant(m, bits), XE, YE), 2)
stamp("after_quant", qq)

# ───────────────────── 9. 충실도 (교사와 같은 답을 하는가) ─────────────────────
print("\n── 충실도 ──", flush=True)
tp = logits_of(TM[16], XE).argmax(1)
fid = {}
for tag in ["A_ce_10k", "C_kd_transfer"]:
    m = Net(4); m.load_state_dict(torch.load(f"{CK}/s_{tag}.pt"))
    sp = logits_of(m, XE).argmax(1)
    fid[tag] = {"교사와_같은_예측_%": round(100 * (sp == tp).float().mean().item(), 2),
                "정확도_%": round(100 * (sp == YE).float().mean().item(), 2)}
fid["teacher_정확도_%"] = round(100 * (tp == YE).float().mean().item(), 2)
stamp("fidelity", fid)

# ───────────────────── 10. 클래스별 정확도 (어디가 좋아졌나) ─────────────────────
CLS = ["티셔츠", "바지", "풀오버", "드레스", "코트", "샌들", "셔츠", "스니커즈", "가방", "앵클부츠"]
percls = {}
for tag in ["A_ce_10k", "C_kd_transfer"]:
    m = Net(4); m.load_state_dict(torch.load(f"{CK}/s_{tag}.pt"))
    sp = logits_of(m, XE).argmax(1)
    percls[tag] = {CLS[c]: round(100 * (sp[YE == c] == c).float().mean().item(), 2) for c in range(10)}
stamp("per_class", percls)

print("\n완료.", flush=True)
