"""4·5주차의 digits 데이터로 증류가 되는지 확인 — 데이터셋을 바꾼 이유의 근거."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, json
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.set_num_threads(1)
d = load_digits()
X, y = d.images.astype(np.float32) / 16.0, d.target
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
Xtr = torch.from_numpy(Xtr).unsqueeze(1); Xte = torch.from_numpy(Xte).unsqueeze(1)
ytr = torch.from_numpy(ytr).long(); yte = torch.from_numpy(yte).long()
print(f"digits: 학습 {len(Xtr)} / 테스트 {len(Xte)}", flush=True)


class C(nn.Module):
    def __init__(s, w):
        super().__init__()
        s.c1 = nn.Conv2d(1, w, 3, padding=1); s.c2 = nn.Conv2d(w, 2 * w, 3, padding=1)
        s.f = nn.Linear(2 * w * 4 * 4, 10)

    def forward(s, x):
        x = F.relu(s.c1(x)); x = F.max_pool2d(F.relu(s.c2(x)), 2)
        return s.f(x.flatten(1))


def npar(m): return sum(p.numel() for p in m.parameters())


def ev(m):
    m.eval()
    with torch.no_grad():
        return 100.0 * (m(Xte).argmax(1) == yte).float().mean().item()


def tr(m, ep=60, seed=0, tl=None, T=4.0, a=0.7):
    torch.manual_seed(seed)
    o = torch.optim.Adam(m.parameters(), lr=1e-3)
    sc = torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=ep)
    g = torch.Generator().manual_seed(seed + 1)
    for _ in range(ep):
        m.train()
        p = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(Xtr), 64):
            b = p[i:i + 64]; out = m(Xtr[b])
            if tl is None:
                L = F.cross_entropy(out, ytr[b])
            else:
                L = a * F.kl_div(F.log_softmax(out / T, 1), F.log_softmax(tl[b] / T, 1),
                                 reduction="batchmean", log_target=True) * T * T \
                    + (1 - a) * F.cross_entropy(out, ytr[b])
            o.zero_grad(); L.backward(); o.step()
        sc.step()
    return ev(m)


res = {}
big = C(16); res["교사(w=16)"] = round(np.mean([tr(C(16), seed=s) for s in range(3)]), 2)
tm = C(16); tr(tm, seed=0)
tm.eval()
with torch.no_grad():
    TL = tm(Xtr)
res["학생(w=4) CE"] = round(np.mean([tr(C(4), seed=s) for s in range(3)]), 2)
res["학생(w=4) KD"] = round(np.mean([tr(C(4), seed=s, tl=TL) for s in range(3)]), 2)
res["파라미터_교사"] = npar(C(16)); res["파라미터_학생"] = npar(C(4))
res["교사_학습데이터_정확도"] = round(100 * (TL.argmax(1) == ytr).float().mean().item(), 2)
print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)
json.dump(res, open("/root/ondevice-ai/lab06/digits.json", "w"), ensure_ascii=False, indent=1)
