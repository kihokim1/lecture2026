import torch, torch.nn as nn, torch.nn.functional as F
import torch.nn.utils.prune as prune
import numpy as np, copy, time, os, zipfile
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(1)

d = load_digits()                                    # 8×8 손글씨 숫자 1,797장
X = torch.tensor(d.images.astype(np.float32) / 16.0).unsqueeze(1)
X = F.interpolate(X, size=(28, 28), mode="bilinear", align_corners=False)
y = torch.tensor(d.target).long()
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

class Net(nn.Module):
    def __init__(s, c1=32, c2=64):
        super().__init__()
        s.c1, s.c2 = c1, c2
        s.conv1 = nn.Conv2d(1, c1, 3, padding=1)
        s.conv2 = nn.Conv2d(c1, c2, 3, padding=1)
        s.fc1   = nn.Linear(c2*7*7, 128)
        s.fc2   = nn.Linear(128, 10)
    def forward(s, x):
        x = F.max_pool2d(F.relu(s.conv1(x)), 2)
        x = F.max_pool2d(F.relu(s.conv2(x)), 2)
        return s.fc2(F.relu(s.fc1(x.flatten(1))))

def train(m, epochs=12, lr=1e-3):
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for _ in range(epochs):
        m.train(); perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 64):
            j = perm[i:i+64]
            opt.zero_grad(); F.cross_entropy(m(Xtr[j]), ytr[j]).backward(); opt.step()
    return m

def acc(m):
    m.eval()
    with torch.no_grad(): return (m(Xte).argmax(1) == yte).float().mean().item()

base = train(Net())
print(f"원본 정확도 : {acc(base):.2%}")
print(f"파라미터    : {sum(p.numel() for p in base.parameters()):,}개")
for n, p in base.named_parameters():
    if p.dim() > 1: print(f"   {n:12s} {str(tuple(p.shape)):16s} {p.numel():>8,}")

def prune_global(model, amount):
    m = copy.deepcopy(model)
    ps = [(mod, "weight") for mod in m.modules() if isinstance(mod, (nn.Conv2d, nn.Linear))]
    prune.global_unstructured(ps, pruning_method=prune.L1Unstructured, amount=amount)
    for mod, n in ps: prune.remove(mod, n)      # 마스크를 가중치에 확정
    return m

def sparsity(m):
    z = t = 0
    for mod in m.modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            z += int((mod.weight == 0).sum()); t += mod.weight.nelement()
    return z / t

models = {}
print(f"{'제거율':>7} {'실제 희소성':>11} {'정확도':>9}")
for amt in [0.0, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]:
    m = base if amt == 0 else prune_global(base, amt)
    models[amt] = m
    print(f"{amt:>6.0%} {sparsity(m):>11.1%} {acc(m):>9.2%}")

for name, mod in models[0.9].named_modules():
    if isinstance(mod, (nn.Conv2d, nn.Linear)):
        w = mod.weight
        print(f"  {name:8s} {str(tuple(w.shape)):16s} {float((w==0).sum())/w.nelement():>7.2%}")

def nonzero(m):
    nz = tot = 0
    for mod in m.modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            nz += int((mod.weight != 0).sum()); tot += mod.weight.nelement()
    return nz, tot

for amt in sorted(models):
    nz, tot = nonzero(models[amt])
    print(f"{amt:>6.0%}  0이 아닌 가중치 {nz:>8,} / {tot:,}   장부상 {tot/nz:>5.1f}배 감소")

def bench_all(models, bs=32, runs=40, reps=5):
    x = torch.randn(bs, 1, 28, 28)
    with torch.no_grad():
        for _ in range(50):
            for m in models.values(): m(x)       # 전체 워밍업
        res = {k: [] for k in models}
        for _ in range(reps):                    # 라운드로빈 — 순서 효과 제거
            for k, m in models.items():
                s = []
                for _ in range(runs):
                    t0 = time.perf_counter(); m(x); s.append((time.perf_counter()-t0)*1000)
                res[k].append(np.median(s))
    return {k: (float(np.median(v)), float(np.std(v))) for k, v in res.items()}

r = bench_all(models); b = r[0.0][0]
print(f"{'제거율':>7} {'지연(ms)':>13} {'속도비':>8}")
for amt in sorted(models):
    md, sd = r[amt]
    print(f"{amt:>6.0%} {md:>9.2f}±{sd:>3.2f} {b/md:>8.2f}")

def sizes(m, tag):
    torch.save(m.state_dict(), f"{tag}.pt")
    with zipfile.ZipFile(f"{tag}.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.write(f"{tag}.pt")
    return os.path.getsize(f"{tag}.pt"), os.path.getsize(f"{tag}.zip")

print(f"{'제거율':>7} {'.pt 파일':>14} {'압축(zip)':>14} {'압축비':>8}")
for amt in sorted(models):
    raw, comp = sizes(models[amt], f"m{int(amt*100):02d}")
    print(f"{amt:>6.0%} {raw:>11,} B {comp:>11,} B {raw/comp:>7.1f}x")

def shrink(model, keep_ratio):
    """conv2의 출력 채널을 L1 기준으로 골라 실제로 제거한 '더 작은 모델'을 만든다."""
    c2_new = int(round(model.c2 * keep_ratio))
    imp  = model.conv2.weight.detach().abs().sum(dim=(1,2,3))       # 필터별 L1 중요도
    keep = torch.argsort(imp, descending=True)[:c2_new].sort().values
    new = Net(c1=model.c1, c2=c2_new)
    new.conv1.load_state_dict(model.conv1.state_dict())
    new.conv2.weight.data = model.conv2.weight.data[keep].clone()
    new.conv2.bias.data   = model.conv2.bias.data[keep].clone()
    W = model.fc1.weight.data.view(128, model.c2, 49)               # 채널당 7×7=49열
    new.fc1.weight.data = W[:, keep, :].reshape(128, c2_new*49).clone()
    new.fc1.bias.data   = model.fc1.bias.data.clone()
    new.fc2.load_state_dict(model.fc2.state_dict())
    return new

s50, s75 = shrink(base, 0.5), shrink(base, 0.25)
a0 = {"구조적50%(32ch)": acc(s50), "구조적75%(16ch)": acc(s75)}   # 미세조정 전에 미리 잰다
sm = {"원본(64ch)": base, "구조적50%(32ch)": s50, "구조적75%(16ch)": s75}
rs = bench_all(sm); bb = rs["원본(64ch)"][0]

print(f"{'모델':>18} {'파라미터':>11} {'지연(ms)':>13} {'속도비':>7} {'자른 직후':>9} {'미세조정후':>10}")
for k, m in sm.items():
    md, sd = rs[k]; n = sum(p.numel() for p in m.parameters())
    if k == "원본(64ch)":
        print(f"{k:>18} {n:>11,} {md:>9.2f}±{sd:>3.2f} {bb/md:>7.2f} {acc(m):>9.2%} {'—':>10}")
    else:
        ft = train(copy.deepcopy(m), epochs=10, lr=5e-4)
        print(f"{k:>18} {n:>11,} {md:>9.2f}±{sd:>3.2f} {bb/md:>7.2f} {a0[k]:>9.2%} {acc(ft):>10.2%}")

torch.save(s50.state_dict(), "shrunk50.pt")
print(f"\n구조적 50% 모델 파일 : {os.path.getsize('shrunk50.pt'):,} B "
      f"(원본 {os.path.getsize('m00.pt'):,} B)")
