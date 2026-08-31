# -*- coding: utf-8 -*-
"""실험 1 — MCU에서 먼저 바닥나는 것은 Flash 인가 SRAM 인가."""
import json, pathlib, torch, torchvision
import mem_common as M
from tiny_models import MODELS

OUT = pathlib.Path("/root/lab08"); (OUT / "onnx").mkdir(exist_ok=True)
res = {}

# MLPerf Tiny 참조 보드 예산 (STM32F746 Discovery)
SRAM_KB, FLASH_KB = 320, 1024


def export(model, shape, path):
    model.eval()
    torch.onnx.export(model, torch.randn(*shape), path,
                      input_names=["input"], output_names=["out"],
                      opset_version=13, dynamo=False)
    return path


def macs_of(model, shape):
    """Conv/Linear 의 MAC 수를 forward hook 으로 실제로 센다."""
    tot = [0]

    def hk(mod, i, o):
        if isinstance(mod, torch.nn.Conv2d):
            tot[0] += o.numel() * mod.in_channels // mod.groups * \
                mod.kernel_size[0] * mod.kernel_size[1]
        elif isinstance(mod, torch.nn.Linear):
            tot[0] += o.numel() * mod.in_features
    hs = [m.register_forward_hook(hk) for m in model.modules()
          if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]
    model.eval()
    with torch.no_grad():
        model(torch.randn(*shape))
    for h in hs:
        h.remove()
    return tot[0]


def row(path):
    m = M.load(path)
    a = M.analyze(m, bpe=1, inplace=False)
    b = M.analyze(m, bpe=1, inplace=True)
    return dict(flash=a["flash"], naive=a["sum_all"], sram=a["greedy"],
                sram_inplace=b["greedy"], n_alias=b["n_alias"],
                peak_node=a["watermark_at"], n=a["n_nodes"],
                per_node=a["per_node"], profile=a["live_profile"])


# ── A. MLPerf Tiny 참조 모델 4종 ────────────────────────────────────────────
print("[A] MLPerf Tiny 참조 모델 — 파라미터 순위와 SRAM 순위는 다르다")
res["tiny"] = {}
for name, (cls, shape) in MODELS.items():
    m = cls()
    p = str(OUT / "onnx" / (name.split()[0] + ".onnx"))
    export(m, shape, p)
    r = row(p)
    r["params"] = sum(q.numel() for q in m.parameters())
    r["macs"] = macs_of(cls(), shape)
    r["shape"] = list(shape)
    res["tiny"][name] = r
    print(f"  {name:30s} 파라미터 {r['params']:8,d} | MACs {r['macs']:11,d} | "
          f"Flash {M.kb(r['flash']):7.1f} KB | SRAM {M.kb(r['sram_inplace']):7.1f} KB")

# ── B. MobileNetV2 — 해상도만 바꾼다 ────────────────────────────────────────
print("\n[B] MobileNetV2 — 파라미터는 1바이트도 안 변하는데 SRAM 만 변한다")
res["res_sweep"] = {}
for hw in [224, 192, 160, 128, 96, 64, 32]:
    m = torchvision.models.mobilenet_v2(weights=None)
    p = str(OUT / "onnx" / f"mbv2_{hw}.onnx")
    export(m, (1, 3, hw, hw), p)
    r = row(p)
    r["params"] = sum(q.numel() for q in m.parameters())
    r["macs"] = macs_of(torchvision.models.mobilenet_v2(weights=None), (1, 3, hw, hw))
    res["res_sweep"][hw] = r
    ok = "SRAM OK" if M.kb(r["sram_inplace"]) <= SRAM_KB else "SRAM 초과"
    print(f"  {hw:3d}²  파라미터 {r['params']:,}  Flash {M.kb(r['flash']):7.1f} KB  "
          f"SRAM {M.kb(r['sram']):8.1f} → in-place {M.kb(r['sram_inplace']):8.1f} KB   {ok}")

res["budget"] = dict(sram_kb=SRAM_KB, flash_kb=FLASH_KB)
(OUT / "mem.json").write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
print("\n저장: mem.json")
