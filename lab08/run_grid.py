# -*- coding: utf-8 -*-
"""실험 3 — 세 개의 손잡이(정밀도 · 해상도 · 버퍼 재사용)는 곱해진다."""
import json, itertools
import mem_common as M

SRAM_KB, FLASH_KB = 320, 1024
out = {}

print("[C] 320 KB SRAM / 1 MB Flash 예산 안에 들어오는가 — MobileNetV2")
print("해상도  정밀도 |  Flash(KB)      재사용없음     수명기반    in-place |  판정")
grid = []
for hw in [224, 160, 96, 64]:
    m = M.load(f"/root/lab08/onnx/mbv2_{hw}.onnx")
    for bpe, tag in [(4, "FP32"), (1, "INT8")]:
        a = M.analyze(m, bpe=bpe, inplace=False)
        b = M.analyze(m, bpe=bpe, inplace=True)
        fl, sr = M.kb(a["flash"]), M.kb(b["greedy"])
        v = ("통과" if (fl <= FLASH_KB and sr <= SRAM_KB)
             else ("SRAM만 통과" if sr <= SRAM_KB else "둘 다 초과"))
        grid.append(dict(hw=hw, bpe=bpe, flash=a["flash"], naive=a["sum_all"],
                         life=a["greedy"], inplace=b["greedy"], verdict=v))
        print(f" {hw:3d}²   {tag}  | {fl:9.1f}   {M.kb(a['sum_all']):11.1f} "
              f"{M.kb(a['greedy']):11.1f} {sr:11.1f} | {v}")
out["grid"] = grid

print("\n[D] MLPerf Tiny 4종 — INT8 기준 예산 판정")
tin = []
for name, f in [("ResNet-8", "ResNet-8"), ("DS-CNN", "DS-CNN"),
                ("MobileNetV1-0.25@96", "MobileNetV1"), ("FC-AutoEncoder", "FC-AutoEncoder")]:
    m = M.load(f"/root/lab08/onnx/{f}.onnx")
    a = M.analyze(m, 1, False); b = M.analyze(m, 1, True)
    fl, sr = M.kb(a["flash"]), M.kb(b["greedy"])
    tin.append(dict(name=name, flash=a["flash"], sram=b["greedy"],
                    flash_pct=fl / FLASH_KB * 100, sram_pct=sr / SRAM_KB * 100))
    print(f"  {name:22s} Flash {fl:7.1f} KB ({fl/FLASH_KB*100:5.1f}% 사용) | "
          f"SRAM {sr:6.1f} KB ({sr/SRAM_KB*100:5.1f}% 사용)")
out["tiny_budget"] = tin

json.dump(out, open("/root/lab08/grid.json", "w"), ensure_ascii=False, indent=1)
print("\n저장: grid.json")
