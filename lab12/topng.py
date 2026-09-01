# -*- coding: utf-8 -*-
import pathlib, cairosvg
from PIL import Image
SRC = pathlib.Path("/root/ondevice-ai/img/week12")
DST = pathlib.Path("/root/ondevice-ai/wikidocs-repo/assets")
PPT = pathlib.Path("/root/ondevice-ai/pptx_assets/week12"); PPT.mkdir(parents=True, exist_ok=True)
for o in ["w12_p1_centralized_vs_fl_01", "w12_p1_fedavg_02", "w12_p1_noniid_03",
          "w12_p2_lora_peft_04", "w12_p2_privacy_05", "w12_p3_fedavg_sim_06"]:
    for d in (DST, PPT, SRC):
        for ext in ("png", "svg"):
            p = d / f"{o}.{ext}"
            if p.exists() and not (d == SRC and o in
                                   [s.stem for s in SRC.glob("*.svg")] and False):
                pass
NEW = {s.stem for s in SRC.glob("*.svg")}
for o in ["w12_p1_centralized_vs_fl_01", "w12_p1_fedavg_02", "w12_p1_noniid_03",
          "w12_p2_lora_peft_04", "w12_p2_privacy_05", "w12_p3_fedavg_sim_06"]:
    if o not in NEW:
        for d in (DST, PPT):
            p = d / f"{o}.png"
            if p.exists():
                p.unlink()
for s in sorted(SRC.glob("*.svg")):
    t = s.read_text(encoding="utf-8").replace(
        "'Segoe UI',Arial,sans-serif", "'Noto Sans CJK KR','Segoe UI',sans-serif")
    cairosvg.svg2png(bytestring=t.encode(), write_to=str(DST / f"{s.stem}.png"),
                     scale=2.2, background_color="white")
    o = PPT / f"{s.stem}.png"
    cairosvg.svg2png(bytestring=t.encode(), write_to=str(o), scale=3.0, background_color="white")
    w, h = Image.open(o).size
    print(f'  {s.stem}: {w/h:.3f},')
