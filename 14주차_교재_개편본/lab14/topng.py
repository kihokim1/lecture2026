# -*- coding: utf-8 -*-
import pathlib, cairosvg
from PIL import Image

SRC = pathlib.Path("/root/ondevice-ai/img/week14")
DST = pathlib.Path("/root/ondevice-ai/wikidocs-repo/assets")
PPT = pathlib.Path("/root/ondevice-ai/pptx_assets/week14"); PPT.mkdir(parents=True, exist_ok=True)

OLD = ["w14_p0_timeline_01", "w14_p0_rubric_02"]
NEW = {s.stem for s in SRC.glob("*.svg")}
for o in OLD:
    if o in NEW:
        continue
    for d in (DST, PPT, SRC):
        for ext in ("png", "svg"):
            p = d / f"{o}.{ext}"
            if p.exists():
                p.unlink()
                print(f"  삭제 {p}")

for s in sorted(SRC.glob("*.svg")):
    t = s.read_text(encoding="utf-8").replace(
        "'Segoe UI',Arial,sans-serif", "'Noto Sans CJK KR','Segoe UI',sans-serif")
    cairosvg.svg2png(bytestring=t.encode(), write_to=str(DST / f"{s.stem}.png"),
                     scale=2.2, background_color="white")
    o = PPT / f"{s.stem}.png"
    cairosvg.svg2png(bytestring=t.encode(), write_to=str(o), scale=3.0, background_color="white")
    w, h = Image.open(o).size
    print(f'  {s.stem}: {w/h:.3f},')
