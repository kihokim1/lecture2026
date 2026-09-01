# -*- coding: utf-8 -*-
import pathlib, cairosvg
from PIL import Image

SRC = pathlib.Path("/root/ondevice-ai/img/week13")
DST = pathlib.Path("/root/ondevice-ai/wikidocs-repo/assets")
PPT = pathlib.Path("/root/ondevice-ai/pptx_assets/week13"); PPT.mkdir(parents=True, exist_ok=True)

# 구판 그림 제거 (개편 전 13주차)
OLD = ["w13_p1_paradigm_llm_01", "w13_p1_timeseries_fm_02", "w13_p1_green_aiot_03",
       "w13_p2_paper_tracks_04", "w13_p2_critical_review_05", "w13_p3_roadmap_06"]
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
