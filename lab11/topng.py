# -*- coding: utf-8 -*-
import pathlib, cairosvg
from PIL import Image
SRC = pathlib.Path("/root/ondevice-ai/img/week11")
DST = pathlib.Path("/root/ondevice-ai/wikidocs-repo/assets")
PPT = pathlib.Path("/root/ondevice-ai/pptx_assets/week11"); PPT.mkdir(parents=True, exist_ok=True)
for o in ["w11_p1_ai_compiler_01", "w11_p1_host_device_02", "w11_p1_scratchpad_03",
          "w11_p2_tensorrt_04", "w11_p2_npu_sdk_05", "w11_p3_profiling_06"]:
    p = DST / f"{o}.png"
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
    print(f'  {s.stem[8:]}: {w/h:.3f},')
