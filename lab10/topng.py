# -*- coding: utf-8 -*-
import pathlib, cairosvg
from PIL import Image
SRC = pathlib.Path("/root/ondevice-ai/img/week10")
DST = pathlib.Path("/root/ondevice-ai/wikidocs-repo/assets")
PPT = pathlib.Path("/root/ondevice-ai/pptx_assets/week10"); PPT.mkdir(parents=True, exist_ok=True)
for o in ["w10_p1_attention_bottleneck_01","w10_p1_kv_cache_02","w10_p1_compact_bert_03",
          "w10_p2_quant_4bit_04","w10_p2_speculative_05","w10_p3_bench_06"]:
    p = DST / f"{o}.png"
    if p.exists(): p.unlink()
for s in sorted(SRC.glob("*.svg")):
    t = s.read_text(encoding="utf-8").replace(
        "'Segoe UI',Arial,sans-serif", "'Noto Sans CJK KR','Segoe UI',sans-serif")
    cairosvg.svg2png(bytestring=t.encode(), write_to=str(DST / f"{s.stem}.png"),
                     scale=2.2, background_color="white")
    o = PPT / f"{s.stem}.png"
    cairosvg.svg2png(bytestring=t.encode(), write_to=str(o), scale=3.0, background_color="white")
    w, h = Image.open(o).size
    print(f'  {s.stem[8:]}: {w/h:.3f},')
