"""제출 전 자가 점검 — 여러분의 캡스톤 보고서를 여러분이 먼저 채점한다.

    python3 selfcheck.py 보고서.md

이 도구가 하는 일은 **자동 감사 하나**뿐이고, 그 감사는 우리 실측으로
**오류율 16.7%** 다. 표본의 **35.7%** 는 애초에 수치 주장이 아니었다.
그러니 이 도구의 점수를 믿지 말고, **어느 문장이 걸렸는지**를 보라.

마지막에 사람이 채워야 할 제출 체크리스트를 함께 내민다.
"""
import re, sys, pathlib

UNITS = (r"ms|s|분|시간|%p|%|배|dB|MB|KB|GB|GiB|MiB|FPS|fps|tok/s|"
         r"mA|µA|uA|mW|W|mJ|J|Hz|kHz|MHz|GHz|개|회|장|층|비트|bit|"
         r"GFLOPs|MFLOPs|FLOPs|MACs|파라미터")
NUM = re.compile(r"(?<![A-Za-z0-9_.])(\d[\d,]*(?:\.\d+)?)\s*(" + UNITS + r")(?![A-Za-z0-9])")
COND = re.compile(
    r"배치|스레드|해상도|시드|문맥|버전|대비|기준|조건|온도|전원|워밍업|반복|"
    r"CPU|GPU|MCU|NPU|Jetson|Raspberry|Cortex|ARM|x86|nRF|ESP32|RTX|Mali|Snapdragon|"
    r"MNIST|CIFAR|ImageNet|COCO|KWS|데이터셋|검증셋|테스트셋|"
    r"ONNX|TFLite|PyTorch|TensorRT|onnxruntime|NNAPI|delegate|"
    r"FP32|FP16|INT8|INT4|양자화|프루닝|증류|"
    r"에서|기준으로|일 때|설정|\bB=|배치 ?\d")
META = re.compile(r"분\)|배정|교시|주차|쪽|페이지|절|문항|점|명|팀|년|월|일")

AXES = {"정확도": r"정확도|accuracy|mAP|F1|WER|BLEU",
        "지연": r"지연|latency|ms\b|FPS|fps|tok/s",
        "메모리": r"메모리|memory|피크|peak|RSS|SRAM|RAM",
        "크기": r"모델 크기|파일 크기|MB\b|KB\b|파라미터"}
COND4 = {"하드웨어": r"CPU|GPU|MCU|NPU|Jetson|Raspberry|Cortex|ARM|폰|보드|스레드",
         "데이터": r"데이터셋|검증셋|테스트셋|MNIST|CIFAR|ImageNet|COCO|장 |샘플",
         "설정": r"배치|해상도|문맥|시드|입력 크기|반복|워밍업",
         "버전": r"버전|v\d|\d+\.\d+\.\d+|커밋|commit"}

path = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "보고서.md")
if not path.exists():
    print(f"파일이 없습니다: {path}\n\n    python3 selfcheck.py 보고서.md")
    sys.exit(1)
txt = path.read_text(encoding="utf-8")

# ── ① 조건 없는 수치 문장 찾기 ────────────────────────────────────
bare, withc, in_code = [], 0, False
for ln in txt.splitlines():
    if ln.lstrip().startswith("```"):
        in_code = not in_code
        continue
    if in_code or re.match(r"^\s*(\||#|>)", ln):
        continue
    for sent in re.split(r"(?<=다)\.|(?<=\.)\s|\. ", ln):
        hits = [h for h in NUM.findall(sent) if not META.search(h[1])]
        if not hits:
            continue
        if COND.search(sent):
            withc += 1
        else:
            bare.append(sent.strip()[:150])

tot = withc + len(bare)
print("=" * 74)
print(f"제출 전 자가 점검 — {path.name}")
print("=" * 74)
print(f"\n① 수치가 든 문장 {tot}개 중 같은 문장에 조건이 없는 것 {len(bare)}개")
if tot:
    print(f"   조건 명시율 {100*withc/tot:.1f}%  "
          f"(참고: 이 과목 교재 자체는 사람이 세어 44.4%였다)")
for b in bare[:12]:
    print(f"   · {b}")
if len(bare) > 12:
    print(f"   … 외 {len(bare)-12}개")
print("\n   ※ 전부 고칠 필요는 없습니다. 표에 조건이 있으면 문장은 짧아도 됩니다.")
print("     다만 **발표에서 소리 내어 읽을 문장**에는 조건을 붙이십시오 —")
print("     인용은 문장 단위로 잘려 나갑니다(13주차 1.2 ① 유형).")

# ── ② 네 축이 다 있는가 ──────────────────────────────────────────
print("\n② 결과 요약표의 네 축")
for k, pat in AXES.items():
    ok = bool(re.search(pat, txt))
    print(f"   [{'O' if ok else ' '}] {k}")

# ── ③ 측정 조건 네 가지 ──────────────────────────────────────────
print("\n③ 측정 조건 네 가지 (표 머리에 적혔는가)")
for k, pat in COND4.items():
    ok = bool(re.search(pat, txt))
    print(f"   [{'O' if ok else ' '}] {k}")

# ── ④ 감점 사유 자동 탐지 ────────────────────────────────────────
print("\n④ 자동으로 잡히는 감점 신호")
sig = [("워밍업 언급 없음", not re.search(r"워밍업|warm ?up", txt)),
       ("반복 측정 횟수 언급 없음", not re.search(r"반복|회 측정|n ?= ?\d|시행", txt)),
       ("평균만 있고 분포(중앙값·꼬리·표준편차)가 없음",
        bool(re.search(r"평균", txt)) and not re.search(r"중앙값|p9[05]|표준편차|분포|꼬리|±", txt)),
       ("\"체감\" 류 표현이 있음", bool(re.search(r"체감|느낌|훨씬 빨라|매우 빨라", txt))),
       ("프로파일링 근거 언급 없음", not re.search(r"프로파일|profil|연산자별|op ?별", txt))]
hit = [n for n, bad in sig if bad]
for n, bad in sig:
    print(f"   [{'!' if bad else 'O'}] {n}")

print(f"""
⑤ 여기까지가 자동으로 되는 전부입니다 — 그리고 이 감사는 오류율 16.7% 입니다.
   나머지는 사람이 확인해야 합니다.

   ┌──────────────────────────────────────────────────────────────┐
   │ ① 원본과 최적화본을 **같은 조건**에서 쟀는가                  │
   │ ② 타깃 등급(A~D)과 그 선택 이유를 첫 슬라이드에 적었는가      │
   │ ③ 프로파일링으로 병목 → 개선을 데이터로 이었는가              │
   │ ④ 인용한 논문의 게재처·연도를 확인했는가 (13주차 student.py)  │
   │ ⑤ 다른 사람이 README 대로 돌려서 같은 표가 나오는가           │
   └──────────────────────────────────────────────────────────────┘

   자동 점검 신호 {len(hit)}건: {', '.join(hit) if hit else '없음'}""")
