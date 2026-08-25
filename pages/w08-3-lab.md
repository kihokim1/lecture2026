# 08주차 3교시. MCU 배포 파이프라인과 Peak Memory

**실습 목표** — 5주차에서 만든 INT8 모델을 MCU에 올리는 **배포 파이프라인**을 끝까지 따라가 본다. 모델을 펌웨어용 **C 배열로 바꿔 Flash 소요량을 확정**하고, **Peak Memory를 추정한 뒤 실제 파일로 검증**한다. 실제 보드가 없어도 노트북에서 두 예산을 모두 숫자로 확인할 수 있다.

> 준비물: 노트북과 **5주차 3.3에서 만든 `tinycnn_int8.tflite`**(30KB). 그때 못 만들었거나 파일을 지웠다면 조교 배포본을 받아 쓴다. 별도 설치는 필요 없다 — C 배열 변환은 파이썬만으로 하고, Peak Memory 검증에만 `ai-edge-litert`(약 30MB)를 쓴다.

---

## 3.1 배포 파이프라인 (10분)

TinyML 배포는 정해진 단계를 거친다.

![MCU 배포 파이프라인 — 학습 → 양자화(INT8) → 변환(.tflite/TFLM) → xxd(C 배열) → 펌웨어 빌드·플래시. 각 단계에서 Flash·RAM(Peak Memory) 예산을 확인한다](../assets/w08_p3_deploy_06.png)

`학습(서버)` → `양자화(INT8, 5주차)` → `변환(.tflite, TFLM용)` → `xxd로 C 배열화` → `펌웨어 빌드·플래시`.

각 단계에서 **두 예산**을 계속 확인한다: 모델이 **Flash**에 들어가는가, 실행 중 **Peak Memory**가 **RAM**을 넘지 않는가. 하나라도 넘으면 보드에 아예 안 올라가거나 실행 중 죽는다.

> 관찰 포인트: 양자화가 파이프라인 앞단에 있는 이유 — INT8이라야 크기(Flash)와 정수 연산(FPU 부재)을 모두 감당할 수 있기 때문이다(1교시).

---

## 3.2 모델을 펌웨어로 — C 배열화 (15분)

MCU에는 파일 시스템이 없는 경우가 많다. 그래서 모델을 **C 소스의 바이트 배열**로 바꿔 펌웨어에 **직접 포함**시킨다. 현장의 표준 도구는 `xxd`다.

```bash
xxd -i tinycnn_int8.tflite > tinycnn_int8.h
```

다만 `xxd`는 OS마다 있기도 하고 없기도 하다(맥·대부분의 리눅스에는 있고, Windows는 Git Bash가 필요하며, 최소 설치 리눅스에는 아예 빠져 있다). **결과가 같으니 파이썬으로 하자.** 어디서나 돌아가고, 무엇을 하는 도구인지도 훨씬 잘 보인다.

```python
# to_c_array.py — xxd -i 와 같은 결과를 파이썬만으로
import sys, pathlib

src = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "tinycnn_int8.tflite")
name = src.name.replace(".", "_").replace("-", "_")
data = src.read_bytes()

lines = [f"unsigned char {name}[] = {{"]
for i in range(0, len(data), 12):
    chunk = ", ".join(f"0x{b:02x}" for b in data[i:i+12])
    lines.append(f"  {chunk},")
lines.append("};")
lines.append(f"unsigned int {name}_len = {len(data)};")

out = src.with_suffix(".h")
out.write_text("\n".join(lines) + "\n")
print(f"{out} 생성 — 모델 {len(data):,} bytes = Flash 소요량")
```

```
$ python to_c_array.py tinycnn_int8.tflite
tinycnn_int8.h 생성 — 모델 30,360 bytes = Flash 소요량
```

생성된 헤더의 앞뒤는 이렇다.

```c
unsigned char tinycnn_int8_tflite[] = {
  0x1c, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33, 0x14, 0x00, 0x20, 0x00,
  0x1c, 0x00, 0x18, 0x00, 0x14, 0x00, 0x10, 0x00, 0x0c, 0x00, 0x00, 0x00,
  /* ... */
};
unsigned int tinycnn_int8_tflite_len = 30360;   // ← 모델 크기(Flash 소요)
```

이 배열을 펌웨어에 넣고, TFLM 인터프리터가 이 바이트를 읽어 실행한다. `..._len` 값이 곧 모델이 차지하는 **Flash 용량**이다.

> 다섯 번째 바이트부터 `0x54 0x46 0x4c 0x33`, 즉 아스키로 **`TFL3`** 이 보인다. FlatBuffer 형식의 서명이다. 30KB짜리 파일이 정말 TFLite 모델이 맞다는 것을 눈으로 확인할 수 있다.

> 관찰 포인트: `_len`(30,360 B ≈ 30KB)을 보드의 Flash 용량과 비교하라. Arduino Nano 33 BLE Sense는 1MB, ESP32-S3는 보통 8MB다. 이 숫자가 배포 가능 여부의 1차 관문이다.

---

## 3.3 Peak Memory 추정과 검증 (22분)

RAM 예산은 **연산 중 동시에 살아있는 버퍼의 합**으로 정해진다. 간단한 층 구성에서 이를 추정해 본다(순수 파이썬, 라이브러리 불필요).

```python
# 각 레이어의 출력 Feature Map 크기(원소 수, INT8=1바이트 가정)
#  (예시: 채널 x 높이 x 너비)
fmaps = [
    ("input",  3*96*96),     # 27,648
    ("conv1", 16*48*48),     # 36,864
    ("conv2", 32*24*24),     # 18,432
    ("conv3", 64*12*12),     #  9,216
    ("pool",  64*1*1),       #     64
]

# 단순 추정: 한 레이어를 계산할 때 '입력 버퍼 + 출력 버퍼'가 동시에 필요
peak = 0
for i in range(1, len(fmaps)):
    live = fmaps[i-1][1] + fmaps[i][1]      # 이전(입력) + 현재(출력)
    peak = max(peak, live)
    print(f"{fmaps[i][0]:>6}: live={live:,} bytes")
print(f"\nPeak Memory ≈ {peak:,} bytes  (RAM 예산과 비교!)")
```

관찰의 핵심:

1. Peak는 대개 **초반 레이어**(Feature Map이 가장 클 때)에서 발생한다.
2. 이 값이 기기 RAM(예: 256KB)을 넘으면 배포 불가 → In-place 재활용, 타일링, 더 작은 입력 해상도로 낮춰야 한다.
3. "모델 크기(Flash)"와 "실행 메모리(RAM/Peak)"는 **다른 예산**이다 — 둘 다 통과해야 한다.

**손으로 센 값이 맞는지 확인하자.** 위 `fmaps`는 추정이다. 실제 `.tflite` 안에 어떤 크기의 버퍼가 잡혀 있는지 직접 열어 보면 된다.

```bash
pip install ai-edge-litert
```

```python
import numpy as np
from ai_edge_litert.interpreter import Interpreter

itp = Interpreter(model_path="tinycnn_int8.tflite")
itp.allocate_tensors()

print("실제 .tflite의 활성 텐서 크기")
for d in itp.get_tensor_details():
    s = d["shape"]
    if len(s) == 4 and s[0] == 1 and d["dtype"] == np.int8:   # 배치 1짜리 Feature Map만
        print(f"  {int(np.prod(s)):>8,} bytes   {[int(v) for v in s]}")
```

```
실제 .tflite의 활성 텐서 크기
    27,648 bytes   [1, 96, 96, 3]
    36,864 bytes   [1, 48, 48, 16]
    18,432 bytes   [1, 24, 24, 32]
     9,216 bytes   [1, 12, 12, 64]
```

위 `fmaps`에 손으로 적은 숫자와 **하나도 틀리지 않고 일치한다.** 그리고 Peak가 잡히는 지점도 그대로다 — 입력 27,648 + conv1 출력 36,864 = **64,512 bytes**. 이 모델은 RAM 256KB 보드에는 올라가고, 64KB 보드에는 못 올라간다.

> 관찰 포인트: 종이에서 센 값과 파일에서 읽은 값이 맞아떨어지는 경험이 중요하다. 이 다음부터는 모델 구조만 보고도 "이건 그 보드에 안 들어간다"를 회의 자리에서 즉석으로 판단할 수 있다.

### 과제 (3분 안내)
1. **예산 점검표** — 위 코드의 `fmaps`를 본인 관심 모델 구조로 바꿔 Peak Memory를 추정하고, 가상의 RAM 예산(예: 256KB)과 비교해 배포 가능 여부를 판정한다.
2. **입력 해상도 실험** — 입력을 96×96 → 48×48로 줄이면 Peak가 어떻게 변하는지 계산하고, 정확도-메모리 트레이드오프를 3~4문장으로 논한다.
3. **개념 연결** — Duty Cycling과 Depthwise Separable이 각각 '전력'과 '연산량'의 어느 문제를 푸는지 구분해 설명한다.
4. **두 예산 판정표** — `tinycnn_int8.tflite`를 아래 세 보드에 올릴 수 있는지 Flash·RAM 두 축으로 판정하고, 안 되는 칸은 무엇을 줄여야 하는지 한 줄씩 적는다.
   | 보드 | Flash | RAM |
   |---|--:|--:|
   | Arduino Nano 33 BLE Sense | 1 MB | 256 KB |
   | ESP32-S3 (일반 모듈) | 8 MB | 512 KB |
   | STM32F103 "Blue Pill" | 128 KB | 20 KB |
5. **다음 주 예습** — 9주차(엣지 비전)에서는 MobileNet·YOLO 등 경량 비전 모델을 다룬다. Depthwise Separable이 어디에 쓰이는지 미리 찾아온다.

> 교수님을 위한 Tip: 이 실습의 승부처는 **3.3의 검증 단계**입니다. 손으로 센 27,648과 파일에서 읽은 27,648이 일치하는 순간, 학생들은 "추정이 아니라 계산이었구나"를 체감합니다. 그 전까지는 숫자를 미리 보여주지 마시고, 각자 세어 본 뒤에 파일을 열게 하세요.
>
> 실제 보드(Arduino Nano 33 BLE Sense, ESP32-S3)가 있으면 만들어진 헤더를 펌웨어에 넣어 빌드 크기를 보여주면 좋습니다. **없어도 실습은 완결됩니다.** 한 걸음 더 나가고 싶으시면 **Renode**(무료 오픈소스 시뮬레이터)에서 TFLM을 실제로 구동할 수 있고, 실행 명령 수·메모리 접근 횟수까지 뽑아 줍니다 — 1주차부터 깔아 온 "진짜 병목은 메모리"를 숫자로 다시 확인시키기에 좋은 도구입니다.

---

### 3교시 정리
- MCU 배포 파이프라인(양자화→변환→C 배열화→플래시)을 이해했다.
- 5주차에서 만든 `.tflite`를 C 배열로 바꿔 **Flash 소요량 30,360 B**를 확정했다.
- Peak Memory를 손으로 추정하고 **실제 파일로 검증**해 **64,512 B**를 얻었다. Flash와 RAM이 별개의 예산임을 수치로 구분했다.
- 다음 주부터는 이 기반 위에서 비전·언어 등 실제 응용으로 넘어간다.
