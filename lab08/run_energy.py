# -*- coding: utf-8 -*-
"""실험 2 — 추론을 두 배 빠르게 하면 배터리는 얼마나 오래 가나.

전제(모두 1차 자료의 값을 조건까지 명시해 인용한다):
  MCU      nRF52840 Product Specification v1.11, 3 V · 25 °C
    I_active 3.3 mA   ICPU0: CoreMark @64 MHz, 플래시에서 실행, HFXO, **DC/DC 레귤레이터**
                       (같은 조건에서 LDO 를 쓰면 6.3 mA 다 — 조건을 안 밝히면 2배가 갈린다)
    I_sleep  3.16 uA  ION_RAMON_RTC: System ON, **256 kB RAM 전체 유지**, RTC 로 기상
                       (RAM 을 안 지키면 1.50 uA 다 — 모델 가중치를 RAM 에 두면 이 값을 쓸 수 없다)
  배터리    Panasonic CR2032, 공칭 3 V · **225 mAh**, 자기방전 코인형 **연 1.0 %**
  t_inf    100 ms (추론 1회)
"""
import json

CAP_MAH = 225.0
I_A = 3.3e-3         # A
I_S = 3.16e-6        # A
T_INF = 0.100        # s
SELF_A = CAP_MAH * 0.01 / 8766.0 * 1e-3   # 연 1.0 % 자기방전을 전류로 환산 [A]

out = {"assume": dict(cap_mah=CAP_MAH, i_active_a=I_A, i_sleep_a=I_S,
                      t_inf_s=T_INF, self_discharge_a=SELF_A)}
print(f"자기방전 등가 전류      {SELF_A*1e6:.3f} uA (연 1.0 %)")


def avg_current(period, t_inf=T_INF, i_a=I_A, i_s=I_S, selfd=True):
    t_inf = min(t_inf, period)
    load = (i_a * t_inf + i_s * (period - t_inf)) / period
    return load + (SELF_A if selfd else 0.0)


def life_days(period, **kw):
    return (CAP_MAH / 1000.0) / avg_current(period, **kw) / 24.0


# ── 1. 손익분기 주기 — 활성 전하 = 수면 전하가 되는 지점 ────────────────────
T_star = T_INF * (1 + I_A / I_S)
out["ratio_active_sleep"] = I_A / I_S
out["breakeven_s"] = T_star
print(f"활성/수면 전류비        {I_A/I_S:,.0f} 배")
print(f"손익분기 주기 T*        {T_star:,.1f} 초 = {T_star/60:.2f} 분")

# ── 2. 주기별 수명 · 추론을 절반으로 줄였을 때의 이득 ───────────────────────
print("\n주기        평균전류     배터리 수명       활성 비중   추론 1/2 수명      이득")
rows = []
for T in [0.5, 1, 5, 10, 60, 104.5, 300, 3600, 86400]:
    ia = avg_current(T)
    frac = I_A * min(T_INF, T) / (ia * T)
    g = ia / avg_current(T, t_inf=T_INF / 2)
    rows.append(dict(period_s=T, i_avg_a=ia, life_d=life_days(T), active_frac=frac,
                     life_half_d=life_days(T, t_inf=T_INF / 2), gain=g))
    print(f"{T:>8.1f}s  {ia*1e6:9.2f} uA  {life_days(T):8.1f} 일 "
          f"({life_days(T)/365:5.2f} 년)  {frac*100:6.1f}%  "
          f"{life_days(T, t_inf=T_INF/2):8.1f} 일   {g:5.3f} 배")
out["sweep"] = rows

# ── 3. 세 개의 손잡이 — 같은 노력을 어디에 쓸 것인가 ────────────────────────
print("\n[같은 노력, 다른 결과] 각각을 절반으로 만들었을 때 수명 배수")
print("주기          추론시간 1/2   활성전류 1/2   수면전류 1/2   (수면전류 0)")
knob = []
for T in [1, 60, 3600, 86400]:
    base = life_days(T)
    a = life_days(T, t_inf=T_INF / 2) / base
    b = life_days(T, i_a=I_A / 2) / base
    c = life_days(T, i_s=I_S / 2) / base
    z = life_days(T, i_s=0.0) / base
    knob.append(dict(period_s=T, halve_time=a, halve_iactive=b,
                     halve_isleep=c, zero_isleep=z))
    print(f"{T:>8}s        {a:6.3f} 배      {b:6.3f} 배      {c:6.3f} 배     {z:6.3f} 배")
out["knobs"] = knob

# ── 4. 자기방전이 차지하는 몫 ───────────────────────────────────────────────
print("\n[자기방전의 몫] 부하가 작아질수록 배터리 자신이 주된 소비자가 된다")
sd = []
for T in [1, 60, 3600, 86400]:
    load = avg_current(T, selfd=False)
    share = SELF_A / (load + SELF_A)
    sd.append(dict(period_s=T, load_a=load, self_share=share,
                   life_with=life_days(T),
                   life_without=(CAP_MAH / 1000) / load / 24))
    print(f"{T:>8}s  부하 {load*1e6:8.3f} uA | 자기방전 몫 {share*100:5.1f}% | "
          f"자기방전 무시하면 {(CAP_MAH/1000)/load/24/365:6.2f} 년 → 반영하면 {life_days(T)/365:6.2f} 년")
out["selfd"] = sd

# ── 5. 2단 캐스케이드 — 작은 문지기 + 큰 모델 ───────────────────────────────
print("\n[2단 캐스케이드] 문지기 10 ms/1초 + 본모델 100 ms, 통과율 p")
T_G, T_GATE, T_BIG = 1.0, 0.010, 0.100
always = avg_current(T_G, t_inf=T_BIG)
casc = []
print(f"  항상 본모델                  {always*1e6:8.1f} uA   {life_days(T_G):7.2f} 일")
for p in [0.0, 0.01, 0.05, 0.1, 0.3, 0.9, 1.0]:
    on = T_GATE + T_BIG * p
    i = (I_A * on + I_S * (T_G - on)) / T_G + SELF_A
    d = (CAP_MAH / 1000) / i / 24
    casc.append(dict(p=p, i_avg_a=i, life_d=d, gain=always / i))
    print(f"  p={p:<5.2f}                      {i*1e6:8.1f} uA   {d:7.2f} 일   {always/i:5.2f} 배")
p_break = (T_BIG - T_GATE) / T_BIG
out["cascade"] = dict(period_s=T_G, t_gate=T_GATE, t_big=T_BIG,
                      always_i=always, rows=casc, breakeven_p=p_break)
print(f"  → 통과율이 {p_break*100:.0f}% 를 넘으면 캐스케이드가 오히려 손해다")

json.dump(out, open("/root/lab08/energy.json", "w"), ensure_ascii=False, indent=1)
print("\n저장: energy.json")
