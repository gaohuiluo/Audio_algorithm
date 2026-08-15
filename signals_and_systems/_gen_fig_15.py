# -*- coding: utf-8 -*-
"""生成第 15 篇配图：
   fig_15_1_1.png —— FIR 窗函数法这条链：理想 sinc → 矩形截断的 Gibbs 振铃 → 加窗压制
   fig_15_4_1.png —— FIR vs IIR 对比：幅频 / 群延迟 / 冲激响应
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

plt.rcParams["axes.unicode_minus"] = False

fs = 16000
fc = 2000.0          # 截止频率 (Hz)


# ========== 图一：窗函数法这条链 ==========
def ideal_lp_h(N, fc, fs):
    """理想低通的截断冲激响应：中心化 sinc（不加窗，等于矩形窗）"""
    n = np.arange(N) - (N - 1) / 2.0
    wc = 2 * np.pi * fc / fs
    h = np.sinc(wc * n / np.pi) * wc / np.pi     # = sin(wc n)/(pi n)
    return h


fig1, ax = plt.subplots(2, 2, figsize=(11, 7.5))

# (a) 理想 sinc 冲激响应（截断到 61 抽头，看它的双向拖尾）
N = 61
h_ideal = ideal_lp_h(N, fc, fs)
n_axis = np.arange(N) - (N - 1) // 2
ax[0, 0].stem(n_axis, h_ideal, basefmt=" ")
ax[0, 0].set_title("(a) Ideal low-pass h[n] = sinc (truncated to 61 taps)")
ax[0, 0].set_xlabel("n (sample)")
ax[0, 0].set_ylabel("h[n]")
ax[0, 0].grid(True, alpha=0.3)

# (b) 矩形截断：阶数越高，Gibbs 振铃幅度不降（约 9%），只是变密
for N_ in (21, 61, 201):
    h = ideal_lp_h(N_, fc, fs)
    w, H = signal.freqz(h, 1, worN=4096, fs=fs)
    ax[0, 1].plot(w, np.abs(H), label=f"N={N_} (rect)", alpha=0.8)
ax[0, 1].axvline(fc, color="gray", ls="--", alpha=0.6)
ax[0, 1].set_title("(b) Rectangular truncation: Gibbs ripple won't shrink")
ax[0, 1].set_xlim(0, fs / 2)
ax[0, 1].set_xlabel("Frequency (Hz)")
ax[0, 1].set_ylabel("|H| (linear)")
ax[0, 1].legend(fontsize=8)
ax[0, 1].grid(True, alpha=0.3)

# (c) 窗形状：矩形 vs 汉明
N = 61
rect = np.ones(N)
hamm = np.hamming(N)
ax[1, 0].plot(rect, label="Rectangular", lw=2)
ax[1, 0].plot(hamm, label="Hamming", lw=2)
ax[1, 0].set_title("(c) Window shapes: rectangular vs Hamming")
ax[1, 0].set_xlabel("n (sample)")
ax[1, 0].set_ylabel("w[n]")
ax[1, 0].legend(fontsize=9)
ax[1, 0].grid(True, alpha=0.3)

# (d) 同阶数下：矩形窗(振铃大) vs 汉明窗(振铃压平, 过渡带变宽), dB 看阻带
N = 61
h_rect = ideal_lp_h(N, fc, fs) * rect
h_hamm = ideal_lp_h(N, fc, fs) * hamm
h_hamm = h_hamm / np.sum(h_hamm)     # 归一化直流增益=1
h_rect = h_rect / np.sum(h_rect)
for h, lab in ((h_rect, "Rectangular window"), (h_hamm, "Hamming window")):
    w, H = signal.freqz(h, 1, worN=4096, fs=fs)
    ax[1, 1].plot(w, 20 * np.log10(np.abs(H) + 1e-9), label=lab)
ax[1, 1].axvline(fc, color="gray", ls="--", alpha=0.6)
ax[1, 1].set_title("(d) Same N=61: Hamming crushes stopband ripple")
ax[1, 1].set_xlim(0, fs / 2)
ax[1, 1].set_ylim(-90, 8)
ax[1, 1].set_xlabel("Frequency (Hz)")
ax[1, 1].set_ylabel("Gain (dB)")
ax[1, 1].legend(fontsize=8)
ax[1, 1].grid(True, alpha=0.3)

fig1.suptitle("FIR window method: ideal sinc -> truncation ringing -> windowing",
              fontsize=12)
fig1.tight_layout(rect=[0, 0, 1, 0.97])
fig1.savefig("fig_15_1_1.png", dpi=110)
print("saved fig_15_1_1.png")


# ========== 图二：FIR vs IIR 对比 ==========
# 目标：都做一个 ~2000Hz 低通
N_fir = 101
h_fir = signal.firwin(N_fir, cutoff=fc, fs=fs)               # 线性相位 FIR
b_iir, a_iir = signal.butter(N=6, Wn=fc / (fs / 2), btype="low")   # 6 阶 IIR

w_fir, H_fir = signal.freqz(h_fir, 1, worN=4096, fs=fs)
w_iir, H_iir = signal.freqz(b_iir, a_iir, worN=4096, fs=fs)

# 群延迟
wg_fir, gd_fir = signal.group_delay((h_fir, 1), w=4096, fs=fs)
wg_iir, gd_iir = signal.group_delay((b_iir, a_iir), w=4096, fs=fs)

# 冲激响应：FIR 就是系数本身；IIR 用 lfilter 敲一个冲激
imp = np.zeros(120); imp[0] = 1.0
h_iir_imp = signal.lfilter(b_iir, a_iir, imp)

fig2, ax = plt.subplots(1, 3, figsize=(15, 4.5))

# (a) 幅频
ax[0].plot(w_fir, 20 * np.log10(np.abs(H_fir) + 1e-9),
           label=f"FIR firwin (N={N_fir})")
ax[0].plot(w_iir, 20 * np.log10(np.abs(H_iir) + 1e-9),
           label="IIR butter (order 6)")
ax[0].axvline(fc, color="gray", ls="--", alpha=0.6, label="cutoff 2kHz")
ax[0].axhline(-3, color="r", ls=":", alpha=0.5)
ax[0].set_title("(a) Magnitude: 6-order IIR ~ 101-tap FIR")
ax[0].set_xlim(0, fs / 2); ax[0].set_ylim(-90, 8)
ax[0].set_xlabel("Frequency (Hz)"); ax[0].set_ylabel("Gain (dB)")
ax[0].legend(fontsize=8); ax[0].grid(True, alpha=0.3)

# (b) 群延迟
ax[1].plot(wg_fir, gd_fir, label="FIR (flat = linear phase)")
ax[1].plot(wg_iir, gd_iir, label="IIR (varies with freq)")
ax[1].set_title("(b) Group delay: FIR constant, IIR bends")
ax[1].set_xlim(0, fs / 2); ax[1].set_ylim(0, 70)
ax[1].set_xlabel("Frequency (Hz)"); ax[1].set_ylabel("Group delay (samples)")
ax[1].legend(fontsize=8); ax[1].grid(True, alpha=0.3)

# (c) 冲激响应
ax[2].stem(np.arange(N_fir), h_fir, linefmt="C0-", markerfmt="C0.",
           basefmt=" ", label="FIR: finite (101 taps)")
ax[2].plot(np.arange(120), h_iir_imp, "C1-", alpha=0.9,
           label="IIR: infinite decaying tail")
ax[2].set_title("(c) Impulse response: FIR finite vs IIR infinite")
ax[2].set_xlim(0, 120)
ax[2].set_xlabel("n (sample)"); ax[2].set_ylabel("h[n]")
ax[2].legend(fontsize=8); ax[2].grid(True, alpha=0.3)

fig2.tight_layout()
fig2.savefig("fig_15_4_1.png", dpi=110)
print("saved fig_15_4_1.png")
