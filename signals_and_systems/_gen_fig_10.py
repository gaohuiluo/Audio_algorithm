# -*- coding: utf-8 -*-
"""生成第 10 篇配图 fig_10_1_1.png，并（可选）重生成 voice_eq.wav。
只依赖 NumPy + Matplotlib + scipy（音频用 scipy.io.wavfile，不用 soundfile）。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

HERE = os.path.dirname(os.path.abspath(__file__))
fs = 16000  # 采样率 16 kHz

# ---------- 1. 用 firwin 造一个截止 3kHz 的线性相位 FIR 低通，拿到冲激响应 h[n] ----------
N = 101
fc = 3000.0
h = signal.firwin(N, cutoff=fc, fs=fs)          # 对称 -> 天然线性相位，直流增益=1

# ---------- 2. 冲激响应 -> 频率响应：freqz 就是在做 DTFT 采样 ----------
w, H = signal.freqz(h, worN=4096, fs=fs)        # w 单位 Hz
mag_db = 20 * np.log10(np.abs(H) + 1e-12)
_, gd = signal.group_delay((h, 1), w=4096, fs=fs)

# ---------- 3. 最小移动平均 h=[0.5,0.5] 的解析 |H|，呼应正文手算 ----------
wn = np.linspace(0, np.pi, 512)                 # 数字角频率 rad/sample
mag_ma = np.abs(np.cos(wn / 2))                 # |H(e^jw)| = |cos(w/2)|
f_ma = wn / np.pi * (fs / 2)                    # 换算成 Hz

# ---------- 4. 画 2x2 ----------
fig, ax = plt.subplots(2, 2, figsize=(12, 8))

ax[0, 0].stem(np.arange(N), h, basefmt=" ")
ax[0, 0].set_title("(a) Impulse Response h[n]  (time domain, 101-tap FIR)")
ax[0, 0].set_xlabel("Sample n"); ax[0, 0].set_ylabel("Amplitude")

ax[0, 1].plot(w, mag_db, lw=1.6)
ax[0, 1].axvline(fc, color="r", ls="--", label="cutoff 3 kHz")
ax[0, 1].set_title("(b) |H(e^jw)| in dB  =  the EQ curve")
ax[0, 1].set_xlabel("Frequency (Hz)"); ax[0, 1].set_ylabel("Gain (dB)")
ax[0, 1].set_ylim(-90, 5); ax[0, 1].legend(loc="lower left")

ax[1, 0].plot(f_ma, mag_ma, color="tab:green", lw=1.8)
ax[1, 0].set_title("(c) h=[0.5,0.5] moving avg:  |H| = |cos(w/2)|  (hand-calc)")
ax[1, 0].set_xlabel("Frequency (Hz)"); ax[1, 0].set_ylabel("|H|  (linear)")
ax[1, 0].axhline(1.0, color="gray", ls=":", lw=0.8)
ax[1, 0].set_ylim(0, 1.1)

ax[1, 1].plot(w, gd, color="tab:purple", lw=1.6)
ax[1, 1].axhline((N - 1) / 2, color="r", ls="--", label=f"(N-1)/2 = {(N-1)//2}")
ax[1, 1].set_title("(d) Group delay  (flat -> linear phase)")
ax[1, 1].set_xlabel("Frequency (Hz)"); ax[1, 1].set_ylabel("Delay (samples)")
ax[1, 1].set_ylim(0, N); ax[1, 1].legend(loc="upper right")

plt.tight_layout()
out_png = os.path.join(HERE, "fig_10_1_1.png")
plt.savefig(out_png, dpi=110)
print("saved", out_png)

# ---------- 5. 重生成 voice_eq.wav：对一段人声做低通 EQ 演示 ----------
src = os.path.join(HERE, "Test-voice.wav")
if os.path.exists(src):
    sr, x = wavfile.read(src)
    x = x.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)          # 取单声道
    x /= (np.max(np.abs(x)) + 1e-12)
else:
    sr = fs
    t = np.arange(int(sr * 1.5)) / sr
    # 合成一段带高频"嘶嘶"底噪的元音
    x = 0.6 * np.sin(2 * np.pi * 180 * t) + 0.3 * np.sin(2 * np.pi * 900 * t)
    x += 0.2 * np.random.randn(t.size)   # 宽带噪声，含大量高频

# 用同一个 FIR（按目标采样率重设计），时域卷积
h_eq = signal.firwin(N, cutoff=3000.0, fs=sr)
y = np.convolve(x, h_eq)
y /= (np.max(np.abs(y)) + 1e-12)
y = (y * 0.98 * 32767).astype(np.int16)   # 留 headroom，防削顶
out_wav = os.path.join(HERE, "voice_eq.wav")
wavfile.write(out_wav, sr, y)
print("saved", out_wav, "sr=", sr, "len=", y.size)
