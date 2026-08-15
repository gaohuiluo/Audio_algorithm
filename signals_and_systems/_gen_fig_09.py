# -*- coding: utf-8 -*-
"""生成第 9 篇配图 fig_09_1_1.png，并重新生成混叠演示 aliased_sweep.wav。
依赖: numpy, matplotlib, scipy。音频用 scipy.io.wavfile.write。"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.io import wavfile

plt.rcParams["font.family"] = "DejaVu Sans"  # 纯英文标注，避免缺字

# ============================================================
# 图：上=时域混叠（6kHz 被 8kHz 采成 2kHz 鬼影），
#     下左=频谱复制不重叠(够快)，下右=频谱复制重叠=混叠(太慢)
# ============================================================
fig = plt.figure(figsize=(11, 8))
gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 1.0], hspace=0.38, wspace=0.22)

# ---------- 上：时域 ----------
ax0 = fig.add_subplot(gs[0, :])
fs_true = 400000                    # 高分辨率模拟"连续"真波
T = 0.002                           # 2 ms 窗口
t_fine = np.arange(0, T, 1 / fs_true)

f_sig = 6000                        # 真实信号 6 kHz
true_wave = np.sin(2 * np.pi * f_sig * t_fine)

fs_low = 8000                       # 采样率 8 kHz -> 奈奎斯特 4 kHz
t_samp = np.arange(0, T, 1 / fs_low)
samples = np.sin(2 * np.pi * f_sig * t_samp)

f_alias = abs(f_sig - fs_low)       # |6000 - 8000| = 2000 Hz
alias_wave = np.sin(2 * np.pi * f_alias * t_fine)

ax0.plot(t_fine * 1000, true_wave, "b-", lw=1.0, alpha=0.5,
         label=f"True signal {f_sig} Hz")
ax0.plot(t_fine * 1000, alias_wave, "g--", lw=1.8,
         label=f"Aliased ghost {f_alias} Hz")
ax0.plot(t_samp * 1000, samples, "ro", ms=9,
         label=f"Samples @ {fs_low} Hz")
ax0.set_title("Time domain: a 6 kHz tone sampled at 8 kHz collapses onto a fake 2 kHz tone")
ax0.set_xlabel("Time (ms)")
ax0.set_ylabel("Amplitude")
ax0.legend(loc="upper right", fontsize=9)
ax0.grid(alpha=0.3)

# ---------- 频域复制的画法 ----------
def draw_spectrum(ax, fmax, fs, title):
    # 基带三角谱 + 以 fs 为周期的搬移副本
    def tri(f, center):
        y = 1 - np.abs(f - center) / fmax
        return np.clip(y, 0, None)
    f = np.linspace(-2.4 * fs, 2.4 * fs, 4000)
    ax.plot(f / 1000, tri(f, 0), "b-", lw=2, label="Baseband")
    for k in range(-2, 3):
        if k == 0:
            continue
        ax.plot(f / 1000, tri(f, k * fs), "r-", lw=1.3, alpha=0.8,
                label="Shifted copies" if k == 1 else None)
    # 奈奎斯特线
    ax.axvline(fs / 2 / 1000, color="k", ls=":", lw=1)
    ax.axvline(-fs / 2 / 1000, color="k", ls=":", lw=1)
    ax.text(fs / 2 / 1000, 1.08, "fs/2", ha="center", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylim(0, 1.25)
    ax.set_xlim(-2.4 * fs / 1000, 2.4 * fs / 1000)
    ax.grid(alpha=0.3)

# ---------- 下左：够快，副本不重叠 ----------
ax1 = fig.add_subplot(gs[1, 0])
draw_spectrum(ax1, fmax=3000, fs=8000,
              title="fs=8k > 2*fmax=6k : copies stay apart (safe)")
ax1.legend(loc="upper right", fontsize=8)

# ---------- 下右：太慢，副本重叠=混叠 ----------
ax2 = fig.add_subplot(gs[1, 1])
draw_spectrum(ax2, fmax=3000, fs=5000,
              title="fs=5k < 2*fmax=6k : copies overlap = ALIASING")
# 标出重叠区
ax2.axvspan(2500 / 1000, 3000 / 1000, color="orange", alpha=0.3)
ax2.axvspan(-3000 / 1000, -2500 / 1000, color="orange", alpha=0.3)

fig.suptitle("Sampling copies the spectrum every fs Hz; too-slow fs makes the copies collide",
             fontsize=12, y=0.98)
fig.savefig("fig_09_1_1.png", dpi=110, bbox_inches="tight")
print("saved fig_09_1_1.png")

# ============================================================
# 重新生成混叠演示 wav：扫频 200->8000Hz，粗暴抽点降到 6kHz，不做抗混叠滤波
# ============================================================
fs = 44100
dur = 3.0
t = np.arange(0, dur, 1 / fs)
# 线性扫频 200 -> 8000 Hz
f0, f1 = 200.0, 8000.0
sweep = np.sin(2 * np.pi * (f0 * t + (f1 - f0) / (2 * dur) * t ** 2))

fs_out = 6000
factor = fs // fs_out               # 44100 // 6000 = 7
bad = sweep[::factor]               # 直接抽点，故意不滤波 -> 制造混叠
bad = bad / np.max(np.abs(bad))
wavfile.write("aliased_sweep.wav", fs_out, np.int16(bad * 32767))
print(f"saved aliased_sweep.wav ({len(bad)} samples @ {fs_out} Hz)")
