"""
第 14 篇《Z 变换》配图生成脚本。
产物: fig_14_1_1.png —— z 平面单位圆 + 极点零点图, 以及对应的频率响应。
上排: 一阶 IIR  y[n] = x[n] + 0.9 y[n-1]  (H(z)=1/(1-0.9 z^-1), 极点 z=0.9)
下排: 反馈梳状  y[n] = x[n] + 0.85 y[n-D]  (D=8, 单位圆内均匀排开的极点)
依赖: numpy, scipy, matplotlib (不使用 soundfile)
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import tf2zpk, freqz

plt.rcParams["axes.unicode_minus"] = False

fs = 16000


def plot_pz(ax, b, a, title):
    """在 ax 上画 z 平面: 单位圆 + 极点(x) + 零点(o)。"""
    z, p, k = tf2zpk(b, a)
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.5, label="Unit circle")
    ax.axhline(0, color="gray", lw=0.6)
    ax.axvline(0, color="gray", lw=0.6)
    if len(z):
        ax.scatter(z.real, z.imag, marker="o", s=90, facecolors="none",
                   edgecolors="b", label="Zeros")
    if len(p):
        ax.scatter(p.real, p.imag, marker="x", s=90, c="r", label="Poles")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_title(title)
    ax.set_xlabel("Real")
    ax.set_ylabel("Imag")
    ax.legend(loc="upper left", fontsize=8)
    return p


def plot_freq(ax, b, a, title):
    """在 ax 上画幅度频响 (dB), 横轴 Hz。"""
    w, H = freqz(b, a, worN=2048)
    freqs = w / np.pi * (fs / 2)
    ax.plot(freqs, 20 * np.log10(np.abs(H) + 1e-9))
    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.grid(True, alpha=0.3)


fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# ---------- 上排: 一阶 IIR  y[n] = x[n] + 0.9 y[n-1] ----------
b1 = [1.0]
a1 = [1.0, -0.9]              # 1 - 0.9 z^-1
p1 = plot_pz(axes[0, 0], b1, a1,
             "1st-order IIR: y[n]=x[n]+0.9y[n-1]  (pole at z=0.9)")
plot_freq(axes[0, 1], b1, a1, "Freq response: a gentle low-shelf boost")
print("一阶系统极点:", p1, "  |极点|=", np.abs(p1))

# ---------- 下排: 反馈梳状  y[n] = x[n] + 0.85 y[n-D] ----------
D = 8
g = 0.85
b2 = np.zeros(1)
b2[0] = 1.0
a2 = np.zeros(D + 1)
a2[0] = 1.0
a2[D] = -g
p2 = plot_pz(axes[1, 0], b2, a2,
             f"Feedback comb: y[n]=x[n]+{g}y[n-{D}]  ({D} poles on a circle)")
plot_freq(axes[1, 1], b2, a2, f"Freq response: comb teeth spaced fs/D={fs//D} Hz")
print("梳状极点半径:", np.round(np.abs(p2), 4))

plt.tight_layout()
plt.savefig("fig_14_1_1.png", dpi=110)
print("saved fig_14_1_1.png")
