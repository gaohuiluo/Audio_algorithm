# -*- coding: utf-8 -*-
"""
生成 fig_12_1_1.png：频谱泄漏的因果链，两块面板。
  上：把一段"非整数周期"的 440.3 Hz 正弦首尾相接，接缝处出现折角(突变)——
      这正是 DFT 隐含的"周期延拓"假设；加汉宁窗后两头压到 0，接缝平滑。
  下：矩形窗(不加窗) vs 汉宁窗 的 DFT 幅度谱(dB)——看裙边如何被窗压下去。
输入：无需外部文件(纯合成正弦)。正文另有一段对 Test-voice.wav 做 DFT 的可运行代码。
依赖：numpy + matplotlib(不使用 soundfile)
"""
import numpy as np
import matplotlib.pyplot as plt

fs = 16000          # 采样率 16 kHz
N = 1024            # 分析段长度
f0 = 440.3          # 故意非整数周期(不是 fs/N=15.625Hz 的整数倍)

n = np.arange(N)
x = np.sin(2 * np.pi * f0 * n / fs)
w = np.hanning(N)
xw = x * w

# ---------- 上面板：周期延拓的接缝 ----------
# 把两个周期拼起来看首尾接缝(取接缝附近的一小段放大)
tile_rect = np.concatenate([x, x])
tile_hann = np.concatenate([xw, xw])
# 接缝在 index=N 处，放大它前后各 60 个样本
lo, hi = N - 60, N + 60
seam_idx = np.arange(lo, hi)

fig, ax = plt.subplots(2, 1, figsize=(10, 8))

ax[0].plot(seam_idx, tile_rect[lo:hi], color="#c0392b", lw=1.6,
           label="Rectangular (raw cut): a JUMP at the seam")
ax[0].plot(seam_idx, tile_hann[lo:hi], color="#2471a3", lw=1.8,
           label="Hann window: seam smoothed to ~0")
ax[0].axvline(N, color="gray", ls="--", lw=1)
ax[0].text(N, ax[0].get_ylim()[1] * 0.8, "  seam (period boundary)",
           color="gray", fontsize=9)
ax[0].set_title("Panel A: DFT secretly tiles your segment end-to-end. "
                "A non-integer period makes a JUMP at the seam.", fontsize=10)
ax[0].set_xlabel("Sample index (zoomed around the wrap-around seam)")
ax[0].set_ylabel("Amplitude")
ax[0].legend(loc="lower left", fontsize=9)
ax[0].grid(True, alpha=0.3)

# ---------- 下面板：泄漏谱 ----------
freqs = np.fft.rfftfreq(N, d=1 / fs)


def to_db(sig):
    X = np.fft.rfft(sig)
    mag = np.abs(X)
    return 20 * np.log10(mag / mag.max() + 1e-12)


ax[1].plot(freqs, to_db(x), color="#c0392b", alpha=0.85,
           label="Rectangular (no window): skirts leak everywhere")
ax[1].plot(freqs, to_db(xw), color="#2471a3", lw=2,
           label="Hann window: skirts crushed, main lobe a bit fatter")
ax[1].set_xlim(0, 2000)
ax[1].set_ylim(-100, 5)
ax[1].set_title("Panel B: The seam jump costs you a whole skirt of fake "
                "frequencies (spectral leakage)", fontsize=10)
ax[1].set_xlabel("Frequency (Hz)")
ax[1].set_ylabel("Magnitude (dB, normalized)")
ax[1].legend(loc="upper right", fontsize=9)
ax[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("fig_12_1_1.png", dpi=110)
print("saved fig_12_1_1.png  fs=", fs, " N=", N, " f0=", f0,
      " bin spacing=", fs / N, "Hz")
