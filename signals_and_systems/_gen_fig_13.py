# -*- coding: utf-8 -*-
"""
生成 fig_13_1_1.png：用同一段真实人声，长帧 vs 短帧两张语谱图，
直观展示 STFT 的时间-频率取舍（测不准）。
输入：工作目录下的 Test-voice.wav（44.1kHz 立体声 int16）
依赖：numpy + scipy + matplotlib（不使用 soundfile）
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# ---------- 读入真实人声 ----------
fs, data = wavfile.read("Test-voice.wav")
if data.ndim == 2:                      # 立体声 -> 单声道
    data = data.mean(axis=1)
x = data.astype(np.float64)
x /= np.max(np.abs(x)) + 1e-12          # 归一化到 [-1, 1]

# 截取中间一段清晰语音（约 2.5 秒），避免头尾静音
seg_start = int(3.0 * fs)
seg_len = int(2.5 * fs)
x = x[seg_start:seg_start + seg_len]

# ---------- 手写 STFT（揭开黑箱，逐帧加窗 + rfft） ----------
def stft(sig, n_fft, hop, fs):
    win = np.hanning(n_fft)             # Hann 窗，压泄漏
    frames = []
    for start in range(0, len(sig) - n_fft, hop):
        seg = sig[start:start + n_fft] * win   # 取一帧 + 加窗
        spec = np.fft.rfft(seg)                # 这一帧的 FFT（=DFT）
        frames.append(np.abs(spec))
    S = np.array(frames).T                     # 每一列 = 一帧
    S_db = 20 * np.log10(S / (S.max() + 1e-12) + 1e-6)
    freqs = np.fft.rfftfreq(n_fft, d=1 / fs)
    times = np.arange(S.shape[1]) * hop / fs
    return S_db, freqs, times

# ---------- 长帧 vs 短帧 ----------
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
configs = [
    (2048, "Long frame  N=2048  (~46 ms)\nfine frequency, blurry time  [narrowband]"),
    (256,  "Short frame  N=256  (~6 ms)\nsharp time, coarse frequency  [wideband]"),
]
for a, (n_fft, name) in zip(ax, configs):
    S_db, freqs, times = stft(x, n_fft=n_fft, hop=n_fft // 4, fs=fs)
    im = a.pcolormesh(times, freqs, S_db, shading="auto",
                      vmin=-70, vmax=0, cmap="magma")
    a.set_title(name, fontsize=10)
    a.set_xlabel("Time (s)")
    a.set_ylabel("Frequency (Hz)")
    a.set_ylim(0, 5000)
    fig.colorbar(im, ax=a, label="dB")

plt.tight_layout()
plt.savefig("fig_13_1_1.png", dpi=110)
print("saved fig_13_1_1.png  fs=", fs, " seg samples=", len(x))
