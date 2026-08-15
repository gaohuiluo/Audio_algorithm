# -*- coding: utf-8 -*-
"""
第 16 篇配图：对 Test-voice.wav 的一帧浊音做 LPC。
产物 fig_16_1_1.png：
  上 - FFT 幅度谱(灰,布满基频谐波毛刺) + LPC 声道包络(红,共振峰鼓包)
  下 - LPC 逆滤波得到的预测残差 e[n](激励源估计,浊音呈周期脉冲串)
只用 numpy + scipy + matplotlib；不依赖 librosa / soundfile。
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import resample_poly, freqz, lfilter

plt.rcParams["axes.unicode_minus"] = False

# ---------- 1. 读音频 → 单声道 → 重采样到 16 kHz ----------
fs_raw, data = wavfile.read("Test-voice.wav")
if data.ndim == 2:
    data = data.mean(axis=1)
data = data.astype(np.float64)
data /= np.max(np.abs(data)) + 1e-12

fs = 16000
x = resample_poly(data, fs, fs_raw)

# ---------- 2. 找一帧高能量的浊音(元音) ----------
frame_len = int(0.030 * fs)          # 30 ms 帧,短时平稳(呼应第 13 篇 STFT)
hop = frame_len // 2
best_i, best_e = 0, -1.0
for i in range(0, len(x) - frame_len, hop):
    seg = x[i:i + frame_len]
    e = np.sum(seg ** 2)
    # 简单浊音判据:能量高 + 过零率低
    zcr = np.mean(np.abs(np.diff(np.sign(seg)))) / 2.0
    if e > best_e and zcr < 0.15:
        best_e, best_i = e, i
frame = x[best_i:best_i + frame_len].copy()
print("选中帧起点 %.3f s, 能量 %.3f" % (best_i / fs, best_e))

# ---------- 3. 预加重 + 加窗 ----------
pre = np.append(frame[0], frame[1:] - 0.97 * frame[:-1])  # 预加重,提升高频(呼应第5篇 [1,-0.97])
win = pre * np.hamming(len(pre))

# ---------- 4. 自相关法 + Levinson-Durbin 解 LPC ----------
def lpc_levinson(sig, order):
    r = np.correlate(sig, sig, "full")[len(sig) - 1:]  # 自相关
    r = r[:order + 1]
    a = np.zeros(order + 1)
    a[0] = 1.0
    err = r[0]
    for i in range(1, order + 1):
        k = -(a[:i] @ r[i:0:-1]) / err     # 反射系数
        a[:i + 1] += k * a[i::-1]
        err *= (1.0 - k * k)
    return a, err                          # a[0]=1, A(z)=1 - sum a_k z^-k(注意 a[1:] 已含负号)

order = fs // 1000 + 2                      # 经验阶数:16k → 18
a, err = lpc_levinson(win, order)
G = np.sqrt(err)                            # 增益 = 残差能量方根

# ---------- 5. LPC 声道包络 H(z)=G/A(z) ----------
w, H = freqz([G], a, worN=2048, fs=fs)
env_db = 20 * np.log10(np.abs(H) + 1e-9)

# ---------- 6. FFT 幅度谱(同一加窗帧) ----------
NFFT = 2048
S = np.abs(np.fft.rfft(win, NFFT))
freqs = np.fft.rfftfreq(NFFT, 1 / fs)
S_db = 20 * np.log10(S + 1e-9)
shift = np.max(env_db) - np.max(S_db)       # 对齐两条曲线的高度便于叠看
S_db += shift

# ---------- 7. 从 LPC 包络里挑共振峰(极点角度→频率) ----------
roots = np.roots(a)
roots = roots[np.imag(roots) > 0]           # 取上半平面共轭极点
ang = np.arctan2(np.imag(roots), np.real(roots))
formants = sorted(ang * fs / (2 * np.pi))
formants = [f for f in formants if 90 < f < fs / 2 - 100][:4]
print("估计共振峰(Hz):", [round(f) for f in formants])

# ---------- 8. 预测残差 = 激励源估计:用 A(z) 逆滤波 ----------
resid = lfilter(a, [1.0], pre)              # e[n] = s[n] - sum a_k s[n-k]

# ---------- 9. 画图 ----------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.2))

ax1.plot(freqs, S_db, color="0.55", lw=1.0,
         label="FFT spectrum of frame (source x filter)")
ax1.plot(w, env_db, "r", lw=2.6, label="LPC vocal-tract envelope (order %d)" % order)
for k, f in enumerate(formants):
    ax1.axvline(f, ls="--", color="tab:blue", alpha=0.5)
    ax1.text(f + 30, np.max(env_db) - 4 - k * 0, "F%d" % (k + 1),
             color="tab:blue", fontsize=9)
ax1.set_xlim(0, 5000)
ax1.set_xlabel("Frequency (Hz)")
ax1.set_ylabel("Magnitude (dB)")
ax1.set_title("LPC envelope rides on top of the harmonic spikes -> formants")
ax1.legend(loc="upper right", fontsize=9)
ax1.grid(alpha=0.25)

t = np.arange(len(resid)) / fs * 1000
ax2.plot(t, pre / (np.max(np.abs(pre)) + 1e-9), color="0.7", lw=1.0,
         label="speech frame s[n] (normalized)")
ax2.plot(t, resid / (np.max(np.abs(resid)) + 1e-9), color="tab:green", lw=1.0,
         label="LPC residual e[n] = excitation estimate")
ax2.set_xlim(0, t[-1])
ax2.set_xlabel("Time (ms)")
ax2.set_ylabel("Amplitude (norm.)")
ax2.set_title("Residual collapses smooth speech into a spiky pulse train (pitch pulses)")
ax2.legend(loc="upper right", fontsize=9)
ax2.grid(alpha=0.25)

plt.tight_layout()
plt.savefig("fig_16_1_1.png", dpi=110)
print("saved fig_16_1_1.png")
