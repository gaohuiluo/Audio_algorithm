# -*- coding: utf-8 -*-
"""第 18 篇配图：迷你语音 Pipeline 实战。
生成 fig_18_1_1.png (pipeline 各级信号/频谱流转)
     fig_18_2_1.png (VAD 判决叠在波形/能量曲线上)
只用 NumPy + Matplotlib + scipy，不用 soundfile / librosa。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.io import wavfile
from scipy.signal import resample_poly

plt.rcParams["axes.unicode_minus"] = False

# ---------------- 1. 读入真实录音，转单声道 float，重采样到 16k ----------------
fs_raw, x_raw = wavfile.read("Test-voice.wav")
if x_raw.ndim == 2:
    x_raw = x_raw.mean(axis=1)          # 立体声 -> 单声道
x_raw = x_raw.astype(np.float64)
if np.issubdtype(np.dtype(np.int16), np.integer):
    x_raw = x_raw / 32768.0             # int16 -> [-1,1)

FS = 16000
x = resample_poly(x_raw, FS, fs_raw)    # 44100 -> 16000 (先低通再抽, 防混叠)
# 截一段"先静音后语音"的片段，方便看清 VAD 判决
t0, t1 = 13.0, 24.0
x = x[int(t0 * FS): int(t1 * FS)]
x = x / (np.max(np.abs(x)) + 1e-12)
print(f"raw fs={fs_raw}, resampled fs={FS}, len={len(x)} ({len(x)/FS:.2f}s)")


# ---------------- 2. pipeline 基础零件 ----------------
def preemphasis(sig, a=0.97):
    """预加重: y[n] = x[n] - a*x[n-1]，一阶高通，抬中高频。"""
    return np.append(sig[0], sig[1:] - a * sig[:-1])


def frame_signal(sig, flen, hop):
    n = 1 + (len(sig) - flen) // hop
    idx = np.arange(flen)[None, :] + hop * np.arange(n)[:, None]
    return sig[idx]


FLEN = int(FS * 0.025)      # 25 ms 帧长
HOP = int(FS * 0.010)       # 10 ms 帧移
WIN = np.hamming(FLEN)

x_dc = x - np.mean(x)                    # 去直流
x_pe = preemphasis(x_dc, 0.97)          # 预加重
frames = frame_signal(x_pe, FLEN, HOP)  # (帧, 样本)
NFFT = 512
S = np.fft.rfft(frames * WIN, n=NFFT, axis=1)     # STFT
mag = np.abs(S)
power_db = 20 * np.log10(mag + 1e-6)

# 逐帧特征
energy = np.sum((frames * WIN) ** 2, axis=1)
energy_db = 10 * np.log10(energy + 1e-10)
zcr = np.mean(np.abs(np.diff(np.sign(frames), axis=1)) > 0, axis=1)

tf = np.arange(frames.shape[0]) * HOP / FS
t = np.arange(len(x)) / FS

# ==================================================================
# fig_18_1_1: pipeline 各级信号/频谱流转
# ==================================================================
fig, ax = plt.subplots(4, 1, figsize=(11, 9))

ax[0].plot(t, x, lw=0.5, color="#1f77b4")
ax[0].set_title("(1) raw waveform  ->  read as float32 @16kHz  (ch.9 sampling)")
ax[0].set_ylabel("amp")

ax[1].plot(t, x_pe, lw=0.5, color="#2ca02c")
ax[1].set_title("(2) after de-DC + pre-emphasis  y[n]=x[n]-0.97x[n-1]  (ch.15 / ch.5)")
ax[1].set_ylabel("amp")

extent = [tf[0], tf[-1], 0, FS / 2 / 1000]
ax[2].imshow(power_db.T, origin="lower", aspect="auto", extent=extent,
             cmap="magma", vmax=power_db.max(), vmin=power_db.max() - 70)
ax[2].set_title("(3) framing + windowing -> STFT magnitude (dB)  (ch.13)")
ax[2].set_ylabel("kHz")

ax[3].plot(tf, energy_db, color="#d62728", label="frame energy")
ax[3].plot(tf, (zcr - zcr.min()) / (zcr.ptp() + 1e-9) * energy_db.ptp()
           + energy_db.min(), color="#9467bd", lw=0.8, label="ZCR (scaled)")
ax[3].set_title("(4) per-frame features: energy / ZCR  (ch.16 / ch.17)")
ax[3].set_ylabel("dB")
ax[3].set_xlabel("time (s)")
ax[3].legend(loc="upper right", fontsize=8)

for a in ax[:2]:
    a.set_xlim(t[0], t[-1])
ax[3].set_xlim(tf[0], tf[-1])
plt.tight_layout()
plt.savefig("fig_18_1_1.png", dpi=110)
plt.close()
print("saved fig_18_1_1.png")

# ==================================================================
# fig_18_2_1: VAD 判决叠在波形/能量曲线上
# ==================================================================
# 自适应阈值: 用能量最低的 15% 帧估计噪声本底
sorted_e = np.sort(energy_db)
noise_floor = np.mean(sorted_e[: max(1, len(sorted_e) // 7)])
thr = noise_floor + 9.0                 # 本底之上 9 dB
zcr_gate = 0.35                          # 过零率上限，滤掉高频噪声毛刺

raw_vad = (energy_db > thr) & (zcr < zcr_gate)

# hangover 平滑: 连续短空隙不切断，避免一句话被切成碎片
def hangover(dec, hang=8):
    out = dec.copy()
    count = 0
    for i, d in enumerate(dec):
        if d:
            count = hang
        elif count > 0:
            out[i] = True
            count -= 1
    return out

vad = hangover(raw_vad, hang=8)

# 提取语音段 (帧 -> 秒)
segs = []
i = 0
while i < len(vad):
    if vad[i]:
        j = i
        while j < len(vad) and vad[j]:
            j += 1
        if (j - i) >= 5:                 # 丢弃过短的段 (<50ms)
            segs.append((tf[i], tf[min(j, len(tf) - 1)]))
        i = j
    else:
        i += 1

fig, ax = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
ax[0].plot(t, x, lw=0.4, color="#1f77b4")
for (s0, s1) in segs:
    ax[0].axvspan(s0, s1, color="#ff7f0e", alpha=0.25)
ax[0].set_title("waveform + detected speech segments (shaded)")
ax[0].set_ylabel("amp")

ax[1].plot(tf, energy_db, color="#d62728", label="energy (dB)")
ax[1].axhline(thr, color="k", ls="--", lw=1, label=f"adaptive thr = floor+9dB")
ax[1].axhline(noise_floor, color="gray", ls=":", lw=1, label="noise floor")
ax[1].set_title("frame energy vs adaptive threshold")
ax[1].set_ylabel("dB")
ax[1].legend(loc="upper right", fontsize=8)

ax[2].step(tf, vad.astype(int), where="mid", color="#2ca02c", label="VAD (after hangover)")
ax[2].step(tf, raw_vad.astype(int) * 0.9, where="mid", color="#9467bd",
           lw=0.7, alpha=0.6, label="raw decision")
ax[2].set_title("VAD decision (1 = speech)")
ax[2].set_ylabel("speech?")
ax[2].set_xlabel("time (s)")
ax[2].set_ylim(-0.2, 1.2)
ax[2].legend(loc="upper right", fontsize=8)

plt.tight_layout()
plt.savefig("fig_18_2_1.png", dpi=110)
plt.close()
print(f"saved fig_18_2_1.png; segments={len(segs)}, speech frames={vad.sum()}/{len(vad)}")
