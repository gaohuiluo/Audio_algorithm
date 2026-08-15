"""
生成第 17 篇配图 fig_17_1_1.png：
  面板 1：梅尔三角滤波器组（低频窄密、高频宽疏）
  面板 2：一段真实语音 Test-voice.wav 的 MFCC 特征热图
  面板 3：一个浊音帧的实倒谱，标出对应基频 F0 的那根尖峰
不依赖 librosa / soundfile；音频用 scipy.io.wavfile，Mel/DCT 自实现。
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.io import wavfile
from scipy.signal import resample_poly
from scipy.fftpack import dct

# ---------- 让中文能显示（找不到中文字体就退回英文标题） ----------
ZH = None
for name in ["Microsoft YaHei", "SimHei", "SimSun"]:
    try:
        font_manager.findfont(name, fallback_to_default=False)
        ZH = name
        break
    except Exception:
        continue
if ZH:
    plt.rcParams["font.sans-serif"] = [ZH]
    plt.rcParams["axes.unicode_minus"] = False


def T(zh, en):
    return zh if ZH else en


# ---------- Mel 刻度换算（学名：Mel scale，一把"低频密高频疏"的尺子） ----------
def hz2mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel2hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


# ---------- 构造梅尔三角滤波器组 ----------
def mel_filterbank(n_filters, nfft, fs, fmin=0, fmax=None):
    fmax = fmax or fs / 2
    mel_pts = np.linspace(hz2mel(fmin), hz2mel(fmax), n_filters + 2)
    hz_pts = mel2hz(mel_pts)
    bins = np.floor((nfft + 1) * hz_pts / fs).astype(int)
    fb = np.zeros((n_filters, nfft // 2 + 1))
    for i in range(1, n_filters + 1):
        l, c, r = bins[i - 1], bins[i], bins[i + 1]
        for k in range(l, c):
            fb[i - 1, k] = (k - l) / max(c - l, 1)
        for k in range(c, r):
            fb[i - 1, k] = (r - k) / max(r - c, 1)
    return fb, hz_pts


# ---------- MFCC 主流程（每一步都对应正文流水线的一环） ----------
def extract_mfcc(sig, fs, frame_ms=25, hop_ms=10, n_filters=26, n_mfcc=13, nfft=512):
    flen, hop = int(fs * frame_ms / 1000), int(fs * hop_ms / 1000)
    win = np.hamming(flen)
    fb, _ = mel_filterbank(n_filters, nfft, fs)
    feats = []
    for start in range(0, len(sig) - flen, hop):
        frame = sig[start:start + flen] * win           # ① 分帧 + 加窗
        spec = np.abs(np.fft.rfft(frame, nfft)) ** 2     # ② FFT 功率谱
        mel_e = fb @ spec                                # ③ 过梅尔滤波器组
        log_e = np.log(mel_e + 1e-10)                    # ④ 取对数
        c = dct(log_e, type=2, norm="ortho")             # ⑤ DCT = 倒谱
        feats.append(c[1:n_mfcc])                        # ⑥ 取 1..12（丢弃 c0 能量项）
    return np.array(feats).T


# ---------- 读入真实语音，转单声道 16 kHz ----------
fs0, x = wavfile.read("Test-voice.wav")
if x.ndim > 1:
    x = x.mean(axis=1)
x = x.astype(np.float64)
x /= (np.max(np.abs(x)) + 1e-12)
fs = 16000
if fs0 != fs:
    # 有理重采样到 16 kHz（44100 -> 16000 约等于 160/441）
    from math import gcd
    g = gcd(fs0, fs)
    x = resample_poly(x, fs // g, fs0 // g)

# 截取中间一段清晰语音（避开首尾静音）
seg = x[int(2.0 * fs):int(4.0 * fs)]

# ================= 面板 1：梅尔三角滤波器组 =================
nfft = 512
fb, hz_pts = mel_filterbank(26, nfft, fs)
freqs = np.linspace(0, fs / 2, nfft // 2 + 1)

fig, axes = plt.subplots(3, 1, figsize=(11, 11))

ax = axes[0]
for i in range(fb.shape[0]):
    ax.plot(freqs, fb[i], lw=1.0)
ax.set_title(T("① 梅尔三角滤波器组：低频窄而密、高频宽而疏（模仿人耳分辨率）",
               "Mel filterbank: narrow/dense at low freq, wide/sparse at high freq"))
ax.set_xlabel(T("频率 Hz", "Frequency (Hz)"))
ax.set_ylabel(T("权重", "Weight"))
ax.set_xlim(0, fs / 2)

# ================= 面板 2：真实语音的 MFCC 热图 =================
mfcc = extract_mfcc(seg, fs)
ax = axes[1]
im = ax.imshow(mfcc, aspect="auto", origin="lower", cmap="viridis",
               extent=[0, seg.shape[0] / fs, 1, mfcc.shape[0] + 1])
ax.set_title(T("② 真实语音的 MFCC 特征（每列 = 一帧压成的 12 个数，纹理随音素流动）",
               "MFCC of real speech (each column = one frame -> 12 numbers)"))
ax.set_xlabel(T("时间 s", "Time (s)"))
ax.set_ylabel(T("MFCC 系数序号", "MFCC coefficient"))
fig.colorbar(im, ax=ax)

# ================= 面板 3：一个浊音帧的倒谱，标基频尖峰 =================
# 在语音段里找一个能量较高的帧（大概率是浊音元音）
flen = int(0.03 * fs)  # 30 ms
best_start, best_e = 0, -1
for s in range(0, len(seg) - flen, flen // 2):
    e = np.sum(seg[s:s + flen] ** 2)
    if e > best_e:
        best_e, best_start = e, s
frame = seg[best_start:best_start + flen] * np.hamming(flen)

# 实倒谱：c[n] = IFFT( log|FFT(x)| )
spec = np.fft.rfft(frame, 2048)
logmag = np.log(np.abs(spec) + 1e-10)
# 用实数逆变换回到 quefrency 轴
cep = np.fft.irfft(logmag)
q = np.arange(len(cep)) / fs  # quefrency 轴，单位：秒

# 只在人声基频合理范围（60~400 Hz -> 2.5~16.7 ms）里找尖峰
qmin, qmax = int(fs / 400), int(fs / 60)
peak_idx = qmin + np.argmax(cep[qmin:qmax])
f0 = fs / peak_idx

ax = axes[2]
ax.plot(q[:qmax + 20] * 1000, cep[:qmax + 20], lw=1.0)
ax.axvline(peak_idx / fs * 1000, color="r", ls="--", lw=1.2)
ax.annotate(T(f"基频尖峰\n{peak_idx/fs*1000:.1f} ms → F0≈{f0:.0f} Hz",
              f"pitch peak\n{peak_idx/fs*1000:.1f} ms -> F0~{f0:.0f} Hz"),
            xy=(peak_idx / fs * 1000, cep[peak_idx]),
            xytext=(peak_idx / fs * 1000 + 2, cep[peak_idx] * 0.8),
            color="r", fontsize=10,
            arrowprops=dict(arrowstyle="->", color="r"))
ax.set_title(T("③ 一个浊音帧的倒谱：低倒频率=声道包络，高倒频率那根尖峰=基音周期",
               "Cepstrum of a voiced frame: low quefrency=vocal tract, peak=pitch period"))
ax.set_xlabel(T("倒频率 quefrency (毫秒)", "Quefrency (ms)"))
ax.set_ylabel(T("倒谱幅度", "Cepstrum amplitude"))

plt.tight_layout()
plt.savefig("fig_17_1_1.png", dpi=110)
print("saved fig_17_1_1.png")
print("MFCC shape (coef, frames):", mfcc.shape)
print("detected F0:", round(f0, 1), "Hz  at", round(peak_idx / fs * 1000, 2), "ms")
