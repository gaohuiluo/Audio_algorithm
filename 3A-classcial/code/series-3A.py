#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""系列 3A · ANS 原理篇配套代码：谱减法 vs 维纳滤波（真实风扇录音版）。

教学参考实现（NumPy/SciPy），讲透原理，不追生产性能。
数据：voice-with-noise/fan-1.wav / fan-2.wav / fan-3.wav
      （用户在风扇下录的真人语音，16kHz 单声道，噪声强度/频谱各异）

运行：python code/series-3A.py（在 3A-classcial/ 目录下执行）
产物：
    figures/s3a_fanN_spectrograms.png   Noisy / Spectral Sub. / Wiener 三联语谱图
    figures/s3a_fanN_gain_map.png       维纳增益 M(t,f) 热力图
    figures/s3a_fanN_waveforms.png      Noisy / Spectral Sub. / Wiener 时域三联
    figures/s3a_gain_curves.png         全局理论曲线：增益 M vs SNR
    audio/s3a_fanN_specsub.wav          谱减法处理后音频
    audio/s3a_fanN_wiener.wav           维纳滤波处理后音频

真实录音无干净参考 clean，因此不再用需要 clean 的 seg_snr，
改用无参考指标：纯噪声段处理前后的能量下降 dB（降噪量）。
图内文字一律英文，避免中文字体缺失报错。
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无显示环境下也能出图，必须在 pyplot 之前设置

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import istft, stft

# ----------------------------------------------------------------------------
# 全局参数（符号遵循 STYLE.md）
# ----------------------------------------------------------------------------
FS = 16000          # f_s 采样率 (Hz)
N = 512             # N   STFT 窗长（样本）
H = 256             # H   帧移 hop = N/2
EPS = 1e-12         # 防除零小量

BASE = Path(__file__).resolve().parent.parent          # 3A-classcial/
FIG_DIR = BASE / "figures"
AUDIO_DIR = BASE / "audio"
VOICE_DIR = BASE.parent / "voice-with-noise"           # e:/Android/Tech-Blog/voice-with-noise
FIG_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

FILES = ["fan-1.wav", "fan-2.wav", "fan-3.wav"]        # 待处理三段录音


# ----------------------------------------------------------------------------
# 1. 音频 IO 工具：读 wav（float64/单声道/峰值归一化）、写 wav（float→int16）
# ----------------------------------------------------------------------------
def load_wav(path):
    """读取 wav → x # [n] float64 单声道，峰值归一化到 [-1, 1]。

    用 scipy.io.wavfile（无需 soundfile）。int16/int32/uint8 均转 float 并按
    位深归一化，再做峰值归一化。
    """
    fs, x = wavfile.read(path)
    x = x.astype(np.float64)
    if x.ndim > 1:                                     # 多声道 → 取均值转单声道
        x = x.mean(axis=1)
    # 按整数位深归一化到 [-1, 1) 附近（uint8 需去偏置）
    if np.issubdtype(np.dtype(x.dtype), np.floating):
        pass
    peak = np.max(np.abs(x)) + EPS
    x = x / peak                                       # 峰值归一化
    return x, fs


def save_wav(path, x, fs=FS):
    """写 wav：float → int16，先 clip 到 [-1, 1] 再放大到 int16 量程。"""
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, -1.0, 1.0)
    x_i16 = (x * 32767.0).astype(np.int16)
    wavfile.write(path, fs, x_i16)


# ----------------------------------------------------------------------------
# 2. STFT / iSTFT 封装
# ----------------------------------------------------------------------------
def forward_stft(x):
    """返回 X # [F, T] 复数谱，f # [F] 频率轴，tt # [T] 帧时间轴。"""
    f, tt, X = stft(x, fs=FS, window="hann", nperseg=N, noverlap=N - H,
                    boundary="zeros", padded=True)
    return X, f, tt


def inverse_stft(X):
    """由复数谱重建时域 x_rec # [n]。"""
    _, x_rec = istft(X, fs=FS, window="hann", nperseg=N, noverlap=N - H,
                     boundary=True)
    return x_rec


# ----------------------------------------------------------------------------
# 3. 稳健的纯噪声段选取 + λ_d 估计
#    真实录音里静音段不一定在开头，用滑动窗找"能量最低窗口"当纯噪声段。
# ----------------------------------------------------------------------------
def find_quietest_window(X, win_frames=15):
    """滑动窗找能量最低窗口，返回起始帧索引 q0（该窗口作纯噪声段）。

    X # [F, T]。逐帧能量沿频率轴求和，再用累积和求每个长 win_frames
    窗口的能量，取最小者。不写死"前若干帧"，对静音在任意位置都稳健。
    """
    frame_e = (np.abs(X) ** 2).sum(axis=0)             # [T] 逐帧能量
    T = frame_e.shape[0]
    K = int(min(win_frames, T))                        # 窗口不超过总帧数
    csum = np.cumsum(np.insert(frame_e, 0, 0.0))       # [T+1]
    win_e = csum[K:] - csum[:-K]                       # [T-K+1] 各窗口能量
    q0 = int(np.argmin(win_e))                         # 最安静窗口起始帧
    return q0, K


def estimate_lambda_d(X, q0, K):
    """λ_d # [F, 1]：纯噪声段 [q0, q0+K) 内 |X|² 的时间平均（沿帧轴）。"""
    noise_mag2 = np.abs(X[:, q0:q0 + K]) ** 2          # [F, K]
    lam = np.mean(noise_mag2, axis=1, keepdims=True)   # λ_d # [F, 1]
    return lam


# ----------------------------------------------------------------------------
# 4. 两种降噪：谱减法 / 维纳滤波
# ----------------------------------------------------------------------------
def spectral_subtraction(X, lam_d):
    """幅度谱减法：|Ŝ| = max(|X| - sqrt(λ_d), 0)，相位沿用带噪相位 ∠X。

    等价增益 M_SS = max(1 - sqrt(λ_d)/|X|, 0)（半波整流）。
    """
    mag = np.abs(X)                                    # |X| # [F, T]
    noise_mag = np.sqrt(lam_d)                         # 噪声幅度估计 # [F, 1]
    gain = np.maximum(1.0 - noise_mag / (mag + EPS), 0.0)  # M_SS # [F, T]
    S_hat = gain * X                                   # 复数谱：幅度缩放，相位不变
    return S_hat, gain


def wiener_filter(X, lam_d):
    """维纳增益 M = ξ/(1+ξ)。

    后验 SNR γ_post = |X|²/λ_d；先验 SNR 用极大似然估计 ξ = max(γ-1, 0)。
    （判决引导 DD 的更稳估计留到 3B。）
    """
    gamma = (np.abs(X) ** 2) / (lam_d + EPS)           # γ_post # [F, T]
    xi = np.maximum(gamma - 1.0, 0.0)                  # ξ_prior # [F, T]
    gain = xi / (1.0 + xi)                             # M # [F, T]
    S_hat = gain * X
    return S_hat, gain, xi, gamma


# ----------------------------------------------------------------------------
# 5. 无参考降噪指标（真实录音无 clean，不能用 seg_snr）
# ----------------------------------------------------------------------------
def noise_reduction_db(X_in, X_out, q0, K):
    """纯噪声段能量下降 dB：10log10(P_in_noise / P_out_noise)。

    在纯噪声段 [q0, q0+K) 上，比较处理前后的谱能量，
    衡量算法在"应该被完全抑制"的静音段上压低了多少噪声（越大越好）。
    """
    p_in = (np.abs(X_in[:, q0:q0 + K]) ** 2).sum() + EPS
    p_out = (np.abs(X_out[:, q0:q0 + K]) ** 2).sum() + EPS
    return 10.0 * np.log10(p_in / p_out)


def global_suppression_db(x_in, x_out):
    """整体噪声抑制量 dB：全信号处理前后的总能量下降。

    衡量整段（含语音+噪声）被压低的总量，配合纯噪声段指标一起看：
    静音段降得多、整体降得少 → 说明主要压的是噪声而非语音。
    """
    m = min(len(x_in), len(x_out))
    p_in = np.sum(x_in[:m] ** 2) + EPS
    p_out = np.sum(x_out[:m] ** 2) + EPS
    return 10.0 * np.log10(p_in / p_out)


# ----------------------------------------------------------------------------
# 绘图工具
# ----------------------------------------------------------------------------
def db_spec(X):
    """转 dB 幅度谱 # [F, T]，用于 imshow。"""
    return 20 * np.log10(np.abs(X) + 1e-6)


def plot_spectrograms(specs, titles, f, tt, path):
    """三联语谱图对比。specs 每个 # [F, T]。"""
    vmax = max(db_spec(s).max() for s in specs)
    vmin = vmax - 80
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, s, title in zip(axes.ravel(), specs, titles):
        im = ax.pcolormesh(tt, f / 1000, db_spec(s), vmin=vmin, vmax=vmax,
                           shading="auto", cmap="magma")
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (kHz)")
        fig.colorbar(im, ax=ax, label="dB")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_gain_curves(path):
    """增益 M vs SNR(dB)：谱减 vs 维纳，展示两者形状差异（纯理论曲线）。"""
    snr_db = np.linspace(-20, 20, 400)                 # SNR 轴 (dB) # [400]
    snr_lin = 10 ** (snr_db / 10)                      # 线性 SNR # [400]
    m_wiener = snr_lin / (1.0 + snr_lin)               # 维纳: M=ξ/(1+ξ)
    # 谱减以后验 SNR γ 计：M_SS = max(1 - 1/sqrt(γ), 0)
    m_ss = np.maximum(1.0 - 1.0 / np.sqrt(snr_lin), 0.0)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(snr_db, m_wiener, label="Wiener  M = xi/(1+xi)", lw=2)
    ax.plot(snr_db, m_ss, label="Spectral Sub.  M = max(1 - 1/sqrt(gamma), 0)",
            lw=2, ls="--")
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.axvline(0, color="gray", ls=":", lw=1)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Gain M(t,f)")
    ax.set_title("Suppression Gain vs SNR: Wiener is smooth, Spectral Sub. is hard-cut")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_gain_map(gain, f, tt, path):
    """维纳增益 M(t,f) 热力图 # [F, T]，直观看"每个频点旋钮"。"""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.pcolormesh(tt, f / 1000, gain, vmin=0, vmax=1,
                       shading="auto", cmap="viridis")
    ax.set_title("Wiener Gain Map M(t,f):  1 = keep, 0 = suppress")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (kHz)")
    fig.colorbar(im, ax=ax, label="M")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_waveforms(noisy, ss, wf, path):
    """时域波形对比 # [n]：Noisy / Spectral Subtraction / Wiener。"""
    m = min(map(len, [noisy, ss, wf]))
    t = np.arange(m) / FS
    fig, axes = plt.subplots(3, 1, figsize=(11, 6.5), sharex=True)
    for ax, sig, title in zip(
        axes,
        [noisy[:m], ss[:m], wf[:m]],
        ["Noisy (fan noise)", "Spectral Subtraction", "Wiener"],
    ):
        ax.plot(t, sig, lw=0.6)
        ax.set_ylabel(title, fontsize=9)
        ax.set_ylim(-1.1, 1.1)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Waveforms: noisy / denoised")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 单文件处理流程
# ----------------------------------------------------------------------------
def process_file(fname, tag):
    """处理一个录音文件：加载→STFT→估 λ_d→谱减+维纳→iSTFT→存音频→出图。

    参数：
        fname  文件名，如 "fan-1.wav"
        tag    输出前缀标签，如 "fan1"
    """
    print(f"\n========== {fname}  (tag={tag}) ==========")
    # --- 加载 ---
    noisy, fs = load_wav(VOICE_DIR / fname)            # noisy # [n]
    dur = len(noisy) / fs
    print(f"[load] fs={fs}, dur={dur:.2f}s, n={len(noisy)}")

    # --- STFT ---
    Xn, f, tt = forward_stft(noisy)                    # Xn # [F, T]
    print(f"[shape] STFT X # [F, T] = {Xn.shape}  (F={Xn.shape[0]}, T={Xn.shape[1]})")

    # --- 稳健纯噪声段 + λ_d 估计 ---
    q0, K = find_quietest_window(Xn, win_frames=15)
    lam_d = estimate_lambda_d(Xn, q0, K)               # λ_d # [F, 1]
    print(f"[noise] 纯噪声段起始帧 q0={q0} (~{tt[q0]:.2f}s), 窗长 K={K} 帧")
    print(f"[shape] lambda_d # [F, 1] = {lam_d.shape}")

    # --- 两种降噪 ---
    S_ss, g_ss = spectral_subtraction(Xn, lam_d)       # 谱减
    S_wf, g_wf, xi, gamma = wiener_filter(Xn, lam_d)   # 维纳
    print(f"[shape] Wiener gain M # [F, T] = {g_wf.shape}, "
          f"xi # [F, T] = {xi.shape}, gamma # [F, T] = {gamma.shape}")

    # --- 重建时域 ---
    x_ss = inverse_stft(S_ss)                          # x_ss # [n]
    x_wf = inverse_stft(S_wf)                           # x_wf # [n]

    # --- 无参考指标 ---
    nr_ss = noise_reduction_db(Xn, S_ss, q0, K)        # 纯噪声段降噪 dB
    nr_wf = noise_reduction_db(Xn, S_wf, q0, K)
    gs_ss = global_suppression_db(noisy, x_ss)         # 整体抑制 dB
    gs_wf = global_suppression_db(noisy, x_wf)
    print("\n===== 无参考降噪指标 =====")
    print(f"  纯噪声段能量下降 (dB, 越大越好):")
    print(f"     谱减法   : {nr_ss:6.2f}")
    print(f"     维纳滤波 : {nr_wf:6.2f}")
    print(f"  整体能量抑制量 (dB):")
    print(f"     谱减法   : {gs_ss:6.2f}")
    print(f"     维纳滤波 : {gs_wf:6.2f}")

    # --- 保存处理后音频 ---
    p_ss = AUDIO_DIR / f"s3a_{tag}_specsub.wav"
    p_wf = AUDIO_DIR / f"s3a_{tag}_wiener.wav"
    save_wav(p_ss, x_ss)
    save_wav(p_wf, x_wf)
    print(f"\n[audio] audio/{p_ss.name}")
    print(f"[audio] audio/{p_wf.name}")

    # --- 配图 ---
    p_spec = FIG_DIR / f"s3a_{tag}_spectrograms.png"
    p_gain = FIG_DIR / f"s3a_{tag}_gain_map.png"
    p_wave = FIG_DIR / f"s3a_{tag}_waveforms.png"
    plot_spectrograms(
        [Xn, S_ss, S_wf],
        ["Noisy", "Spectral Subtraction", "Wiener"],
        f, tt, p_spec,
    )
    plot_gain_map(g_wf, f, tt, p_gain)
    plot_waveforms(noisy, x_ss, x_wf, p_wave)
    print(f"[figures] figures/{p_spec.name}")
    print(f"[figures] figures/{p_gain.name}")
    print(f"[figures] figures/{p_wave.name}")

    return {
        "fname": fname, "tag": tag, "stft_shape": Xn.shape,
        "q0": q0, "K": K, "q0_sec": float(tt[q0]),
        "nr_ss": nr_ss, "nr_wf": nr_wf, "gs_ss": gs_ss, "gs_wf": gs_wf,
    }


# ----------------------------------------------------------------------------
# 主流程：三个文件循环处理 + 一张全局理论曲线
# ----------------------------------------------------------------------------
def main():
    tags = {"fan-1.wav": "fan1", "fan-2.wav": "fan2", "fan-3.wav": "fan3"}
    results = []
    for fname in FILES:
        results.append(process_file(fname, tags[fname]))

    # 全局的、与具体文件无关的理论增益曲线
    plot_gain_curves(FIG_DIR / "s3a_gain_curves.png")
    print("\n[figures] figures/s3a_gain_curves.png  (全局理论曲线)")

    print("\n===== series-3A 汇总 =====")
    for r in results:
        print(f"  {r['fname']}: STFT={r['stft_shape']}, "
              f"noise@frame {r['q0']}(~{r['q0_sec']:.2f}s), "
              f"NR_ss={r['nr_ss']:.2f}dB, NR_wf={r['nr_wf']:.2f}dB, "
              f"GS_ss={r['gs_ss']:.2f}dB, GS_wf={r['gs_wf']:.2f}dB")
    print("\n[done] series-3A 全部完成。")


if __name__ == "__main__":
    main()
