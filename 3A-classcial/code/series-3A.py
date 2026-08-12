#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""系列 3A · ANS 原理篇配套代码：谱减法 vs 维纳滤波。

教学参考实现（NumPy/SciPy），讲透原理，不追生产性能。
运行：python code/series-3A.py
产物：figures/s3a_*.png
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无显示环境下也能出图，必须在 pyplot 之前设置

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import istft, lfilter, stft

# ----------------------------------------------------------------------------
# 全局参数（符号遵循 STYLE.md）
# ----------------------------------------------------------------------------
FS = 16000          # f_s 采样率 (Hz)
N = 512             # N   STFT 窗长（样本）
H = 256             # H   帧移 hop = N/2
RNG = np.random.default_rng(2026)

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------
# 1. 合成一段"类语音"干净信号（无外部音频依赖）
# ----------------------------------------------------------------------------
def make_speech(fs=FS, dur=2.5):
    """合成谐波+共振峰+音节包络的类语音信号。

    返回 clean # [n]，前 0.3s 为静音（供噪声估计当"纯噪声帧"）。
    """
    n_total = int(dur * fs)
    t = np.arange(n_total) / fs                     # t # [n] 时间轴 (s)

    # 基频 f0 在 110~150 Hz 缓慢起伏（模拟语调）
    f0 = 120 + 20 * np.sin(2 * np.pi * 0.7 * t)     # f0 # [n]
    phase = 2 * np.pi * np.cumsum(f0) / fs          # 相位积分
    voiced = np.zeros(n_total)                       # voiced # [n]
    formants = [500, 1500, 2500]                     # 三个共振峰中心 (Hz)
    for k in range(1, 30):                           # 叠加 30 次谐波
        fk = k * f0                                  # 第 k 次谐波频率 # [n]
        env = np.zeros(n_total)
        for fc in formants:                          # 共振峰包络加权
            env += np.exp(-((fk - fc) ** 2) / (2 * 250.0 ** 2))
        voiced += env * np.sin(k * phase)

    # 音节门控包络：制造"说话-停顿"的起伏
    syl = np.zeros(n_total)
    for (start, end) in [(0.30, 0.75), (0.85, 1.30), (1.45, 2.05), (2.15, 2.45)]:
        i0, i1 = int(start * fs), int(end * fs)
        w = np.hanning(i1 - i0)                      # 每个音节用汉宁窗淡入淡出
        syl[i0:i1] = w
    clean = voiced * syl
    clean /= np.max(np.abs(clean)) + 1e-12           # 归一化到 [-1, 1]
    return clean.astype(np.float64)


# ----------------------------------------------------------------------------
# 2. 两种稳态噪声：白噪 + "风扇"低频噪声
# ----------------------------------------------------------------------------
def make_white(n):
    return RNG.standard_normal(n)                    # white # [n]


def make_fan(n, fs=FS):
    """风扇/空调噪声：低通有色噪声 + 两条低频嗡鸣 tonal。"""
    w = RNG.standard_normal(n)
    # 一阶低通（AR(1)）把能量压到低频，模拟机械噪声的"闷"
    colored = lfilter([1.0], [1.0, -0.95], w)        # colored # [n]
    t = np.arange(n) / fs
    hum = 0.3 * np.sin(2 * np.pi * 120 * t) + 0.2 * np.sin(2 * np.pi * 240 * t)
    fan = colored / (np.std(colored) + 1e-12) + hum
    return fan


def mix_at_snr(clean, noise, snr_db):
    """按目标全局 SNR 混合，只在语音活跃段计功率更贴近真实感知。"""
    active = np.abs(clean) > 0.01
    ps = np.mean(clean[active] ** 2) if active.any() else np.mean(clean ** 2)
    pn = np.mean(noise ** 2)
    target_pn = ps / (10 ** (snr_db / 10))
    noise_scaled = noise * np.sqrt(target_pn / (pn + 1e-12))
    return clean + noise_scaled, noise_scaled


# ----------------------------------------------------------------------------
# 3. STFT / iSTFT 封装
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
# 4. 噪声功率谱 λ_d 估计：取前若干"纯噪声帧"平均
# ----------------------------------------------------------------------------
def estimate_lambda_d(X, n_noise_frames=10):
    """λ_d # [F, 1]：前 n_noise_frames 帧 |X|² 的均值（沿时间轴）。"""
    noise_mag2 = np.abs(X[:, :n_noise_frames]) ** 2   # [F, K]
    lam = np.mean(noise_mag2, axis=1, keepdims=True)  # λ_d # [F, 1]
    return lam


# ----------------------------------------------------------------------------
# 5. 两种降噪：谱减法 / 维纳滤波
# ----------------------------------------------------------------------------
def spectral_subtraction(X, lam_d):
    """幅度谱减法：|Ŝ| = max(|X| - sqrt(λ_d), 0)，相位沿用带噪相位 ∠X。

    等价增益 M_SS = max(1 - sqrt(λ_d)/|X|, 0)（半波整流）。
    """
    mag = np.abs(X)                                   # |X| # [F, T]
    noise_mag = np.sqrt(lam_d)                        # 噪声幅度估计 # [F, 1]
    gain = np.maximum(1.0 - noise_mag / (mag + 1e-12), 0.0)  # M_SS # [F, T]
    S_hat = gain * X                                  # 复数谱：幅度缩放，相位不变
    return S_hat, gain


def wiener_filter(X, lam_d):
    """维纳增益 M = ξ/(1+ξ)。

    后验 SNR γ_post = |X|²/λ_d；先验 SNR 用极大似然估计 ξ = max(γ-1, 0)。
    （判决引导 DD 的更稳估计留到 3B。）
    """
    gamma = (np.abs(X) ** 2) / (lam_d + 1e-12)        # γ_post # [F, T]
    xi = np.maximum(gamma - 1.0, 0.0)                 # ξ_prior # [F, T]
    gain = xi / (1.0 + xi)                            # M # [F, T]
    S_hat = gain * X
    return S_hat, gain, xi, gamma


# ----------------------------------------------------------------------------
# 6. 指标：分段 SNR（越大越好）
# ----------------------------------------------------------------------------
def seg_snr(clean, test):
    """对齐长度后计算全局 SNR (dB)：10log10(P_signal / P_error)。"""
    m = min(len(clean), len(test))
    c, x = clean[:m], test[:m]
    err = c - x
    return 10 * np.log10((np.sum(c ** 2) + 1e-12) / (np.sum(err ** 2) + 1e-12))


# ----------------------------------------------------------------------------
# 绘图工具
# ----------------------------------------------------------------------------
def db_spec(X):
    """转 dB 幅度谱 # [F, T]，用于 imshow。"""
    return 20 * np.log10(np.abs(X) + 1e-6)


def plot_spectrograms(specs, titles, f, tt, path):
    """2x2 语谱图对比。specs 每个 # [F, T]。"""
    vmax = max(db_spec(s).max() for s in specs)
    vmin = vmax - 80
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
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
    """增益 M vs SNR(dB)：谱减 vs 维纳，展示两者形状差异。"""
    snr_db = np.linspace(-20, 20, 400)                # SNR 轴 (dB) # [400]
    snr_lin = 10 ** (snr_db / 10)                     # 线性 SNR # [400]
    m_wiener = snr_lin / (1.0 + snr_lin)              # 维纳: M=ξ/(1+ξ)
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


def plot_waveforms(clean, noisy, ss, wf, path):
    """时域波形对比 # [n]。"""
    m = min(map(len, [clean, noisy, ss, wf]))
    t = np.arange(m) / FS
    fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
    for ax, sig, title in zip(
        axes,
        [clean[:m], noisy[:m], ss[:m], wf[:m]],
        ["Clean", "Noisy (fan noise)", "Spectral Subtraction", "Wiener"],
    ):
        ax.plot(t, sig, lw=0.6)
        ax.set_ylabel(title, fontsize=9)
        ax.set_ylim(-1.1, 1.1)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Waveforms: clean / noisy / denoised")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main():
    clean = make_speech()                             # clean # [n]

    # --- 场景：风扇噪声（低频有色）为主线，白噪作对照 ---
    fan = make_fan(len(clean))                         # fan # [n]
    noisy, _ = mix_at_snr(clean, fan, snr_db=3.0)      # noisy # [n]

    white = make_white(len(clean))
    noisy_white, _ = mix_at_snr(clean, white, snr_db=3.0)

    # --- STFT ---
    Xc, f, tt = forward_stft(clean)                    # Xc # [F, T]
    Xn, _, _ = forward_stft(noisy)                     # Xn # [F, T]
    print(f"[shape] STFT X # [F, T] = {Xn.shape}  (F={Xn.shape[0]}, T={Xn.shape[1]})")

    # --- 噪声功率谱估计（前 10 帧为静音，当纯噪声帧）---
    lam_d = estimate_lambda_d(Xn, n_noise_frames=10)   # λ_d # [F, 1]
    print(f"[shape] lambda_d # [F, 1] = {lam_d.shape}")

    # --- 两种降噪 ---
    S_ss, g_ss = spectral_subtraction(Xn, lam_d)       # 谱减
    S_wf, g_wf, xi, gamma = wiener_filter(Xn, lam_d)   # 维纳
    print(f"[shape] Wiener gain M # [F, T] = {g_wf.shape}, "
          f"xi # [F, T] = {xi.shape}, gamma # [F, T] = {gamma.shape}")

    # --- 重建时域 ---
    x_ss = inverse_stft(S_ss)                          # x_ss # [n]
    x_wf = inverse_stft(S_wf)                           # x_wf # [n]

    # --- 指标 ---
    snr_in = seg_snr(clean, noisy)
    snr_ss = seg_snr(clean, x_ss)
    snr_wf = seg_snr(clean, x_wf)
    print("\n===== 全局 SNR (dB) =====")
    print(f"  带噪输入      : {snr_in:6.2f}")
    print(f"  谱减法输出    : {snr_ss:6.2f}  (Δ {snr_ss - snr_in:+.2f})")
    print(f"  维纳滤波输出  : {snr_wf:6.2f}  (Δ {snr_wf - snr_in:+.2f})")

    # --- 配图 ---
    plot_spectrograms(
        [Xc, Xn, S_ss, S_wf],
        ["Clean", "Noisy (SNR=3dB)", "Spectral Subtraction", "Wiener"],
        f, tt, FIG_DIR / "s3a_spectrograms.png",
    )
    plot_gain_curves(FIG_DIR / "s3a_gain_curves.png")
    plot_gain_map(g_wf, f, tt, FIG_DIR / "s3a_gain_map.png")
    plot_waveforms(clean, noisy, x_ss, x_wf, FIG_DIR / "s3a_waveforms.png")

    print("\n[figures] 已生成：")
    for name in ["s3a_spectrograms.png", "s3a_gain_curves.png",
                 "s3a_gain_map.png", "s3a_waveforms.png"]:
        print(f"  figures/{name}")
    print("\n[done] series-3A 全部完成。")


if __name__ == "__main__":
    main()
