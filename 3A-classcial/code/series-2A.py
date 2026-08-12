#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""系列 2A · AEC 原理篇 配套代码。

把「回声墙」建模成一个待辨识的 FIR 滤波器，用 NLMS 做声学回声消除 (AEC)：
    1) 合成远端参考 x[n] + 随机房间冲激响应 h -> 回声
    2) 叠加近端语音 + 噪声 -> 麦克风信号 d[n]
    3) NLMS 在线辨识回声路径，输出误差 e[n] 即消回声后信号
    4) 画 ERLE 收敛曲线、消回声前后波形/频谱对比
    5) 演示远端未对齐（时延失配）时 ERLE 崩塌，并用互相关估计时延

运行（cwd=项目根目录）：
    python code/series-2A.py
所有配图输出到 figures/，前缀 s2a_。
"""

import matplotlib
matplotlib.use("Agg")  # 无显示环境后端，必须在 pyplot 之前设置

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

# ----------------------------------------------------------------------------
# 全局常量（符号遵循 STYLE.md）
# ----------------------------------------------------------------------------
FS = 16000            # 采样率 f_s (Hz)，全系列默认 16k
RNG = np.random.default_rng(2025)  # 固定随机种子，保证配图可复现
FIG_DIR = os.path.join("figures")
os.makedirs(FIG_DIR, exist_ok=True)


# ----------------------------------------------------------------------------
# 1. 信号合成
# ----------------------------------------------------------------------------
def make_room_ir(n_taps=512, rt60_taps=300):
    """生成一条随机房间冲激响应 h（强直达峰 + 指数衰减的早期反射/混响）。

    返回 h  # [n_taps]  —— 这就是我们要辨识的「回声墙」真值。
    """
    h = RNG.standard_normal(n_taps)                 # [n_taps] 白噪声骨架
    decay = np.exp(-np.arange(n_taps) / rt60_taps)  # [n_taps] 指数衰减包络
    h = 0.35 * h * decay                            # 反射/混响部分（弱于直达）
    h[:8] = 0.0                                      # 直达延迟前留空
    h[8] = 1.0                                       # 直达声主峰（第 8 抽头，能量占优）
    h /= np.linalg.norm(h)                           # 归一化能量，方便控制回声强度
    return h.astype(np.float64)


def make_farend(n_samples):
    """合成远端参考信号 x[n]：语音谱着色的宽带噪声 + 音节级调幅包络。

    用宽带（而非纯正弦）激励有两个原因：
      1) 系统辨识需要「持续激励」—— 宽带信号才能激起冲激响应的所有抽头；
      2) 宽带信号自相关近似冲激，互相关时延估计才有唯一尖峰（正弦会周期性歧义）。

    返回 x  # [n_samples]
    """
    t = np.arange(n_samples) / FS                    # [n_samples] 时间轴 (s)
    white = RNG.standard_normal(n_samples)           # [n_samples] 白噪声激励
    # 短低通核做「谱着色」，让能量偏向低频，更像语音谱
    kernel = np.array([0.15, 0.25, 0.30, 0.20, 0.10])
    x = np.convolve(white, kernel, mode="same")      # [n_samples] 有色宽带噪声
    # 2 Hz 音节级调幅包络，制造「说一句停一下」的节奏
    env = 0.5 * (1 + np.sin(2 * np.pi * 2.0 * t))
    x = x * env
    x /= np.max(np.abs(x)) + 1e-12                    # 归一化到 [-1, 1]
    return x.astype(np.float64)


def make_nearend(n_samples, active_from):
    """合成近端语音 s[n]：仅在 active_from 之后出现（模拟单讲->双讲）。

    返回 s  # [n_samples]
    """
    t = np.arange(n_samples) / FS
    s = 0.5 * np.sin(2 * np.pi * 330 * t) + 0.3 * np.sin(2 * np.pi * 700 * t)
    gate = np.zeros(n_samples)                        # [n_samples] 开关门
    gate[active_from:] = 0.5 * (1 + np.sin(2 * np.pi * 3.0 * t[active_from:]))
    s = s * gate
    return s.astype(np.float64)


# ----------------------------------------------------------------------------
# 2. NLMS 自适应滤波器（AEC 核心引擎，源自系列 1）
# ----------------------------------------------------------------------------
def nlms_aec(x, d, L=512, mu=0.5, eps=1e-6):
    """用 NLMS 辨识回声路径并消回声。

    参数:
        x  # [n]  远端参考（滤波器输入）
        d  # [n]  麦克风信号（期望信号 = 回声 + 近端 + 噪声）
        L         滤波器抽头数
        mu        归一化步长 (0,2)
        eps       防止除零的小正数
    返回:
        e  # [n]  误差 = 消回声后信号
        w  # [L]  收敛后的抽头（对回声路径的估计）
    """
    n = len(x)
    w = np.zeros(L)                       # [L] 抽头系数，初始全零
    e = np.zeros(n)                       # [n] 误差输出
    x_buf = np.zeros(L)                   # [L] 输入滑动窗（最近 L 个样本，新样本在前）
    for k in range(n):
        x_buf[1:] = x_buf[:-1]            # 右移一位
        x_buf[0] = x[k]                   # 压入当前样本
        y_hat = np.dot(w, x_buf)          # 标量：滤波器对回声的预测 ŷ[k]
        e[k] = d[k] - y_hat               # e[k] = d[k] - ŷ[k]
        norm = np.dot(x_buf, x_buf) + eps # 输入能量 ||x||^2
        w = w + mu * e[k] * x_buf / norm  # NLMS 更新
    return e, w


# ----------------------------------------------------------------------------
# 3. 评估指标
# ----------------------------------------------------------------------------
def erle_curve(d, e, frame=1024):
    """分帧计算 ERLE = 10*log10(E[d^2]/E[e^2])。

    返回:
        centers  # [n_frames]  每帧中心样本索引（画图横轴）
        erle_db  # [n_frames]  每帧 ERLE (dB)
    """
    n = len(d)
    n_frames = n // frame
    centers = np.zeros(n_frames)
    erle_db = np.zeros(n_frames)
    for i in range(n_frames):
        seg = slice(i * frame, (i + 1) * frame)
        pd = np.mean(d[seg] ** 2) + 1e-12
        pe = np.mean(e[seg] ** 2) + 1e-12
        erle_db[i] = 10.0 * np.log10(pd / pe)
        centers[i] = i * frame + frame / 2
    return centers, erle_db


def estimate_delay_xcorr(x, d, max_lag=2000):
    """用互相关估计麦克风相对参考的时延（样本数）。

    返回 lag_hat（>0 表示 d 落后于 x 若干样本）。
    """
    # 只取前段做估计，够用且省算力
    seg = min(len(x), 40000)
    xc = x[:seg] - np.mean(x[:seg])
    dc = d[:seg] - np.mean(d[:seg])
    corr = fftconvolve(dc, xc[::-1], mode="full")     # [2*seg-1] 互相关
    zero = seg - 1                                     # lag=0 对应的索引
    lags = np.arange(-max_lag, max_lag + 1)
    window = corr[zero - max_lag: zero + max_lag + 1]
    lag_hat = lags[np.argmax(np.abs(window))]
    return int(lag_hat), lags, window


# ----------------------------------------------------------------------------
# 4. 主流程
# ----------------------------------------------------------------------------
def main():
    dur = 8.0                                  # 时长 (s)
    n = int(dur * FS)                          # 总样本数
    L = 512                                    # 滤波器抽头数（>= 冲激响应长度）

    # --- 合成 ---
    h = make_room_ir(n_taps=L)                 # [L] 真·回声路径
    x = make_farend(n)                         # [n] 远端参考
    echo = fftconvolve(x, h, mode="full")[:n]  # [n] 回声 = x * h（卷积后截断）
    near_from = int(5.0 * FS)                  # 5s 后近端才开口（前段是单讲）
    s = make_nearend(n, near_from)             # [n] 近端语音
    noise = 0.002 * RNG.standard_normal(n)     # [n] 背景噪声
    d = echo + s + noise                       # [n] 麦克风信号 d[n]

    # --- 对齐良好情形：NLMS 消回声 ---
    e_ok, w_hat = nlms_aec(x, d, L=L, mu=0.5)
    c_ok, erle_ok = erle_curve(d, e_ok)

    # --- 时延失配情形：把参考人为延迟 200 样本再喂给滤波器 ---
    shift = 200
    x_mis = np.zeros(n)
    x_mis[shift:] = x[:n - shift]              # [n] 错位的参考
    e_bad, _ = nlms_aec(x_mis, d, L=L, mu=0.5)
    c_bad, erle_bad = erle_curve(d, e_bad)

    # --- 互相关时延估计（验证能找回失配量）---
    # 构造一个「麦克风比参考晚 true_delay 样本」的场景来演示时延估计。
    # 回声以直达声为主（延迟 true_delay、带房间衰减），叠加近端 -> 麦克风。
    true_delay = 200
    d_delayed = np.zeros(n)
    d_delayed[true_delay:] = 0.7 * x[:n - true_delay] + s[true_delay:]
    lag_hat, lags, corr_win = estimate_delay_xcorr(x, d_delayed)

    # 单讲段（近端未开口，0~5s）的稳态 ERLE，用于报告
    steady = erle_ok[(c_ok < near_from)]
    steady_erle = float(np.mean(steady[len(steady) // 2:]))  # 取后半段均值
    print(f"[对齐良好] 单讲段稳态 ERLE ~ {steady_erle:.1f} dB")
    print(f"[时延失配] 单讲段稳态 ERLE ~ "
          f"{np.mean(erle_bad[(c_bad < near_from)][2:]):.1f} dB")
    print(f"[互相关估计] 真实时延={true_delay} 样本, 估计={lag_hat} 样本")

    t = np.arange(n) / FS  # [n] 时间轴 (s)

    # ------------------------------------------------------------------
    # 图 1：ERLE 收敛曲线（对齐 vs 失配）
    # ------------------------------------------------------------------
    plt.figure(figsize=(9, 4.5))
    plt.plot(c_ok / FS, erle_ok, label="Aligned reference", lw=2)
    plt.plot(c_bad / FS, erle_bad, label="Misaligned (200-sample delay)",
             lw=2, color="crimson", alpha=0.85)
    plt.axvline(near_from / FS, ls="--", color="gray",
                label="Near-end talker starts (double-talk)")
    plt.xlabel("Time (s)")
    plt.ylabel("ERLE (dB)")
    plt.title("ERLE convergence: alignment is the make-or-break of AEC")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    p1 = os.path.join(FIG_DIR, "s2a_erle_convergence.png")
    plt.savefig(p1, dpi=130)
    plt.close()

    # ------------------------------------------------------------------
    # 图 2：消回声前后波形对比（取单讲段 1.0~1.3s 放大）
    # ------------------------------------------------------------------
    seg = slice(int(1.0 * FS), int(1.3 * FS))
    fig, ax = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    ax[0].plot(t[seg], d[seg], color="tab:blue")
    ax[0].set_title("Before AEC: microphone signal d[n] (echo dominates)")
    ax[0].set_ylabel("Amplitude")
    ax[0].grid(alpha=0.3)
    ax[1].plot(t[seg], e_ok[seg], color="tab:green")
    ax[1].set_title("After AEC: error signal e[n] (echo removed)")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Amplitude")
    ax[1].grid(alpha=0.3)
    # 统一纵轴范围，直观看能量下降
    ymax = np.max(np.abs(d[seg])) * 1.1
    ax[0].set_ylim(-ymax, ymax)
    ax[1].set_ylim(-ymax, ymax)
    plt.tight_layout()
    p2 = os.path.join(FIG_DIR, "s2a_waveform_before_after.png")
    plt.savefig(p2, dpi=130)
    plt.close()

    # ------------------------------------------------------------------
    # 图 3：消回声前后频谱（单讲段平均幅度谱）
    # ------------------------------------------------------------------
    seg2 = slice(int(1.0 * FS), int(4.5 * FS))  # 单讲段
    def avg_spectrum(sig):
        win = 2048
        acc = np.zeros(win // 2 + 1)
        cnt = 0
        for i in range(seg2.start, seg2.stop - win, win // 2):
            frame = sig[i:i + win] * np.hanning(win)
            acc += np.abs(np.fft.rfft(frame))
            cnt += 1
        return acc / max(cnt, 1)
    freqs = np.fft.rfftfreq(2048, 1 / FS)  # [1025] 频率轴 (Hz)
    Sd = avg_spectrum(d)
    Se = avg_spectrum(e_ok)
    plt.figure(figsize=(9, 4.5))
    plt.semilogy(freqs, Sd + 1e-9, label="Before AEC  |D(f)|", color="tab:blue")
    plt.semilogy(freqs, Se + 1e-9, label="After AEC  |E(f)|", color="tab:green")
    plt.xlim(0, 4000)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Average magnitude (log)")
    plt.title("Spectrum before vs after AEC (single-talk segment)")
    plt.legend()
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    p3 = os.path.join(FIG_DIR, "s2a_spectrum_before_after.png")
    plt.savefig(p3, dpi=130)
    plt.close()

    # ------------------------------------------------------------------
    # 图 4：真·冲激响应 vs 学到的抽头 + 互相关时延估计
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(h, label="True room IR  h", color="tab:blue", lw=1.2)
    ax[0].plot(w_hat, label="Estimated taps  w", color="crimson",
               lw=1.0, alpha=0.8)
    ax[0].set_title("System identification: w learns the echo wall h")
    ax[0].set_xlabel("Tap index")
    ax[0].set_ylabel("Amplitude")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    ax[1].plot(lags, np.abs(corr_win), color="tab:purple")
    ax[1].axvline(lag_hat, ls="--", color="crimson",
                  label=f"Estimated delay = {lag_hat} samples")
    ax[1].set_title("Cross-correlation delay estimate")
    ax[1].set_xlabel("Lag (samples)")
    ax[1].set_ylabel("|cross-corr|")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    plt.tight_layout()
    p4 = os.path.join(FIG_DIR, "s2a_ir_and_delay.png")
    plt.savefig(p4, dpi=130)
    plt.close()

    print("已生成配图：")
    for p in (p1, p2, p3, p4):
        print("  ", p)


if __name__ == "__main__":
    main()
