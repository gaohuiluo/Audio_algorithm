#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""系列 2B · AEC 工程篇 —— 双讲、发散与非线性残留 · 配套代码

本脚本在系列 2A 的 NLMS 回声消除基础上，演示三件工程上绕不开的事：

  实验一  双讲(double-talk)会污染误差信号，NLMS 若不做双讲检测就会"学歪"→
          失调(misalignment)上升、ERLE 崩塌；
  实验二  用 Geigel 算法做双讲检测，检测到双讲就冻结更新(μ→0)，滤波器恢复稳定；
  实验三  扬声器软削波(soft-clipping)非线性使真实回声 != 线性卷积，线性 AEC 存在
          原理性残留；再挂一级频域残余回声抑制(RES, 维纳式增益)把残留压下去。

运行:
    python code/series-2B.py
产物:
    figures/s2b_divergence.png
    figures/s2b_geigel.png
    figures/s2b_res.png
"""
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无显示环境后端，必须在 import pyplot 前设置
import matplotlib.pyplot as plt
from scipy.signal import lfilter, stft, istft
from scipy.ndimage import maximum_filter1d

GEIGEL_ENV = 80  # Geigel 分子的短时包络窗(样本)，5ms@16k，抗单点抖动

# ----------------------------------------------------------------------------
# 全局配置
# ----------------------------------------------------------------------------
FS = 16000                       # 采样率 (Hz)，全系列默认 16k
RNG = np.random.default_rng(2024)  # 固定随机种子，保证配图可复现
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------
# 信号与房间脉冲响应生成工具
# ----------------------------------------------------------------------------
def speech_like(n_samples, seed, active_mask=None):
    """生成"类语音"信号：有色噪声 + 音节起伏包络。

    仅用于教学演示 ERLE / 失调的动态行为，不追求真实语音质量。

    返回 sig  # [n_samples] float64
    """
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n_samples)                 # [n_samples]
    # AR(1) 着色，让频谱偏低频，像浊音
    colored = lfilter([1.0], [1.0, -0.95], white)          # [n_samples]
    # 音节包络：把慢变随机门限平滑成一段段"说话/停顿"
    gate = (rng.standard_normal(n_samples) > 0.3).astype(float)  # [n_samples]
    env = lfilter(np.ones(1600) / 1600.0, [1.0], gate)     # [n_samples] 100ms 平滑
    sig = colored * env                                    # [n_samples]
    if active_mask is not None:
        sig = sig * active_mask                            # 只在指定区间有能量
    # 归一化到单位标准差，方便后面按 dB 调能量
    sig = sig / (sig.std() + 1e-12)
    return sig


def make_rir(length, seed=7, direct_tap=12):
    """生成一条"有主峰 + 衰减混响尾"的房间脉冲响应(回声路径 h)。

    主峰(直达/最强反射)集中在 direct_tap 处，幅度为 1；后面挂一条指数衰减的
    随机混响尾。有明显主峰，Geigel 那种"看幅度比"的双讲判据才站得住脚。

    返回 h  # [length] float64，峰值归一到 1
    """
    rng = np.random.default_rng(seed)
    h = np.zeros(length)                                   # [length]
    idx = np.arange(length)                                # [length]
    tail = rng.standard_normal(length) * np.exp(-idx / (length / 5.0)) * 0.35  # [length]
    tail[:direct_tap] = 0.0                                 # 主峰之前无能量(纯时延)
    h = h + tail
    h[direct_tap] = 1.0                                     # 直达主峰
    h = h / np.max(np.abs(h))                               # 峰值归一到 1
    return h


def framewise_erle(d, e, frame=320):
    """分帧计算 ERLE = 10log10( E[d^2] / E[e^2] )，单位 dB。

    d  # [n] 麦克风信号(含回声)
    e  # [n] AEC 误差(残留)
    返回 (t_sec[frames], erle_db[frames])
    """
    n = min(len(d), len(e))
    n_frames = n // frame
    d = d[:n_frames * frame].reshape(n_frames, frame)      # [F, frame]
    e = e[:n_frames * frame].reshape(n_frames, frame)      # [F, frame]
    p_d = np.sum(d ** 2, axis=1) + 1e-12                   # [F]
    p_e = np.sum(e ** 2, axis=1) + 1e-12                   # [F]
    erle = 10.0 * np.log10(p_d / p_e)                      # [F]
    t = (np.arange(n_frames) * frame + frame / 2) / FS     # [F] 帧中心时间(s)
    return t, erle


# ----------------------------------------------------------------------------
# NLMS 自适应滤波器（可选 Geigel 双讲检测冻结更新）
# ----------------------------------------------------------------------------
def nlms_aec(x, d, L=256, mu=0.5, delta=1e-3,
             use_dtd=False, geigel_thr=1.2, hold=800,
             h_ref=None, rec_hop=160):
    """NLMS 回声消除，附可选 Geigel 双讲检测。

    x        # [n] 远端参考(扬声器数字信号)
    d        # [n] 麦克风采集(回声 + 可能的近端语音 + 噪声)
    L        # 滤波器抽头数
    mu       # 归一化步长
    delta    # 归一化分母正则项，防止除零
    use_dtd  # 是否启用 Geigel 双讲检测并冻结更新
    geigel_thr  # Geigel 判据阈值 T
    hold     # 检测到双讲后额外保持冻结的样本数(迟滞，防抖)
    h_ref    # [L] 真实回声路径(补零到 L)，用于计算失调；None 则不记录
    rec_hop  # 每隔多少样本记录一次失调/权重范数

    返回 dict:
        e            # [n] 误差
        dtd_flag     # [n] 每样本是否判为双讲(0/1)
        geigel_stat  # [n] Geigel 判决统计量
        rec_t        # [K] 记录点时间(s)
        misalign_db  # [K] 失调(dB)，需 h_ref
        wnorm        # [K] 权重范数
    """
    n = len(x)
    w = np.zeros(L)                                        # [L] 抽头，从零起步
    e = np.zeros(n)                                        # [n]
    dtd_flag = np.zeros(n)                                 # [n]

    # Geigel 判据: stat[k] = max(|d[k-W..k]|) / max(|x[k-L+1..k]|)
    # 分子用短时包络(而非单点 |d[k]|)抗抖动，分母是最近 L 个远端样本的幅度峰值。
    d_env = maximum_filter1d(np.abs(d), size=GEIGEL_ENV)   # [n] 近端短时包络
    run_max = maximum_filter1d(np.abs(x), size=L, origin=(L - 1) // 2)  # [n]
    geigel_stat = d_env / (run_max + 1e-9)                # [n]

    rec_t, misalign_db, wnorm = [], [], []
    freeze_counter = 0                                     # 冻结迟滞倒计时

    for k in range(L, n):
        xk = x[k - L + 1:k + 1][::-1]                     # [L] 参考帧(最新样本在前)
        y_hat = np.dot(w, xk)                             # 标量：估计回声
        ek = d[k] - y_hat                                 # 标量：误差
        e[k] = ek

        update = True
        if use_dtd:
            if geigel_stat[k] > geigel_thr:              # 触发双讲
                freeze_counter = hold                    # 重置迟滞窗
            if freeze_counter > 0:
                update = False                           # 冻结更新: μ→0
                dtd_flag[k] = 1.0
                freeze_counter -= 1

        if update:
            norm = np.dot(xk, xk) + delta                # 标量：参考帧能量
            w = w + mu * ek * xk / norm                  # [L] NLMS 更新

        if h_ref is not None and (k % rec_hop == 0):
            err_vec = w - h_ref                          # [L]
            mis = 10.0 * np.log10(
                (np.dot(err_vec, err_vec) + 1e-12)
                / (np.dot(h_ref, h_ref) + 1e-12))
            rec_t.append(k / FS)
            misalign_db.append(mis)
            wnorm.append(np.sqrt(np.dot(w, w)))

    return {
        "e": e,
        "dtd_flag": dtd_flag,
        "geigel_stat": geigel_stat,
        "rec_t": np.array(rec_t),
        "misalign_db": np.array(misalign_db),
        "wnorm": np.array(wnorm),
    }


# ----------------------------------------------------------------------------
# 实验一 & 二：双讲导致发散，Geigel 检测冻结更新恢复稳定
# ----------------------------------------------------------------------------
def experiment_doubletalk():
    """构造含双讲段的场景，对比 无DTD / 有DTD 两条曲线。"""
    dur = 6.0
    n = int(dur * FS)                                     # [标量] 总样本数
    L, P = 256, 128                                       # 滤波器长 / 回声路径长

    # 双讲区间: 2.5s ~ 4.0s
    dt_start, dt_end = int(2.5 * FS), int(4.0 * FS)
    near_mask = np.zeros(n)                               # [n]
    near_mask[dt_start:dt_end] = 1.0

    # 远端全程活跃
    x = speech_like(n, seed=11)                          # [n]
    # 回声路径 + 线性回声
    h = make_rir(P, seed=7)                              # [P] 带主峰的房间响应
    ECHO = 0.35                                          # 回声路径总增益(相对远端)
    echo = lfilter(h, 1.0, x) * ECHO                     # [n] 线性回声
    # 近端语音仅在双讲区间，且比回声明显更响(近端就在自己麦跟前，约 +10dB)——
    # 这正是 Geigel"看幅度比"判据成立的物理前提。
    near = speech_like(n, seed=29, active_mask=near_mask)  # [n]
    near *= 3.0 * echo[dt_start:dt_end].std() / (near[dt_start:dt_end].std() + 1e-12)
    # 底噪
    noise = 0.001 * RNG.standard_normal(n)               # [n]
    d = echo + near + noise                              # [n] 麦克风信号

    # 真实回声路径补零到 L，用于计算失调
    h_ref = np.zeros(L)                                  # [L]
    h_ref[:P] = h * ECHO                                 # w 应收敛到 ECHO*h

    res_no = nlms_aec(x, d, L=L, mu=0.5, use_dtd=False, h_ref=h_ref)
    res_dtd = nlms_aec(x, d, L=L, mu=0.5, use_dtd=True,
                       geigel_thr=1.2, hold=800, h_ref=h_ref)

    t_no, erle_no = framewise_erle(d, res_no["e"])
    t_dtd, erle_dtd = framewise_erle(d, res_dtd["e"])

    # ---- 图1: 失调 + ERLE 对比 ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    dt0, dt1 = dt_start / FS, dt_end / FS

    ax1.axvspan(dt0, dt1, color="orange", alpha=0.15, label="double-talk region")
    ax1.plot(res_no["rec_t"], res_no["misalign_db"], "r-", lw=1.6,
             label="no DTD (diverges)")
    ax1.plot(res_dtd["rec_t"], res_dtd["misalign_db"], "b-", lw=1.6,
             label="Geigel DTD (frozen)")
    ax1.set_ylabel("Misalignment (dB)")
    ax1.set_title("Double-talk without DTD corrupts the filter; Geigel DTD keeps it stable")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2.axvspan(dt0, dt1, color="orange", alpha=0.15)
    ax2.plot(t_no, erle_no, "r-", lw=1.0, alpha=0.8, label="no DTD")
    ax2.plot(t_dtd, erle_dtd, "b-", lw=1.0, alpha=0.8, label="Geigel DTD")
    ax2.set_ylabel("ERLE (dB)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylim(-15, 40)
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "s2b_divergence.png", dpi=130)
    plt.close(fig)

    # ---- 图2: Geigel 判决统计量 + 检测标志 ----
    t_axis = np.arange(n) / FS                            # [n]
    fig2, ax = plt.subplots(figsize=(10, 4))
    ax.axvspan(dt0, dt1, color="orange", alpha=0.15, label="true double-talk")
    # 抽稀绘制，避免图太密
    step = 20
    ax.plot(t_axis[::step], res_dtd["geigel_stat"][::step], color="gray",
            lw=0.6, alpha=0.7, label="Geigel statistic |d|/max|x|")
    ax.axhline(1.2, color="green", ls="--", lw=1.5, label="threshold T=1.2")
    # 检测到双讲的样本，画在底部
    det = res_dtd["dtd_flag"] > 0.5
    ax.plot(t_axis[det][::step], np.full(det.sum(), 0.2)[::step], "b.",
            ms=2, label="detected & frozen")
    ax.set_ylim(0, 3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Geigel statistic")
    ax.set_title("Geigel double-talk detector: statistic crosses threshold during double-talk")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(FIG_DIR / "s2b_geigel.png", dpi=130)
    plt.close(fig2)

    # 控制台摘要
    def tail_mean(t, v, lo, hi):
        m = (t >= lo) & (t <= hi)
        return float(np.mean(v[m])) if m.any() else float("nan")

    print("[实验一/二 双讲]")
    print(f"  双讲前失调(2.0-2.5s): 无DTD = {tail_mean(res_no['rec_t'], res_no['misalign_db'], 2.0, 2.5):+.1f} dB"
          f" | 有DTD = {tail_mean(res_dtd['rec_t'], res_dtd['misalign_db'], 2.0, 2.5):+.1f} dB (收敛一致)")
    print(f"  双讲中失调(2.5-4.0s): 无DTD = {tail_mean(res_no['rec_t'], res_no['misalign_db'], 2.5, 4.0):+.1f} dB"
          f" | 有DTD = {tail_mean(res_dtd['rec_t'], res_dtd['misalign_db'], 2.5, 4.0):+.1f} dB (无DTD学歪)")
    print(f"  双讲后失调(4.5-6s):   无DTD = {tail_mean(res_no['rec_t'], res_no['misalign_db'], 4.5, 6.0):+.1f} dB"
          f" | 有DTD = {tail_mean(res_dtd['rec_t'], res_dtd['misalign_db'], 4.5, 6.0):+.1f} dB")
    print(f"  双讲检出率 = {(res_dtd['dtd_flag'][int(2.5*FS):int(4.0*FS)] > 0.5).mean():.1%}"
          f" | 单讲误检率 = {(res_dtd['dtd_flag'][:int(2.5*FS)] > 0.5).mean():.1%}")


# ----------------------------------------------------------------------------
# 实验三：扬声器非线性 → 线性 AEC 残留 → 频域 RES 抑制
# ----------------------------------------------------------------------------
def residual_suppression(e, x, fs=FS, nperseg=512, noverlap=384,
                         gain_min=0.1, overest=1.4):
    """频域残余回声抑制(RES)：维纳式增益压残留。

    思路：假设残余回声幅度谱正比于远端幅度谱，
          |R_est(t,f)| = leak(f) * |X(t,f)|，
          leak(f) 从"远端活跃"帧上以 |E|/|X| 的中位数在线估计。
          维纳式增益  G = max(1 - overest * |R_est|^2 / |E|^2 , gain_min)。

    e  # [n] 线性 AEC 残留误差
    x  # [n] 远端参考
    返回 e_res  # [n] RES 之后的残留
    """
    f, t, E = stft(e, fs=fs, nperseg=nperseg, noverlap=noverlap)  # E [F, T]
    _, _, X = stft(x, fs=fs, nperseg=nperseg, noverlap=noverlap)  # X [F, T]
    magE = np.abs(E)                                     # [F, T]
    magX = np.abs(X)                                     # [F, T]

    # 逐频点估计泄漏系数 leak(f)：远端有能量的帧上取 |E|/|X| 中位数
    active = magX > (0.1 * magX.max())                   # [F, T] 远端活跃掩码
    leak = np.ones(magE.shape[0])                        # [F]
    for fi in range(magE.shape[0]):
        m = active[fi]
        if m.sum() > 5:
            leak[fi] = np.median(magE[fi, m] / (magX[fi, m] + 1e-9))
    R_est = leak[:, None] * magX                         # [F, T] 残余回声幅度估计

    # 维纳式增益：残留越像回声(|E|≈|R_est|)压得越狠，越像近端(|E|>>|R_est|)越保留
    gain = 1.0 - overest * (R_est ** 2) / (magE ** 2 + 1e-12)  # [F, T]
    gain = np.clip(gain, gain_min, 1.0)                  # [F, T]

    E_res = gain * E                                     # [F, T] 增益作用于复数谱
    _, e_res = istft(E_res, fs=fs, nperseg=nperseg, noverlap=noverlap)  # [~n]
    e_res = e_res[:len(e)]                               # 对齐长度
    if len(e_res) < len(e):
        e_res = np.pad(e_res, (0, len(e) - len(e_res)))
    return e_res


def experiment_nonlinear():
    """扬声器软削波 → 线性 AEC 残留 → RES 抑制。"""
    dur = 5.0
    n = int(dur * FS)
    L, P = 256, 128

    x = speech_like(n, seed=101)                         # [n] 远端数字信号(AEC 唯一可用参考)
    h = make_rir(P, seed=7)                              # [P] 回声路径

    # 扬声器/功放非线性：软削波(soft clipping)。
    # 把信号推到较高电平，让 tanh 引入明显谐波失真。
    drive = 3.0
    x_loud = np.tanh(drive * x) / drive                  # [n] 扬声器实际发声(失真)
    echo = lfilter(h, 1.0, x_loud) * 0.5                 # [n] 真实回声 = 失真信号过房间
    noise = 0.001 * RNG.standard_normal(n)               # [n]
    d = echo + noise                                     # [n] 单讲(仅回声)，便于量化残留

    # 线性 AEC：参考只有线性的 x，学不到非线性成分
    res = nlms_aec(x, d, L=L, mu=0.5, use_dtd=False)
    e_lin = res["e"]                                     # [n] 线性 AEC 残留

    # 后置 RES
    e_res = residual_suppression(e_lin, x)               # [n]

    # 收敛后段(3-5s)做量化
    seg = slice(int(3.0 * FS), n)
    def erle_seg(err):
        return 10.0 * np.log10(
            (np.sum(d[seg] ** 2) + 1e-12) / (np.sum(err[seg] ** 2) + 1e-12))
    erle_lin = erle_seg(e_lin)
    erle_res = erle_seg(e_res)

    # ---- 图3: 平均幅度谱 + ERLE 对比 ----
    def avg_spec(sig):
        f, _, S = stft(sig[seg], fs=FS, nperseg=512, noverlap=384)  # [F, T]
        return f, np.mean(np.abs(S), axis=1)             # [F]
    f_axis, sp_d = avg_spec(d)
    _, sp_lin = avg_spec(e_lin)
    _, sp_res = avg_spec(e_res)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    eps = 1e-9
    ax1.semilogy(f_axis, sp_d + eps, "k-", lw=1.2, label="mic echo d[n]")
    ax1.semilogy(f_axis, sp_lin + eps, "r-", lw=1.2, label="after linear AEC (residual)")
    ax1.semilogy(f_axis, sp_res + eps, "b-", lw=1.2, label="after AEC + RES")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Avg magnitude")
    ax1.set_title("Nonlinear echo leaves residual that linear AEC cannot remove")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3, which="both")

    bars = ax2.bar(["linear AEC", "AEC + RES"], [erle_lin, erle_res],
                   color=["#d9534f", "#2b6cb0"], width=0.5)
    ax2.set_ylabel("ERLE (dB), 3-5s")
    ax2.set_title("RES lifts ERLE by suppressing residual echo")
    for b, v in zip(bars, [erle_lin, erle_res]):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f} dB",
                 ha="center", va="bottom", fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_ylim(0, max(erle_lin, erle_res) * 1.25)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "s2b_res.png", dpi=130)
    plt.close(fig)

    print("[实验三 非线性 + RES]")
    print(f"  线性 AEC ERLE = {erle_lin:.1f} dB | +RES ERLE = {erle_res:.1f} dB"
          f" | 提升 {erle_res - erle_lin:+.1f} dB")


def main():
    experiment_doubletalk()
    experiment_nonlinear()
    print("\n配图已生成于 figures/ : s2b_divergence.png, s2b_geigel.png, s2b_res.png")


if __name__ == "__main__":
    main()
