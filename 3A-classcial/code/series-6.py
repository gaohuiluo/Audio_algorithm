#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""系列 6 · 多麦阵列信号处理 配套代码。

从「延迟求和 (Delay-and-Sum, DAS)」到「MVDR」，把单帧频域增益推广到空域滤波：
    1) 构造均匀线阵 (ULA) 的几何 + 平面波到达的导向矢量 a(θ)
    2) 合成目标 + 干扰 + 传感器噪声的多通道快拍，估计空间协方差 R
    3) 实现 DAS 与 MVDR 两种波束权 w，画波束方向图 (beampattern) 对比
    4) 定量对比阵列增益 / 干扰抑制 / 输出 SINR
    5) 演示麦克风增益失配 -> 导向矢量失配 -> 波束方向图畸变

运行（cwd=项目根目录）：
    python code/series-6.py
所有配图输出到 figures/，前缀 s6_。复数矩阵运算统一走 np.linalg。
"""

import matplotlib
matplotlib.use("Agg")  # 无显示环境后端，必须在 pyplot 之前设置

import os
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# 全局常量（符号遵循 STYLE.md：R 空间协方差、θ 到达角）
# ----------------------------------------------------------------------------
C = 343.0                          # 声速 (m/s)，室温近似
F0 = 2000.0                        # 窄带处理的设计频率 (Hz)
OMEGA = 2 * np.pi * F0             # 角频率 ω = 2πf
M = 8                              # 阵元（麦克风）数
D = C / (2 * F0)                   # 阵元间距 = 半波长，避开空间混叠
RNG = np.random.default_rng(2025)  # 固定随机种子，保证配图可复现
FIG_DIR = os.path.join("figures")
os.makedirs(FIG_DIR, exist_ok=True)


# ----------------------------------------------------------------------------
# 1. 导向矢量：均匀线阵 + 远场平面波
# ----------------------------------------------------------------------------
def steering_vector(theta_deg, m=M, d=D, omega=OMEGA):
    """远场平面波下，ULA 对来向 θ 的导向矢量 a(θ)。

    第 m 个阵元相对参考阵元的时延 τ_m = m·d·sinθ / c，
    对应相位 exp(-j·ω·τ_m)。

    返回 a  # [M]  复数导向矢量
    """
    theta = np.deg2rad(theta_deg)
    m_idx = np.arange(m)                          # [M] 阵元索引 0..M-1
    tau = m_idx * d * np.sin(theta) / C           # [M] 各阵元相对时延 (s)
    a = np.exp(-1j * omega * tau)                 # [M] 相位对齐因子
    return a.astype(np.complex128)


# ----------------------------------------------------------------------------
# 2. 多通道快拍合成 + 空间协方差估计
# ----------------------------------------------------------------------------
def simulate_snapshots(theta_s, theta_i, n_snap=2000,
                       snr_db=10.0, inr_db=20.0, gain_mismatch=None):
    """合成阵列接收的窄带复数快拍 X。

    模型：X[:,t] = g ⊙ (a(θ_s)·s_t + a(θ_i)·i_t) + noise_t
        s_t 目标信号、i_t 干扰，均为单位功率复高斯；
        噪声为各阵元独立复高斯（空间白）。
        g 为逐阵元增益失配（None 表示理想一致，全 1）。

    参数:
        theta_s  目标来向 (deg)
        theta_i  干扰来向 (deg)
        snr_db   目标相对噪声的信噪比 (dB)
        inr_db   干扰相对噪声的干噪比 (dB)
        gain_mismatch  # [M] 或 None，逐阵元幅度增益
    返回:
        X  # [M, n_snap]  复数快拍矩阵
    """
    a_s = steering_vector(theta_s)                # [M]
    a_i = steering_vector(theta_i)                # [M]

    amp_s = 10 ** (snr_db / 20.0)                 # 目标幅度（噪声功率归一化为 1）
    amp_i = 10 ** (inr_db / 20.0)                 # 干扰幅度

    # 单位功率复高斯：实虚部各 1/sqrt(2)，合成功率为 1
    def cgauss(shape):
        return (RNG.standard_normal(shape) + 1j * RNG.standard_normal(shape)) / np.sqrt(2)

    s = amp_s * cgauss(n_snap)                    # [n_snap] 目标源
    i = amp_i * cgauss(n_snap)                    # [n_snap] 干扰源
    noise = cgauss((M, n_snap))                   # [M, n_snap] 空间白噪声

    X = np.outer(a_s, s) + np.outer(a_i, i) + noise   # [M, n_snap]

    if gain_mismatch is not None:
        X = X * gain_mismatch[:, None]            # 逐阵元乘增益（广播）[M, n_snap]
    return X.astype(np.complex128)


def spatial_covariance(X, diag_load=1e-3):
    """样本空间协方差 R = X X^H / T，附加对角加载稳健化。

    返回 R  # [M, M]  厄米特正定复矩阵
    """
    n_snap = X.shape[1]
    R = (X @ X.conj().T) / n_snap                 # [M, M] 样本协方差
    R = R + diag_load * np.trace(R).real / M * np.eye(M)  # 对角加载
    return R


# ----------------------------------------------------------------------------
# 3. 两种波束权
# ----------------------------------------------------------------------------
def das_weights(theta_s):
    """延迟求和 (DAS) 波束权：w = a(θ_s) / M。

    对齐目标方向的相位后等权平均，满足 w^H a(θ_s) = 1。
    返回 w  # [M]
    """
    a_s = steering_vector(theta_s)                # [M]
    w = a_s / M                                   # [M]
    return w


def mvdr_weights(R, theta_s):
    """MVDR 波束权：w = R^-1 a / (a^H R^-1 a)。

    在保持目标方向增益为 1 的约束下，最小化输出总功率 w^H R w。
    复数线性方程组用 np.linalg.solve（比显式求逆更稳）。
    返回 w  # [M]
    """
    a_s = steering_vector(theta_s)                # [M]
    Rinv_a = np.linalg.solve(R, a_s)              # [M]  = R^-1 a
    denom = a_s.conj() @ Rinv_a                   # 标量 a^H R^-1 a（实正）
    w = Rinv_a / denom                            # [M]
    return w


# ----------------------------------------------------------------------------
# 4. 波束方向图与指标
# ----------------------------------------------------------------------------
def beampattern(w, angles_deg):
    """波束方向图 B(θ) = |w^H a(θ)|²，随扫描角变化。

    返回 b_db  # [len(angles)]  归一化后的响应 (dB)
    """
    resp = np.array([w.conj() @ steering_vector(t) for t in angles_deg])  # [A]
    p = np.abs(resp) ** 2                          # [A] 功率响应
    b_db = 10 * np.log10(p / np.max(p) + 1e-12)    # 峰值归一化到 0 dB
    return b_db


def output_sinr(w, theta_s, theta_i, snr_db, inr_db):
    """闭式计算波束输出 SINR = 目标功率 / (干扰+噪声功率)。

    返回 sinr_db  标量 (dB)
    """
    a_s = steering_vector(theta_s)                 # [M]
    a_i = steering_vector(theta_i)                 # [M]
    amp_s2 = 10 ** (snr_db / 10.0)                 # 目标功率
    amp_i2 = 10 ** (inr_db / 10.0)                 # 干扰功率
    p_sig = amp_s2 * np.abs(w.conj() @ a_s) ** 2   # 输出目标功率
    p_int = amp_i2 * np.abs(w.conj() @ a_i) ** 2   # 输出干扰功率
    p_noise = np.abs(w.conj() @ w)                 # 输出白噪声功率 (σ²=1) = ||w||²
    return 10 * np.log10(p_sig / (p_int + p_noise + 1e-12))


# ----------------------------------------------------------------------------
# 主流程：生成 4 张配图
# ----------------------------------------------------------------------------
def main():
    theta_s = 20.0     # 目标来向 (deg)
    theta_i = -30.0    # 干扰来向 (deg)，落在 DAS 主瓣旁的高旁瓣处
    snr_db = 10.0
    inr_db = 30.0      # 干扰比目标强 20 dB，逼出 DAS 的软肋
    angles = np.linspace(-90, 90, 721)  # [721] 扫描角网格 (deg)

    # ---- 快拍 + 协方差 + 两种权 ----
    X = simulate_snapshots(theta_s, theta_i, snr_db=snr_db, inr_db=inr_db)  # [M, T]
    R = spatial_covariance(X)                                              # [M, M]
    w_das = das_weights(theta_s)                                          # [M]
    w_mvdr = mvdr_weights(R, theta_s)                                     # [M]

    b_das = beampattern(w_das, angles)     # [721]
    b_mvdr = beampattern(w_mvdr, angles)   # [721]

    sinr_das = output_sinr(w_das, theta_s, theta_i, snr_db, inr_db)
    sinr_mvdr = output_sinr(w_mvdr, theta_s, theta_i, snr_db, inr_db)
    sinr_in = snr_db - inr_db  # 单阵元输入 SINR（干扰远强于目标）

    print(f"[cfg] M={M} mics, d={D*100:.2f} cm (half-wavelength @ {F0:.0f} Hz)")
    print(f"[cfg] target {theta_s:+.0f} deg, interference {theta_i:+.0f} deg, "
          f"SNR={snr_db}dB, INR={inr_db}dB")
    print(f"[SINR] input (single mic) ~ {sinr_in:+.1f} dB")
    print(f"[SINR] DAS  output        = {sinr_das:+.1f} dB")
    print(f"[SINR] MVDR output        = {sinr_mvdr:+.1f} dB")

    # ------------------------------------------------------------------
    # 图 1：阵列几何 + 平面波到达时延示意
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    mic_x = np.arange(M) * D           # [M] 阵元 x 坐标 (m)
    ax[0].scatter(mic_x * 100, np.zeros(M), s=80, c="tab:blue", zorder=3,
                  label="Microphones")
    # 画一束来自 theta_s 的平面波波前
    th = np.deg2rad(theta_s)
    for off in np.linspace(-0.05, 0.15, 5):
        px = np.array([-0.1, 0.3])
        # 波前垂直于传播方向：x*sin + y*cos = const
        py = (off - px * np.sin(th)) / np.cos(th)
        ax[0].plot(px * 100, py * 100, color="tab:orange", alpha=0.5, lw=1)
    ax[0].set_title(f"ULA geometry, plane wave from {theta_s:.0f} deg")
    ax[0].set_xlabel("x (cm)")
    ax[0].set_ylabel("y (cm)")
    ax[0].legend(loc="upper right")
    ax[0].grid(alpha=0.3)
    ax[0].set_ylim(-10, 20)

    # 各阵元相对时延随来向变化
    for tdeg, col in [(0, "tab:green"), (20, "tab:orange"), (60, "tab:red")]:
        tau = np.arange(M) * D * np.sin(np.deg2rad(tdeg)) / C * 1e6  # [M] μs
        ax[1].plot(np.arange(M), tau, "o-", color=col, label=f"{tdeg} deg")
    ax[1].set_title("Per-mic arrival delay vs DOA")
    ax[1].set_xlabel("Mic index m")
    ax[1].set_ylabel("Delay tau_m (us)")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    plt.tight_layout()
    p1 = os.path.join(FIG_DIR, "s6_geometry.png")
    plt.savefig(p1, dpi=130)
    plt.close()

    # ------------------------------------------------------------------
    # 图 2：DAS vs MVDR 波束方向图对比
    # ------------------------------------------------------------------
    plt.figure(figsize=(9, 4.8))
    plt.plot(angles, b_das, label="DAS beampattern", color="tab:blue", lw=1.5)
    plt.plot(angles, b_mvdr, label="MVDR beampattern", color="crimson", lw=1.5)
    plt.axvline(theta_s, ls="--", color="tab:green", alpha=0.8,
                label=f"Target {theta_s:.0f} deg")
    plt.axvline(theta_i, ls="--", color="tab:orange", alpha=0.8,
                label=f"Interference {theta_i:.0f} deg")
    plt.ylim(-60, 3)
    plt.xlabel("Angle theta (deg)")
    plt.ylabel("Response (dB, peak-normalized)")
    plt.title("DAS vs MVDR: MVDR carves a null toward interference")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    p2 = os.path.join(FIG_DIR, "s6_beampattern.png")
    plt.savefig(p2, dpi=130)
    plt.close()

    # ------------------------------------------------------------------
    # 图 3：阵列增益 / 干扰抑制 / 输出 SINR 对比
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    # 左：干扰方向响应（越低越好）
    resp_i_das = 10 * np.log10(np.abs(w_das.conj() @ steering_vector(theta_i)) ** 2 + 1e-12)
    resp_i_mvdr = 10 * np.log10(np.abs(w_mvdr.conj() @ steering_vector(theta_i)) ** 2 + 1e-12)
    ax[0].bar(["DAS", "MVDR"], [resp_i_das, resp_i_mvdr],
              color=["tab:blue", "crimson"])
    ax[0].set_title("Gain toward interference (lower = better)")
    ax[0].set_ylabel("Response @ interference (dB)")
    ax[0].grid(alpha=0.3, axis="y")
    for k, v in enumerate([resp_i_das, resp_i_mvdr]):
        ax[0].text(k, v, f"{v:.1f}", ha="center",
                   va="bottom" if v < 0 else "top")
    # 右：输出 SINR
    ax[1].bar(["Input\n(1 mic)", "DAS", "MVDR"],
              [sinr_in, sinr_das, sinr_mvdr],
              color=["gray", "tab:blue", "crimson"])
    ax[1].set_title("Output SINR comparison")
    ax[1].set_ylabel("SINR (dB)")
    ax[1].grid(alpha=0.3, axis="y")
    for k, v in enumerate([sinr_in, sinr_das, sinr_mvdr]):
        ax[1].text(k, v, f"{v:+.1f}", ha="center",
                   va="bottom" if v >= 0 else "top")
    plt.tight_layout()
    p3 = os.path.join(FIG_DIR, "s6_suppression.png")
    plt.savefig(p3, dpi=130)
    plt.close()

    # ------------------------------------------------------------------
    # 图 4：麦克风增益失配 -> 波束方向图畸变
    # ------------------------------------------------------------------
    # 理想一致：全 1；失配：±30% 随机幅度扰动
    g_ideal = np.ones(M)                                   # [M]
    g_bad = 1.0 + 0.3 * RNG.standard_normal(M)             # [M] 幅度失配
    g_bad = np.clip(g_bad, 0.3, 1.7)

    X_bad = simulate_snapshots(theta_s, theta_i, snr_db=snr_db, inr_db=inr_db,
                               gain_mismatch=g_bad)         # [M, T]
    R_bad = spatial_covariance(X_bad)                       # [M, M]
    # 波束权仍按「理想」导向矢量设计（因为我们不知道失配）-> 失配
    w_mvdr_bad = mvdr_weights(R_bad, theta_s)               # [M]
    b_mvdr_bad = beampattern(w_mvdr_bad, angles)            # [721]

    plt.figure(figsize=(9, 4.8))
    plt.plot(angles, b_mvdr, label="MVDR (calibrated)", color="crimson", lw=1.5)
    plt.plot(angles, b_mvdr_bad, label="MVDR (gain mismatch)",
             color="tab:purple", lw=1.5, alpha=0.85)
    plt.axvline(theta_s, ls="--", color="tab:green", alpha=0.8,
                label=f"Target {theta_s:.0f} deg")
    plt.axvline(theta_i, ls="--", color="tab:orange", alpha=0.8,
                label=f"Interference {theta_i:.0f} deg")
    plt.ylim(-60, 3)
    plt.xlabel("Angle theta (deg)")
    plt.ylabel("Response (dB, peak-normalized)")
    plt.title("Gain mismatch distorts beampattern: null drifts off interference")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    p4 = os.path.join(FIG_DIR, "s6_mismatch.png")
    plt.savefig(p4, dpi=130)
    plt.close()

    # 定量：失配后目标方向是否还是 1、干扰是否还压得住
    g_target_bad = np.abs(w_mvdr_bad.conj() @ (g_bad * steering_vector(theta_s)))
    print(f"[mismatch] MVDR response to true target |w^H (g*a_s)| = "
          f"{g_target_bad:.3f} (should be 1.0 if calibrated)")

    print("[figures] generated:")
    for p in (p1, p2, p3, p4):
        print("  ", p)


if __name__ == "__main__":
    main()
