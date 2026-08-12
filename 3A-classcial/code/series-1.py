# -*- coding: utf-8 -*-
"""系列 1 配套代码：LMS / NLMS 自适应滤波器辨识一个已知 FIR 系统。

运行:
    python code/series-1.py

产出 (figures/ 下, 前缀 s1_):
    s1_system.png          未知系统的真实冲激响应
    s1_learning_mu.png     LMS 不同步长 μ 的学习曲线 (误差能量随迭代下降)
    s1_lms_vs_nlms.png     LMS vs NLMS 收敛速度对比
    s1_weight_track.png    NLMS 抽头系数收敛到真实系统的过程
    s1_taps_short.png      抽头长度不足 (欠定) 导致的稳态误差地板

图内文字一律英文, 避免中文字体缺失报错。
"""
import matplotlib
matplotlib.use("Agg")  # 无显示环境下也能出图, 必须在 pyplot 之前设置

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 结果可复现
RNG = np.random.default_rng(2026)

# 配图输出目录 (相对项目根: 脚本在 code/ 下, 图存 figures/)
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# 待辨识的"未知系统": 一个已知的 FIR 冲激响应 (我们假装不知道它, 让滤波器学)
# ----------------------------------------------------------------------
def make_unknown_system(L_true: int = 16) -> np.ndarray:
    """构造一个衰减振荡的 FIR 冲激响应, 模拟房间回声路径的粗略形状。

    返回:
        h_true  # [L_true]  真实抽头系数
    """
    n = np.arange(L_true)                      # [L_true]
    h = np.cos(0.6 * n) * np.exp(-0.25 * n)    # [L_true] 衰减余弦
    h[0] = 1.0                                 # 直达声主抽头
    return h.astype(np.float64)                # [L_true]


def run_filter(x: np.ndarray, h_true: np.ndarray, L: int, mu: float,
               mode: str = "lms", eps: float = 1e-6, noise_std: float = 1e-3):
    """用 LMS 或 NLMS 在线辨识未知系统 h_true。

    参数:
        x         # [T]        输入信号 (激励)
        h_true    # [L_true]   未知系统真实抽头
        L         # 标量        自适应滤波器抽头长度 (可与 L_true 不同)
        mu        # 标量        步长
        mode      # "lms" | "nlms"
        eps       # NLMS 分母正则项, 防止除零
        noise_std # 期望信号叠加的观测噪声标准差

    返回:
        e         # [T]     误差信号 e[n] = d[n] - y_hat[n]
        w_hist    # [T, L]  每步的抽头系数 (用于画收敛轨迹)
    """
    T = x.shape[0]                              # 标量: 样本数
    L_true = h_true.shape[0]                    # 标量

    # 真实系统输出 d[n] = h_true^T x_true[n] + 观测噪声
    d_clean = np.convolve(x, h_true)[:T]        # [T] 线性卷积后截断到 T
    d = d_clean + noise_std * RNG.standard_normal(T)  # [T] 期望信号

    w = np.zeros(L)                             # [L] 抽头初始化为 0
    e = np.zeros(T)                             # [T] 误差记录
    w_hist = np.zeros((T, L))                   # [T, L] 抽头轨迹

    for n in range(T):
        # 取当前输入向量 x[n] = [x[n], x[n-1], ..., x[n-L+1]]^T
        if n >= L - 1:
            x_vec = x[n - L + 1: n + 1][::-1]   # [L] 反转成 [x[n], x[n-1], ...]
        else:
            # 开头不足 L 个样本, 左侧补零
            x_vec = np.zeros(L)                 # [L]
            seg = x[: n + 1][::-1]              # [n+1]
            x_vec[: n + 1] = seg

        y_hat = w @ x_vec                       # 标量: 滤波器输出 ŷ[n] = w^T x[n]
        e[n] = d[n] - y_hat                     # 标量: 误差 e[n]

        if mode == "lms":
            # LMS: w(n+1) = w(n) + μ · e[n] · x[n]
            w = w + mu * e[n] * x_vec           # [L]
        elif mode == "nlms":
            # NLMS: 用输入瞬时功率归一化步长 μ / (ε + ||x||²)
            norm = eps + x_vec @ x_vec          # 标量: ε + ||x||²
            w = w + (mu / norm) * e[n] * x_vec  # [L]
        else:
            raise ValueError(f"unknown mode: {mode}")

        w_hist[n] = w                           # [L] 记录本步抽头

    return e, w_hist


def smooth_energy(e: np.ndarray, win: int = 64) -> np.ndarray:
    """把误差平方做滑动平均, 得到平滑的学习曲线 (dB)。

    参数:
        e    # [T] 误差
    返回:
        db   # [T] 10*log10 的平滑误差能量
    """
    p = e ** 2                                  # [T] 瞬时误差能量
    kernel = np.ones(win) / win                 # [win]
    p_smooth = np.convolve(p, kernel, mode="same")  # [T]
    return 10.0 * np.log10(p_smooth + 1e-12)    # [T] dB


# ----------------------------------------------------------------------
# 主流程: 生成激励, 分别跑各实验并出图
# ----------------------------------------------------------------------
def main():
    T = 4000                                    # 标量: 迭代样本数
    L_true = 16                                 # 标量: 真实系统阶数
    h_true = make_unknown_system(L_true)        # [L_true]

    # 激励用白噪声: 各抽头被均匀"照亮", 便于辨识
    x = RNG.standard_normal(T)                  # [T]
    power = float(np.mean(x ** 2))              # 标量: 输入功率 ≈ λ_max(白噪声)

    # 白噪声下相关矩阵 R ≈ power·I, 故 λ_max ≈ power, 稳定上界 μ < 2/λ_max
    mu_max = 2.0 / power                         # 标量: LMS 稳定步长上界(近似)
    print(f"[info] input power ≈ {power:.4f}, LMS stable bound mu < 2/lambda_max ≈ {mu_max:.4f}")

    # === 图1: 未知系统真实冲激响应 ===
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.stem(np.arange(L_true), h_true, basefmt=" ")
    ax.set_title("Unknown System: True FIR Impulse Response")
    ax.set_xlabel("tap index")
    ax.set_ylabel("amplitude")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "s1_system.png", dpi=130)
    plt.close(fig)

    # === 图2: LMS 不同步长 μ 的学习曲线 ===
    # 白噪声下均方稳定的实用上界更紧: μ < 2/tr(R) ≈ 2/(L·power)
    mu_mse_bound = 2.0 / (L_true * power)        # 标量: 均方稳定实用上界
    print(f"[info] practical MSE-stable bound mu < 2/tr(R) ≈ {mu_mse_bound:.4f}")
    # 递增步长: 最后一个刻意逼近上界, 展示"更快但稳态误差更高"(misadjustment)
    mus = [0.005, 0.02, 0.06, 0.11]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for mu in mus:
        e, _ = run_filter(x, h_true, L=L_true, mu=mu, mode="lms")
        ax.plot(smooth_energy(e), label=f"mu={mu:g}  (={mu/mu_mse_bound:.2f}*2/tr(R))")
    ax.set_title("LMS Learning Curves for Different Step Sizes")
    ax.set_xlabel("iteration n")
    ax.set_ylabel("smoothed error energy (dB)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "s1_learning_mu.png", dpi=130)
    plt.close(fig)

    # === 图3: LMS vs NLMS 收敛速度 ===
    e_lms, _ = run_filter(x, h_true, L=L_true, mu=0.02, mode="lms")
    e_nlms, _ = run_filter(x, h_true, L=L_true, mu=0.5, mode="nlms")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(smooth_energy(e_lms), label="LMS  (mu=0.02)")
    ax.plot(smooth_energy(e_nlms), label="NLMS (mu=0.5)")
    ax.set_title("LMS vs NLMS Convergence (white-noise excitation)")
    ax.set_xlabel("iteration n")
    ax.set_ylabel("smoothed error energy (dB)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "s1_lms_vs_nlms.png", dpi=130)
    plt.close(fig)

    # === 图4: NLMS 抽头收敛到真实系统 (画前 4 个抽头轨迹) ===
    _, w_hist = run_filter(x, h_true, L=L_true, mu=0.5, mode="nlms")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for k in range(4):
        ax.plot(w_hist[:, k], label=f"w[{k}] -> {h_true[k]:.2f}")
        ax.axhline(h_true[k], color="gray", ls="--", lw=0.7, alpha=0.5)
    ax.set_title("NLMS Weight Trajectories Converging to True Taps")
    ax.set_xlabel("iteration n")
    ax.set_ylabel("tap value")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "s1_weight_track.png", dpi=130)
    plt.close(fig)

    # === 图5: 抽头长度不足 vs 充足 -> 稳态误差地板 ===
    # 真实系统 16 阶; 滤波器只给 6 阶 (欠定), 无论如何学不干净
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for L in [6, 10, 16, 24]:
        e, _ = run_filter(x, h_true, L=L, mu=0.5, mode="nlms")
        tag = "under-modeled" if L < L_true else ("matched" if L == L_true else "over-sized")
        ax.plot(smooth_energy(e), label=f"L={L} ({tag})")
    ax.set_title("Filter Length vs Steady-State Error Floor (NLMS)")
    ax.set_xlabel("iteration n")
    ax.set_ylabel("smoothed error energy (dB)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "s1_taps_short.png", dpi=130)
    plt.close(fig)

    print("[done] figures saved to", FIG_DIR)


if __name__ == "__main__":
    main()
