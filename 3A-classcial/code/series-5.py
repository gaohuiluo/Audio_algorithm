# -*- coding: utf-8 -*-
"""系列 5 配套代码：带 Attack/Release 的下压缩器 + 峰值/响度归一化对比。

运行:
    python code/series-5.py

产出 (figures/ 下, 前缀 s5_):
    s5_static_curve.png     压缩器静态特性曲线 (不同压缩比 R + 软/硬拐点)
    s5_envelope_gain.png    动态信号的输入/输出包络 + 增益轨迹 (dB)
    s5_attack_release.png    不同 Attack/Release 时间常数的增益轨迹对比 (pumping)
    s5_peak_vs_rms.png       峰值归一化 vs RMS(响度)归一化的波形与电平对比

图内文字一律英文, 避免中文字体缺失报错。
"""
import matplotlib
matplotlib.use("Agg")  # 无显示环境下也能出图, 必须在 pyplot 之前设置

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 结果可复现
RNG = np.random.default_rng(2026)

FS = 16000  # 采样率 (Hz), 全系列默认
EPS = 1e-12  # 防 log(0)

# 配图输出目录 (相对项目根: 脚本在 code/ 下, 图存 figures/)
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# dB <-> 线性 幅度换算 (以满量程 1.0 为 0 dBFS)
# ----------------------------------------------------------------------
def to_db(amp: np.ndarray) -> np.ndarray:
    """线性幅度 -> dBFS。amp 可为标量或数组。"""
    return 20.0 * np.log10(np.abs(amp) + EPS)


def from_db(db: np.ndarray) -> np.ndarray:
    """dB -> 线性增益。"""
    return 10.0 ** (db / 20.0)


# ----------------------------------------------------------------------
# 压缩器静态特性曲线 (dB 域), 带软拐点 (soft knee)
# ----------------------------------------------------------------------
def static_curve(x_db: np.ndarray, T: float, R: float, W: float = 0.0) -> np.ndarray:
    """输入电平 -> 输出电平 的静态映射 (全部 dB 域)。

    参数:
        x_db  # [K]   输入电平 (dBFS)
        T     # 标量   阈值 threshold (dBFS)
        R     # 标量   压缩比 ratio (>1)
        W     # 标量   拐点宽度 knee width (dB), 0 表示硬拐点
    返回:
        y_db  # [K]   输出电平 (dBFS)
    """
    x_db = np.asarray(x_db, dtype=np.float64)     # [K]
    over = x_db - T                                # [K] 超过阈值的量
    y = np.empty_like(x_db)                        # [K]

    if W <= 0:
        # 硬拐点: 阈值以下原样, 阈值以上按 1/R 斜率压缩
        below = over <= 0
        y[below] = x_db[below]
        y[~below] = T + over[~below] / R
        return y

    # 软拐点: 在 [T-W/2, T+W/2] 内用二次曲线平滑过渡
    lo = 2 * over < -W          # [K] 远低于阈值
    hi = 2 * over > W           # [K] 远高于阈值
    mid = ~(lo | hi)            # [K] 拐点过渡区
    y[lo] = x_db[lo]
    y[hi] = T + over[hi] / R
    # 过渡区: 二次插值, 斜率从 1 平滑降到 1/R
    y[mid] = x_db[mid] + (1.0 / R - 1.0) * (over[mid] + W / 2.0) ** 2 / (2.0 * W)
    return y


# ----------------------------------------------------------------------
# 带 Attack/Release 的下压缩器 (gain-smoothing 拓扑)
# ----------------------------------------------------------------------
def compressor(x: np.ndarray, T: float, R: float, W: float,
               atk_ms: float, rel_ms: float,
               det_ms: float = 5.0, makeup_db: float = 0.0,
               gate_db: float = -np.inf):
    """对时域信号做下压缩, 返回输出与中间轨迹 (用于画图)。

    参数:
        x         # [Tn]   输入时域信号 (线性, 峰值 ~1)
        T,R,W               静态曲线阈值/压缩比/拐点宽度
        atk_ms    # 标量    Attack 时间常数 (ms)
        rel_ms    # 标量    Release 时间常数 (ms)
        det_ms    # 标量    电平检测器 (包络) 时间常数 (ms)
        makeup_db # 标量    补偿增益 (dB)
        gate_db   # 标量    静音冻结门限: 检测电平低于此值时增益保持不变(不抬噪声)
    返回:
        y         # [Tn]   压缩后信号
        env_db    # [Tn]   检测到的输入电平包络 (dBFS)
        g_db      # [Tn]   平滑后施加的增益 (dB, 含 makeup)
    """
    Tn = x.shape[0]                                 # 标量: 样本数
    # 一阶时间常数 -> 平滑系数 alpha = exp(-1/(tau·f_s))
    a_det = np.exp(-1.0 / (det_ms * 1e-3 * FS))     # 标量: 检测器系数
    a_atk = np.exp(-1.0 / (atk_ms * 1e-3 * FS))     # 标量: attack 系数
    a_rel = np.exp(-1.0 / (rel_ms * 1e-3 * FS))     # 标量: release 系数

    env_db = np.empty(Tn)                           # [Tn] 电平包络 (dB)
    g_db = np.empty(Tn)                             # [Tn] 施加增益 (dB)

    env = EPS                                       # 标量: 线性包络状态
    g = 0.0                                         # 标量: 当前增益 (dB, 减量为负)

    for n in range(Tn):
        # --- 电平检测: 对 |x| 做单极点平滑, 得到瞬时电平 ---
        rect = abs(x[n])                            # 标量: 整流
        env = a_det * env + (1.0 - a_det) * rect    # 标量: 平滑包络
        lvl_db = to_db(env)                         # 标量: 当前电平 (dBFS)
        env_db[n] = lvl_db

        # --- 增益计算: 静态曲线给出目标增益(减量) ---
        y_db = static_curve(np.array([lvl_db]), T, R, W)[0]  # 标量
        g_target = y_db - lvl_db                     # 标量: 目标增益 (<=0)

        # --- 静音冻结: 电平过低则不更新增益, 避免把噪声抬起来 ---
        if lvl_db < gate_db:
            g_target = g

        # --- Attack/Release 平滑: 增益变小(更多压制)用 attack, 变大(松开)用 release ---
        coef = a_atk if g_target < g else a_rel      # 标量
        g = coef * g + (1.0 - coef) * g_target       # 标量: g(t)=α·g(t-1)+(1-α)·g_target
        g_db[n] = g + makeup_db

    y = x * from_db(g_db)                             # [Tn] 施加线性增益
    return y, env_db, g_db


# ----------------------------------------------------------------------
# 构造一段动态范围很大的测试信号: 忽远忽近的说话人
# ----------------------------------------------------------------------
def make_dynamic_signal():
    """合成一段电平起伏很大的类语音信号 (远->近->突发瞬态->远)。

    返回:
        x    # [Tn]  时域信号 (峰值约 0.9)
        t    # [Tn]  时间轴 (秒)
    """
    dur = 4.0                                        # 标量: 时长 (秒)
    Tn = int(dur * FS)                               # 标量: 样本数
    t = np.arange(Tn) / FS                           # [Tn] 时间轴

    # 载波: 两个共振峰的和, 模拟浊音基频+谐振
    carrier = (np.sin(2 * np.pi * 180 * t)
               + 0.5 * np.sin(2 * np.pi * 550 * t)
               + 0.3 * np.sin(2 * np.pi * 1200 * t))  # [Tn]
    # 叠一点气声噪声
    carrier = carrier + 0.15 * RNG.standard_normal(Tn)  # [Tn]
    carrier /= np.max(np.abs(carrier))                # [Tn] 归一化到 ~1

    # 电平包络: 分段模拟 远(小) -> 近(大) -> 远(小) + 一个突发大声
    env = np.full(Tn, 0.08)                           # [Tn] 默认远场小音量
    seg = lambda a, b: slice(int(a * FS), int(b * FS))
    env[seg(0.8, 1.6)] = 0.9                           # 走近: 大声
    env[seg(2.0, 2.05)] = 1.0                          # 突发瞬态 (拍桌子)
    env[seg(2.4, 3.2)] = 0.5                           # 中等
    # 平滑包络的硬边沿, 避免click
    k = np.ones(int(0.01 * FS)) / int(0.01 * FS)       # [win] 10ms 平滑核
    env = np.convolve(env, k, mode="same")             # [Tn]

    x = carrier * env                                  # [Tn] 最终信号
    x = 0.9 * x / np.max(np.abs(x))                    # [Tn] 峰值归一到 0.9
    return x, t


# ----------------------------------------------------------------------
# 归一化: 峰值 vs RMS(响度代理)
# ----------------------------------------------------------------------
def peak_normalize(x: np.ndarray, target_db: float = -1.0) -> np.ndarray:
    """峰值归一化到 target_db (dBFS)。"""
    peak = np.max(np.abs(x)) + EPS                     # 标量
    gain = from_db(target_db) / peak                   # 标量
    return x * gain                                    # [Tn]


def rms_normalize(x: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """RMS(响度)归一化到 target_db (dBFS RMS)。"""
    rms = np.sqrt(np.mean(x ** 2)) + EPS               # 标量
    gain = from_db(target_db) / rms                    # 标量
    return x * gain                                    # [Tn]


def rms_db(x: np.ndarray) -> float:
    """整段 RMS 电平 (dBFS)。"""
    return float(to_db(np.sqrt(np.mean(x ** 2))))


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main():
    # === 图1: 静态特性曲线 (不同 R + 硬/软拐点) ===
    x_db = np.linspace(-60, 0, 400)                    # [400] 输入电平扫描
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    ax.plot(x_db, x_db, "k--", lw=1.0, label="1:1 (bypass)")
    T = -25.0
    for R in [2, 4, 8]:
        ax.plot(x_db, static_curve(x_db, T=T, R=R, W=0.0),
                label=f"hard knee R={R}:1")
    ax.plot(x_db, static_curve(x_db, T=T, R=4, W=12.0),
            "r", lw=2.0, alpha=0.8, label="soft knee R=4:1 (W=12dB)")
    ax.axvline(T, color="gray", ls=":", lw=1.0)
    ax.text(T + 0.5, -58, f"threshold T={T:.0f} dB", fontsize=8, color="gray")
    ax.set_title("Compressor Static Characteristic (dB domain)")
    ax.set_xlabel("input level (dBFS)")
    ax.set_ylabel("output level (dBFS)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "s5_static_curve.png", dpi=130)
    plt.close(fig)

    # 构造动态信号
    x, t = make_dynamic_signal()                       # [Tn], [Tn]
    print(f"[info] input: peak={to_db(np.max(np.abs(x))):.1f} dBFS, "
          f"RMS={rms_db(x):.1f} dBFS")

    # === 图2: 输入/输出包络 + 增益轨迹 ===
    y, env_db, g_db = compressor(x, T=-25, R=4, W=6,
                                 atk_ms=5, rel_ms=120,
                                 makeup_db=6, gate_db=-55)
    print(f"[info] output: peak={to_db(np.max(np.abs(y))):.1f} dBFS, "
          f"RMS={rms_db(y):.1f} dBFS")

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 7.2), sharex=True)
    axes[0].plot(t, x, lw=0.4, color="#2b6cb0")
    axes[0].set_ylabel("input amp")
    axes[0].set_title("Input signal (near/far speaker + transient)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, y, lw=0.4, color="#c98a00")
    axes[1].set_ylabel("output amp")
    axes[1].set_title("Compressed + makeup gain output")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, env_db, label="detected level (dBFS)", color="#2b6cb0")
    axes[2].axhline(-25, color="gray", ls=":", label="threshold")
    ax2b = axes[2].twinx()
    ax2b.plot(t, g_db, label="applied gain (dB)", color="#e53e3e", lw=1.2)
    axes[2].set_ylabel("level (dBFS)")
    ax2b.set_ylabel("gain (dB)")
    axes[2].set_xlabel("time (s)")
    axes[2].set_title("Level envelope (blue) vs applied gain trajectory (red)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=8, loc="lower left")
    ax2b.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "s5_envelope_gain.png", dpi=130)
    plt.close(fig)

    # === 图3: 不同 Attack/Release 的增益轨迹对比 ===
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    configs = [
        ("fast atk / slow rel (5/150ms)", 5, 150, "#2b6cb0"),
        ("slow atk / slow rel (60/150ms)", 60, 150, "#38a169"),
        ("fast atk / fast rel (5/15ms)  -> pumping", 5, 15, "#e53e3e"),
    ]
    for label, atk, rel, c in configs:
        _, _, g = compressor(x, T=-25, R=4, W=6, atk_ms=atk, rel_ms=rel,
                             makeup_db=0, gate_db=-55)
        ax.plot(t, g, label=label, color=c, lw=1.1)
    ax.set_title("Gain Trajectory under Different Attack/Release")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("gain reduction (dB)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "s5_attack_release.png", dpi=130)
    plt.close(fig)

    # === 图4: 峰值归一化 vs RMS(响度)归一化 ===
    xp = peak_normalize(x, target_db=-1.0)             # [Tn]
    xr = rms_normalize(x, target_db=-20.0)             # [Tn]
    print(f"[info] peak-norm : peak={to_db(np.max(np.abs(xp))):.1f} dBFS, "
          f"RMS={rms_db(xp):.1f} dBFS")
    print(f"[info] rms-norm  : peak={to_db(np.max(np.abs(xr))):.1f} dBFS, "
          f"RMS={rms_db(xr):.1f} dBFS")

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 5.6), sharex=True)
    axes[0].plot(t, xp, lw=0.4, color="#2b6cb0",
                 label=f"peak-norm (peak={to_db(np.max(np.abs(xp))):.1f}, "
                       f"RMS={rms_db(xp):.1f} dB)")
    axes[0].axhline(from_db(-1.0), color="gray", ls=":", lw=0.8)
    axes[0].axhline(-from_db(-1.0), color="gray", ls=":", lw=0.8)
    axes[0].set_ylabel("amp")
    axes[0].set_title("Peak normalization: peaks aligned, loudness still low")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, xr, lw=0.4, color="#c98a00",
                 label=f"RMS-norm (peak={to_db(np.max(np.abs(xr))):.1f}, "
                       f"RMS={rms_db(xr):.1f} dB)")
    axes[1].set_ylabel("amp")
    axes[1].set_xlabel("time (s)")
    axes[1].set_title("RMS (loudness) normalization: matched loudness (may clip)")
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "s5_peak_vs_rms.png", dpi=130)
    plt.close(fig)

    print("[done] figures saved to", FIG_DIR)


if __name__ == "__main__":
    main()
