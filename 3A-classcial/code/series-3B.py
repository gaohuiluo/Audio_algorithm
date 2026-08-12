# -*- coding: utf-8 -*-
"""系列 3B 配套代码：Musical Noise 与噪声估计。

对比 "纯谱减" vs "判决引导(DD)+维纳" 的降噪结果，展示音乐噪声差异；
并实现最小值统计 (minimum statistics) 的在线噪声功率估计。

运行:
    python code/series-3B.py

产出 (figures/ 下, 前缀 s3b_):
    s3b_gain_vs_snr.png    增益曲线：谱减 vs 维纳，随后验 SNR 的形状差异
    s3b_gain_track.png     纯噪声频点上的增益时序：谱减剧烈抖动 vs DD 平滑
    s3b_spectrograms.png   语谱图四联：干净 / 带噪 / 纯谱减(孤立亮点) / DD维纳(平滑)
    s3b_minstat.png        最小值统计对 λ_d 的跟踪曲线 (估计 vs 真实)

图内文字一律英文, 避免中文字体缺失报错。
"""
import matplotlib
matplotlib.use("Agg")  # 无显示环境也能出图, 必须在 pyplot 之前设置

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from pathlib import Path

# 结果可复现
RNG = np.random.default_rng(2026)

# 配图输出目录 (脚本在 code/ 下, 图存 figures/)
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 全局 STFT 参数 (STYLE.md: f_s 默认 16000)
FS = 16000        # 采样率 Hz
N = 512           # 帧长 (窗长) 样本
H = 128           # 帧移 hop, 75% overlap 满足 Hann 的 COLA 可完美重构
EPS = 1e-12       # 防除零小量


# ----------------------------------------------------------------------
# 1. 造信号: 合成一段 "语音样" 的干净信号 + 加性噪声
#    不依赖外部音频文件, 用调频扫频 + 谐波模拟浊音的时变谱结构。
# ----------------------------------------------------------------------
def make_clean_speech(dur: float = 2.5) -> np.ndarray:
    """合成一段带时变谐波与静音间隙的 "类语音" 信号。

    返回:
        s  # [T]  干净信号, 幅度归一化到约 [-1, 1]
    """
    T = int(dur * FS)                              # 标量: 总样本数
    t = np.arange(T) / FS                          # [T] 时间轴 (秒)

    # 基频在 120~180 Hz 之间缓慢起伏 (模拟语调)
    f0 = 150.0 + 30.0 * np.sin(2 * np.pi * 0.7 * t)  # [T]
    phase = 2 * np.pi * np.cumsum(f0) / FS          # [T] 瞬时相位积分
    s = np.zeros(T)                                 # [T]
    # 叠加前 6 次谐波, 高次谐波能量递减 (类似浊音频谱包络)
    for k in range(1, 7):
        s += (1.0 / k) * np.sin(k * phase)          # [T]

    # 用几个 "音节" 门控: 制造语音活动段与静音段 (给噪声估计留纯噪声帧)
    env = np.zeros(T)                               # [T] 幅度包络
    syllables = [(0.15, 0.55), (0.75, 1.15), (1.35, 1.75), (1.95, 2.35)]
    for (a, b) in syllables:
        ia, ib = int(a * FS), int(b * FS)
        win = np.hanning(ib - ia)                   # [ib-ia] 平滑起落, 避免爆音
        env[ia:ib] = win
    s = s * env                                     # [T] 加窗门控
    s = s / (np.max(np.abs(s)) + EPS)               # [T] 归一化
    return s.astype(np.float64)                     # [T]


def add_noise(s: np.ndarray, snr_db: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """按目标 SNR 叠加白噪声 (随机起伏是音乐噪声的温床)。

    参数:
        s        # [T]  干净信号
        snr_db   # 标量  目标信噪比 (dB)
    返回:
        x        # [T]  带噪信号
        noise    # [T]  实际叠加的噪声 (留作参考真值)
    """
    T = s.shape[0]                                  # 标量
    noise = RNG.standard_normal(T)                  # [T] 高斯白噪声
    # 只按 "有话段" 的功率算 SNR, 避免静音段拉低平均功率
    p_s = np.mean(s[np.abs(s) > 1e-3] ** 2) + EPS   # 标量: 语音段平均功率
    p_n = np.mean(noise ** 2) + EPS                 # 标量: 噪声功率
    scale = np.sqrt(p_s / (p_n * 10 ** (snr_db / 10.0)))  # 标量: 噪声缩放
    noise = noise * scale                           # [T]
    x = s + noise                                   # [T] 带噪
    return x.astype(np.float64), noise.astype(np.float64)


# ----------------------------------------------------------------------
# 2. STFT / iSTFT 封装 (scipy)
# ----------------------------------------------------------------------
def stft(x: np.ndarray):
    """短时傅里叶变换。

    返回:
        f   # [F]      频点 (Hz)
        tt  # [Tf]     帧时间 (s)
        X   # [F, Tf]  复数谱 X(t,f) (注意 scipy 返回 [freq, frame])
    """
    f, tt, X = signal.stft(x, fs=FS, window="hann",
                           nperseg=N, noverlap=N - H, boundary="zeros", padded=True)
    return f, tt, X                                 # X: [F, Tf]


def istft(X: np.ndarray) -> np.ndarray:
    """逆变换回时域 (沿用处理后的复数谱)。

    参数:
        X   # [F, Tf]  复数谱
    返回:
        y   # [T]      重构时域信号
    """
    _, y = signal.istft(X, fs=FS, window="hann",
                        nperseg=N, noverlap=N - H, boundary=True)
    return y                                        # [T]


# ----------------------------------------------------------------------
# 3. 噪声估计: 最小值统计 (minimum statistics)
#    无 VAD 时, 对每个频点在滑动时间窗内取功率最小值近似 λ_d。
#    直觉: 即便有话段, 谱功率也会短暂 "落回" 噪声底; 取窗内最小值 ≈ 噪声底。
# ----------------------------------------------------------------------
def minimum_statistics(P: np.ndarray, win_frames: int = 40,
                       bias: float = 1.5, alpha_p: float = 0.85) -> np.ndarray:
    """最小值统计估计噪声功率谱 λ_d。

    参数:
        P           # [F, Tf]  带噪功率谱 |X(t,f)|²
        win_frames  # 标量      回看窗长 (帧数); 越大越稳但延迟越大
        bias        # 标量      最小值偏低的补偿因子 (min 是有偏估计, 乘回来)
        alpha_p     # 标量      对功率谱先做一阶平滑, 削掉毛刺再取 min
    返回:
        lam_d       # [F, Tf]  噪声功率谱估计 λ_d(t,f)
    """
    F, Tf = P.shape                                 # 标量
    # 3.1 先对功率谱做时间平滑, 否则单帧毛刺会把 min 拉得过低
    Ps = np.zeros_like(P)                            # [F, Tf]
    Ps[:, 0] = P[:, 0]
    for t in range(1, Tf):
        Ps[:, t] = alpha_p * Ps[:, t - 1] + (1 - alpha_p) * P[:, t]

    # 3.2 滑动窗内取每个频点的功率最小值
    lam_d = np.zeros_like(P)                          # [F, Tf]
    for t in range(Tf):
        a = max(0, t - win_frames + 1)                # 窗左边界
        lam_d[:, t] = np.min(Ps[:, a:t + 1], axis=1)  # [F] 窗内最小
    lam_d *= bias                                     # 补偿最小值的负偏差
    return lam_d                                      # [F, Tf]


# ----------------------------------------------------------------------
# 4. 纯谱减法 (每个频点独立硬减 —— 音乐噪声的源头)
# ----------------------------------------------------------------------
def spectral_subtraction(X: np.ndarray, lam_d: np.ndarray,
                         over_sub: float = 2.0, floor: float = 0.0) -> np.ndarray:
    """功率谱减法。

    参数:
        X         # [F, Tf]  带噪复数谱
        lam_d     # [F, Tf]  噪声功率估计
        over_sub  # 标量      过减因子 (>1 减得更狠)
        floor     # 标量      谱下限 β (0 表示硬减到 0, 音乐噪声最重)
    返回:
        Y         # [F, Tf]  降噪后复数谱 (沿用带噪相位)
    """
    P = np.abs(X) ** 2                                 # [F, Tf] 带噪功率
    # 减掉 over_sub 倍噪声功率, 不足下限则用 floor·|X|² 兜底
    P_hat = P - over_sub * lam_d                       # [F, Tf]
    P_hat = np.maximum(P_hat, floor * P)               # [F, Tf] 谱下限
    gain = np.sqrt(P_hat / (P + EPS))                  # [F, Tf] 幅度增益 M(t,f)
    Y = gain * X                                       # [F, Tf] 沿用带噪相位
    return Y


# ----------------------------------------------------------------------
# 5. 判决引导 (DD) 估计先验 SNR + 维纳增益
#    DD: ξ_prior(t) = α·|Ŝ(t-1)|²/λ_d + (1-α)·max(γ_post-1, 0)
#    维纳增益: M = ξ / (1 + ξ)
# ----------------------------------------------------------------------
def dd_wiener(X: np.ndarray, lam_d: np.ndarray,
              alpha: float = 0.98, g_min: float = 0.08) -> tuple[np.ndarray, np.ndarray]:
    """判决引导(Decision-Directed)先验 SNR + 维纳滤波。

    参数:
        X       # [F, Tf]  带噪复数谱
        lam_d   # [F, Tf]  噪声功率估计
        alpha   # 标量      DD 平滑系数 (越接近 1 越平滑, 越抑制音乐噪声)
        g_min   # 标量      增益下限 (软谱下限, 保留一点底噪听感更自然)
    返回:
        Y       # [F, Tf]  降噪后复数谱
        xi      # [F, Tf]  先验 SNR ξ_prior (用于观察平滑效果)
    """
    F, Tf = X.shape                                    # 标量
    P = np.abs(X) ** 2                                 # [F, Tf] 带噪功率
    gamma = P / (lam_d + EPS)                          # [F, Tf] 后验 SNR γ_post

    xi = np.zeros_like(P)                              # [F, Tf] 先验 SNR
    gain = np.zeros_like(P)                            # [F, Tf] 维纳增益 M(t,f)
    S_prev = np.zeros(F)                               # [F] 上一帧幅度平方 |Ŝ(t-1)|²

    for t in range(Tf):
        gamma_t = gamma[:, t]                          # [F]
        # DD 两项: 前项=上一帧估计的先验SNR; 后项=当前帧的最大似然瞬时估计
        xi_t = alpha * (S_prev / (lam_d[:, t] + EPS)) \
             + (1 - alpha) * np.maximum(gamma_t - 1.0, 0.0)  # [F]
        xi_t = np.maximum(xi_t, EPS)                   # [F] 防负
        g_t = xi_t / (1.0 + xi_t)                      # [F] 维纳增益
        g_t = np.maximum(g_t, g_min)                   # [F] 软下限
        gain[:, t] = g_t
        xi[:, t] = xi_t
        # 更新: 本帧估计的干净幅度平方, 供下一帧 DD 前项使用
        S_prev = (g_t ** 2) * P[:, t]                  # [F] |Ŝ(t)|²

    Y = gain * X                                       # [F, Tf] 沿用带噪相位
    return Y, xi


# ----------------------------------------------------------------------
# 6. 一个粗糙的 "音乐噪声" 量化指标: 静音段残余谱的时频起伏程度
#    孤立亮点越多、帧间越不连续, 该值越大。
# ----------------------------------------------------------------------
def musical_noise_index(Y: np.ndarray, active_mask: np.ndarray) -> float:
    """用静音段的对数功率谱的 "帧间+频间" 差分能量近似音乐噪声强度。

    参数:
        Y            # [F, Tf]  降噪后复数谱
        active_mask  # [Tf]     True=有话帧, False=静音帧
    返回:
        idx          # 标量      越大表示残余越 "斑驳" (音乐噪声越重)
    """
    logP = np.log(np.abs(Y) ** 2 + EPS)                # [F, Tf]
    sil = logP[:, ~active_mask]                        # [F, Tf_sil] 只看静音段
    if sil.shape[1] < 3:
        return float("nan")
    d_time = np.diff(sil, axis=1)                      # [F, Tf_sil-1] 帧间差分
    d_freq = np.diff(sil, axis=0)                      # [F-1, Tf_sil] 频间差分
    return float(np.var(d_time) + np.var(d_freq))      # 标量


# ----------------------------------------------------------------------
# 7. 绘图
# ----------------------------------------------------------------------
def plot_gain_vs_snr():
    """静态对比: 谱减增益 vs 维纳增益 随后验 SNR γ_post 的形状。"""
    gamma = np.linspace(0.1, 20, 400)                  # [400] 后验 SNR (线性)
    # 谱减 (假设 ξ≈γ-1): 增益 = sqrt(max(1 - 1/γ, 0))
    g_ss = np.sqrt(np.maximum(1.0 - 1.0 / gamma, 0.0)) # [400]
    # 维纳 (先验 SNR ξ ≈ γ-1): 增益 = ξ/(1+ξ)
    xi = np.maximum(gamma - 1.0, EPS)                  # [400]
    g_wiener = xi / (1.0 + xi)                         # [400]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(10 * np.log10(gamma), g_ss, label="Spectral Subtraction", lw=2)
    ax.plot(10 * np.log10(gamma), g_wiener, label="Wiener (DD prior)", lw=2)
    ax.axvline(0, color="gray", ls="--", lw=1, alpha=0.7)
    ax.set_xlabel("a posteriori SNR  gamma_post (dB)")
    ax.set_ylabel("gain  M(t,f)")
    ax.set_title("Gain curves: hard subtraction vs smooth Wiener")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "s3b_gain_vs_snr.png", dpi=130)
    plt.close(fig)


def plot_gain_track(g_ss_track, g_dd_track, fbin_hz):
    """纯噪声频点上的增益时序: 谱减剧烈抖动 vs DD 平滑。"""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(g_ss_track, label="Spectral Subtraction gain", lw=1.2, alpha=0.9)
    ax.plot(g_dd_track, label="DD-Wiener gain", lw=1.8)
    ax.set_xlabel("frame index t")
    ax.set_ylabel("gain  M(t,f)")
    ax.set_title(f"Gain over time at a noise-dominated bin (~{fbin_hz:.0f} Hz)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "s3b_gain_track.png", dpi=130)
    plt.close(fig)


def plot_spectrograms(specs, titles, f):
    """四联语谱图: 对比孤立亮点 (音乐噪声) 与平滑残余。"""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    vmax = np.max([20 * np.log10(np.abs(s) + EPS) for s in specs])
    for ax, Y, title in zip(axes.ravel(), specs, titles):
        db = 20 * np.log10(np.abs(Y) + EPS)            # [F, Tf]
        im = ax.pcolormesh(np.arange(db.shape[1]), f / 1000.0, db,
                           vmin=vmax - 70, vmax=vmax, cmap="magma", shading="auto")
        ax.set_title(title)
        ax.set_ylim(0, 4)                              # 只看 0~4 kHz, 语音主要能量区
    for ax in axes[-1]:
        ax.set_xlabel("frame index t")
    for ax in axes[:, 0]:
        ax.set_ylabel("frequency (kHz)")
    fig.colorbar(im, ax=axes, shrink=0.8, label="magnitude (dB)")
    fig.suptitle("Spectrograms: musical noise (isolated specks) vs smooth residual")
    fig.savefig(FIG_DIR / "s3b_spectrograms.png", dpi=130)
    plt.close(fig)


def plot_minstat(P, lam_est, lam_true, f, fbin):
    """最小值统计对 λ_d 的跟踪曲线 (单频点)。"""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(P[fbin], label="noisy power |X|^2", lw=1, alpha=0.55)
    ax.plot(lam_true[fbin], label="true noise power (reference)", lw=1.6, ls="--")
    ax.plot(lam_est[fbin], label="min-statistics estimate lambda_d", lw=1.8)
    ax.set_xlabel("frame index t")
    ax.set_ylabel("power")
    ax.set_title(f"Minimum-statistics noise tracking at ~{f[fbin]:.0f} Hz")
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "s3b_minstat.png", dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------
# 8. 主流程
# ----------------------------------------------------------------------
def main():
    # --- 造信号 ---
    s = make_clean_speech()                            # [T] 干净
    x, noise = add_noise(s, snr_db=5.0)                # [T],[T] 带噪+噪声真值

    # --- STFT ---
    f, tt, X = stft(x)                                 # X: [F, Tf]
    _, _, S = stft(s)                                  # 干净参考谱 [F, Tf]
    _, _, Dn = stft(noise)                             # 噪声参考谱 [F, Tf]
    P = np.abs(X) ** 2                                 # [F, Tf] 带噪功率
    lam_true = np.abs(Dn) ** 2                         # [F, Tf] 真实噪声功率 (参考)
    F, Tf = X.shape

    # --- 噪声估计: 最小值统计 ---
    lam_est = minimum_statistics(P, win_frames=40)     # [F, Tf]

    # --- 两种降噪 (都用同一份最小值统计的噪声估计, 公平对比) ---
    Y_ss = spectral_subtraction(X, lam_est, over_sub=2.0, floor=0.0)   # 硬减
    Y_dd, xi = dd_wiener(X, lam_est, alpha=0.98, g_min=0.08)           # DD+维纳

    # --- 增益时序: 选一个语音能量弱、噪声主导的高频点 ---
    fbin_track = int(np.argmin(np.abs(f - 6000)))      # ~6 kHz 频点索引
    g_ss = np.sqrt(np.abs(Y_ss) ** 2 / (P + EPS))      # [F, Tf]
    g_dd = np.sqrt(np.abs(Y_dd) ** 2 / (P + EPS))      # [F, Tf]

    # --- 有话/静音帧标记 (用干净谱能量阈值, 给音乐噪声指标用) ---
    frame_energy = np.mean(np.abs(S) ** 2, axis=0)     # [Tf]
    thr = 0.05 * np.max(frame_energy)                  # 标量
    active = frame_energy > thr                        # [Tf] True=有话

    # --- 量化音乐噪声 ---
    mn_ss = musical_noise_index(Y_ss, active)
    mn_dd = musical_noise_index(Y_dd, active)

    # --- iSTFT 回时域 (验证可重构, 也可落地保存) ---
    y_ss = istft(Y_ss)                                 # [T]
    y_dd = istft(Y_dd)                                 # [T]

    # --- 绘图 ---
    plot_gain_vs_snr()
    plot_gain_track(g_ss[fbin_track], g_dd[fbin_track], f[fbin_track])
    plot_spectrograms(
        [S, X, Y_ss, Y_dd],
        ["Clean", "Noisy (SNR=5dB)",
         f"Spectral Subtraction (MN idx={mn_ss:.2f})",
         f"DD + Wiener (MN idx={mn_dd:.2f})"],
        f,
    )
    fbin_ms = int(np.argmin(np.abs(f - 3000)))          # ~3 kHz 观察噪声跟踪
    plot_minstat(P, lam_est, lam_true, f, fbin_ms)

    print("=== 系列 3B 运行报告 ===")
    print(f"信号: T={s.shape[0]} 样本 ({s.shape[0]/FS:.2f}s), STFT 谱 shape={X.shape}")
    print(f"音乐噪声指标 (静音段谱起伏, 越小越好):")
    print(f"    纯谱减        : {mn_ss:.3f}")
    print(f"    DD + 维纳     : {mn_dd:.3f}")
    print(f"    改善倍数      : {mn_ss / (mn_dd + EPS):.2f}x")
    print(f"重构信号长度: y_ss={y_ss.shape[0]}, y_dd={y_dd.shape[0]}")
    print(f"图已存至: {FIG_DIR}")
    print("生成: s3b_gain_vs_snr.png, s3b_gain_track.png, "
          "s3b_spectrograms.png, s3b_minstat.png")


if __name__ == "__main__":
    main()



