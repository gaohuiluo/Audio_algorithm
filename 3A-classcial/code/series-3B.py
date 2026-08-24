# -*- coding: utf-8 -*-
"""系列 3B 配套代码：Musical Noise 与噪声估计（真实风扇录音版）。

对比 "纯谱减" vs "判决引导(DD)+维纳" 的降噪结果，展示音乐噪声差异；
用最小值统计 (minimum statistics) 做无 VAD 的在线噪声功率盲估。

数据：voice-with-noise/fan-1.wav / fan-2.wav / fan-3.wav
      （用户在风扇下录的真人语音，16kHz 单声道，无干净参考 clean）

运行:
    python code/series-3B.py（在 3A-classcial/ 目录下执行）

产出:
    figures/s3b_fanN_spectrograms.png  Noisy / Spectral Sub.(硬减) / DD+Wiener(平滑)
    figures/s3b_fanN_gain_track.png    高频噪声点上 谱减增益(乱跳) vs DD维纳(平滑)
    figures/s3b_fanN_minstat.png       某频点 |X|^2 与最小值统计估计 λ_d 的跟踪
    figures/s3b_gain_vs_snr.png        全局理论曲线：谱减 vs 维纳增益随后验 SNR
    audio/s3b_fanN_specsub.wav         纯谱减处理后音频
    audio/s3b_fanN_ddwiener.wav        DD+维纳处理后音频

真实录音无干净参考，故 musical_noise_index 改为用"带噪信号自身"划分静音帧：
帧能量接近最小值统计噪声底的帧视为静音帧，从而在无 clean 时仍能量化音乐噪声。
图内文字一律英文, 避免中文字体缺失报错。
"""
import matplotlib
matplotlib.use("Agg")  # 无显示环境也能出图, 必须在 pyplot 之前设置

import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
from pathlib import Path

# 全局 STFT 参数 (STYLE.md: f_s 默认 16000)
FS = 16000        # 采样率 Hz
N = 512           # 帧长 (窗长) 样本
H = 128           # 帧移 hop, 75% overlap 满足 Hann 的 COLA 可完美重构
EPS = 1e-12       # 防除零小量

BASE = Path(__file__).resolve().parent.parent          # 3A-classcial/
FIG_DIR = BASE / "figures"
AUDIO_DIR = BASE / "audio"
VOICE_DIR = BASE.parent / "voice-with-noise"           # e:/Android/Tech-Blog/voice-with-noise
FIG_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

FILES = ["fan-1.wav", "fan-2.wav", "fan-3.wav"]        # 待处理三段录音


# ----------------------------------------------------------------------
# 1. 音频 IO: 读 wav (float64/单声道/峰值归一化), 写 wav (float→int16)
# ----------------------------------------------------------------------
def load_wav(path):
    """读取 wav → x # [n] float64 单声道, 峰值归一化到 [-1, 1]。"""
    fs, x = wavfile.read(path)
    x = x.astype(np.float64)
    if x.ndim > 1:                                     # 多声道 → 取均值转单声道
        x = x.mean(axis=1)
    x = x / (np.max(np.abs(x)) + EPS)                  # 峰值归一化
    return x, fs


def save_wav(path, x, fs=FS):
    """写 wav: float → int16, 先 clip 到 [-1, 1] 再放大到 int16 量程。"""
    x = np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)
    wavfile.write(path, fs, (x * 32767.0).astype(np.int16))


# ----------------------------------------------------------------------
# 2. STFT / iSTFT 封装 (scipy)
# ----------------------------------------------------------------------
def stft(x):
    """短时傅里叶变换。返回 f # [F], tt # [Tf], X # [F, Tf] 复数谱。"""
    f, tt, X = signal.stft(x, fs=FS, window="hann",
                           nperseg=N, noverlap=N - H, boundary="zeros", padded=True)
    return f, tt, X                                    # X: [F, Tf]


def istft(X):
    """逆变换回时域 y # [T] (沿用处理后的复数谱)。"""
    _, y = signal.istft(X, fs=FS, window="hann",
                        nperseg=N, noverlap=N - H, boundary=True)
    return y


# ----------------------------------------------------------------------
# 3. 噪声估计: 最小值统计 (minimum statistics)
#    无 VAD 时, 对每个频点在滑动时间窗内取功率最小值近似 λ_d。
#    直觉: 即便有话段, 谱功率也会短暂 "落回" 噪声底; 取窗内最小值 ≈ 噪声底。
# ----------------------------------------------------------------------
def minimum_statistics(P, win_frames=40, bias=1.5, alpha_p=0.85):
    """最小值统计估计噪声功率谱 λ_d。

    参数:
        P           # [F, Tf]  带噪功率谱 |X(t,f)|²
        win_frames  # 标量      回看窗长 (帧数); 越大越稳但延迟越大
        bias        # 标量      最小值偏低的补偿因子 (min 是有偏估计, 乘回来)
        alpha_p     # 标量      对功率谱先做一阶平滑, 削掉毛刺再取 min
    返回:
        lam_d       # [F, Tf]  噪声功率谱估计 λ_d(t,f)
    """
    F, Tf = P.shape                                    # 标量
    # 3.1 先对功率谱做时间平滑, 否则单帧毛刺会把 min 拉得过低
    Ps = np.zeros_like(P)                              # [F, Tf]
    Ps[:, 0] = P[:, 0]
    for t in range(1, Tf):
        Ps[:, t] = alpha_p * Ps[:, t - 1] + (1 - alpha_p) * P[:, t]

    # 3.2 滑动窗内取每个频点的功率最小值
    lam_d = np.zeros_like(P)                           # [F, Tf]
    for t in range(Tf):
        a = max(0, t - win_frames + 1)                 # 窗左边界
        lam_d[:, t] = np.min(Ps[:, a:t + 1], axis=1)   # [F] 窗内最小
    lam_d *= bias                                      # 补偿最小值的负偏差
    return lam_d                                       # [F, Tf]


# ----------------------------------------------------------------------
# 4. 纯谱减法 (每个频点独立硬减 —— 音乐噪声的源头)
# ----------------------------------------------------------------------
def spectral_subtraction(X, lam_d, over_sub=2.0, floor=0.0):
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
    P_hat = P - over_sub * lam_d                       # [F, Tf] 减掉 over_sub 倍噪声功率
    P_hat = np.maximum(P_hat, floor * P)               # [F, Tf] 谱下限
    gain = np.sqrt(P_hat / (P + EPS))                  # [F, Tf] 幅度增益 M(t,f)
    Y = gain * X                                       # [F, Tf] 沿用带噪相位
    return Y


# ----------------------------------------------------------------------
# 5. 判决引导 (DD) 估计先验 SNR + 维纳增益
#    DD: ξ_prior(t) = α·|Ŝ(t-1)|²/λ_d + (1-α)·max(γ_post-1, 0)
#    维纳增益: M = ξ / (1 + ξ)
# ----------------------------------------------------------------------
def dd_wiener(X, lam_d, alpha=0.98, g_min=0.08):
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
# 6. 音乐噪声量化指标: 静音段残余谱的时频起伏程度
#    真实录音无 clean, 改用"带噪信号自身 + 最小值统计噪声底"划分静音帧:
#    帧总功率接近噪声底 (低于噪声底若干倍阈值) 的帧视为静音帧。
# ----------------------------------------------------------------------
def silent_mask_from_noisy(X, lam_d, ratio=3.0):
    """用带噪功率与最小值统计噪声底之比划分静音帧 (无需 clean)。

    参数:
        X       # [F, Tf]  带噪复数谱
        lam_d   # [F, Tf]  最小值统计噪声功率估计
        ratio   # 标量      帧能量 / 噪声底能量 < ratio 视为静音帧
    返回:
        sil     # [Tf]     True = 静音帧 (接近噪声底)
    """
    P = np.abs(X) ** 2                                 # [F, Tf]
    frame_e = P.sum(axis=0)                            # [Tf] 带噪帧能量
    floor_e = lam_d.sum(axis=0)                        # [Tf] 噪声底帧能量
    sil = frame_e < ratio * floor_e                    # [Tf] 接近噪声底 → 静音
    return sil


def musical_noise_index(Y, silent_mask):
    """用静音段对数功率谱的"帧间+频间"差分方差近似音乐噪声强度。

    参数:
        Y            # [F, Tf]  降噪后复数谱
        silent_mask  # [Tf]     True=静音帧
    返回:
        idx          # 标量      越大表示残余越 "斑驳" (音乐噪声越重)
    """
    logP = np.log(np.abs(Y) ** 2 + EPS)                # [F, Tf]
    sil = logP[:, silent_mask]                         # [F, Tf_sil] 只看静音段
    if sil.shape[1] < 3:
        return float("nan")
    d_time = np.diff(sil, axis=1)                      # [F, Tf_sil-1] 帧间差分
    d_freq = np.diff(sil, axis=0)                      # [F-1, Tf_sil] 频间差分
    return float(np.var(d_time) + np.var(d_freq))      # 标量


# ----------------------------------------------------------------------
# 7. 绘图
# ----------------------------------------------------------------------
def plot_gain_vs_snr():
    """全局理论曲线: 谱减增益 vs 维纳增益 随后验 SNR γ_post 的形状。"""
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


def plot_gain_track(g_ss_track, g_dd_track, fbin_hz, path):
    """噪声主导频点上的增益时序: 谱减剧烈抖动 vs DD 平滑。"""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(g_ss_track, label="Spectral Subtraction gain", lw=1.2, alpha=0.9)
    ax.plot(g_dd_track, label="DD-Wiener gain", lw=1.8)
    ax.set_xlabel("frame index t")
    ax.set_ylabel("gain  M(t,f)")
    ax.set_title(f"Gain over time at a noise-dominated bin (~{fbin_hz:.0f} Hz)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_spectrograms(specs, titles, f, path):
    """三联语谱图: 对比孤立亮点 (音乐噪声) 与平滑残余。"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)
    vmax = np.max([20 * np.log10(np.abs(s) + EPS) for s in specs])
    for ax, Y, title in zip(axes.ravel(), specs, titles):
        db = 20 * np.log10(np.abs(Y) + EPS)            # [F, Tf]
        im = ax.pcolormesh(np.arange(db.shape[1]), f / 1000.0, db,
                           vmin=vmax - 70, vmax=vmax, cmap="magma", shading="auto")
        ax.set_title(title)
        ax.set_ylim(0, 4)                              # 只看 0~4 kHz, 语音主要能量区
        ax.set_xlabel("frame index t")
    axes[0].set_ylabel("frequency (kHz)")
    fig.colorbar(im, ax=axes, shrink=0.8, label="magnitude (dB)")
    fig.suptitle("Spectrograms: musical noise (isolated specks) vs smooth residual")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_minstat(P, lam_est, f, fbin, path):
    """最小值统计对 λ_d 的跟踪曲线 (单频点)。真实录音无真值参考线。"""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(P[fbin], label="noisy power |X|^2", lw=1, alpha=0.55)
    ax.plot(lam_est[fbin], label="min-statistics estimate lambda_d", lw=1.8)
    ax.set_xlabel("frame index t")
    ax.set_ylabel("power")
    ax.set_title(f"Minimum-statistics noise tracking at ~{f[fbin]:.0f} Hz")
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------
# 8. 单文件处理流程
# ----------------------------------------------------------------------
def process_file(fname, tag):
    """处理一个录音文件：加载→STFT→最小值统计盲估 λ_d→谱减/DD维纳→出图存音频。"""
    print(f"\n========== {fname}  (tag={tag}) ==========")
    # --- 加载 ---
    x, fs = load_wav(VOICE_DIR / fname)                # [n] 带噪
    print(f"[load] fs={fs}, dur={len(x)/fs:.2f}s, n={len(x)}")

    # --- STFT ---
    f, tt, X = stft(x)                                 # X: [F, Tf]
    P = np.abs(X) ** 2                                 # [F, Tf] 带噪功率
    F, Tf = X.shape
    print(f"[shape] STFT X # [F, T] = {X.shape}  (F={F}, T={Tf})")

    # --- 噪声估计: 最小值统计 (无 VAD 盲估, 不依赖纯噪声段先验) ---
    lam_est = minimum_statistics(P, win_frames=40)     # [F, Tf]
    print(f"[shape] lambda_d (min-stat) # [F, T] = {lam_est.shape}")

    # --- 两种降噪 (都用同一份最小值统计的噪声估计, 公平对比) ---
    Y_ss = spectral_subtraction(X, lam_est, over_sub=2.0, floor=0.0)   # 硬减
    Y_dd, xi = dd_wiener(X, lam_est, alpha=0.98, g_min=0.08)           # DD+维纳

    # --- 静音帧划分 (无 clean, 用带噪能量 vs 噪声底) + 音乐噪声指标 ---
    sil = silent_mask_from_noisy(X, lam_est, ratio=3.0)   # [Tf]
    n_sil = int(sil.sum())
    mn_ss = musical_noise_index(Y_ss, sil)
    mn_dd = musical_noise_index(Y_dd, sil)

    # --- 增益时序: 选一个语音能量弱、噪声主导的高频点 ---
    fbin_track = int(np.argmin(np.abs(f - 6000)))      # ~6 kHz 频点索引
    g_ss = np.sqrt(np.abs(Y_ss) ** 2 / (P + EPS))      # [F, Tf]
    g_dd = np.sqrt(np.abs(Y_dd) ** 2 / (P + EPS))      # [F, Tf]

    # --- iSTFT 回时域 ---
    y_ss = istft(Y_ss)                                 # [T]
    y_dd = istft(Y_dd)                                 # [T]

    # --- 保存音频 ---
    p_ss = AUDIO_DIR / f"s3b_{tag}_specsub.wav"
    p_dd = AUDIO_DIR / f"s3b_{tag}_ddwiener.wav"
    save_wav(p_ss, y_ss)
    save_wav(p_dd, y_dd)

    # --- 绘图 ---
    p_spec = FIG_DIR / f"s3b_{tag}_spectrograms.png"
    p_track = FIG_DIR / f"s3b_{tag}_gain_track.png"
    p_ms = FIG_DIR / f"s3b_{tag}_minstat.png"
    plot_spectrograms(
        [X, Y_ss, Y_dd],
        ["Noisy",
         f"Spectral Subtraction (MN idx={mn_ss:.2f})",
         f"DD + Wiener (MN idx={mn_dd:.2f})"],
        f, p_spec,
    )
    plot_gain_track(g_ss[fbin_track], g_dd[fbin_track], f[fbin_track], p_track)
    fbin_ms = int(np.argmin(np.abs(f - 3000)))         # ~3 kHz 观察噪声跟踪
    plot_minstat(P, lam_est, f, fbin_ms, p_ms)

    # --- 打印报告 ---
    print(f"[silent] 静音帧数 (带噪能量近噪声底)={n_sil}/{Tf}")
    print("===== Musical Noise 指标 (静音段谱起伏, 越小越好) =====")
    print(f"    纯谱减        : {mn_ss:.3f}")
    print(f"    DD + 维纳     : {mn_dd:.3f}")
    print(f"    改善倍数      : {mn_ss / (mn_dd + EPS):.2f}x")
    print(f"[recon] 重构信号长度: y_ss={y_ss.shape[0]}, y_dd={y_dd.shape[0]}")
    print(f"[audio] audio/{p_ss.name}")
    print(f"[audio] audio/{p_dd.name}")
    print(f"[figures] figures/{p_spec.name}")
    print(f"[figures] figures/{p_track.name}")
    print(f"[figures] figures/{p_ms.name}")

    return {
        "fname": fname, "tag": tag, "stft_shape": X.shape,
        "n_sil": n_sil, "mn_ss": mn_ss, "mn_dd": mn_dd,
        "improve": mn_ss / (mn_dd + EPS),
        "len_ss": y_ss.shape[0], "len_dd": y_dd.shape[0],
    }


# ----------------------------------------------------------------------
# 9. 主流程：三个文件循环 + 一张全局理论曲线
# ----------------------------------------------------------------------
def main():
    tags = {"fan-1.wav": "fan1", "fan-2.wav": "fan2", "fan-3.wav": "fan3"}
    results = []
    for fname in FILES:
        results.append(process_file(fname, tags[fname]))

    # 全局理论增益曲线 (与具体文件无关)
    plot_gain_vs_snr()
    print("\n[figures] figures/s3b_gain_vs_snr.png  (全局理论曲线)")

    print("\n===== series-3B 汇总 =====")
    for r in results:
        print(f"  {r['fname']}: STFT={r['stft_shape']}, "
              f"MN_ss={r['mn_ss']:.3f}, MN_dd={r['mn_dd']:.3f}, "
              f"improve={r['improve']:.2f}x, "
              f"len(ss={r['len_ss']}, dd={r['len_dd']})")
    print("\n[done] series-3B 全部完成。")


if __name__ == "__main__":
    main()
