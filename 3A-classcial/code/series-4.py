# -*- coding: utf-8 -*-
"""系列 4 配套代码：VAD 语音端点检测 —— 从能量+过零率双门限到频域似然比 (LRT)。

运行:
    python code/series-4.py

产出 (figures/ 下, 前缀 s4_):
    s4_signal_overview.png   合成信号总览: 静音/浊音/清音/静音段 + 真值标注
    s4_state_machine.png     双门限判决状态机示意 (静音->可能起始->语音->结束)
    s4_double_threshold.png  波形 / 短时能量 / 过零率 / 判决 四图对齐
    s4_lrt_vs_energy.png     稳态强噪声下 能量门限 vs 频域LRT 的鲁棒性对比

图内文字一律英文, 避免中文字体缺失报错。
"""
import matplotlib
matplotlib.use("Agg")  # 无显示环境也能出图, 必须在 import pyplot 之前

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

RNG = np.random.default_rng(2026)  # 结果可复现

FS = 16000                                   # 采样率 (Hz), 全系列默认
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ======================================================================
# 1. 合成一段带 静音 / 浊音 / 清音 / 静音 的信号 (自带真值标签)
# ======================================================================
# 每段: (起始秒, 结束秒, 类型)。voiced=浊音(低ZCR高能量), unvoiced=清音(高ZCR低能量)
SEGMENTS = [
    (0.00, 0.50, "silence"),
    (0.50, 1.15, "voiced"),    # 一段元音
    (1.15, 1.35, "unvoiced"),  # 摩擦清辅音 (能量低但过零率高)
    (1.35, 2.00, "silence"),
    (2.00, 2.80, "voiced"),
    (2.80, 3.00, "silence"),
    (3.00, 3.55, "voiced"),
    (3.55, 4.00, "silence"),
]
DURATION = 4.0                               # 总时长 (秒)


def synth_signal():
    """按 SEGMENTS 合成时域信号, 并生成样本级真值语音掩码。

    返回:
        x     # [T]  干净信号 (仅含极小本底噪声)
        gt    # [T]  样本级真值, 1=语音(浊音或清音) 0=静音
    """
    T = int(DURATION * FS)                   # 标量: 总样本数
    x = np.zeros(T)                          # [T]
    gt = np.zeros(T)                         # [T] 真值掩码
    x += 1e-3 * RNG.standard_normal(T)       # [T] 录音本底噪声 (很小)

    for (t0, t1, kind) in SEGMENTS:
        i0, i1 = int(t0 * FS), int(t1 * FS)  # 段样本下标
        n = np.arange(i1 - i0)               # [seg] 段内局部索引
        tt = n / FS                          # [seg] 段内时间(秒)
        if kind == "voiced":
            # 浊音: 基频 f0 的谐波叠加 + 共振峰包络, 再乘缓慢起落的音节包络
            f0 = 130.0                       # 基频 (Hz)
            sig = np.zeros_like(tt)          # [seg]
            for k, amp in enumerate([1.0, 0.7, 0.5, 0.35, 0.2], start=1):
                sig += amp * np.sin(2 * np.pi * f0 * k * tt)  # 第k次谐波
            env = 0.5 * (1 - np.cos(2 * np.pi * np.clip(tt / tt[-1], 0, 1)))  # [seg] Hann 型音节包络
            x[i0:i1] += 0.35 * env * sig     # 写入浊音
            gt[i0:i1] = 1.0
        elif kind == "unvoiced":
            # 清音: 高通白噪声 -> 高过零率、能量偏低
            w = RNG.standard_normal(i1 - i0)             # [seg]
            hp = np.diff(np.concatenate([[0.0], w]))     # [seg] 一阶差分=简易高通
            x[i0:i1] += 0.06 * hp
            gt[i0:i1] = 1.0
        # silence 段不叠加 (仅保留本底噪声), gt 保持 0
    return x, gt


# ======================================================================
# 2. 分帧 + 短时能量 + 过零率
# ======================================================================
def frame_signal(x, N, H):
    """把一维信号切成重叠帧。

    参数:
        x   # [T]  输入信号
        N   # 标量 帧长(样本)
        H   # 标量 帧移(样本)
    返回:
        frames  # [num_frames, N]
        centers # [num_frames]  每帧中心样本下标 (用于时间对齐)
    """
    T = x.shape[0]
    num = 1 + (T - N) // H                    # 标量: 帧数
    idx = np.arange(N)[None, :] + H * np.arange(num)[:, None]  # [num, N] 每帧样本下标
    frames = x[idx]                           # [num, N]
    centers = H * np.arange(num) + N // 2      # [num]
    return frames, centers


def short_time_energy(frames):
    """短时能量 E(t) = Σ x[n]^2  (逐帧求平方和)。

    参数: frames # [num, N]
    返回: E      # [num]
    """
    return np.sum(frames ** 2, axis=1)         # [num]


def zero_crossing_rate(frames):
    """过零率 ZCR(t) = (1/2N) Σ |sign(x[n]) - sign(x[n-1])|, 取值 [0,1]。

    参数: frames # [num, N]
    返回: zcr    # [num]  每帧符号翻转样本占比
    """
    s = np.sign(frames)                        # [num, N] 符号
    s[s == 0] = 1.0                            # 约定 0 视作正, 避免虚假过零
    flips = np.abs(np.diff(s, axis=1))         # [num, N-1] 相邻符号差 (翻转处=2)
    return 0.5 * np.mean(flips, axis=1)        # [num]


# ======================================================================
# 3. 双门限 + 状态机 VAD
# ======================================================================
def double_threshold_vad(E, Z, noise_frames=15,
                          k_high=8.0, k_low=2.5, z_scale=1.8,
                          min_on=3, hangover=8):
    """能量+过零率双门限, 配 4 态状态机 (静音->可能起始->语音->结束) 做防抖。

    阈值从开头 noise_frames 帧(假设为静音)自适应估计。
    参数:
        E, Z         # [num]  短时能量 / 过零率
        noise_frames # 标量    用于估噪声底的帧数
        k_high/k_low # 能量高/低门限相对噪声底的倍数
        z_scale      # 过零率门限相对静音段 ZCR 的倍数 (抓清音)
        min_on       # 连续多少帧超限才确认起始 (滤毛刺)
        hangover     # 掉到门限下后再挂起多少帧才判结束 (防止字内换气被切断)
    返回:
        decision # [num]  1=语音 0=静音
        info     # dict   记录三条门限, 供画图
    """
    e_noise = np.mean(E[:noise_frames])                       # 标量: 能量噪声底
    e_std = np.std(E[:noise_frames]) + 1e-12
    E_high = e_noise + k_high * e_std                          # 能量高门限
    E_low = e_noise + k_low * e_std                           # 能量低门限
    Z_th = z_scale * (np.mean(Z[:noise_frames]) + 1e-6)       # 过零率门限

    # 帧级“候选活跃”: 能量够高, 或 (能量过低门限 且 过零率高) -> 抓清辅音
    active = (E > E_high) | ((E > E_low) & (Z > Z_th))        # [num] bool

    decision = np.zeros_like(E)                               # [num]
    state = "SIL"                                              # 静音
    on_cnt = 0                                                # 连续活跃计数
    hang = 0                                                  # 挂起倒计时
    onset_idx = 0                                             # 起始候选帧
    for t in range(E.shape[0]):
        if state == "SIL":
            if active[t]:
                state, on_cnt, onset_idx = "MAYBE_ON", 1, t
        elif state == "MAYBE_ON":
            if active[t]:
                on_cnt += 1
                if on_cnt >= min_on:
                    decision[onset_idx:t + 1] = 1.0            # 回填起始段
                    state = "SPEECH"
            else:
                state, on_cnt = "SIL", 0                       # 毛刺, 打回静音
        elif state == "SPEECH":
            decision[t] = 1.0
            if not active[t]:
                state, hang = "MAYBE_OFF", hangover
        elif state == "MAYBE_OFF":
            decision[t] = 1.0                                  # 挂起期仍算语音
            if active[t]:
                state = "SPEECH"
            else:
                hang -= 1
                if hang <= 0:
                    state = "SIL"
    info = dict(E_high=E_high, E_low=E_low, Z_th=Z_th)
    return decision, info


def fixed_threshold_vad(E, Z, E_high, E_low, Z_th, min_on=3, hangover=8):
    """用外部给定的固定门限跑同一套状态机 (模拟安静环境标定后拿去吵环境用)。

    参数:
        E, Z                    # [num]  能量 / 过零率
        E_high, E_low, Z_th     # 标量    预先标定好的固定门限
    返回:
        decision # [num]  1=语音 0=静音
    """
    active = (E > E_high) | ((E > E_low) & (Z > Z_th))       # [num] bool
    decision = np.zeros_like(E)                              # [num]
    state, on_cnt, hang, onset_idx = "SIL", 0, 0, 0
    for t in range(E.shape[0]):
        if state == "SIL":
            if active[t]:
                state, on_cnt, onset_idx = "MAYBE_ON", 1, t
        elif state == "MAYBE_ON":
            if active[t]:
                on_cnt += 1
                if on_cnt >= min_on:
                    decision[onset_idx:t + 1] = 1.0
                    state = "SPEECH"
            else:
                state, on_cnt = "SIL", 0
        elif state == "SPEECH":
            decision[t] = 1.0
            if not active[t]:
                state, hang = "MAYBE_OFF", hangover
        elif state == "MAYBE_OFF":
            decision[t] = 1.0
            if active[t]:
                state = "SPEECH"
            else:
                hang -= 1
                if hang <= 0:
                    state = "SIL"
    return decision


# ======================================================================
# 4. 频域似然比 (LRT) VAD  —— Sohn 统计模型
# ======================================================================
def lrt_vad(x, N, H, noise_frames=15, alpha=0.98, eta=0.4):
    """频域高斯统计模型的对数似然比 VAD。

    每个频点带噪谱在 H0(仅噪声)/H1(语音+噪声) 下建高斯模型,
    似然比 Λ_f = 1/(1+ξ) · exp( γξ/(1+ξ) ), 对全频点取几何平均后与门限 η 比较。
    参数:
        alpha # 判决引导 (DD) 平滑系数, 估计先验 SNR ξ
        eta   # 对数似然比几何平均的判决门限
    返回:
        decision # [num]  1=语音
        logLR    # [num]  每帧对数似然比几何平均 (log Λ)
        centers  # [num]  帧中心样本下标
    """
    frames, centers = frame_signal(x, N, H)         # [num, N]
    win = np.hanning(N)                              # [N] 分析窗
    X = np.fft.rfft(frames * win[None, :], axis=1)   # [num, F] 复数谱
    P = np.abs(X) ** 2                               # [num, F] 功率谱
    num, F = P.shape

    lam = np.mean(P[:noise_frames], axis=0) + 1e-10  # [F] 噪声功率谱 λ_d (开头静音帧估)
    logLR = np.zeros(num)                            # [num]
    xi_prev_pow = P[0].copy()                         # [F] 上一帧估计的“干净谱功率”

    for t in range(num):
        gamma = np.minimum(P[t] / lam, 1e3)          # [F] 后验 SNR γ = |X|²/λ_d
        # 判决引导估计先验 SNR ξ: 平滑上一帧干净谱 + 当前瞬时估计
        xi = alpha * (xi_prev_pow / lam) + (1 - alpha) * np.maximum(gamma - 1.0, 0.0)  # [F]
        xi = np.maximum(xi, 1e-6)                     # [F] 防止 log/除零
        # 每频点对数似然比: log Λ_f = γξ/(1+ξ) - log(1+ξ)
        llr_f = gamma * xi / (1.0 + xi) - np.log1p(xi)  # [F]
        logLR[t] = np.mean(llr_f)                      # 标量: 全频点几何平均 (log 域取均值)
        # 更新: 用维纳增益 G=ξ/(1+ξ) 得到本帧干净谱功率, 供下一帧 DD 使用
        G = xi / (1.0 + xi)                            # [F]
        xi_prev_pow = (G ** 2) * P[t]                  # [F]

    decision = (logLR > eta).astype(float)             # [num]
    return decision, logLR, centers


# ======================================================================
# 5. 工具: 帧级真值 + 准确率
# ======================================================================
def frame_ground_truth(gt_samples, centers, N, H):
    """把样本级真值按帧内语音占比 (>0.5) 转成帧级真值。

    参数:
        gt_samples # [T]   样本级真值
        centers    # [num] 帧中心
    返回:
        gt_frame   # [num]
    """
    frames, _ = frame_signal(gt_samples, N, H)         # [num, N]
    return (np.mean(frames, axis=1) > 0.5).astype(float)  # [num]


def accuracy(pred, gt):
    """帧级判决准确率 (%)。"""
    return 100.0 * np.mean(pred == gt)


def shade_gt(ax, gt_frame, centers, fs=FS, color="tab:green", alpha=0.12, label="ground-truth speech"):
    """在坐标轴上用浅色阴影标出真值语音区间 (帧级)。"""
    t = centers / fs
    on = False
    start = 0.0
    first = True
    for i in range(len(gt_frame)):
        if gt_frame[i] > 0.5 and not on:
            on, start = True, t[i]
        elif gt_frame[i] <= 0.5 and on:
            ax.axvspan(start, t[i], color=color, alpha=alpha,
                       label=(label if first else None))
            on, first = False, False
    if on:
        ax.axvspan(start, t[-1], color=color, alpha=alpha,
                   label=(label if first else None))


# ======================================================================
# 主流程
# ======================================================================
def main():
    N = 400                       # 帧长 25ms @16k
    H = 160                       # 帧移 10ms @16k
    x, gt = synth_signal()        # [T], [T]
    T = x.shape[0]
    t_axis = np.arange(T) / FS    # [T]
    print(f"[info] fs={FS} N={N}({1000*N/FS:.0f}ms) H={H}({1000*H/FS:.0f}ms) 总时长={DURATION}s")

    # ---- 特征 ----
    frames, centers = frame_signal(x, N, H)   # [num, N], [num]
    E = short_time_energy(frames)             # [num]
    Z = zero_crossing_rate(frames)            # [num]
    gt_frame = frame_ground_truth(gt, centers, N, H)   # [num]
    tc = centers / FS                         # [num] 帧中心时间

    # === 图1: 合成信号总览 + 段类型标注 ===
    fig, ax = plt.subplots(figsize=(9.5, 3.2))
    ax.plot(t_axis, x, lw=0.5, color="tab:blue")
    color_map = {"silence": "gray", "voiced": "tab:green", "unvoiced": "tab:orange"}
    seen = set()
    for (t0, t1, kind) in SEGMENTS:
        lab = kind if kind not in seen else None
        seen.add(kind)
        ax.axvspan(t0, t1, color=color_map[kind], alpha=0.12, label=lab)
    ax.set_title("Synthetic Signal: silence / voiced / unvoiced segments")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("amplitude")
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "s4_signal_overview.png", dpi=130)
    plt.close(fig)

    # === 图2: 双门限状态机示意图 (纯示意, 不依赖数据) ===
    fig, ax = plt.subplots(figsize=(9.5, 2.6))
    states = ["SILENCE", "MAYBE\nONSET", "SPEECH", "MAYBE\nOFFSET"]
    xs = [0.5, 3.0, 5.5, 8.0]
    for name, xc in zip(states, xs):
        box = FancyBboxPatch((xc - 0.7, 0.4), 1.4, 1.2,
                             boxstyle="round,pad=0.08", fc="#d6ebff", ec="#2b6cb0")
        ax.add_patch(box)
        ax.text(xc, 1.0, name, ha="center", va="center", fontsize=9)

    def arrow(x0, x1, text, y=1.75, rad=0.0):
        ar = FancyArrowPatch((x0, 1.0), (x1, 1.0),
                             connectionstyle=f"arc3,rad={rad}",
                             arrowstyle="-|>", mutation_scale=14, color="#c98a00", lw=1.4)
        ax.add_patch(ar)
        ax.text((x0 + x1) / 2, y, text, ha="center", fontsize=7.5, color="#8a5a00")

    arrow(1.2, 2.3, "active", rad=-0.35)
    arrow(3.7, 4.8, "on_cnt>=min_on", rad=-0.35)
    arrow(2.3, 1.2, "drop (glitch)", y=0.15, rad=-0.35)
    arrow(6.2, 7.3, "not active", rad=-0.35)
    arrow(7.3, 6.2, "active again", y=0.15, rad=-0.35)
    ax.text(8.0, 2.15, "hangover-- -> SILENCE", ha="center", fontsize=7.5, color="#8a5a00")
    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(-0.2, 2.6)
    ax.axis("off")
    ax.set_title("Double-Threshold VAD State Machine (anti-chatter)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "s4_state_machine.png", dpi=130)
    plt.close(fig)

    # === 图3: 波形 / 能量 / 过零率 / 判决 四联图对齐 ===
    dec, info = double_threshold_vad(E, Z)     # [num], dict
    acc_dt = accuracy(dec, gt_frame)
    print(f"[info] double-threshold VAD frame accuracy = {acc_dt:.1f}%")

    fig, axes = plt.subplots(4, 1, figsize=(9.5, 8.0), sharex=True)
    axes[0].plot(t_axis, x, lw=0.4, color="tab:blue")
    shade_gt(axes[0], gt_frame, centers)
    axes[0].set_ylabel("waveform")
    axes[0].legend(loc="upper right", fontsize=7)
    axes[0].set_title("Double-Threshold VAD: waveform / energy / ZCR / decision")

    axes[1].plot(tc, E, color="tab:red", lw=1.0, label="short-time energy")
    axes[1].axhline(info["E_high"], color="k", ls="--", lw=0.8, label="E_high")
    axes[1].axhline(info["E_low"], color="gray", ls=":", lw=0.8, label="E_low")
    shade_gt(axes[1], gt_frame, centers, label=None)
    axes[1].set_ylabel("energy")
    axes[1].legend(loc="upper right", fontsize=7)

    axes[2].plot(tc, Z, color="tab:purple", lw=1.0, label="zero-crossing rate")
    axes[2].axhline(info["Z_th"], color="k", ls="--", lw=0.8, label="Z_th")
    shade_gt(axes[2], gt_frame, centers, label=None)
    axes[2].set_ylabel("ZCR")
    axes[2].legend(loc="upper right", fontsize=7)

    axes[3].plot(tc, dec, color="tab:blue", lw=1.2, drawstyle="steps-mid", label="VAD decision")
    axes[3].plot(tc, gt_frame, color="tab:green", lw=1.0, ls="--", drawstyle="steps-mid", label="ground truth")
    axes[3].set_ylim(-0.15, 1.25)
    axes[3].set_ylabel("decision")
    axes[3].set_xlabel("time (s)")
    axes[3].legend(loc="upper right", fontsize=7)
    for a in axes:
        a.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "s4_double_threshold.png", dpi=130)
    plt.close(fig)

    # === 图4: 稳态强噪声下 "静音室标定的固定能量门限" vs 频域LRT ===
    # 关键: 能量门限在安静环境标定好 (Ehi/Elo/Zth), 部署时环境变吵。
    # 稳态噪声把能量底整体抬高, 固定门限被持续顶穿 -> 满屏误判为语音;
    # LRT 用噪声功率 λ_d 做归一化(在 SNR 域判决), 门限随噪声自适应, 因此更鲁棒。
    _, info_clean = double_threshold_vad(E, Z)           # 安静环境标定的门限
    Ehi, Elo, Zth = info_clean["E_high"], info_clean["E_low"], info_clean["Z_th"]

    noise = 0.05 * RNG.standard_normal(T)                # [T] 稳态白噪
    hum = 0.05 * np.sin(2 * np.pi * 220 * t_axis)        # [T] 220Hz 稳态嗡声(风扇/工频)
    x_noisy = x + noise + hum                            # [T]

    frames_n, centers_n = frame_signal(x_noisy, N, H)
    E_n = short_time_energy(frames_n)                     # [num]
    Z_n = zero_crossing_rate(frames_n)                    # [num]
    dec_e = fixed_threshold_vad(E_n, Z_n, Ehi, Elo, Zth)  # 用安静环境的固定门限
    dec_l, logLR, centers_l = lrt_vad(x_noisy, N, H, eta=0.4)  # 频域 LRT
    gt_f = frame_ground_truth(gt, centers_n, N, H)
    acc_e = accuracy(dec_e, gt_f)
    acc_l = accuracy(dec_l, gt_f)
    print(f"[info] noisy: fixed-energy-threshold acc={acc_e:.1f}%  |  LRT acc={acc_l:.1f}%")

    tcn = centers_n / FS
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 6.4), sharex=True)
    axes[0].plot(t_axis, x_noisy, lw=0.4, color="tab:blue")
    shade_gt(axes[0], gt_f, centers_n)
    axes[0].set_ylabel("noisy wave")
    axes[0].legend(loc="upper right", fontsize=7)
    axes[0].set_title("Steady Noise: Room-Calibrated Energy Threshold vs Freq-Domain LRT")

    axes[1].plot(tcn, dec_e, color="tab:red", lw=1.2, drawstyle="steps-mid",
                 label=f"fixed energy+ZCR (acc={acc_e:.0f}%)")
    axes[1].plot(tcn, gt_f, color="tab:green", lw=1.0, ls="--", drawstyle="steps-mid", label="ground truth")
    axes[1].set_ylim(-0.15, 1.25)
    axes[1].set_ylabel("energy VAD")
    axes[1].legend(loc="upper right", fontsize=7)

    axes[2].plot(tcn, dec_l, color="tab:blue", lw=1.2, drawstyle="steps-mid",
                 label=f"LRT (acc={acc_l:.0f}%)")
    axes[2].plot(tcn, gt_f, color="tab:green", lw=1.0, ls="--", drawstyle="steps-mid", label="ground truth")
    axes[2].set_ylim(-0.15, 1.25)
    axes[2].set_ylabel("LRT VAD")
    axes[2].set_xlabel("time (s)")
    axes[2].legend(loc="upper right", fontsize=7)
    for a in axes:
        a.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "s4_lrt_vs_energy.png", dpi=130)
    plt.close(fig)

    print("[done] figures saved to", FIG_DIR)


if __name__ == "__main__":
    main()
