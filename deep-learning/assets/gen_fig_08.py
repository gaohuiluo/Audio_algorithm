# -*- coding: utf-8 -*-
"""
第 8 篇《损失函数》配图生成脚本。
运行：python assets/gen_fig_08.py
输出：assets/fig08_*.png（4 张）

依赖：matplotlib、numpy、torch。中文字体用 Microsoft YaHei。
风格与第 1 篇 gen_fig_01.py 严格对齐（浅底、克制配色、直觉先行）。
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # torch+mpl 的 OpenMP 冲突绕过
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- 全局风格（与第 1 篇一致）----
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 12
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.dpi"] = 130
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.facecolor"] = "white"

# 系列统一配色
C_WAVE = "#2c6fbb"     # 波形/时间/干净目标
C_FREQ = "#c0392b"     # 频率/残差噪声
C_BATCH = "#8e44ad"    # 批次/掩码
C_CHAN = "#16a085"     # 通道/对齐分量·重建
C_GRAD = "#e67e22"     # 梯度/估计信号
C_PAD = "#bdc3c7"      # padding 灰
C_INK = "#2c3e50"      # 主文字


def save(fig, name):
    path = os.path.join(HERE, name)
    fig.savefig(path)
    plt.close(fig)
    print("saved", path)


def _box(ax, xy, w, h, text, fc, ec=None, tc="white", fs=12, r=0.06):
    ec = ec or fc
    ax.add_patch(FancyBboxPatch(
        (xy[0] - w / 2, xy[1] - h / 2), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={r}",
        linewidth=1.6, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(xy[0], xy[1], text, ha="center", va="center",
            color=tc, fontsize=fs, zorder=3, linespacing=1.3)


def _arrow(ax, p0, p1, color=C_INK, ls="-", lw=1.8, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=16,
        linewidth=lw, color=color, linestyle=ls,
        connectionstyle=f"arc3,rad={rad}", zorder=1))


# ============ 图 1：整体放大 → MSE 暴涨、SI-SNR 纹丝不动（真实跑数） ============
def fig_scale_sensitivity():
    torch.manual_seed(0)
    T = 16000
    s = torch.randn(T)
    est0 = s + 0.1 * torch.randn(T)     # 结构基本对的估计

    def si_snr(est, s, eps=1e-8):
        est = est - est.mean()
        s = s - s.mean()
        dot = (est * s).sum()
        s_target = dot / ((s * s).sum() + eps) * s
        e_noise = est - s_target
        ratio = (s_target ** 2).sum() / ((e_noise ** 2).sum() + eps)
        return (10 * torch.log10(ratio + eps)).item()

    def mse(est, s):
        return ((est - s) ** 2).mean().item()

    # 扫一遍整体增益 g：把估计乘以不同倍数（人耳只觉得音量变，结构没变）
    gains = np.linspace(0.2, 5.0, 40)
    mse_vals = [mse(est0 * g, s) for g in gains]
    sisnr_vals = [si_snr(est0 * g, s) for g in gains]

    fig, (axm, axs) = plt.subplots(1, 2, figsize=(10.4, 4.0))

    axm.plot(gains, mse_vals, color=C_FREQ, lw=2.2)
    axm.axvline(1.0, color=C_PAD, ls="--", lw=1.2)
    axm.set_title("MSE：整体音量一变就暴涨（人耳无感，它却狂罚）",
                  fontsize=11.5, color=C_INK)
    axm.set_xlabel("估计信号整体增益 g（×倍数）")
    axm.set_ylabel("MSE")
    axm.grid(alpha=0.25)
    axm.scatter([5.0], [mse(est0 * 5.0, s)], color=C_FREQ, zorder=5, s=40)
    axm.annotate("×5 → 误差暴涨", xy=(5.0, mse(est0 * 5.0, s)),
                 xytext=(2.4, mse(est0 * 5.0, s) * 0.7), color=C_FREQ, fontsize=10,
                 arrowprops=dict(arrowstyle="->", color=C_FREQ))

    axs.plot(gains, sisnr_vals, color=C_WAVE, lw=2.2)
    axs.axvline(1.0, color=C_PAD, ls="--", lw=1.2)
    axs.set_title("SI-SNR：整体音量随便变，数值几乎一条水平线",
                  fontsize=11.5, color=C_INK)
    axs.set_xlabel("估计信号整体增益 g（×倍数）")
    axs.set_ylabel("SI-SNR / dB（越大越好）")
    axs.grid(alpha=0.25)
    span = max(sisnr_vals) - min(sisnr_vals)
    axs.set_ylim(np.mean(sisnr_vals) - max(3, span * 6),
                 np.mean(sisnr_vals) + max(3, span * 6))
    axs.text(2.6, np.mean(sisnr_vals) + 0.4,
             "投影那一步已把增益吸收\n→ 尺度不变", color=C_WAVE, fontsize=10,
             ha="center", va="bottom")

    fig.suptitle("同一段估计只是「整体变响/变轻」：MSE 当大错狂罚，SI-SNR 视而不见",
                 fontsize=13, color=C_INK, y=1.02)
    fig.subplots_adjust(wspace=0.28, bottom=0.16)
    save(fig, "fig08_scale_sensitivity.png")


# ============ 图 2：幅度压缩（pow0.3 / log）拉平响弱频段话语权 ============
def fig_mag_compression():
    x = np.linspace(0, 1.0, 400)
    y_lin = x
    y_pow = x ** 0.3
    y_log = np.log(1 + x) / np.log(2)   # 归一化到 [0,1] 便于同图对比

    fig, (axc, axb) = plt.subplots(1, 2, figsize=(10.6, 4.0),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    # 左：压缩曲线
    axc.plot(x, y_lin, color=C_PAD, lw=2.0, label="不压缩（线性）")
    axc.plot(x, y_pow, color=C_WAVE, lw=2.4, label="幂律压缩 $x^{0.3}$")
    axc.plot(x, y_log, color=C_CHAN, lw=2.4, ls="--", label="对数压缩 log(1+x)")
    axc.set_title("压缩函数：小数抬得多、大数压得狠", fontsize=11.5, color=C_INK)
    axc.set_xlabel("原始幅度 |S|（归一化）")
    axc.set_ylabel("压缩后的值")
    axc.legend(fontsize=9.5, loc="lower right")
    axc.grid(alpha=0.25)
    # 标注：弱成分被抬高
    axc.annotate("弱频段被抬高\n→ 在损失里有了话语权",
                 xy=(0.08, 0.08 ** 0.3), xytext=(0.28, 0.42),
                 color=C_WAVE, fontsize=9.5,
                 arrowprops=dict(arrowstyle="->", color=C_WAVE))

    # 右：柱状对比——响/弱两个频段在损失里的“话语权”
    labels = ["响频段\n|S|=0.9", "弱频段\n|S|=0.05"]
    xpos = np.arange(2)
    w = 0.36
    lin_pair = np.array([0.9, 0.05])
    pow_pair = np.array([0.9, 0.05]) ** 0.3
    axb.bar(xpos - w / 2, lin_pair / lin_pair.max(), w,
            color=C_PAD, label="线性 MSE")
    axb.bar(xpos + w / 2, pow_pair / pow_pair.max(), w,
            color=C_WAVE, label="压缩后")
    axb.set_xticks(xpos); axb.set_xticklabels(labels, fontsize=9.5)
    axb.set_ylabel("在损失里的相对权重")
    axb.set_title("同一对响/弱频段：压缩后差距被拉平", fontsize=11.5, color=C_INK)
    axb.legend(fontsize=9.5)
    axb.grid(alpha=0.25, axis="y")
    axb.text(1, pow_pair[1] / pow_pair.max() + 0.05, "从近乎被忽略\n→ 抬起来",
             ha="center", fontsize=9, color=C_WAVE)

    fig.suptitle("压缩幅度再算 MSE：治「对能量大处偏心」这第一宗罪，一行 .pow(0.3) 的事",
                 fontsize=12.5, color=C_INK, y=1.02)
    fig.subplots_adjust(wspace=0.26, bottom=0.18)
    save(fig, "fig08_mag_compression.png")


# ============ 图 3：Mask Loss 流程 —— mask ⊙ 带噪谱 = 重建谱 ≈ 干净谱 ============
def fig_mask_loss():
    # 造一段可视化的“谱”：干净谱是几条谐波，噪声铺底
    rng = np.random.default_rng(3)
    F, Tt = 48, 60
    ff = np.arange(F)[:, None]
    tt = np.arange(Tt)[None, :]
    clean = np.zeros((F, Tt))
    for h in (6, 12, 20, 30):
        clean += np.exp(-((ff - h) ** 2) / 6.0) * (0.6 + 0.4 * np.sin(tt / 8.0))
    noise = 0.35 * rng.random((F, Tt))
    noisy = clean + noise
    # 理想比值掩码 IRM 当作“网络学到的 mask”的示意
    mask = clean / (clean + noise + 1e-6)
    recon = mask * noisy

    fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.7))
    specs = [(noisy, "带噪谱 |Y|\n[F, T]", "magma"),
             (mask, "网络预测掩码 $\\hat{M}$\nsigmoid∈[0,1]", "viridis"),
             (recon, "重建谱 $\\hat{M}\\odot|Y|$\n[F, T]", "magma"),
             (clean, "干净谱 |S|（标签）\n[F, T]", "magma")]
    for ax, (data, title, cmap) in zip(axes, specs):
        ax.imshow(data, origin="lower", aspect="auto", cmap=cmap)
        ax.set_title(title, fontsize=10.5, color=C_INK)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("T 帧", fontsize=8.5)
    axes[0].set_ylabel("F 频率", fontsize=8.5)

    # 运算符标注（子图之间的间隙中心，随 wspace 精算）
    ops = [(0.311, "$\\odot$", C_BATCH), (0.5125, "$=$", C_INK)]
    for xf, sym, col in ops:
        fig.text(xf, 0.46, sym, ha="center", va="center", fontsize=20,
                 color=col, fontweight="bold")
    # 重建 ≈ 干净：损失就压这个差
    fig.text(0.7139, 0.46, "≈\n损失压这里", ha="center", va="center",
             fontsize=11, color=C_FREQ, fontweight="bold")

    fig.suptitle("Mask Loss：损失定义在「掩码⊙带噪谱」的重建结果上，而非掩码本身"
                 "——让重建谱逼近干净谱，网络自己学出最优掩码",
                 fontsize=12.5, color=C_INK, y=1.04)
    fig.subplots_adjust(wspace=0.18, bottom=0.12, top=0.80)
    save(fig, "fig08_mask_loss.png")


# ============ 图 4：SI-SNR 向量投影分解（真实 numpy 几何） ============
def fig_sisnr_projection():
    # 二维几何示意：目标 s，估计 est；把 est 投影到 s 方向
    s = np.array([4.0, 1.0])                 # 干净目标方向
    est = np.array([2.6, 2.4])               # 网络估计（方向对但有偏、幅度不同）
    # 投影：s_target = <est,s>/||s||^2 * s
    coef = est.dot(s) / s.dot(s)
    s_target = coef * s                       # 对齐目标分量
    e_noise = est - s_target                  # 正交残差

    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    ax.set_xlim(-0.6, 5.0); ax.set_ylim(-0.6, 3.4)
    ax.set_aspect("equal"); ax.axis("off")

    origin = np.array([0, 0])

    def vec(v, color, label, lw=2.6, lab_off=(0.08, 0.08)):
        ax.add_patch(FancyArrowPatch(origin, v, arrowstyle="-|>",
                     mutation_scale=18, color=color, lw=lw, zorder=4))
        ax.text(v[0] + lab_off[0], v[1] + lab_off[1], label, color=color,
                fontsize=12, fontweight="bold", zorder=5)

    # 目标方向（画一条延伸虚线表示“目标张成的方向”）
    ext = s / np.linalg.norm(s) * 4.8
    ax.plot([0, ext[0]], [0, ext[1]], color=C_WAVE, ls=":", lw=1.4, zorder=1)
    vec(s, C_WAVE, "目标 s")
    vec(est, C_GRAD, "估计 $\\hat{s}$")
    vec(s_target, C_CHAN, "$s_{target}$ 对齐分量", lw=3.0, lab_off=(0.05, -0.28))

    # 残差 e_noise：从 s_target 指向 est（与 s 正交）
    ax.add_patch(FancyArrowPatch(s_target, est, arrowstyle="-|>",
                 mutation_scale=16, color=C_FREQ, lw=2.4, zorder=4))
    mid = (s_target + est) / 2
    ax.text(mid[0] + 0.12, mid[1] + 0.02, "$e_{noise}$ 残差\n(与 s 正交)",
            color=C_FREQ, fontsize=11, zorder=5)

    # 直角标记
    d1 = (s_target - est); d1 = d1 / np.linalg.norm(d1) * 0.22
    d2 = (origin - s_target); d2 = d2 / np.linalg.norm(d2) * 0.22
    corner = s_target
    p1 = corner + d1; p2 = corner + d2; p3 = corner + d1 + d2
    ax.plot([p1[0], p3[0], p2[0]], [p1[1], p3[1], p2[1]],
            color=C_INK, lw=1.2, zorder=4)

    # 公式与说明
    ax.text(2.4, 3.15,
            "把估计投影到目标方向：$s_{target}=\\dfrac{\\langle\\hat{s},s\\rangle}{\\|s\\|^2}s$"
            "，增益差在这一步被吸收",
            ha="center", fontsize=11, color=C_INK)
    sisnr = 10 * np.log10((s_target ** 2).sum() / (e_noise ** 2).sum())
    ax.text(2.4, -0.5,
            "SI-SNR $=10\\log_{10}\\dfrac{\\|s_{target}\\|^2}{\\|e_{noise}\\|^2}"
            f"={sisnr:.1f}$ dB　（对齐能量 ÷ 没对上的能量，越大越干净）",
            ha="center", fontsize=10.5, color=C_INK)

    fig.suptitle("SI-SNR 的几何：估计 = 对齐目标的分量 + 与目标正交的残差",
                 fontsize=13, color=C_INK, y=0.98)
    fig.subplots_adjust(bottom=0.06, top=0.9)
    save(fig, "fig08_sisnr_projection.png")


if __name__ == "__main__":
    fig_scale_sensitivity()
    fig_mag_compression()
    fig_mask_loss()
    fig_sisnr_projection()

