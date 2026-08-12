# -*- coding: utf-8 -*-
"""
第 1 篇《从 STFT 到 Tensor》配图生成脚本。
运行：python assets/gen_fig_01.py
输出：assets/fig01_*.png（5 张）

依赖：matplotlib、numpy、torch。中文字体用 Microsoft YaHei。
所有图统一走 setup_style()，风格与系列基调一致（浅底、克制配色、直觉先行）。
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

# ---- 全局风格 ----
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 12
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.dpi"] = 130
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.facecolor"] = "white"

# 系列统一配色
C_WAVE = "#2c6fbb"     # 波形/时间
C_FREQ = "#c0392b"     # 频率
C_BATCH = "#8e44ad"    # 批次
C_CHAN = "#16a085"     # 通道
C_GRAD = "#e67e22"     # 梯度/反向
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


# ============ 图 1：autograd 计算图，前向织图 / 反向倒推梯度 ============
def fig_autograd():
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.2); ax.axis("off")
    ax.set_title("Tensor 是「活」的：前向织一张计算图，反向沿图倒推梯度",
                 fontsize=13.5, color=C_INK, pad=12)

    # 节点：w, x -> mul -> sum(y) -> L
    yw, yx = 4.0, 2.0
    _box(ax, (1.3, yw), 1.4, 0.9, "w\n(requires_grad)", C_BATCH, fs=10.5)
    _box(ax, (1.3, yx), 1.4, 0.9, "x\n(输入)", "#7f8c8d", fs=10.5)
    _box(ax, (4.0, 3.0), 1.5, 0.9, "* 逐元素乘", C_WAVE, fs=11)
    _box(ax, (6.6, 3.0), 1.5, 0.9, ".sum()", C_WAVE, fs=11)
    _box(ax, (9.0, 3.0), 1.3, 0.9, "y = 5", C_INK, fs=11)

    # 前向箭头（黑）
    _arrow(ax, (2.0, yw), (3.3, 3.15), rad=-0.12)
    _arrow(ax, (2.0, yx), (3.3, 2.85), rad=0.12)
    _arrow(ax, (4.75, 3.0), (5.85, 3.0))
    _arrow(ax, (7.35, 3.0), (8.35, 3.0))
    ax.text(5.0, 4.55, "前向 forward：算 y，同时把「怎么算出来的」记进计算图 →",
            color=C_WAVE, fontsize=11, ha="center")

    # 反向箭头（橙，虚线，回到 w）
    _arrow(ax, (8.35, 2.55), (7.35, 2.55), color=C_GRAD, ls="--")
    _arrow(ax, (5.85, 2.55), (4.75, 2.55), color=C_GRAD, ls="--")
    _arrow(ax, (3.3, 2.55), (2.0, 3.7), color=C_GRAD, ls="--", rad=0.15)
    ax.text(5.0, 1.15, "← 反向 backward：从 L 顺着图倒推，每个参数「该往哪调」= 梯度",
            color=C_GRAD, fontsize=11, ha="center")
    ax.text(1.3, 5.0, "w.grad = [1., 1.]  ← PyTorch 替你算好",
            color=C_GRAD, fontsize=10.5, ha="left", style="italic")

    save(fig, "fig01_autograd.png")


# ============ 图 2：STFT，波形 [T] → 复数谱 [F,T]（用正文那段 chirp 真实跑） ============
def fig_stft():
    fs, dur = 16000, 1.0
    t = torch.arange(int(fs * dur)) / fs
    f0, f1 = 300.0, 3000.0
    phase = 2 * torch.pi * (f0 * t + (f1 - f0) / (2 * dur) * t ** 2)
    wave = 0.6 * torch.sin(phase)
    n_fft, hop = 512, 128
    win = torch.hann_window(n_fft)
    spec = torch.stft(wave, n_fft=n_fft, hop_length=hop,
                      window=win, return_complex=True)
    mag = spec.abs().numpy()
    mag_db = 20 * np.log10(mag + 1e-6)
    F, T = mag.shape

    fig, (axw, axs) = plt.subplots(
        1, 2, figsize=(10.2, 3.9), gridspec_kw={"width_ratios": [1, 1.25]})

    # 左：一维波形
    axw.plot(t.numpy(), wave.numpy(), color=C_WAVE, lw=0.7)
    axw.set_title("波形 wave  shape = [T] = [16000]", fontsize=11.5, color=C_INK)
    axw.set_xlabel("时间 / s"); axw.set_ylabel("振幅")
    axw.set_xlim(0, 1); axw.grid(alpha=0.25)
    axw.text(0.5, -0.9, "一条抖动的曲线：只有「时间」一个维度",
             ha="center", color=C_WAVE, fontsize=10)

    # 右：语谱图 [F,T]
    im = axs.imshow(mag_db, origin="lower", aspect="auto", cmap="magma",
                    extent=[0, T, 0, fs / 2 / 1000])
    axs.set_title(f"复数谱取幅度  shape = [F, T] = [{F}, {T}]",
                  fontsize=11.5, color=C_INK)
    axs.set_xlabel("帧 T（一帧一张频率快照）")
    axs.set_ylabel("频率 / kHz（F 个 bin）")
    cb = fig.colorbar(im, ax=axs, fraction=0.046, pad=0.02)
    cb.set_label("能量 / dB", fontsize=9)

    fig.suptitle("torch.stft：「时间」这一维被拆成「频率 × 帧」两维",
                 fontsize=13.5, color=C_INK, y=1.02)
    # 中间大箭头
    fig.text(0.485, 0.5, "STFT\n→", ha="center", va="center",
             fontsize=15, color=C_FREQ, fontweight="bold")
    fig.subplots_adjust(wspace=0.32, bottom=0.2)
    save(fig, "fig01_stft.png")


# ============ 图 3：相位不是垃圾（幅度不动、相位打乱 → 波形失真） ============
def fig_phase():
    fs, dur = 16000, 1.0
    t = torch.arange(int(fs * dur)) / fs
    f0, f1 = 300.0, 3000.0
    phase = 2 * torch.pi * (f0 * t + (f1 - f0) / (2 * dur) * t ** 2)
    wave = 0.6 * torch.sin(phase)
    n_fft, hop = 512, 128
    win = torch.hann_window(n_fft)
    spec = torch.stft(wave, n_fft=n_fft, hop_length=hop,
                      window=win, return_complex=True)

    mag = spec.abs()
    torch.manual_seed(0)
    ph_rand = torch.rand_like(spec.angle()) * 2 * torch.pi - torch.pi
    spec_bad = mag * torch.exp(1j * ph_rand)
    wave_bad = torch.istft(spec_bad, n_fft=n_fft, hop_length=hop, window=win)

    # 取中间一小段放大看波形结构
    a, b = 6000, 6600
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.2, 4.6), sharex=True)
    ax1.plot(t[a:b].numpy(), wave[a:b].numpy(), color=C_WAVE, lw=1.1)
    ax1.set_title("原始波形：幅度 + 相位都对，结构规整（这才是能听的语音）",
                  fontsize=11, color=C_INK)
    ax1.grid(alpha=0.25); ax1.set_ylabel("振幅")

    n = min(len(wave_bad), b) - a
    ax2.plot(t[a:a + n].numpy(), wave_bad[a:a + n].numpy(), color=C_FREQ, lw=1.1)
    ax2.set_title("幅度原样保留、相位换成随机数后重建：波形彻底乱掉 → 听感是糊掉的「机器人音」",
                  fontsize=11, color=C_INK)
    ax2.grid(alpha=0.25); ax2.set_ylabel("振幅"); ax2.set_xlabel("时间 / s（放大同一段）")

    fig.suptitle("相位携带波形的对齐/结构信息——每次 .abs() 扔掉相位，心里都要「咯噔」一下",
                 fontsize=13, color=C_INK, y=1.0)
    fig.subplots_adjust(hspace=0.32)
    save(fig, "fig01_phase.png")


# ============ 图 4：shape 之旅 [T] → [F,T] → [B,C,F,T]（本篇核心，维度方块图） ============
def fig_shape_journey():
    fig, ax = plt.subplots(figsize=(10.4, 4.2))
    ax.set_xlim(0, 13); ax.set_ylim(0, 5.4); ax.axis("off")
    ax.set_title("shape 之旅：牢记这条维度顺序  [B, C, F, T]",
                 fontsize=14, color=C_INK, pad=10)

    # 阶段一：[T] 一维条
    ax.add_patch(FancyBboxPatch((0.5, 2.6), 2.0, 0.35,
                 boxstyle="round,pad=0.01,rounding_size=0.03",
                 fc=C_WAVE, ec=C_WAVE, zorder=2))
    ax.text(1.5, 3.35, "[T]", ha="center", fontsize=13, color=C_WAVE, fontweight="bold")
    ax.text(1.5, 2.15, "波形\n一维时间序列", ha="center", va="top", fontsize=9.5, color=C_INK)

    # 阶段二：[F,T] 二维格
    x0, y0, w, h = 3.9, 1.9, 2.0, 1.6
    ax.add_patch(FancyBboxPatch((x0, y0), w, h,
                 boxstyle="square,pad=0", fc="#d6e4f0", ec=C_FREQ, lw=1.8, zorder=2))
    for i in range(1, 4):
        ax.plot([x0, x0 + w], [y0 + h * i / 4] * 2, color="white", lw=0.8, zorder=3)
    for j in range(1, 5):
        ax.plot([x0 + w * j / 5] * 2, [y0, y0 + h], color="white", lw=0.8, zorder=3)
    ax.text(x0 + w / 2, y0 + h + 0.28, "[F, T]", ha="center", fontsize=13,
            color=C_FREQ, fontweight="bold")
    ax.annotate("", xy=(x0 - 0.25, y0 - 0.05), xytext=(x0 - 0.25, y0 + h + 0.05),
                arrowprops=dict(arrowstyle="<->", color=C_FREQ))
    ax.text(x0 - 0.45, y0 + h / 2, "F 频率", rotation=90, va="center", ha="right",
            fontsize=9.5, color=C_FREQ)
    ax.text(x0 + w / 2, y0 - 0.28, "T 帧", ha="center", fontsize=9.5, color=C_INK)

    # 阶段三：[B,C,F,T] 堆叠方块
    bx, by = 8.6, 1.7
    n_stack = 4
    for k in range(n_stack):
        off = k * 0.26
        fc = "#e8dff0" if k else "#d6e4f0"
        ax.add_patch(FancyBboxPatch((bx + off, by + off, ), 1.7, 1.7,
                     boxstyle="square,pad=0",
                     fc=fc, ec=C_BATCH if k else C_FREQ, lw=1.5, zorder=2 + k))
    ax.text(bx + 0.85 + 0.4, by + 1.7 + 0.55, "[B, C, F, T]", ha="center",
            fontsize=13, color=C_BATCH, fontweight="bold")
    ax.text(bx + 2.9, by + 1.5, "B 批次\n(堆 N 条)", fontsize=9.5, color=C_BATCH, va="center")
    ax.text(bx + 2.9, by + 0.4, "C 通道\n(单声道=1)", fontsize=9.5, color=C_CHAN, va="center")

    # 箭头 + 操作标注
    ax.add_patch(FancyArrowPatch((2.6, 2.78), (3.75, 2.7), arrowstyle="-|>",
                 mutation_scale=18, color=C_INK, lw=2))
    ax.text(3.2, 3.35, "torch.stft", ha="center", fontsize=10, color=C_FREQ)
    ax.text(3.2, 2.2, "时间→频率×帧", ha="center", fontsize=8.5, color="#7f8c8d")

    ax.add_patch(FancyArrowPatch((6.05, 2.7), (8.45, 2.55), arrowstyle="-|>",
                 mutation_scale=18, color=C_INK, lw=2))
    ax.text(7.2, 3.35, "unsqueeze / stack", ha="center", fontsize=10, color=C_BATCH)
    ax.text(7.2, 2.2, "补通道维 + 攒批次", ha="center", fontsize=8.5, color="#7f8c8d")

    ax.text(6.5, 0.5, "batch 维不是数学需要，是硬件需要——GPU 一次吃「一整个方块」才跑得满",
            ha="center", fontsize=10.5, color=C_INK, style="italic")
    save(fig, "fig01_shape_journey.png")


# ============ 图 5：padding + length mask（变长音频组批） ============
def fig_padding_mask():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.6, 4.2),
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    lengths = [100, 126, 88]
    Tmax = max(lengths)
    labels = ["条1: 真实 100 帧", "条2: 真实 126 帧", "条3: 真实 88 帧"]

    # 左：三条谱补零对齐成方块
    axL.set_xlim(0, Tmax + 6); axL.set_ylim(-0.6, 3); axL.axis("off")
    axL.set_title("padding：以最长的 126 帧为准，短的在时间轴末尾补零",
                  fontsize=11, color=C_INK)
    for i, (L, lab) in enumerate(zip(lengths, labels)):
        y = 2 - i * 0.85
        # 真实部分
        axL.add_patch(FancyBboxPatch((0, y), L, 0.55, boxstyle="square,pad=0",
                      fc=C_WAVE, ec=C_WAVE, zorder=2))
        # padding 部分
        if L < Tmax:
            axL.add_patch(FancyBboxPatch((L, y), Tmax - L, 0.55,
                          boxstyle="square,pad=0", fc=C_PAD, ec="#95a5a6",
                          hatch="///", zorder=2))
        axL.text(-2, y + 0.27, lab, ha="right", va="center", fontsize=9, color=C_INK)
    axL.text(Tmax / 2, -0.4, "真实语音", ha="center", color=C_WAVE, fontsize=9.5)
    axL.text((Tmax + 100) / 2 + 6, -0.4, "补的零（不是真实语音）",
             ha="center", color="#7f8c8d", fontsize=9.5)
    axL.annotate("", xy=(Tmax, 2.75), xytext=(0, 2.75),
                 arrowprops=dict(arrowstyle="<->", color=C_INK))
    axL.text(Tmax / 2, 2.9, "对齐后 T = 126", ha="center", fontsize=9, color=C_INK)

    # 右：length mask 热图
    mask = np.zeros((3, Tmax), dtype=float)
    for i, L in enumerate(lengths):
        mask[i, :L] = 1.0
    axR.imshow(mask, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
               extent=[0, Tmax, 2.5, -0.5])
    axR.set_title("length mask：True=真实帧 / False=padding",
                  fontsize=11, color=C_INK)
    axR.set_yticks([0, 1, 2]); axR.set_yticklabels(["条1", "条2", "条3"])
    axR.set_xlabel("帧 T")
    for i, L in enumerate(lengths):
        if L < Tmax:
            axR.text((L + Tmax) / 2, i, "False\n(不计损失)", ha="center", va="center",
                     fontsize=8, color="#7f2d2d")
        axR.text(L / 2, i, "True", ha="center", va="center", fontsize=9, color="#14532d")

    fig.suptitle("变长音频靠 padding 组批，但补出来的零必须用 mask 标记、算损失时排除",
                 fontsize=13, color=C_INK, y=1.01)
    fig.subplots_adjust(wspace=0.28, left=0.13)
    save(fig, "fig01_padding_mask.png")


if __name__ == "__main__":
    fig_autograd()
    fig_stft()
    fig_phase()
    fig_shape_journey()
    fig_padding_mask()
