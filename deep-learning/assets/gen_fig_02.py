# -*- coding: utf-8 -*-
"""
第 2 篇《nn.Module 与数据管道》配图生成脚本。
运行：python assets/gen_fig_02.py
输出：assets/fig02_*.png（5 张）

依赖：matplotlib、numpy、torch。中文字体用 Microsoft YaHei。
风格与第 1 篇 gen_fig_01.py 完全一致（浅底、克制配色、直觉先行）。
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # torch+mpl 的 OpenMP 冲突绕过
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

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
C_HID = "#5d6d7e"      # 隐藏层/中间维（新增，同灰蓝色系，不冲突）


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


# ============ 图 1：nn.Module = 带旋钮的函数，forward 你写、backward 白送 ============
def fig_module_knobs():
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
    ax.set_title("nn.Module = 一台「带旋钮的调音台」：你只写 forward（正向走线），backward（反向梯度）自动白送",
                 fontsize=12.5, color=C_INK, pad=12)

    # 中间的大容器：调音台 / nn.Module
    cx, cy, cw, ch = 6.0, 3.1, 4.6, 2.6
    ax.add_patch(FancyBboxPatch((cx - cw / 2, cy - ch / 2), cw, ch,
                 boxstyle="round,pad=0.02,rounding_size=0.12",
                 fc="#eef3f8", ec=C_INK, lw=2.0, zorder=2))
    ax.text(cx, cy + ch / 2 - 0.28, "nn.Module（调音台）", ha="center",
            fontsize=12, color=C_INK, fontweight="bold", zorder=4)

    # 旋钮：几个圆当可学习参数
    knob_x = [4.55, 5.55, 6.55, 7.55]
    for i, kx in enumerate(knob_x):
        ax.add_patch(Circle((kx, cy - 0.15), 0.34, fc=C_BATCH, ec="white",
                     lw=1.5, zorder=3))
        ax.add_patch(FancyArrowPatch((kx, cy - 0.15), (kx + 0.22, cy + 0.12),
                     arrowstyle="-", lw=1.8, color="white", zorder=4))
    ax.text(cx, cy - 0.9, "可学习参数（旋钮）：训练就是自动拧到「该拧的位置」",
            ha="center", fontsize=9.5, color=C_BATCH, zorder=4)

    # 输入 / 输出
    _box(ax, (1.4, cy), 1.9, 0.95, "输入张量\n[B, T, F]", C_WAVE, fs=10.5)
    _box(ax, (10.6, cy), 1.9, 0.95, "输出 mask\n[B, T, F]", C_CHAN, fs=10.5)

    # 前向箭头（蓝，上方）
    _arrow(ax, (2.4, cy + 0.35), (cx - cw / 2 - 0.02, cy + 0.35), color=C_WAVE, lw=2.2)
    _arrow(ax, (cx + cw / 2 + 0.02, cy + 0.35), (9.6, cy + 0.35), color=C_WAVE, lw=2.2)
    ax.text(cx, 5.05, "forward：数据正着流（你亲手写的走线）→",
            ha="center", fontsize=11, color=C_WAVE)

    # 反向箭头（橙，虚线，下方）
    _arrow(ax, (9.6, cy - 0.75), (cx + cw / 2 + 0.02, cy - 0.75),
           color=C_GRAD, ls="--", lw=2.0)
    _arrow(ax, (cx - cw / 2 - 0.02, cy - 0.75), (2.4, cy - 0.75),
           color=C_GRAD, ls="--", lw=2.0)
    ax.text(cx, 1.35, "← backward：梯度反着流（autograd 白送，你永远不用手写）",
            ha="center", fontsize=11, color=C_GRAD)

    # 两个部分标注
    ax.text(1.4, 5.4, "__init__：安装旋钮\n（注册各层，自动登记参数）",
            ha="center", fontsize=9.5, color=C_INK, va="center")
    ax.text(10.6, 5.4, "forward：定义走线\n（数据怎么一层层流过）",
            ha="center", fontsize=9.5, color=C_INK, va="center")
    save(fig, "fig02_module_knobs.png")


# ============ 图 2：forward 数据流——Linear 只动最后一维，B/T 不变 ============
def fig_forward_flow():
    fig, ax = plt.subplots(figsize=(11.2, 4.4))
    ax.set_xlim(0, 15.5); ax.set_ylim(0, 5.2); ax.axis("off")
    ax.set_title("跑一次 forward：Linear 把「最后一维」当特征向量逐帧变换——B、T 全程不动，只有特征维 F 在变",
                 fontsize=12, color=C_INK, pad=10)

    y = 3.1
    # (节点文本, 特征维标签, 颜色, 宽)
    stages = [
        ("输入\n带噪谱", "[4, 126, 257]", C_WAVE, 1.7),
        ("Linear\n257→512", "[4, 126, 512]", C_HID, 1.7),
        ("ReLU", "[4, 126, 512]", "#95a5a6", 1.35),
        ("Linear\n512→512", "[4, 126, 512]", C_HID, 1.7),
        ("Linear\n512→257", "[4, 126, 257]", C_HID, 1.7),
        ("Sigmoid\n压到[0,1]", "mask\n[4, 126, 257]", C_CHAN, 1.7),
    ]
    xs = np.linspace(1.4, 14.1, len(stages))
    for i, ((txt, shp, col, w), x) in enumerate(zip(stages, xs)):
        _box(ax, (x, y), w, 1.0, txt, col, fs=9.5)
        # 特征维标注在下方
        ax.text(x, y - 1.0, shp, ha="center", va="top", fontsize=8.8,
                color=C_INK, fontweight="bold")
        if i > 0:
            _arrow(ax, (xs[i - 1] + w / 2 - 0.15, y), (x - w / 2 + 0.05, y), lw=1.8)

    # 中间省略 ReLU 之后接第二个 Linear，用小注释串起
    ax.text(xs[2], y + 0.95, "掰弯", ha="center", fontsize=8.5, color="#7f8c8d")

    # 底部大注释：B、T 不动
    ax.text(7.7, 0.75,
            "看「最后一维」：257→512→512→257 一路在变；而 B=4、T=126 从头到尾纹丝不动",
            ha="center", fontsize=10.5, color=C_FREQ, style="italic")
    # 顶部注释：为何用 [B,T,F]
    ax.text(7.7, 4.55,
            "维度顺序 [B, T, F]：把每一帧的 257 维频谱当一个特征向量喂给全连接（用什么层，就把它要处理的轴放最后）",
            ha="center", fontsize=9.8, color=C_INK)
    save(fig, "fig02_forward_flow.png")


# ============ 图 3：参数自动登记——从 named_parameters 抓真实 shape 画嵌套树 ============
def fig_param_registry():
    # 用正文那个真实网络，跑出真实参数名/shape/数量
    net = nn.Sequential(
        nn.Linear(257, 512), nn.ReLU(),
        nn.Linear(512, 512), nn.ReLU(),
        nn.Linear(512, 257), nn.Sigmoid(),
    )
    named = [(n, tuple(p.shape), p.numel()) for n, p in net.named_parameters()]
    total = sum(x[2] for x in named)

    fig, ax = plt.subplots(figsize=(10.4, 5.0))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.2); ax.axis("off")
    ax.set_title("你只是 self.net = nn.Sequential(...)，参数就被「自动登记」进一棵可遍历的树",
                 fontsize=12.5, color=C_INK, pad=12)

    # 根节点：model
    _box(ax, (2.0, 5.3), 2.4, 0.75, "model\n(DenoiseMaskNet)", C_INK, fs=10)
    # 中间层：self.net (Sequential)
    _box(ax, (2.0, 4.0), 2.4, 0.7, "self.net\n(Sequential)", C_WAVE, fs=10)
    _arrow(ax, (2.0, 4.94), (2.0, 4.38), lw=1.6)

    # 叶子：各 Linear 的 weight/bias（真实 shape）
    leaf_y = np.linspace(3.15, 0.55, len(named))
    for (name, shp, num), ly in zip(named, leaf_y):
        _box(ax, (2.0, ly), 2.4, 0.42, name, C_BATCH, fs=8.5, r=0.04)
        _arrow(ax, (2.0, 3.62), (1.05, ly + 0.05), color=C_HID, lw=1.0, rad=0.18)
        # 右侧列出 shape 与数量
        ax.text(3.5, ly, f"shape = {shp}", ha="left", va="center",
                fontsize=9.5, color=C_INK)
        ax.text(7.9, ly, f"{num:,} 个", ha="left", va="center",
                fontsize=9.5, color=C_FREQ)

    # 汇总条
    ax.text(3.5, 0.12, "model.parameters() 一把抓全部 →",
            ha="left", fontsize=9.5, color=C_INK, style="italic")
    ax.text(7.9, 0.12, f"合计 {total:,} 个", ha="left", fontsize=10.5,
            color=C_FREQ, fontweight="bold")

    # 右侧大括注：这三件苦活
    ax.text(9.9, 4.6,
            "自动登记的红利：\n① parameters() 交给优化器\n② state_dict 一键存/取\n③ .to()/.train()/.eval()\n   一句操作整棵树",
            ha="left", va="center", fontsize=10, color=C_INK,
            linespacing=1.6,
            bbox=dict(boxstyle="round,pad=0.5", fc="#f4ecf7", ec=C_BATCH, lw=1.2))
    save(fig, "fig02_param_registry.png")


# ============ 图 4：mask 是一层「透光度膜」——用真实 istft 前的谱数据演示乘法压噪 ============
def fig_mask_concept():
    # 造一张有结构的"干净谱"（几条谐波带）+ 噪声，再画一个学好的理想 mask 去噪
    F, T = 96, 120
    rng = np.random.default_rng(0)
    freqs = np.linspace(0, 1, F)[:, None]
    time = np.linspace(0, 1, T)[None, :]
    # 干净：三条随时间漂移的谐波带
    clean = np.zeros((F, T))
    for k, base in enumerate([0.12, 0.30, 0.52]):
        center = base + 0.05 * np.sin(2 * np.pi * (time + 0.1 * k))
        clean += np.exp(-((freqs - center) ** 2) / (2 * 0.015 ** 2))
    clean *= 0.9
    noise = 0.35 * rng.random((F, T))
    noisy = clean + noise
    # 理想 mask：干净能量占比（越像人声越接近 1）
    mask = clean / (noisy + 1e-6)
    mask = np.clip(mask, 0, 1)
    est = mask * noisy

    fig, axes = plt.subplots(1, 4, figsize=(12.6, 3.5))
    titles = ["带噪谱 noisy\n(输入)", "网络输出 mask\n[0,1] 透光度膜",
              "乘  ×", "估计干净谱\nest = mask × noisy"]
    data = [noisy, mask, None, est]
    cmaps = ["magma", "viridis", None, "magma"]
    for ax, title, d, cm in zip(axes, titles, data, cmaps):
        if d is None:
            ax.axis("off")
            ax.text(0.5, 0.5, "×", ha="center", va="center",
                    fontsize=40, color=C_INK)
            ax.set_title(title, fontsize=10.5, color=C_INK)
            continue
        ax.imshow(d, origin="lower", aspect="auto", cmap=cm,
                  extent=[0, T, 0, F], vmin=0, vmax=(1 if cm == "viridis" else None))
        ax.set_title(title, fontsize=10.5, color=C_INK)
        ax.set_xlabel("帧 T", fontsize=9)
        ax.set_yticks([])
    axes[0].set_ylabel("频率 F", fontsize=9)

    fig.suptitle("mask（masking）：网络吐一层「透光度不同的膜」，噪声点调到≈0（挡住）、人声点≈1（放行），乘回带噪谱就压掉噪声",
                 fontsize=11.8, color=C_INK, y=1.03)
    fig.text(0.5, -0.02,
             "Sigmoid 把网络输出天然压进 [0,1]，正好当「透光度」——这就是频域降噪/分离最主流的范式",
             ha="center", fontsize=10, color=C_FREQ, style="italic")
    fig.subplots_adjust(wspace=0.18, bottom=0.16)
    save(fig, "fig02_mask_concept.png")


# ============ 图 5：Dataset→DataLoader 传送带 + collate_fn padding 组批 ============
def fig_dataloader_belt():
    fig, (axT, axB) = plt.subplots(2, 1, figsize=(11.4, 6.2),
                                   gridspec_kw={"height_ratios": [1, 1.15]})
    # ---------- 上半：传送带流程 ----------
    axT.set_xlim(0, 14); axT.set_ylim(0, 4); axT.axis("off")
    axT.set_title("Dataset 定义「单条长啥样」→ DataLoader 成批打乱预取 → collate_fn 在打包这一刻 padding 对齐",
                  fontsize=12, color=C_INK, pad=8)

    # Dataset：变长的单条样本
    lens_demo = [95, 112, 88]
    for i, L in enumerate(lens_demo):
        y = 3.0 - i * 0.95
        w = L / 95 * 1.7
        axT.add_patch(FancyBboxPatch((0.4, y), w, 0.6, boxstyle="square,pad=0",
                      fc=C_WAVE, ec=C_WAVE, zorder=2))
        axT.text(0.4 + w / 2, y + 0.3, f"[{L},257]", ha="center", va="center",
                 fontsize=8, color="white", zorder=3)
    axT.text(1.4, 3.85, "Dataset\n__getitem__ 单条(变长!)", ha="center",
             fontsize=9.5, color=C_INK)

    # 箭头到 DataLoader
    axT.add_patch(FancyArrowPatch((3.2, 2.0), (4.5, 2.0), arrowstyle="-|>",
                  mutation_scale=16, lw=2, color=C_INK))
    _box(axT, (5.6, 2.0), 2.0, 1.1,
         "DataLoader\nbatch_size=8\nshuffle 打乱\nnum_workers 预取", C_HID, fs=9)

    # 箭头到 collate_fn
    axT.add_patch(FancyArrowPatch((6.7, 2.0), (8.0, 2.0), arrowstyle="-|>",
                  mutation_scale=16, lw=2, color=C_INK))
    _box(axT, (9.2, 2.0), 2.2, 1.1,
         "collate_fn\n补零对齐 Tmax\n生成 length mask", C_BATCH, fs=9.5)
    axT.text(9.2, 0.75, "默认 torch.stack 摞不齐变长 → 报错\ncollate_fn 是官方入口",
             ha="center", fontsize=8.5, color=C_FREQ)

    # 箭头到成品 batch
    axT.add_patch(FancyArrowPatch((10.4, 2.0), (11.7, 2.0), arrowstyle="-|>",
                  mutation_scale=16, lw=2, color=C_INK))
    _box(axT, (12.8, 2.0), 2.0, 1.1,
         "整齐 batch\nx [8,Tmax,257]\nmask [8,Tmax]", C_CHAN, fs=9)

    # ---------- 下半：真实 collate 跑一批，画对齐方块 + mask ----------
    torch.manual_seed(3)
    lengths = torch.randint(80, 130, (8,))
    Tmax = int(lengths.max())
    axB.set_xlim(-14, Tmax + 4); axB.set_ylim(-0.8, 8.2); axB.axis("off")
    axB.set_title(f"collate_fn 组批实况：8 条帧数不一，以最长 Tmax={Tmax} 为准末尾补零，同时记 length mask（绿=真实/灰=padding）",
                  fontsize=10.8, color=C_INK)

    for i, L in enumerate(lengths.tolist()):
        y = 7 - i * 0.9
        # 真实部分
        axB.add_patch(FancyBboxPatch((0, y), L, 0.6, boxstyle="square,pad=0",
                      fc=C_WAVE, ec=C_WAVE, zorder=2))
        # padding
        if L < Tmax:
            axB.add_patch(FancyBboxPatch((L, y), Tmax - L, 0.6,
                          boxstyle="square,pad=0", fc=C_PAD, ec="#95a5a6",
                          hatch="///", zorder=2))
        axB.text(-1.5, y + 0.3, f"条{i+1}:{L}帧", ha="right", va="center",
                 fontsize=8.5, color=C_INK)
        # 右侧 mask 条
        mx = Tmax + 0.6
        axB.add_patch(FancyBboxPatch((mx, y + 0.12), L / Tmax * 2.2, 0.36,
                      boxstyle="square,pad=0", fc="#27ae60", ec="none", zorder=2))
        if L < Tmax:
            axB.add_patch(FancyBboxPatch((mx + L / Tmax * 2.2, y + 0.12),
                          (Tmax - L) / Tmax * 2.2, 0.36, boxstyle="square,pad=0",
                          fc=C_PAD, ec="none", zorder=2))

    axB.annotate("", xy=(Tmax, 7.9), xytext=(0, 7.9),
                 arrowprops=dict(arrowstyle="<->", color=C_INK))
    axB.text(Tmax / 2, 8.05, f"对齐后统一到 Tmax={Tmax}", ha="center",
             fontsize=9, color=C_INK)
    axB.text(Tmax / 2, -0.55, "真实语音（蓝）+ 补的零（灰斜纹）",
             ha="center", fontsize=9, color="#7f8c8d")
    axB.text(Tmax + 1.7, -0.55, "length\nmask", ha="center", fontsize=8.5,
             color="#27ae60")
    axB.text(-7, -0.55, "mask 一路传到第 8 篇算损失时排除 padding → 填了第 1 篇的坑",
             ha="center", fontsize=9, color=C_FREQ, style="italic")

    fig.subplots_adjust(hspace=0.35)
    save(fig, "fig02_dataloader_belt.png")


if __name__ == "__main__":
    fig_module_knobs()
    fig_forward_flow()
    fig_param_registry()
    fig_mask_concept()
    fig_dataloader_belt()
