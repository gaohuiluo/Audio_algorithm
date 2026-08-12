# -*- coding: utf-8 -*-
"""
第 9 篇《优化器 AdamW：怎么下山，是一门越来越聪明的手艺》配图生成脚本。
运行：python assets/gen_fig_09.py
输出：assets/fig09_*.png（5 张）

依赖：matplotlib、numpy、torch。中文字体用 Microsoft YaHei。
所有图统一走全局风格，配色沿用系列色板；不同优化器取不同颜色但同出一板。
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

# 优化器专用色（都从统一色板取值）
C_SGD = C_FREQ         # SGD 用红（问题色）
C_MOM = C_GRAD         # Momentum 用橙
C_ADAM = C_WAVE        # Adam 用蓝
C_ADAMW = C_CHAN       # AdamW 用青绿（Adam 同系新增）
C_DECAY = C_BATCH      # 权重衰减概念用紫


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


# ============ 2D 损失面与优化器数值迭代（供轨迹图真实使用） ============
def _contour(ax, A, xlim, ylim, n=240):
    """画二次型 f = 0.5*(A[0]*x^2 + A[1]*y^2) 的等高线。"""
    xs = np.linspace(*xlim, n)
    ys = np.linspace(*ylim, n)
    X, Y = np.meshgrid(xs, ys)
    Z = 0.5 * (A[0] * X ** 2 + A[1] * Y ** 2)
    levels = np.linspace(Z.min(), Z.max(), 14) ** 1
    ax.contour(X, Y, Z, levels=levels, colors="#b8c4d0", linewidths=0.8, zorder=1)
    ax.plot(0, 0, marker="*", ms=16, color="#f1c40f",
            mec=C_INK, mew=0.8, zorder=6)


def _run_sgd(A, start, lr, steps, momentum=0.0):
    p = np.array(start, dtype=float)
    v = np.zeros(2)
    path = [p.copy()]
    for _ in range(steps):
        g = np.array([A[0] * p[0], A[1] * p[1]])
        v = momentum * v + g
        p = p - lr * v
        path.append(p.copy())
    return np.array(path)


def _run_adam(A, start, lr, steps, b1=0.9, b2=0.999, eps=1e-8):
    p = np.array(start, dtype=float)
    m = np.zeros(2); v = np.zeros(2)
    path = [p.copy()]
    for t in range(1, steps + 1):
        g = np.array([A[0] * p[0], A[1] * p[1]])
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g ** 2
        mh = m / (1 - b1 ** t)
        vh = v / (1 - b2 ** t)
        p = p - lr * mh / (np.sqrt(vh) + eps)
        path.append(p.copy())
    return np.array(path)


# ============ 图 1：SGD 学习率两难——太大震荡 / 太小龟速 ============
def fig_sgd_dilemma():
    # 一维抛物线碗 f = 0.5 x^2，梯度 g = x，更新 x <- x - lr*x
    xs = np.linspace(-10, 10, 400)
    f = 0.5 * xs ** 2

    def descend(x0, lr, steps):
        x = float(x0); pts = [x]
        for _ in range(steps):
            x = x - lr * x
            pts.append(x)
        return np.array(pts)

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.1), sharey=True)
    configs = [
        ("学习率太大 (η=2.1)：来回横跳，发散", 2.1, 7, C_SGD),
        ("学习率刚好 (η=0.6)：几步到底", 0.6, 8, C_ADAM),
        ("学习率太小 (η=0.05)：龟速挪动", 0.05, 8, C_MOM),
    ]
    for ax, (title, lr, steps, col) in zip(axes, configs):
        ax.plot(xs, f, color="#b8c4d0", lw=2, zorder=1)
        path = descend(9.0, lr, steps)
        fpath = 0.5 * path ** 2
        for i in range(len(path) - 1):
            _arrow(ax, (path[i], fpath[i]), (path[i + 1], fpath[i + 1]),
                   color=col, lw=1.6, rad=0.0)
        ax.scatter(path, fpath, color=col, s=22, zorder=4)
        ax.plot(0, 0, marker="*", ms=16, color="#f1c40f", mec=C_INK, mew=0.8, zorder=5)
        ax.set_title(title, fontsize=11, color=C_INK)
        ax.set_xlabel("参数 θ"); ax.set_xlim(-11, 11); ax.set_ylim(-4, 58)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("损失 L(θ)")
    axes[0].text(0, 50, "冲过头\n越跳越高", ha="center", fontsize=9, color=C_SGD)
    axes[2].text(6.5, 30, "几万步\n还没到底", ha="center", fontsize=9, color=C_MOM)
    fig.suptitle("SGD 的两难：同一个碗，步长定生死——梯度只给方向，不给合适的步子",
                 fontsize=13.5, color=C_INK, y=1.02)
    fig.subplots_adjust(wspace=0.08, bottom=0.14)
    save(fig, "fig09_sgd_dilemma.png")


# ============ 图 2：Momentum 加惯性——冲过鞍点 + 抑制横向震荡 ============
def fig_momentum():
    # 狭长峡谷型损失：y 方向陡（易震荡）、x 方向缓（要前进）。A=[ax, ay]
    A = [0.3, 22.0]
    start = [9.5, 2.4]
    lr = 0.085
    p_sgd = _run_sgd(A, start, lr, 45, momentum=0.0)
    p_mom = _run_sgd(A, start, lr, 45, momentum=0.9)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.6), sharey=True)
    for ax, path, col, name in [
        (ax1, p_sgd, C_SGD, "SGD：陡方向来回横跳，进展慢"),
        (ax2, p_mom, C_MOM, "Momentum：横向震荡相消，纵向下山加速"),
    ]:
        _contour(ax, A, (-10.5, 10.5), (-3.2, 3.2))
        ax.plot(path[:, 0], path[:, 1], color=col, lw=1.6, zorder=4)
        ax.scatter(path[:, 0], path[:, 1], color=col, s=14, zorder=5)
        ax.scatter([start[0]], [start[1]], color=C_INK, s=45, marker="o",
                   zorder=6, label="起点")
        ax.set_title(name, fontsize=11.5, color=C_INK)
        ax.set_xlabel("参数 θ1（缓坡方向）")
        ax.set_xlim(-10.5, 10.5); ax.set_ylim(-3.2, 3.2)
    ax1.set_ylabel("参数 θ2（陡坡方向）")
    ax1.text(0, 2.7, "球没惯性，被陡壁\n反复弹来弹去", ha="center", fontsize=9, color=C_SGD)
    ax2.text(0, 2.7, "攒着冲劲，一路\n平滑滚向谷底 ★", ha="center", fontsize=9, color=C_MOM)
    fig.suptitle("Momentum：给下山加惯性——v ← βv + g，冲过小坑鞍点，压住横向震荡",
                 fontsize=13.5, color=C_INK, y=1.01)
    fig.subplots_adjust(wspace=0.08, bottom=0.14)
    save(fig, "fig09_momentum.png")


# ============ 图 3：三种优化器下山轨迹 + 收敛曲线对比 ============
def fig_race():
    A = [0.6, 14.0]
    start = [9.0, 2.6]
    steps = 60
    p_sgd = _run_sgd(A, start, 0.06, steps, momentum=0.0)
    p_mom = _run_sgd(A, start, 0.06, steps, momentum=0.9)
    p_adam = _run_adam(A, start, 0.35, steps)

    def loss_of(path):
        return 0.5 * (A[0] * path[:, 0] ** 2 + A[1] * path[:, 1] ** 2)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.4, 4.8),
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    # 左：等高线上的三条下山路径
    _contour(axL, A, (-10.5, 10.5), (-3.4, 3.4))
    for path, col, name in [
        (p_sgd, C_SGD, "SGD"),
        (p_mom, C_MOM, "Momentum"),
        (p_adam, C_ADAM, "Adam"),
    ]:
        axL.plot(path[:, 0], path[:, 1], color=col, lw=1.8, zorder=4, label=name)
        axL.scatter(path[::4, 0], path[::4, 1], color=col, s=12, zorder=5)
    axL.scatter([start[0]], [start[1]], color=C_INK, s=45, zorder=6)
    axL.text(start[0], start[1] + 0.35, "同一起点", ha="center", fontsize=9, color=C_INK)
    axL.set_title("同一个损失面，三种下山路径", fontsize=11.5, color=C_INK)
    axL.set_xlabel("参数 θ1"); axL.set_ylabel("参数 θ2")
    axL.set_xlim(-10.5, 10.5); axL.set_ylim(-3.4, 3.4)
    axL.legend(loc="lower left", fontsize=9.5, framealpha=0.9)

    # 右：loss 下降曲线（对数纵轴）
    for path, col, name in [
        (p_sgd, C_SGD, "SGD"),
        (p_mom, C_MOM, "Momentum"),
        (p_adam, C_ADAM, "Adam"),
    ]:
        axR.plot(loss_of(path), color=col, lw=2, label=name)
    axR.set_yscale("log")
    axR.set_title("loss 下降对比：Adam 前几十步就俯冲", fontsize=11.5, color=C_INK)
    axR.set_xlabel("迭代步 step"); axR.set_ylabel("损失 L（对数轴）")
    axR.grid(alpha=0.25, which="both")
    axR.legend(fontsize=9.5)
    fig.suptitle("下山进化史：Adam = Momentum 的惯性方向 + 每参数自适应步长，收敛最快",
                 fontsize=13.5, color=C_INK, y=1.02)
    fig.subplots_adjust(wspace=0.24, bottom=0.14)
    save(fig, "fig09_race.png")


# ============ 图 4：AdamW 把 weight decay 解耦（错接 vs 正确） ============
def fig_decouple():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.7), sharey=True)
    for ax in (ax1, ax2):
        ax.set_xlim(0, 10); ax.set_ylim(0, 6.2); ax.axis("off")

    # 左：Adam（L2 正则）——衰减混进梯度，被自适应分母扭曲
    ax1.set_title("Adam：把权重衰减混进梯度（接错的线）", fontsize=11.5, color=C_SGD)
    _box(ax1, (2.0, 5.2), 2.4, 0.85, "真实梯度 g", C_ADAM, fs=10.5)
    _box(ax1, (2.0, 3.9), 2.4, 0.85, "衰减项 λθ", C_DECAY, fs=10.5)
    _box(ax1, (5.4, 4.55), 1.9, 1.0, "混进\ng + λθ", "#7f8c8d", fs=10.5)
    _arrow(ax1, (3.2, 5.2), (4.5, 4.75), color=C_ADAM, rad=-0.1)
    _arrow(ax1, (3.2, 3.9), (4.5, 4.35), color=C_DECAY, rad=0.1)
    _box(ax1, (8.1, 4.55), 2.2, 1.15, "÷ √v+ε\n自适应分母", C_SGD, fs=10)
    _arrow(ax1, (6.35, 4.55), (7.0, 4.55), color=C_INK)
    ax1.text(5.0, 1.9,
             "衰减项一起被分母缩放：\n梯度大的参数，衰减反被除小\n→ 该重罚的没罚到，线接错了",
             ha="center", fontsize=10, color=C_SGD, linespacing=1.5)

    # 右：AdamW——衰减独立施加，不经分母
    ax2.set_title("AdamW：权重衰减独立施加（修对的线）", fontsize=11.5, color=C_ADAMW)
    _box(ax2, (2.0, 5.2), 2.4, 0.85, "真实梯度 g", C_ADAM, fs=10.5)
    _box(ax2, (5.2, 5.2), 2.2, 1.0, "÷ √v+ε\n自适应分母", C_ADAM, fs=10)
    _arrow(ax2, (3.2, 5.2), (4.1, 5.2), color=C_ADAM)
    _box(ax2, (8.2, 5.2), 1.9, 1.0, "Adam\n更新量", C_ADAM, fs=10.5)
    _arrow(ax2, (6.3, 5.2), (7.25, 5.2), color=C_ADAM)
    _box(ax2, (2.0, 2.6), 2.4, 0.85, "衰减项 λθ", C_DECAY, fs=10.5)
    _box(ax2, (8.2, 2.6), 2.2, 0.95, "直接拉向零\n(绕过分母)", C_DECAY, fs=9.5)
    _arrow(ax2, (3.2, 2.6), (7.1, 2.6), color=C_DECAY)
    _box(ax2, (8.2, 3.9), 1.0, 0.7, "θ ←", C_INK, fs=11)
    _arrow(ax2, (8.2, 4.7), (8.2, 4.25), color=C_INK)
    _arrow(ax2, (8.2, 3.08), (8.2, 3.55), color=C_INK)
    ax2.text(5.0, 1.15,
             "衰减不进分母、独立作用：\n每个参数衰减力度整齐一致 → 泛化更好",
             ha="center", fontsize=10, color=C_ADAMW, linespacing=1.5)

    fig.suptitle("AdamW 的唯一改动：权重衰减这根线，别混进梯度被自适应缩放，让它独立施加",
                 fontsize=13.5, color=C_INK, y=1.0)
    fig.subplots_adjust(wspace=0.06)
    save(fig, "fig09_decouple.png")


# ============ 图 5：学习率调度 warmup + cosine ============
def fig_schedule():
    lr_max = 1e-3
    total = 200
    warmup = 20
    steps = np.arange(total)
    lr = np.zeros(total)
    # warmup：线性从 0 升到 lr_max
    lr[:warmup] = lr_max * (steps[:warmup] + 1) / warmup
    # cosine：从 lr_max 沿余弦平滑降到接近 0
    prog = (steps[warmup:] - warmup) / (total - warmup)
    lr[warmup:] = 0.5 * lr_max * (1 + np.cos(np.pi * prog))

    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    ax.plot(steps[:warmup + 1], lr[:warmup + 1], color=C_GRAD, lw=2.6,
            label="warmup（线性升温）")
    ax.plot(steps[warmup:], lr[warmup:], color=C_ADAMW, lw=2.6,
            label="cosine（余弦衰减）")
    ax.axvline(warmup, color=C_INK, ls="--", lw=1, alpha=0.6)
    ax.fill_between(steps[:warmup + 1], 0, lr[:warmup + 1], color=C_GRAD, alpha=0.12)
    ax.fill_between(steps[warmup:], 0, lr[warmup:], color=C_ADAMW, alpha=0.12)

    ax.scatter([warmup], [lr_max], color=C_INK, s=40, zorder=5)
    ax.annotate("峰值 lr", xy=(warmup, lr_max), xytext=(warmup + 22, lr_max * 0.95),
                fontsize=10, color=C_INK,
                arrowprops=dict(arrowstyle="->", color=C_INK))
    ax.text(warmup / 2, lr_max * 0.35,
            "初期步子太大\n易冲飞 → 慢慢升",
            ha="center", fontsize=9.5, color=C_GRAD, linespacing=1.4)
    ax.text(125, lr_max * 0.55,
            "末期沿余弦降到近零\n在山谷里精细收敛",
            ha="center", fontsize=9.5, color=C_ADAMW, linespacing=1.4)

    ax.set_title("学习率调度：warmup 升温 + cosine 衰减（和优化器正交的辅助旋钮）",
                 fontsize=13, color=C_INK, pad=10)
    ax.set_xlabel("迭代步 step"); ax.set_ylabel("学习率 lr")
    ax.set_xlim(0, total); ax.set_ylim(0, lr_max * 1.12)
    ax.grid(alpha=0.25); ax.legend(fontsize=10, loc="upper right")
    fig.subplots_adjust(bottom=0.13)
    save(fig, "fig09_schedule.png")


if __name__ == "__main__":
    fig_sgd_dilemma()
    fig_momentum()
    fig_race()
    fig_decouple()
    fig_schedule()
