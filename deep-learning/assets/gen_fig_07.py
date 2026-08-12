# -*- coding: utf-8 -*-
"""
第 7 篇《复数神经网络基础》配图生成脚本。
运行：python assets/gen_fig_07.py
输出：assets/fig07_*.png（4 张）

依赖：matplotlib、numpy、torch。中文字体用 Microsoft YaHei。
风格与第 1 篇 gen_fig_01.py 完全一致（浅底、克制配色、直觉先行）。
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # torch+mpl 的 OpenMP 冲突绕过
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc

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
C_WAVE = "#2c6fbb"     # 波形/时间/实部
C_FREQ = "#c0392b"     # 频率/幅度
C_BATCH = "#8e44ad"    # 批次
C_CHAN = "#16a085"     # 通道/虚部
C_GRAD = "#e67e22"     # 梯度/反向
C_PAD = "#bdc3c7"      # padding 灰
C_INK = "#2c3e50"      # 主文字
# 本篇新增（与 C_FREQ 同为暖色系，专表「相位」）
C_PHASE = "#d35400"    # 相位（暖橙红，和幅度同色系区分开）


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


# ============ 图 1：复平面——复数 = 幅度 + 相位，也 = 实部 + 虚部 ============
def fig_complex_plane():
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    ax.set_xlim(-1.2, 4.2); ax.set_ylim(-1.2, 4.2)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("同一个复数谱点：两种等价的读法\n幅度+相位（极坐标） = 实部+虚部（直角坐标）",
                 fontsize=13, color=C_INK, pad=10)

    # 坐标轴
    ax.annotate("", xy=(4.1, 0), xytext=(-1.1, 0),
                arrowprops=dict(arrowstyle="-|>", color=C_INK, lw=1.4))
    ax.annotate("", xy=(0, 4.1), xytext=(0, -1.1),
                arrowprops=dict(arrowstyle="-|>", color=C_INK, lw=1.4))
    ax.text(4.05, -0.28, "实轴 Re", ha="right", va="top", color=C_WAVE, fontsize=11)
    ax.text(0.12, 4.05, "虚轴 Im", ha="left", va="top", color=C_CHAN, fontsize=11)

    # 复数点 z = x + iy
    x, y = 3.0, 2.2
    r = np.hypot(x, y)
    theta = np.arctan2(y, x)

    # 幅度向量（红，表幅度）
    _arrow(ax, (0, 0), (x, y), color=C_FREQ, lw=2.6)
    ax.text(x / 2 - 0.35, y / 2 + 0.28, "幅度 mag = |z|",
            color=C_FREQ, fontsize=11.5, rotation=np.degrees(theta), ha="center")

    # 相位角弧（橙）
    arc = Arc((0, 0), 1.8, 1.8, angle=0, theta1=0, theta2=np.degrees(theta),
              color=C_PHASE, lw=2.2)
    ax.add_patch(arc)
    ax.text(1.15, 0.42, "相位 φ", color=C_PHASE, fontsize=12)

    # 实部虚部投影（虚线）
    ax.plot([x, x], [0, y], ls="--", color=C_CHAN, lw=1.4)
    ax.plot([0, x], [y, y], ls="--", color=C_WAVE, lw=1.4)
    ax.plot([x], [y], "o", color=C_INK, ms=9, zorder=5)
    ax.text(x + 0.12, y + 0.12, "z = x + iy", color=C_INK, fontsize=12.5,
            fontweight="bold")
    ax.text(x, -0.28, "实部 x", ha="center", va="top", color=C_WAVE, fontsize=11)
    ax.text(-0.15, y, "虚部 y", ha="right", va="center", color=C_CHAN, fontsize=11)

    # 关系注解
    ax.text(1.9, -0.95,
            "极坐标 → 直角坐标：x = mag·cos φ，y = mag·sin φ\n"
            "扔掉相位 φ、只留 mag，就等于把这个点「拍扁」到一条射线上——半张脸没了",
            ha="center", va="top", color=C_INK, fontsize=10, style="italic")
    save(fig, "fig07_complex_plane.png")


# ============ 图 2：ComplexLinear 接线图——一次复数乘法 = 4 次实数乘法拼起来 ============
def fig_complexlinear_wiring():
    fig, ax = plt.subplots(figsize=(10.6, 5.6))
    ax.set_xlim(0, 13.5); ax.set_ylim(0, 7.4); ax.axis("off")
    ax.set_title("ComplexLinear 的内部接线：(A+iB)(x+iy) = (Ax−By) + i(Bx+Ay)",
                 fontsize=13.5, color=C_INK, pad=10)

    # 输入：实部 x、虚部 y
    _box(ax, (1.3, 5.4), 1.7, 0.9, "实部 x\n[B, in]", C_WAVE, fs=10.5)
    _box(ax, (1.3, 2.0), 1.7, 0.9, "虚部 y\n[B, in]", C_CHAN, fs=10.5)

    # 四次实数矩阵乘法（两套 Linear：A=fc_r，B=fc_i）
    _box(ax, (4.7, 6.3), 1.5, 0.8, "A·x  (fc_r)", C_WAVE, fs=10)
    _box(ax, (4.7, 4.6), 1.5, 0.8, "B·y  (fc_i)", C_CHAN, fs=10)
    _box(ax, (4.7, 2.8), 1.5, 0.8, "B·x  (fc_i)", C_CHAN, fs=10)
    _box(ax, (4.7, 1.1), 1.5, 0.8, "A·y  (fc_r)", C_WAVE, fs=10)

    # 从输入连到四次乘法
    _arrow(ax, (2.15, 5.55), (3.95, 6.3), color=C_WAVE, rad=-0.05)   # x -> Ax
    _arrow(ax, (2.15, 5.2), (3.95, 2.85), color=C_WAVE, rad=0.12)    # x -> Bx
    _arrow(ax, (2.15, 2.2), (3.95, 4.6), color=C_CHAN, rad=-0.12)    # y -> By
    _arrow(ax, (2.15, 1.85), (3.95, 1.15), color=C_CHAN, rad=0.05)   # y -> Ay

    # 加减法节点
    _box(ax, (8.2, 5.45), 1.9, 0.95, "Ax − By", C_FREQ, fs=11.5)
    _box(ax, (8.2, 1.95), 1.9, 0.95, "Bx + Ay", C_PHASE, fs=11.5)

    # 连到加减节点
    _arrow(ax, (5.45, 6.3), (7.25, 5.65), rad=-0.05)   # Ax ->
    _arrow(ax, (5.45, 4.6), (7.25, 5.25), rad=0.05)    # By ->
    ax.text(6.35, 6.15, "−", color=C_FREQ, fontsize=18, ha="center", fontweight="bold")
    _arrow(ax, (5.45, 2.8), (7.25, 2.15), rad=-0.05)   # Bx ->
    _arrow(ax, (5.45, 1.1), (7.25, 1.75), rad=0.05)    # Ay ->
    ax.text(6.35, 1.75, "+", color=C_PHASE, fontsize=18, ha="center", fontweight="bold")

    # 输出
    _box(ax, (11.9, 5.45), 1.9, 0.95, "输出实部\n[B, out]", C_FREQ, fs=10.5)
    _box(ax, (11.9, 1.95), 1.9, 0.95, "输出虚部\n[B, out]", C_PHASE, fs=10.5)
    _arrow(ax, (9.15, 5.45), (10.95, 5.45), lw=2)
    _arrow(ax, (9.15, 1.95), (10.95, 1.95), lw=2)

    # 说明
    ax.text(6.75, 0.15,
            "只有两套实数权重 A(fc_r)、B(fc_i)：它们在实部虚部里「交叉共享」——"
            "这才是真复数变换，不是实虚各学各的",
            ha="center", fontsize=10, color=C_INK, style="italic")
    save(fig, "fig07_complexlinear_wiring.png")


# ============ 图 3：复数乘法的几何——幅度相乘（缩放）+ 相位相加（旋转） ============
def fig_complex_mul_geometry():
    # 拆成左右两个面板：左讲「相位相加=旋转」，右讲「幅度相乘=缩放」，各自留足空间
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 5.4))
    fig.suptitle("复数掩码相乘 Ŝ = M·Y：一次同时改「相位」和「幅度」",
                 fontsize=13.5, color=C_INK, y=0.99)

    def axes_frame(ax, xlim, ylim):
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_aspect("equal"); ax.axis("off")
        ax.annotate("", xy=(xlim[1], 0), xytext=(xlim[0], 0),
                    arrowprops=dict(arrowstyle="-|>", color=C_INK, lw=1.1))
        ax.annotate("", xy=(0, ylim[1]), xytext=(0, ylim[0]),
                    arrowprops=dict(arrowstyle="-|>", color=C_INK, lw=1.1))
        ax.text(xlim[1] - 0.05, -0.18, "Re", ha="right", va="top",
                color=C_INK, fontsize=9.5)
        ax.text(0.12, ylim[1] - 0.05, "Im", ha="left", va="top",
                color=C_INK, fontsize=9.5)

    # 用较大的相位差、幅度差，让两种效果都看得清
    rY, aY = 2.4, np.radians(20)          # 带噪谱 Y
    aM = np.radians(55)                    # 掩码相位（旋转量）
    rM = 0.55                              # 掩码幅度（缩放量，<1 = 变小）

    # ---------- 左面板：相位相加 = 旋转 ----------
    axL.set_title("① 相位相加 → 绕原点旋转 +φ_M", fontsize=11.5, color=C_PHASE)
    axes_frame(axL, (-1.6, 3.0), (-0.7, 3.0))
    # 旋转只改角度、幅度暂设相同长度，突出「转」
    xY, yY = rY * np.cos(aY), rY * np.sin(aY)
    xR, yR = rY * np.cos(aY + aM), rY * np.sin(aY + aM)
    _arrow(axL, (0, 0), (xY, yY), color=C_WAVE, lw=2.6)
    _arrow(axL, (0, 0), (xR, yR), color=C_PHASE, lw=2.6)
    axL.text(xY + 0.1, yY - 0.05, "带噪 Y\nφ=20°", color=C_WAVE,
             fontsize=10.5, ha="left", va="center")
    axL.text(xR - 0.1, yR + 0.12, "旋转后\nφ+φ_M", color=C_PHASE,
             fontsize=10.5, ha="right", va="bottom")
    arc = Arc((0, 0), 3.0, 3.0, angle=0, theta1=np.degrees(aY),
              theta2=np.degrees(aY + aM), color=C_PHASE, lw=2.2)
    axL.add_patch(arc)
    mid = aY + aM / 2
    axL.text(1.85 * np.cos(mid), 1.85 * np.sin(mid), "+φ_M",
             color=C_PHASE, fontsize=11, ha="center", va="center", fontweight="bold")

    # ---------- 右面板：幅度相乘 = 缩放 ----------
    axR.set_title("② 幅度相乘 → 沿射线缩放 ×|M|", fontsize=11.5, color=C_CHAN)
    axes_frame(axR, (-1.0, 3.4), (-0.7, 2.2))
    # 缩放只改长度、方向不变（用旋转后的方向，衔接左图结果）
    aS = aY + aM
    xYr, yYr = rY * np.cos(aS), rY * np.sin(aS)
    xS, yS = rM * rY * np.cos(aS), rM * rY * np.sin(aS)
    _arrow(axR, (0, 0), (xYr, yYr), color=C_PHASE, lw=2.2)
    _arrow(axR, (0, 0), (xS, yS), color=C_FREQ, lw=3.0)
    axR.text(xYr + 0.1, yYr, "旋转后长度 |Y|", color=C_PHASE,
             fontsize=10, ha="left", va="center")
    axR.text(xS + 0.12, yS - 0.12, "重建 Ŝ\n|Ŝ|=|M||Y|", color=C_FREQ,
             fontsize=10.5, ha="left", va="top")
    axR.annotate("", xy=(xS, yS), xytext=(xYr, yYr),
                 arrowprops=dict(arrowstyle="-|>", color=C_CHAN, lw=1.8, ls="--"))
    axR.text(0.5 * (xS + xYr) - 0.35, 0.5 * (yS + yYr) + 0.1,
             "×|M|=0.55\n(变小)", color=C_CHAN, fontsize=10,
             ha="right", va="center")

    fig.text(0.5, 0.02,
             "实数掩码只能沿 Y 的射线「调音量」；复数掩码多出「转角度」这一维，把相位也一次修对",
             ha="center", va="bottom", color=C_INK, fontsize=10.5, style="italic")
    fig.subplots_adjust(wspace=0.05, top=0.88, bottom=0.11)
    save(fig, "fig07_complex_mul_geometry.png")


# ============ 图 4：两种摆法——[B,F,T] complex64  vs  [B,2,F,T] 实虚双通道 ============
def fig_two_layouts():
    fig, ax = plt.subplots(figsize=(10.8, 5.0))
    ax.set_xlim(0, 13.5); ax.set_ylim(0, 6.4); ax.axis("off")
    ax.set_title("复数谱的两种摆法：论文严格复数  vs  端侧可部署",
                 fontsize=13.5, color=C_INK, pad=10)

    # 左：整体复数 [B, F, T] complex64
    ax.text(3.1, 5.9, "① 整体复数张量", ha="center", fontsize=12,
            color=C_FREQ, fontweight="bold")
    for k in range(3):
        off = k * 0.28
        ax.add_patch(FancyBboxPatch((1.3 + off, 2.4 + off), 2.6, 2.0,
                     boxstyle="square,pad=0", fc="#f2d7d2", ec=C_FREQ,
                     lw=1.6, zorder=2 + k))
    ax.text(2.6 + 0.56, 4.4 + 0.75, "[B, F, T]", ha="center", fontsize=13,
            color=C_FREQ, fontweight="bold")
    ax.text(2.6 + 0.28, 3.4, "complex64\n每格 = 1 个复数\n(实部, 虚部)",
            ha="center", va="center", fontsize=10, color=C_INK)
    ax.text(3.1, 1.75, "配 ComplexLinear：\nA、B 交叉共享，最干净", ha="center",
            va="top", fontsize=10, color=C_FREQ)
    ax.text(3.1, 0.65, "缺点：手机 NPU / 很多推理引擎\n没有原生 complex64 算子",
            ha="center", va="top", fontsize=9.5, color="#7f2d2d")

    # 中间箭头
    ax.text(6.75, 3.4, "torch.stack\n([re, im], dim=1)\n──→",
            ha="center", va="center", fontsize=10, color=C_INK)

    # 右：实虚双通道 [B, 2, F, T]
    ax.text(10.4, 5.9, "② 实虚拆成双通道", ha="center", fontsize=12,
            color=C_CHAN, fontweight="bold")
    # 实部块
    for k in range(3):
        off = k * 0.24
        ax.add_patch(FancyBboxPatch((8.5 + off, 3.4 + off), 2.0, 1.4,
                     boxstyle="square,pad=0", fc="#d6e4f0", ec=C_WAVE,
                     lw=1.5, zorder=2 + k))
    ax.text(9.5 + 0.36, 4.8 + 0.5, "实部通道", ha="center", fontsize=10,
            color=C_WAVE, fontweight="bold")
    # 虚部块
    for k in range(3):
        off = k * 0.24
        ax.add_patch(FancyBboxPatch((8.5 + off, 1.5 + off), 2.0, 1.4,
                     boxstyle="square,pad=0", fc="#d4ede8", ec=C_CHAN,
                     lw=1.5, zorder=2 + k))
    ax.text(9.5 + 0.36, 1.2, "虚部通道", ha="center", fontsize=10,
            color=C_CHAN, fontweight="bold")
    ax.text(11.9, 3.5, "[B, 2, F, T]\n全 float32\n普通实数网络\n照吃不误",
            ha="left", va="center", fontsize=10.5, color=C_CHAN, fontweight="bold")
    ax.text(10.4, 0.55, "优点：定点化/端侧友好，靠网络自学实虚耦合（对严格复数层的折中）",
            ha="center", va="top", fontsize=9.5, color="#14532d")

    save(fig, "fig07_two_layouts.png")


def main():
    fig_complex_plane()
    fig_complexlinear_wiring()
    fig_complex_mul_geometry()
    fig_two_layouts()


if __name__ == "__main__":
    main()
