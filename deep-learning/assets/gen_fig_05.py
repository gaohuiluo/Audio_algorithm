# -*- coding: utf-8 -*-
"""
第 5 篇《CNN 处理语谱图》配图生成脚本。
运行：python assets/gen_fig_05.py
输出：assets/fig05_*.png（5 张）

依赖：matplotlib、numpy、torch。中文字体用 Microsoft YaHei。
风格与 gen_fig_01.py 完全对齐（浅底、克制配色、直觉先行）。
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # torch+mpl 的 OpenMP 冲突绕过
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

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
C_WAVE = "#2c6fbb"     # 波形/时间/帧
C_FREQ = "#c0392b"     # 频率
C_BATCH = "#8e44ad"    # 批次
C_CHAN = "#16a085"     # 通道
C_GRAD = "#e67e22"     # 梯度/反向
C_PAD = "#bdc3c7"      # padding 灰
C_INK = "#2c3e50"      # 主文字
# 本篇新增（同色系延伸）
C_KERNEL = "#e67e22"   # 卷积核/探测器（暖色，同 C_GRAD 家族）
C_FMAP = "#16a085"     # feature map（同 C_CHAN 通道色）
C_RF = "#8e44ad"       # 感受野（同 C_BATCH 家族）
C_PAST = "#27ae60"     # 因果：过去（绿，安全）


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


# ---- 造一张“像语谱图”的假图：谐波横纹 + 一根爆破音竖条 ----
def _fake_spectrogram(F=48, T=64, seed=0):
    rng = np.random.default_rng(seed)
    img = np.zeros((F, T)) + 0.06 * rng.random((F, T))
    tt = np.arange(T)
    # 谐波：几条随时间缓升的亮纹（模拟浊音谐波、声调滑动）
    for k, base in enumerate([6, 13, 20, 27]):
        center = base + 0.12 * tt          # 谐波随时间略微上滑
        width = 1.4
        amp = 0.9 - 0.12 * k
        for f in range(F):
            img[f] += amp * np.exp(-((f - center) ** 2) / (2 * width ** 2))
    # 高频擦音雾
    img[38:, :] += 0.18 * rng.random((F - 38, T))
    # 一根贯穿全频的爆破音竖条（瞬态）
    img[:, 44] += 0.8
    img[:, 45] += 0.4
    return np.clip(img, 0, 1.2)


# ============ 图 1：语谱图=单通道灰度图，但两轴不对称、不能旋转 ============
def fig_image_view():
    spec = _fake_spectrogram()
    F, T = spec.shape
    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(10.6, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})

    # 左：把语谱图当单通道灰度图看
    im = axA.imshow(spec, origin="lower", aspect="auto", cmap="magma")
    axA.set_title("语谱图 = 单通道灰度图  [B, 1, F, T]",
                  fontsize=12, color=C_INK)
    axA.set_xlabel("图像的“宽” = 时间轴 T（帧）", color=C_WAVE)
    axA.set_ylabel("图像的“高” = 频率轴 F（bin）", color=C_FREQ)
    axA.text(0.5, F - 4, "每个像素亮度 = 该时频点的能量",
             color="white", fontsize=9.5, ha="left")
    axA.annotate("谐波亮纹", xy=(10, 13), xytext=(2, 30),
                 color="white", fontsize=9,
                 arrowprops=dict(arrowstyle="->", color="white", lw=1.2))
    axA.annotate("爆破音竖条", xy=(44, 40), xytext=(30, 44),
                 color="white", fontsize=9,
                 arrowprops=dict(arrowstyle="->", color="white", lw=1.2))

    # 右：转 90 度 → 物理意义全乱
    rot = np.rot90(spec)
    axB.imshow(rot, origin="lower", aspect="auto", cmap="magma")
    axB.set_title("转 90° 后：轴的物理意义全乱了",
                  fontsize=12, color=C_FREQ)
    axB.set_xlabel("此刻横轴成了“频率”？", color="#7f8c8d")
    axB.set_ylabel("此刻纵轴成了“时间”？", color="#7f8c8d")
    axB.text(0.5, rot.shape[0] - 6,
             "照片转 90° 还是猫；\n语谱图转 90° → 一堆\n没有物理意义的乱码",
             color="white", fontsize=9.5, ha="left", va="top", linespacing=1.4)

    fig.suptitle("语谱图能当图看，但两个轴（频率 vs 时间）含义不对称，不能随便旋转",
                 fontsize=13, color=C_INK, y=1.02)
    fig.subplots_adjust(wspace=0.32, bottom=0.14)
    save(fig, "fig05_image_view.png")


# ============ 图 2：卷积滑窗 —— 3x3 核在语谱图上扫，∑K·X 得一个数 ============
def fig_conv_slide():
    fig, ax = plt.subplots(figsize=(10.6, 4.6))
    ax.set_xlim(0, 13.2); ax.set_ylim(0, 6.2); ax.axis("off")
    ax.set_title(r"卷积滑窗：3×3 核贴在一小片上，重叠 9 个值 $\sum K\cdot X$ 相乘再相加 → 特征图上一个点",
                 fontsize=12.5, color=C_INK, pad=10)

    # ---- 左：输入语谱图网格 6x6，高亮 3x3 覆盖窗 ----
    gx0, gy0, cell = 0.6, 1.0, 0.7
    nR, nC = 6, 6
    Xin = np.array([
        [0.1, 0.2, 0.1, 0.0, 0.1, 0.2],
        [0.2, 0.9, 0.8, 0.1, 0.0, 0.1],
        [0.1, 0.8, 0.9, 0.7, 0.1, 0.0],
        [0.0, 0.1, 0.7, 0.8, 0.6, 0.1],
        [0.1, 0.0, 0.1, 0.6, 0.7, 0.2],
        [0.2, 0.1, 0.0, 0.1, 0.2, 0.1],
    ])
    for r in range(nR):
        for c in range(nC):
            v = Xin[r, c]
            x = gx0 + c * cell
            y = gy0 + (nR - 1 - r) * cell
            ax.add_patch(Rectangle((x, y), cell, cell, facecolor=plt.cm.magma(0.15 + 0.6 * v),
                                    edgecolor="white", lw=1.0, zorder=2))
            ax.text(x + cell / 2, y + cell / 2, f"{v:.1f}", ha="center", va="center",
                    fontsize=7.5, color="white", zorder=3)
    # 高亮 3x3 覆盖窗（取 r=1..3, c=1..3）
    hx = gx0 + 1 * cell
    hy = gy0 + (nR - 1 - 3) * cell
    ax.add_patch(Rectangle((hx, hy), 3 * cell, 3 * cell, fill=False,
                           edgecolor=C_KERNEL, lw=3.2, zorder=4))
    ax.text(gx0 + nC * cell / 2, gy0 - 0.55, "输入语谱图（局部 6×6）",
            ha="center", fontsize=10, color=C_INK)
    ax.text(gx0 + nC * cell / 2, gy0 + nR * cell + 0.25, "F 频率 ↑ / T 帧 →",
            ha="center", fontsize=9, color="#7f8c8d")

    # ---- 中：3x3 卷积核 ----
    kx0, ky0 = 5.7, 2.4
    K = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 4.0
    for r in range(3):
        for c in range(3):
            x = kx0 + c * cell
            y = ky0 + (2 - r) * cell
            ax.add_patch(Rectangle((x, y), cell, cell, facecolor="#fdebd0",
                                    edgecolor=C_KERNEL, lw=1.6, zorder=2))
            ax.text(x + cell / 2, y + cell / 2, f"{K[r, c]:.2f}", ha="center", va="center",
                    fontsize=8, color=C_INK, zorder=3)
    ax.text(kx0 + 1.5 * cell, ky0 + 3 * cell + 0.2, "卷积核 K\n(9 个可学习旋钮)",
            ha="center", fontsize=9.5, color=C_KERNEL, va="bottom")

    # 覆盖窗 → 核：对齐箭头
    _arrow(ax, (hx + 3 * cell + 0.05, hy + 1.5 * cell), (kx0 - 0.1, ky0 + 1.5 * cell),
           color=C_KERNEL, lw=1.6)

    # ---- 右：乘加得一个数 → feature map 一个点 ----
    ax.text(9.15, 4.9, r"$\sum_{u,v} K[u,v]\cdot X[i{+}u,\,j{+}v]$",
            fontsize=12.5, color=C_INK, ha="left")
    _arrow(ax, (kx0 + 3 * cell + 0.1, ky0 + 1.5 * cell), (10.4, 3.4), color=C_INK, lw=1.6)
    _box(ax, (11.4, 3.4), 1.5, 0.95, "= 一个数\n(匹配度)", C_FMAP, fs=10)
    ax.text(11.4, 2.6, "越像核的花纹 → 值越大\n不像 → 接近 0",
            ha="center", fontsize=8.5, color="#7f8c8d", va="top")

    ax.text(6.6, 0.35,
            "核滑遍每个位置 → 得到一整张“匹配度地图”（feature map，特征图）；同一个核滑全图 = 权重共享",
            ha="center", fontsize=10, color=C_INK, style="italic")
    save(fig, "fig05_conv_slide.png")


# ============ 图 3：手写“瞬态探测器”核 —— 撞上竖条就亮，平缓区就哑（真实 torch 跑） ============
def fig_transient_detector():
    # 复刻正文代码：造带竖条的假谱，用左负右正的差分核卷积
    torch.manual_seed(0)
    Ff, Tt = 64, 100
    fake = torch.ones(1, 1, Ff, Tt) * 0.2   # 平缓背景
    fake[:, :, :, 50] = 1.0                  # 第 50 帧插一根贯穿全频的竖条
    edge = torch.tensor([[-1., 0., 1.]]).view(1, 1, 1, 3)   # 时间方向差分核
    resp = torch.conv2d(fake, edge, padding=(0, 1))
    energy = resp.abs().sum(dim=2).squeeze().numpy()        # 每帧响应能量 [100]
    fake_img = fake.squeeze().numpy()
    resp_img = resp.squeeze().numpy()

    fig = plt.figure(figsize=(11.0, 4.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05], wspace=0.34)
    ax0 = fig.add_subplot(gs[0]); ax1 = fig.add_subplot(gs[1]); ax2 = fig.add_subplot(gs[2])

    ax0.imshow(fake_img, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax0.set_title("① 假语谱图\n第 50 帧一根爆破音竖条", fontsize=10.5, color=C_INK)
    ax0.set_xlabel("帧 T", color=C_WAVE); ax0.set_ylabel("频率 F", color=C_FREQ)

    ax1.imshow(np.abs(resp_img), origin="lower", aspect="auto", cmap="viridis")
    ax1.set_title("② 手写差分核 [-1,0,1] 的响应\n只有竖条边沿被点亮", fontsize=10.5, color=C_INK)
    ax1.set_xlabel("帧 T", color=C_WAVE)
    ax1.text(0.06, 0.9, "核 = 左负右正\n只对“沿时间的突变”敏感",
             transform=ax1.transAxes, color="white", fontsize=8.5, va="top")

    ax2.plot(energy, color=C_KERNEL, lw=1.6)
    ax2.fill_between(np.arange(len(energy)), energy, color=C_KERNEL, alpha=0.25)
    amax = int(energy.argmax())
    ax2.axvline(amax, color=C_FREQ, ls="--", lw=1.2)
    ax2.set_title("③ 每帧响应能量：竖条处爆发，其余≈0", fontsize=10.5, color=C_INK)
    ax2.set_xlabel("帧 T", color=C_WAVE); ax2.set_ylabel("|响应| 之和")
    ax2.grid(alpha=0.25)
    ax2.annotate(f"最强帧 ≈ {amax}", xy=(amax, energy[amax]),
                 xytext=(amax - 42, energy[amax] * 0.8), color=C_FREQ, fontsize=9,
                 arrowprops=dict(arrowstyle="->", color=C_FREQ))

    fig.suptitle("核 = 局部模式探测器：撞上瞬态竖条就“亮”，撞上平稳区就“哑”（真实卷积输出，非示意）",
                 fontsize=12.5, color=C_INK, y=1.02)
    fig.subplots_adjust(bottom=0.16, top=0.80)
    save(fig, "fig05_transient_detector.png")


# ============ 图 4：感受野随层数增长（同一张语谱图上画嵌套框）+ 三种撑大手段 ============
def fig_receptive_field():
    spec = _fake_spectrogram(F=48, T=64, seed=1)
    F, T = spec.shape
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(11.0, 4.5), gridspec_kw={"width_ratios": [1.25, 1]})

    # ---- 左：语谱图上，一个输出点的感受野随层叠越滚越大 ----
    axL.imshow(spec, origin="lower", aspect="auto", cmap="gray")
    axL.set_title("同一个输出点，感受野随层叠越滚越大", fontsize=11.5, color=C_INK)
    axL.set_xlabel("帧 T", color=C_WAVE); axL.set_ylabel("频率 F", color=C_FREQ)
    cx, cy = 40, 24
    # 3x3 每层 +2：RF = 3,5,7 ...（这里画 3 层）
    rfs = [1, 3, 5]  # 半径：1层看3x3(半径1)、2层5x5(半径2)、3层7x7(半径3) -> 用半径示意
    radii = [1.5, 3.5, 6.5]
    labels = ["1 层 3×3\nRF=3×3", "2 层\nRF=5×5", "3 层\nRF=7×7"]
    cols = ["#f1c40f", C_KERNEL, C_RF]
    for rad, lab, col in zip(radii, labels, cols):
        axL.add_patch(Rectangle((cx - rad, cy - rad), 2 * rad, 2 * rad, fill=False,
                                edgecolor=col, lw=2.4, zorder=4))
    axL.plot(cx, cy, "o", color="#e74c3c", ms=6, zorder=5)
    axL.text(cx + 0.6, cy + 0.6, "输出点", color="#e74c3c", fontsize=8.5, zorder=5)
    # 图例
    for i, (lab, col) in enumerate(zip(labels, cols)):
        axL.text(1, F - 3 - i * 5, lab.replace("\n", "  "), color=col, fontsize=9,
                 va="top", fontweight="bold")

    # ---- 右：三种撑大感受野的手段（方框对比）----
    axR.set_xlim(0, 10); axR.set_ylim(0, 10); axR.axis("off")
    axR.set_title("撑大感受野的三种手段", fontsize=11.5, color=C_INK)
    _box(axR, (5, 8.4), 8.4, 1.35,
         "① 层叠：多堆几层 3×3\nRF = 1 + L×(k−1)，线性增长", C_WAVE, fs=9.5)
    _box(axR, (5, 5.9), 8.4, 1.35,
         "② stride 步长：一步跨 2 格\n图缩一半，等效感受野翻倍（下采样）", C_CHAN, fs=9.5)
    _box(axR, (5, 3.4), 8.4, 1.35,
         "③ dilation 空洞：核点隔着取\n不加参数就撑大 RF，降噪/分离最爱", C_RF, fs=9.5)
    axR.text(5, 1.2,
             "1 秒=100 帧：纯堆 3×3 要 ≈50 层；\ndilation 逐层翻倍(1,2,4,8,16,32) 只需 6 层",
             ha="center", fontsize=9, color=C_INK, style="italic", linespacing=1.4)

    fig.suptitle("感受野 = 一个输出点“看”了原图多大一片上下文；越大越能利用长时频上下文（降噪关键）",
                 fontsize=12.5, color=C_INK, y=1.02)
    fig.subplots_adjust(wspace=0.28, bottom=0.13)
    save(fig, "fig05_receptive_field.png")


# ============ 图 5：频率轴/时间轴卷积不对称 + 因果卷积（不偷看未来帧） ============
def fig_axis_causal():
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(11.0, 4.6), gridspec_kw={"width_ratios": [1.05, 1]})

    # ---- 左：三种核形状铺在语谱图网格上 ----
    axL.set_xlim(0, 10); axL.set_ylim(0, 10); axL.axis("off")
    axL.set_title("两轴不对称 → 核形状要区别对待", fontsize=11.5, color=C_INK)

    def mini_grid(ox, oy, hi_cells, ec, title, sub):
        cell = 0.55
        nr, nc = 5, 5
        for r in range(nr):
            for c in range(nc):
                x = ox + c * cell; y = oy + r * cell
                hot = (r, c) in hi_cells
                axL.add_patch(Rectangle((x, y), cell, cell,
                              facecolor=ec if hot else "#ecf0f1",
                              edgecolor="white", lw=0.8, zorder=2))
        axL.text(ox + nc * cell / 2, oy + nr * cell + 0.25, title,
                 ha="center", fontsize=9.5, color=ec, fontweight="bold")
        axL.text(ox + nc * cell / 2, oy - 0.35, sub, ha="center",
                 fontsize=8, color="#7f8c8d")

    # 沿频率核 F×1（竖条），沿时间核 1×T（横条），2D 方核
    col_freq = [(r, 2) for r in range(5)]
    col_time = [(2, c) for c in range(5)]
    col_2d = [(r, c) for r in range(1, 4) for c in range(1, 4)]
    mini_grid(0.4, 5.4, col_freq, C_FREQ, "沿频率 F×1", "抓频率剖面\n(共振峰/谐波间距)")
    mini_grid(3.8, 5.4, col_time, C_WAVE, "沿时间 1×T", "抓时间起伏\n(持续 or 一闪而过)")
    mini_grid(7.2, 5.4, col_2d, C_KERNEL, "2D 方核 3×3", "抓斜纹\n(声调滑动)")
    axL.text(5, 1.2, "F 竖 / T 横 / 方核管一小片时频邻域——用哪种取决于任务",
             ha="center", fontsize=9, color=C_INK, style="italic")

    # ---- 右：因果卷积 vs 普通卷积（padding 位置）----
    axR.set_xlim(0, 12); axR.set_ylim(0, 10); axR.axis("off")
    axR.set_title("因果性：实时时不能偷看未来帧", fontsize=11.5, color=C_INK)
    cell = 0.72
    ty = 6.6
    tc = 3      # 当前帧 index（在时间条里的位置）
    n = 8
    # 时间帧条
    for i in range(n):
        x = 1.5 + i * cell
        future = i > tc
        fc = C_PAD if future else C_PAST
        axR.add_patch(Rectangle((x, ty), cell * 0.9, cell * 0.9,
                      facecolor=fc, edgecolor="white", lw=1.0, zorder=2))
    axR.text(1.5 + tc * cell + cell * 0.45, ty + cell + 0.15, "当前帧 t",
             ha="center", fontsize=8.5, color=C_INK)
    axR.text(1.5 + 1.2 * cell, ty - 0.45, "过去（已采到）", color=C_PAST, fontsize=8.5)
    axR.text(1.5 + 5.5 * cell, ty - 0.45, "未来（还没发生）", color="#7f8c8d", fontsize=8.5)

    # 普通卷积：核对称罩两侧
    _box(axR, (6, 4.4), 10.4, 1.15,
         "普通卷积：核对称罩 t 两侧 → 用到未来帧 t+1,t+2\n离线降噪 OK（整段都拿到了）", C_FREQ, fs=9)
    # 因果卷积：padding 挪到左边
    _box(axR, (6, 2.0), 10.4, 1.15,
         "因果卷积：padding 全挪到左边（过去）\n核只看当前+历史帧，绝不偷看未来 → 实时降噪/AEC 必须", C_PAST, fs=9)

    fig.suptitle("频率≠时间：核形状要分轴设计；实时场景还多一道因果性硬约束（第 06 篇双向 RNN 会重现）",
                 fontsize=12, color=C_INK, y=1.02)
    fig.subplots_adjust(wspace=0.2, bottom=0.1)
    save(fig, "fig05_axis_causal.png")


if __name__ == "__main__":
    fig_image_view()
    fig_conv_slide()
    fig_transient_detector()
    fig_receptive_field()
    fig_axis_causal()
