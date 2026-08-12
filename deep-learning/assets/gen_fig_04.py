# -*- coding: utf-8 -*-
"""
第 4 篇《神经网络基础：网络凭什么会"学会"》配图生成脚本。
运行：python assets/gen_fig_04.py
输出：assets/fig04_*.png（5 张）

依赖：matplotlib、numpy、torch。中文字体用 Microsoft YaHei。
风格严格对齐第 1 篇（gen_fig_01.py）：浅底、克制配色、直觉先行。
前向统一用蓝 C_WAVE、反向/梯度统一用橙 C_GRAD，和第 1 篇 autograd 图一致。
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # torch+mpl 的 OpenMP 冲突绕过
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- 全局风格（原样复用第 1 篇）----
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 12
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.dpi"] = 130
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.facecolor"] = "white"

# 系列统一配色
C_WAVE = "#2c6fbb"     # 波形/时间 / 前向
C_FREQ = "#c0392b"     # 频率
C_BATCH = "#8e44ad"    # 批次
C_CHAN = "#16a085"     # 通道
C_GRAD = "#e67e22"     # 梯度/反向
C_PAD = "#bdc3c7"      # padding 灰
C_INK = "#2c3e50"      # 主文字
C_GELU = "#f1a208"     # 激活曲线新增色（同暖色系）


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


# ============ 图 1：前向→损失→反向→更新 的完整训练回路 ============
def fig_train_loop():
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.4); ax.axis("off")
    ax.set_title("网络怎么「学」：前向出结果 → 损失量错误 → 反传摊责任 → 更新拧旋钮，转千万圈",
                 fontsize=13, color=C_INK, pad=12)

    # 上排：前向链（蓝）  x -> Wx+b -> 激活 -> y_hat -> L
    yf = 4.6
    _box(ax, (1.4, yf), 1.5, 0.95, "输入 x\n一帧频谱", "#7f8c8d", fs=10)
    _box(ax, (3.8, yf), 1.6, 0.95, "Wx + b\n全连接层", C_WAVE, fs=10.5)
    _box(ax, (6.2, yf), 1.6, 0.95, "σ(·)\n激活掰弯", C_WAVE, fs=10.5)
    _box(ax, (8.6, yf), 1.5, 0.95, "输出 ŷ\n(预测)", C_WAVE, fs=10.5)
    _box(ax, (10.9, yf), 1.5, 0.95, "损失 L\n差多少", C_FREQ, fs=10.5)

    _arrow(ax, (2.15, yf), (3.0, yf), color=C_WAVE)
    _arrow(ax, (4.6, yf), (5.4, yf), color=C_WAVE)
    _arrow(ax, (7.0, yf), (7.85, yf), color=C_WAVE)
    _arrow(ax, (9.35, yf), (10.15, yf), color=C_WAVE)
    ax.text(6.0, 5.55, "① 前向 forward：算出 ŷ，并把「怎么算的」记进计算图  →",
            color=C_WAVE, fontsize=11, ha="center")

    # 下排：反向链（橙，虚线）L 顺着图倒推每个旋钮的责任(.grad)
    yb = 2.6
    _arrow(ax, (10.4, yf - 0.5), (10.9, yb + 0.5), color=C_GRAD, ls="--", rad=0.0)
    _box(ax, (8.6, yb), 1.7, 0.95, "∂L/∂W\n(.grad 责任)", C_GRAD, fs=10)
    _box(ax, (6.2, yb), 1.7, 0.95, "链式法则\n逐层回传", C_GRAD, fs=10.5)
    _box(ax, (3.8, yb), 1.7, 0.95, "拿到每个\n旋钮的责任", C_GRAD, fs=10)
    _arrow(ax, (9.9, yb), (10.9, yb + 0.45), color=C_GRAD, ls="--", rad=0.12)
    _arrow(ax, (7.35, yb), (9.75, yb), color=C_GRAD, ls="--")
    _arrow(ax, (5.05, yb), (7.35, yb), color=C_GRAD, ls="--")
    ax.text(6.4, 1.55, "←  ② 反向 backward：从 L 顺着图倒推，loss.backward() 一句自动摊派责任",
            color=C_GRAD, fontsize=11, ha="center")

    # 更新旋钮 + 回到前向的闭环
    _box(ax, (1.7, yb), 2.0, 0.95, "W ← W − η·∂L/∂W\n③ 拧旋钮", C_INK, fs=9.5)
    _arrow(ax, (2.95, yb), (2.55, yb), color=C_GRAD, ls="--")
    # 闭环回到前向
    ax.add_patch(FancyArrowPatch((1.7, yb + 0.5), (1.4, yf - 0.5),
                 arrowstyle="-|>", mutation_scale=15, color=C_INK,
                 lw=1.6, linestyle=":", connectionstyle="arc3,rad=-0.3", zorder=1))
    ax.text(0.7, 3.6, "④ 再来\n一圈", color=C_INK, fontsize=9, ha="center", style="italic")

    ax.text(6.0, 0.5, "反向的橙色链路是 np.ndarray 没有的——正是它，让「一堆旋钮」变成「会学习的网络」",
            ha="center", fontsize=10.5, color=C_INK, style="italic")
    save(fig, "fig04_train_loop.png")


# ============ 图 2：把「责任」摸出来——链式法则的具体数字 ============
def fig_chain_number():
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5.6); ax.axis("off")
    ax.set_title("把「责任」摸出来：一个巴掌大的例子，顺着责任链走一遍链式法则",
                 fontsize=13, color=C_INK, pad=10)

    # 前向（蓝）：x=2, w=0.5 -> y_hat=1.0 -> L
    yf = 4.0
    _box(ax, (1.3, yf), 1.4, 0.85, "x = 2", "#7f8c8d", fs=11)
    _box(ax, (4.0, yf), 1.7, 0.85, "ŷ = w·x = 1.0", C_WAVE, fs=10.5)
    _box(ax, (7.2, yf), 2.0, 0.85, "L = (ŷ − 1.5)²", C_FREQ, fs=10.5)
    _box(ax, (1.3, yf - 1.4), 1.4, 0.85, "w = 0.5", C_BATCH, fs=11)

    _arrow(ax, (2.0, yf), (3.15, yf), color=C_WAVE)
    _arrow(ax, (2.0, yf - 1.4), (3.3, yf - 0.35), color=C_WAVE, rad=0.15)
    _arrow(ax, (4.85, yf), (6.2, yf), color=C_WAVE)
    ax.text(6.0, 4.95, "① 前向：目标 y_target = 1.5，此刻 ŷ=1.0 偏小了  →",
            color=C_WAVE, fontsize=11, ha="center")

    # 反向（橙）：两段偏导 + 相乘
    yb = 1.9
    ax.text(7.2, yb + 0.95, "∂L/∂ŷ = 2(ŷ−1.5) = −1.0",
            color=C_GRAD, fontsize=11, ha="center")
    ax.text(7.2, yb + 0.45, "（输出错了，损失怎么变）", color="#7f8c8d", fontsize=8.5, ha="center")
    ax.text(4.0, yb + 0.95, "∂ŷ/∂w = x = 2",
            color=C_GRAD, fontsize=11, ha="center")
    ax.text(4.0, yb + 0.45, "（w 动一点，输出动多少 = 传动比）",
            color="#7f8c8d", fontsize=8.5, ha="center")

    _arrow(ax, (6.2, yb + 0.9), (5.2, yb + 0.9), color=C_GRAD, ls="--")

    # 相乘得到梯度
    _box(ax, (4.0, yb - 0.55), 4.2, 0.8,
         "∂L/∂w = (−1.0) × 2 = −2.0   ← w 该背的责任", C_GRAD, fs=11)
    ax.text(6.0, 0.95, "② 反向：两段偏导一乘，就是 w 的梯度",
            color=C_GRAD, fontsize=11, ha="center")

    # 更新
    _box(ax, (9.6, yb - 0.55), 3.6, 0.8,
         "w ← 0.5 − 0.1×(−2.0) = 0.7", C_INK, fs=10.5)
    ax.text(9.6, 0.35,
            "③ w 变大 → 下轮 ŷ=1.4，更接近 1.5：责任算对了，一步就把输出往对的方向拉",
            color=C_INK, fontsize=9.5, ha="center", style="italic")
    save(fig, "fig04_chain_number.png")


# ============ 图 3：激活函数把直线「掰弯」（真实 numpy 画） ============
def fig_bend_line():
    x = torch.linspace(-3, 3, 400)
    # 两层纯线性：还是一条直线
    W1, b1, W2, b2 = 1.4, 0.3, -0.8, 0.5
    lin = W2 * (W1 * x + b1) + b2
    # 加激活（tanh）后：掰出了弯
    bent = W2 * torch.tanh(W1 * x + b1) + b2
    # 再叠一层，弯上加弯
    bent2 = 0.9 * torch.tanh(1.6 * bent + 0.2) + 0.6 * torch.tanh(-2.0 * x - 0.5)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.4))

    ax1.plot(x.numpy(), lin.numpy(), color=C_WAVE, lw=2.4,
             label="两层线性层串联 W2·(W1·x+b1)+b2")
    ax1.axhline(0, color="#cccccc", lw=0.8); ax1.axvline(0, color="#cccccc", lw=0.8)
    ax1.set_title("不加激活：摞多少层，还是一条直线", fontsize=11.5, color=C_INK)
    ax1.grid(alpha=0.25); ax1.legend(fontsize=9, loc="upper left")
    ax1.set_xlabel("输入 x"); ax1.set_ylabel("输出 y")
    ax1.text(0, -3.6, "W2·W1 还是一个矩阵 → 等价于单层 y=Wx+b\n表达不了语音那种拧巴的非线性映射",
             ha="center", fontsize=9.5, color=C_FREQ)

    ax2.plot(x.numpy(), bent.numpy(), color=C_GRAD, lw=2.4,
             label="线性后插一个 tanh：掰出一道弯")
    ax2.plot(x.numpy(), bent2.numpy(), color=C_BATCH, lw=2.0, ls="--",
             label="再叠一层：弯上加弯")
    ax2.axhline(0, color="#cccccc", lw=0.8); ax2.axvline(0, color="#cccccc", lw=0.8)
    ax2.set_title("插入激活函数：直线被掰弯，层层叠加能拟合任意曲线",
                  fontsize=11.5, color=C_INK)
    ax2.grid(alpha=0.25); ax2.legend(fontsize=9, loc="upper left")
    ax2.set_xlabel("输入 x"); ax2.set_ylabel("输出 y")
    ax2.text(0, -3.6, "每层在前一层掰出的弯上再掰一道\n这是网络能表达非线性的唯一来源",
             ha="center", fontsize=9.5, color=C_BATCH)

    fig.suptitle("激活函数 = 把线性直线「掰弯」的那道工序，没有它深度就是白搭",
                 fontsize=13, color=C_INK, y=1.02)
    fig.subplots_adjust(wspace=0.24, bottom=0.24)
    save(fig, "fig04_bend_line.png")


# ============ 图 4：常见激活函数曲线 + 导数对比（真实 torch 画） ============
def fig_activation_curves():
    x = torch.linspace(-5, 5, 500, requires_grad=True)

    def curve_and_grad(fn):
        y = fn(x)
        g, = torch.autograd.grad(y.sum(), x, create_graph=False)
        return y.detach().numpy(), g.detach().numpy()

    relu_y, relu_g = curve_and_grad(F.relu)
    lrelu_y, lrelu_g = curve_and_grad(lambda t: F.leaky_relu(t, 0.1))
    sig_y, sig_g = curve_and_grad(torch.sigmoid)
    tanh_y, tanh_g = curve_and_grad(torch.tanh)
    xn = x.detach().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.2))
    specs = [
        (axes[0, 0], "ReLU：负的砍成 0，正的原样通过（隐藏层默认）",
         relu_y, relu_g, C_WAVE),
        (axes[0, 1], "LeakyReLU：负半轴留条小缝，缓解 Dead ReLU",
         lrelu_y, lrelu_g, C_CHAN),
        (axes[1, 0], "Sigmoid：挤进 (0,1)，天生适合做 mask",
         sig_y, sig_g, C_FREQ),
        (axes[1, 1], "Tanh：压到 (−1,1)、关于 0 对称，适合时域波形",
         tanh_y, tanh_g, C_BATCH),
    ]
    for ax, title, y, g, c in specs:
        ax.plot(xn, y, color=c, lw=2.4, label="函数 f(x)")
        ax.plot(xn, g, color=C_GRAD, lw=1.8, ls="--", label="导数 f'(x)（反传的传动比）")
        ax.axhline(0, color="#cccccc", lw=0.8); ax.axvline(0, color="#cccccc", lw=0.8)
        ax.set_title(title, fontsize=10.5, color=C_INK)
        ax.grid(alpha=0.25); ax.set_xlabel("x"); ax.legend(fontsize=8.5, loc="upper left")
        ax.set_ylim(-1.4, 2.2)

    # 在 sigmoid/tanh 子图标注饱和区（导数趋 0 → 梯度传不回去）
    axes[1, 0].text(3.4, 0.55, "两端饱和\n导数→0\n(第 06 篇梯度消失)",
                    fontsize=8, color=C_GRAD, ha="center")
    axes[1, 1].text(3.4, -0.55, "两端饱和\n导数→0",
                    fontsize=8, color=C_GRAD, ha="center")

    fig.suptitle("四条常见激活曲线（实线）与它们的导数（橙虚线）——导数就是反传时的「传动比」",
                 fontsize=13, color=C_INK, y=1.0)
    fig.subplots_adjust(hspace=0.36, wspace=0.22)
    save(fig, "fig04_activation_curves.png")


# ============ 图 5：sigmoid 输出天然落在 (0,1) → 做降噪 mask ============
def fig_sigmoid_mask():
    fig = plt.figure(figsize=(11.0, 4.6))
    axL = fig.add_axes([0.06, 0.16, 0.34, 0.66])

    # 左：sigmoid 把任意实数挤进 (0,1)
    x = torch.linspace(-8, 8, 400)
    s = torch.sigmoid(x)
    axL.plot(x.numpy(), s.numpy(), color=C_FREQ, lw=2.6)
    axL.axhline(0, color="#cccccc", lw=0.8); axL.axhline(1, color="#cccccc", lw=0.8, ls=":")
    axL.axvline(0, color="#cccccc", lw=0.8)
    axL.fill_between(x.numpy(), 0, s.numpy(), color=C_FREQ, alpha=0.08)
    axL.set_title("sigmoid：任意实数 → 挤进 (0,1)", fontsize=11.5, color=C_INK)
    axL.set_xlabel("网络原始输出（logit，可正可负）")
    axL.set_ylabel("mask 系数")
    axL.set_ylim(-0.15, 1.15); axL.grid(alpha=0.25)
    axL.annotate("→ 1：全保留（人声）", xy=(4.5, 0.99), xytext=(0.3, 0.78),
                 fontsize=9, color=C_CHAN,
                 arrowprops=dict(arrowstyle="->", color=C_CHAN))
    axL.annotate("→ 0：全压掉（噪声）", xy=(-4.5, 0.01), xytext=(-7.6, 0.28),
                 fontsize=9, color=C_INK,
                 arrowprops=dict(arrowstyle="->", color=C_INK))

    # 右：mask 逐点相乘作用在带噪谱上  Ŝ = M ⊙ Y
    rng = np.random.RandomState(4)
    Fbin, Tfr = 32, 40
    # 造一张「人声在中低频、噪声散布」的示意带噪谱
    ff, tt = np.meshgrid(np.linspace(0, 1, Tfr), np.linspace(0, 1, Fbin))
    voice = np.exp(-((tt - 0.35) ** 2) / 0.03) * (0.6 + 0.4 * np.sin(6 * ff))
    noise = 0.35 * rng.rand(Fbin, Tfr)
    Y = voice + noise
    # sigmoid mask：人声区接近 1、噪声区接近 0
    logit = 6 * (voice - 0.3) - 3 * noise
    M = 1 / (1 + np.exp(-logit))
    S_hat = M * Y

    # 三联展示 Y ⊙ M = Ŝ
    sub = fig.add_axes([0.50, 0.18, 0.13, 0.56])
    sub2 = fig.add_axes([0.68, 0.18, 0.13, 0.56])
    sub3 = fig.add_axes([0.86, 0.18, 0.13, 0.56])
    for a, data, ttl, cmap in [
        (sub, Y, "带噪谱 Y", "magma"),
        (sub2, M, "mask M∈(0,1)", "viridis"),
        (sub3, S_hat, "降噪后 Ŝ", "magma")]:
        a.imshow(data, origin="lower", aspect="auto", cmap=cmap)
        a.set_title(ttl, fontsize=9.5, color=C_INK)
        a.set_xticks([]); a.set_yticks([])
    fig.text(0.655, 0.45, "⊙", ha="center", va="center", fontsize=18, color=C_INK)
    fig.text(0.835, 0.45, "=", ha="center", va="center", fontsize=18, color=C_INK)
    fig.text(0.745, 0.03,
             "mask 取值天然该在 [0,1]，sigmoid 输出恰好在 (0,1)——降噪网络输出层清一色用它的根本原因",
             ha="center", fontsize=9, color=C_INK, style="italic")

    fig.suptitle("为什么降噪 mask 用 sigmoid：值域天生匹配「0～1 透光度」",
                 fontsize=13, color=C_INK, y=0.98)
    save(fig, "fig04_sigmoid_mask.png")


if __name__ == "__main__":
    fig_train_loop()
    fig_chain_number()
    fig_bend_line()
    fig_activation_curves()
    fig_sigmoid_mask()
