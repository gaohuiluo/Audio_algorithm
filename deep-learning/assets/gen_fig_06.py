# -*- coding: utf-8 -*-
"""
第 6 篇《RNN / LSTM / GRU 与时序》配图生成脚本。
运行：python assets/gen_fig_06.py
输出：assets/fig06_*.png（4 张）

依赖：matplotlib、numpy。中文字体用 Microsoft YaHei。
风格严格对齐 gen_fig_01.py（浅底、克制配色、直觉先行）。
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # torch+mpl 的 OpenMP 冲突绕过
import numpy as np
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
C_WAVE = "#2c6fbb"     # 波形/时间/正向
C_FREQ = "#c0392b"     # 频率/反向读序
C_BATCH = "#8e44ad"    # 批次
C_CHAN = "#16a085"     # 通道/隐状态记忆
C_GRAD = "#e67e22"     # 梯度/反向
C_PAD = "#bdc3c7"      # padding 灰
C_INK = "#2c3e50"      # 主文字
# 本篇新增（同色系延伸）
C_MEM = C_CHAN         # 隐状态 h（记忆）
C_CELL = "#27ae60"     # 细胞状态 c（传送带/梯度高速路）
C_GATE = "#9b59b6"     # 门控 sigmoid 阀门

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


def _arrow(ax, p0, p1, color=C_INK, ls="-", lw=1.8, rad=0.0, ms=16):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=ms,
        linewidth=lw, color=color, linestyle=ls,
        connectionstyle=f"arc3,rad={rad}", zorder=1))


# ============ 图 1：RNN 沿时间展开，记忆一帧帧传下去 ============
def fig_rnn_unroll():
    fig, ax = plt.subplots(figsize=(10.6, 4.6))
    ax.set_xlim(0, 13); ax.set_ylim(0, 5.6); ax.axis("off")
    ax.set_title("RNN 沿时间展开：同一套旋钮 (W_hh, W_xh) 处理每一帧，记忆 h 一帧帧滚下去",
                 fontsize=13.5, color=C_INK, pad=12)

    # 时间步位置
    xs = [2.0, 5.2, 8.4, 11.6]
    labels = ["t=1", "t=2", "t=3", "  ···  t=100"]
    y_cell = 2.9

    for i, (x, lab) in enumerate(zip(xs, labels)):
        # RNN cell 方框
        _box(ax, (x, y_cell), 1.5, 1.1, "RNN\ntanh", C_WAVE, fs=11)
        # 输入 x_t（下方）
        _box(ax, (x, 0.95), 1.3, 0.7, f"x_{i+1}", "#7f8c8d", fs=11)
        _arrow(ax, (x, 1.35), (x, y_cell - 0.6))
        # 输出 h_t（上方）
        _box(ax, (x, 4.75), 1.3, 0.7, f"h_{i+1}", C_MEM, fs=11)
        _arrow(ax, (x, y_cell + 0.6), (x, 4.4))
        ax.text(x, 0.28, lab, ha="center", fontsize=10, color=C_INK)

    # 记忆回路：h_{t-1} -> h_t（水平橙绿链）
    for i in range(len(xs) - 1):
        _arrow(ax, (xs[i] + 0.78, y_cell), (xs[i + 1] - 0.78, y_cell),
               color=C_CELL, lw=2.4, ms=18)
    # 起点隐状态 h0
    _box(ax, (0.55, y_cell), 0.8, 0.8, "h0", C_PAD, tc=C_INK, fs=10)
    _arrow(ax, (0.98, y_cell), (xs[0] - 0.78, y_cell), color=C_CELL, lw=2.4, ms=18)

    ax.text(6.5, 3.55, "记忆 h 沿时间传递：处理第100帧时，h99 里攥着前99帧的一切",
            ha="center", fontsize=10.5, color=C_CELL, style="italic")
    ax.text(6.5, 1.75, "每一帧：新记忆 h_t = tanh( W_hh·旧记忆 h_{t-1} + W_xh·当前帧 x_t )",
            ha="center", fontsize=10.5, color=C_INK)
    ax.text(6.5, 5.35, "输入 [B, T, Feature]：时间 T 摆中间，一帧一帧走过去（batch_first=True）",
            ha="center", fontsize=10.5, color=C_WAVE)
    save(fig, "fig06_rnn_unroll.png")


# ============ 图 2：梯度沿时间连乘 → 消失 / 爆炸（numpy 真实曲线） ============
def fig_grad_vanish():
    # 从最后一帧(t=100)往回追责，穿过 n 段记忆回路，责任 ∝ factor^n
    n = np.arange(0, 100)          # 往回传经过的记忆回路段数
    g_vanish = 0.9 ** n            # 因子<1：越乘越小 -> 梯度消失
    g_explode = 1.1 ** n           # 因子>1：越乘越炸 -> 梯度爆炸
    g_keep = np.ones_like(n, dtype=float)  # 门控高速路：加法主干，责任基本保住

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.4),
                                   gridspec_kw={"width_ratios": [1, 1]})

    # 左：线性坐标，直观看消失/保持
    axL.plot(n, g_vanish, color=C_GRAD, lw=2.2, label="朴素RNN 因子0.9：0.9^n 越传越小（消失）")
    axL.plot(n, g_keep, color=C_CELL, lw=2.2, ls="--",
             label="LSTM 加法高速路：责任基本保住≈1")
    axL.axhline(0, color="#ccc", lw=0.8)
    axL.set_title("沿时间往回追责：因子<1 时责任被连乘磨没", fontsize=11.5, color=C_INK)
    axL.set_xlabel("往回传经过的记忆回路段数 n（离当前帧越远越大）")
    axL.set_ylabel("远处那帧记忆分到的「责任」（梯度大小）")
    axL.set_ylim(-0.05, 1.1); axL.grid(alpha=0.25)
    axL.legend(fontsize=8.8, loc="upper right")
    axL.annotate("0.9^99≈0.00003\n悄悄话传到第100人\n早已听不清", xy=(99, 0.0),
                 xytext=(60, 0.42), fontsize=9, color=C_GRAD, ha="center",
                 arrowprops=dict(arrowstyle="->", color=C_GRAD))

    # 右：对数坐标，同时展示消失与爆炸的天壤之别
    axR.semilogy(n, g_vanish, color=C_GRAD, lw=2.2, label="因子0.9：消失 → 0")
    axR.semilogy(n, g_explode, color=C_FREQ, lw=2.2, label="因子1.1：爆炸 → NaN")
    axR.semilogy(n, g_keep, color=C_CELL, lw=2.2, ls="--", label="门控高速路：≈1 稳住")
    axR.set_title("对数坐标：>1 炸上天 / <1 掉到地板（对因子极其敏感）",
                  fontsize=11.5, color=C_INK)
    axR.set_xlabel("往回传经过的记忆回路段数 n")
    axR.set_ylabel("责任（对数轴）")
    axR.grid(alpha=0.25, which="both")
    axR.legend(fontsize=8.8, loc="center right")
    axR.text(50, 1.1e4, "1.1^99≈12500\nloss 变 NaN，训练崩",
             fontsize=9, color=C_FREQ, ha="center")

    fig.suptitle("梯度消失/爆炸：第04篇「责任链连乘」沿时间轴拉长几十上百倍的必然结果",
                 fontsize=13, color=C_INK, y=1.02)
    fig.subplots_adjust(wspace=0.28, bottom=0.16)
    save(fig, "fig06_grad_vanish.png")


# ============ 图 3：LSTM 门控——细胞状态传送带 + 三个门（梯度高速路） ============
def fig_lstm_gate():
    fig, ax = plt.subplots(figsize=(11.0, 5.2))
    ax.set_xlim(0, 13); ax.set_ylim(0, 6.2); ax.axis("off")
    ax.set_title("LSTM：细胞状态 c 是一条「加法传送带」，三个门决定擦掉多少 / 添上多少 / 取出多少",
                 fontsize=13, color=C_INK, pad=12)

    # ---- 顶部：细胞状态传送带 c_{t-1} -> c_t（加法高速路，绿粗线）----
    y_c = 5.1
    ax.add_patch(FancyBboxPatch((1.2, y_c - 0.14), 10.6, 0.28,
                 boxstyle="square,pad=0", fc=C_CELL, ec=C_CELL, alpha=0.35, zorder=1))
    _arrow(ax, (1.0, y_c), (12.2, y_c), color=C_CELL, lw=3.0, ms=20)
    _box(ax, (1.0, y_c), 1.0, 0.7, "c_{t-1}", C_CELL, fs=10)
    _box(ax, (12.3, y_c), 0.9, 0.7, "c_t", C_CELL, fs=10)
    ax.text(6.5, 5.75, "细胞状态传送带：主干只做加减（× f_t 擦掉、+ i_t·c~ 添上），不反复相乘 → 梯度高速路，长依赖责任传得回去",
            ha="center", fontsize=9.6, color=C_CELL, style="italic")

    # 传送带上的两个运算节点：×（遗忘缩放）和 +（输入添加）
    ax.add_patch(Circle((4.3, y_c), 0.26, fc="white", ec=C_CELL, lw=1.8, zorder=3))
    ax.text(4.3, y_c, "×", ha="center", va="center", fontsize=15, color=C_CELL, zorder=4)
    ax.add_patch(Circle((7.6, y_c), 0.26, fc="white", ec=C_CELL, lw=1.8, zorder=3))
    ax.text(7.6, y_c, "+", ha="center", va="center", fontsize=15, color=C_CELL, zorder=4)

    # ---- 底部输入：x_t 与 h_{t-1} ----
    _box(ax, (1.5, 0.7), 1.4, 0.7, "x_t 当前帧", "#7f8c8d", fs=9.5)
    _box(ax, (3.6, 0.7), 1.5, 0.7, "h_{t-1} 旧输出", C_MEM, fs=9.5)

    # ---- 三个门（sigmoid 阀门，紫）+ 候选 c̃ ----
    gates = [
        (4.3, 2.9, "遗忘门 f_t\nσ", "擦掉多少旧记忆", C_GATE),
        (6.4, 2.9, "候选 c~\ntanh", "本帧提炼的新内容", C_WAVE),
        (7.6, 2.9, "输入门 i_t\nσ", "添上多少新信息", C_GATE),
        (10.0, 2.9, "输出门 o_t\nσ", "取出多少作输出", C_GATE),
    ]
    for gx, gy, lab, sub, col in gates:
        _box(ax, (gx, gy), 1.35, 0.95, lab, col, fs=9.8)
        ax.text(gx, gy - 0.72, sub, ha="center", fontsize=8.3, color=C_INK)
        # 输入汇入每个门
        _arrow(ax, (2.0, 1.05), (gx - 0.35, gy - 0.5), color="#95a5a6", lw=1.0, rad=0.15, ms=11)
        _arrow(ax, (4.0, 1.05), (gx, gy - 0.5), color=C_MEM, lw=1.0, rad=0.1, ms=11)

    # 门连到传送带
    _arrow(ax, (4.3, 3.4), (4.3, y_c - 0.3), color=C_GATE, lw=1.8)   # f_t -> ×
    _arrow(ax, (6.4, 3.4), (7.3, y_c - 0.28), color=C_WAVE, lw=1.6, rad=0.1)  # c̃ -> +
    _arrow(ax, (7.6, 3.4), (7.6, y_c - 0.3), color=C_GATE, lw=1.8)   # i_t -> +

    # 输出门取传送带内容 -> h_t
    _arrow(ax, (7.86, y_c), (10.0, y_c - 0.1), color=C_CELL, lw=1.4, rad=-0.2, ms=12)
    _box(ax, (10.0, 1.05), 1.3, 0.7, "h_t 本帧输出", C_MEM, fs=9.5)
    _arrow(ax, (10.0, 2.4), (10.0, 1.42), color=C_GATE, lw=1.8)

    ax.text(6.5, 0.05, "遗忘/输入/输出三个门都是 sigmoid，输出∈(0,1) 当「阀门开度」——网络自己学会「选择性记忆与遗忘」",
            ha="center", fontsize=9.5, color=C_INK)
    save(fig, "fig06_lstm_gate.png")


# ============ 图 4：双向 RNN——离线利器 / 实时禁区（因果红线） ============
def fig_bidirectional():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.2, 4.6),
                                   gridspec_kw={"width_ratios": [1, 1]})
    xs = [1.4, 3.2, 5.0, 6.8]
    frames = ["帧1", "帧2", "帧3", "帧4"]

    # ---- 左：双向 RNN（离线）----
    axL.set_xlim(0, 8.2); axL.set_ylim(0, 6.0); axL.axis("off")
    axL.set_title("双向 RNN（离线可用）：正向拿历史 + 反向拿未来", fontsize=11.5, color=C_INK)
    for x, f in zip(xs, frames):
        axL.add_patch(FancyBboxPatch((x - 0.5, 2.7), 1.0, 0.8,
                      boxstyle="round,pad=0.02,rounding_size=0.06",
                      fc="#eef2f5", ec=C_INK, lw=1.3, zorder=2))
        axL.text(x, 3.1, f, ha="center", va="center", fontsize=9.5, color=C_INK)
    # 正向链（蓝，往右）
    for i in range(len(xs) - 1):
        axL.add_patch(FancyArrowPatch((xs[i] + 0.5, 3.85), (xs[i + 1] - 0.5, 3.85),
                      arrowstyle="-|>", mutation_scale=15, color=C_WAVE, lw=2.0, zorder=1))
    axL.text(xs[0] - 0.55, 4.35, "正向：过去→现在", color=C_WAVE, fontsize=9.2, ha="left")
    # 反向链（红，往左）
    for i in range(len(xs) - 1):
        axL.add_patch(FancyArrowPatch((xs[i + 1] - 0.5, 2.35), (xs[i] + 0.5, 2.35),
                      arrowstyle="-|>", mutation_scale=15, color=C_FREQ, lw=2.0, zorder=1))
    axL.text(xs[-1] + 0.55, 1.85, "反向：未来→现在", color=C_FREQ, fontsize=9.2, ha="right")
    axL.text(4.1, 5.3, "每帧「瞻前顾后」，拼接双向记忆，效果更好\n但反向链要从最后一帧起算 → 必须拿到整段",
             ha="center", fontsize=9.4, color=C_INK)
    axL.text(4.1, 0.7, "[能] 离线降噪 / 离线语音识别", ha="center", fontsize=10.5,
             color=C_CELL, fontweight="bold")

    # ---- 右：实时禁区——未来帧还没到 ----
    axR.set_xlim(0, 8.2); axR.set_ylim(0, 6.0); axR.axis("off")
    axR.set_title("实时场景：处理第 t 帧时，未来帧还没被麦克风采到", fontsize=11.5, color=C_INK)
    now = 1  # 当前处理到帧2
    for i, (x, f) in enumerate(zip(xs, frames)):
        arrived = i <= now
        fc = "#eef2f5" if arrived else C_PAD
        ec = C_INK if arrived else "#95a5a6"
        hatch = None if arrived else "///"
        axR.add_patch(FancyBboxPatch((x - 0.5, 2.7), 1.0, 0.8,
                      boxstyle="round,pad=0.02,rounding_size=0.06",
                      fc=fc, ec=ec, lw=1.3, hatch=hatch, zorder=2))
        axR.text(x, 3.1, f if arrived else "?", ha="center", va="center",
                 fontsize=9.5, color=C_INK if arrived else "#7f8c8d")
    # 正向链只连已到达的帧
    for i in range(now):
        axR.add_patch(FancyArrowPatch((xs[i] + 0.5, 3.85), (xs[i + 1] - 0.5, 3.85),
                      arrowstyle="-|>", mutation_scale=15, color=C_WAVE, lw=2.0, zorder=1))
    axR.text(xs[0] - 0.55, 4.35, "正向：只依赖历史 [可]", color=C_WAVE, fontsize=9.2, ha="left")
    # 反向链画成断裂红叉
    axR.add_patch(FancyArrowPatch((xs[now + 1] - 0.5, 2.35), (xs[now] + 0.5, 2.35),
                  arrowstyle="-|>", mutation_scale=15, color=C_FREQ, lw=2.0,
                  ls=":", alpha=0.5, zorder=1))
    axR.text(xs[now + 1], 2.02, "X", ha="center", fontsize=15, color=C_FREQ, fontweight="bold")
    # 竖直的"当下"红线
    axR.axvline((xs[now] + xs[now + 1]) / 2, ymin=0.12, ymax=0.82,
                color=C_FREQ, lw=1.8, ls="--")
    axR.text((xs[now] + xs[now + 1]) / 2, 5.0, "「当下」因果红线\n右边尚未发生",
             ha="center", fontsize=9.2, color=C_FREQ)
    axR.text(4.1, 0.7, "[禁] 双向出局，实时只能用单向 RNN", ha="center", fontsize=10.5,
             color=C_FREQ, fontweight="bold")

    fig.suptitle("「能不能看未来帧」这条因果红线：离线尽管双向，实时永远只能盯着过去",
                 fontsize=13, color=C_INK, y=1.02)
    fig.subplots_adjust(wspace=0.12, bottom=0.05)
    save(fig, "fig06_bidirectional.png")


if __name__ == "__main__":
    fig_rnn_unroll()
    fig_grad_vanish()
    fig_lstm_gate()
    fig_bidirectional()

