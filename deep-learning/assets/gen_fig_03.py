# -*- coding: utf-8 -*-
"""
第 3 篇《显存与混合精度》配图生成脚本。
运行：python assets/gen_fig_03.py
输出：assets/fig03_*.png（5 张）

依赖：matplotlib、numpy。中文字体用 Microsoft YaHei。
风格严格对齐 gen_fig_01.py：浅底、克制配色、直觉先行。
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # torch+mpl 的 OpenMP 冲突绕过
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- 全局风格（与第 1 篇一致）----
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 12
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.dpi"] = 130
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.facecolor"] = "white"

# 系列统一配色（沿用第 1 篇含义）
C_WAVE = "#2c6fbb"     # 波形/时间/forward
C_FREQ = "#c0392b"     # 频率
C_BATCH = "#8e44ad"    # 批次
C_CHAN = "#16a085"     # 通道
C_GRAD = "#e67e22"     # 梯度/反向
C_PAD = "#bdc3c7"      # padding 灰
C_INK = "#2c3e50"      # 主文字

# 本篇新增（同色系延伸，保持含义一致）
C_PARAM = "#1f6f8b"    # 参数（蓝系，网络的"旋钮"）
C_OPT = C_BATCH        # 优化器状态（紫，Adam 的隐形大户）
C_ACT = C_FREQ         # 激活值（红，最吃地方的大户）
C_TABLE = "#8a9299"    # 台面/边框中性灰
C_OK = "#2e8b57"       # 安全绿
C_FP32 = C_WAVE        # fp32
C_FP16 = C_GRAD        # fp16
C_BF16 = C_CHAN        # bf16


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


# ============ 图 1：工作台上摊了什么 —— 显存的四类零件 ============
def fig_workbench():
    """把显存画成一张有限大小的工作台，摊开四类零件，激活值最吃地方。"""
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.2); ax.axis("off")
    ax.set_title("显存 = 一张有限大小的工作台：训练时同时摊着四类零件",
                 fontsize=14, color=C_INK, pad=12)

    # 工作台台面
    ax.add_patch(FancyBboxPatch((0.4, 0.55), 9.2, 4.55,
                 boxstyle="round,pad=0.02,rounding_size=0.08",
                 fc="#f4f6f7", ec=C_TABLE, lw=2.2, zorder=1))
    ax.text(5.0, 5.28, "显存工作台（例：8 GB）—— 台面一满就 OOM",
            ha="center", fontsize=11, color=C_TABLE, style="italic")

    # 四类零件方块：宽度暗示相对占地（激活最大）
    # ①参数 ②梯度 ③优化器状态（2倍） ④激活值（随数据暴涨，最大）
    _box(ax, (1.75, 3.55), 2.2, 1.5,
         "① 参数 W, b\n网络的旋钮\n1 份 · 每个 4 字节", C_PARAM, fs=10.5)
    _box(ax, (1.75, 1.55), 2.2, 1.1,
         "② 梯度\n该往哪调\n1 份（= 参数）", C_WAVE, fs=10.5)
    _box(ax, (4.35, 2.55), 2.2, 3.1,
         "③ 优化器状态\nAdam 的动量/方差\n2 份 = 参数×2\n（隐形大户）", C_OPT, fs=10.5)
    _box(ax, (7.85, 2.55), 3.0, 3.9,
         "④ 激活值\n前向每层的中间输出\n随 B×C×F×T 暴涨\n最吃地方的大户", C_ACT, fs=11.5)

    # 分组标注
    ax.text(3.0, 0.15, "前三类「跟着参数走」：Adam 时 ①+②+③ = 参数量 × 4",
            ha="center", fontsize=9.5, color=C_INK)
    ax.text(7.85, 0.15, "第 ④ 类「跟着数据量走」，且乘起来涨",
            ha="center", fontsize=9.5, color=C_FREQ)
    ax.annotate("", xy=(6.35, 0.35), xytext=(5.9, 0.35),
                arrowprops=dict(arrowstyle="-", color="#cccccc"))
    save(fig, "fig03_workbench.png")


# ============ 图 2：算一笔账 —— 激活值是怎么被 B×C×F×T 乘爆的 ============
def fig_memory_math():
    """真实数值：单层激活 [B,C,F,T] 随各因子翻倍而翻倍，几层就冲破 8GB 台面。"""
    B, C, F, T = 64, 64, 257, 500
    per_layer_gb = B * C * F * T * 4 / 1024**3  # ≈ 2.05 GB

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.0, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})

    # 左：多层激活堆叠，越堆越高，8GB 红线截断
    layers = np.arange(1, 6)
    cum = layers * per_layer_gb
    colors = [C_ACT if v <= 8 else "#7f2d2d" for v in cum]
    ax1.bar(layers, [per_layer_gb] * len(layers), bottom=(layers - 1) * per_layer_gb,
            width=0.62, color=C_ACT, ec="white", lw=1.2)
    for i, v in enumerate(cum):
        ax1.text(layers[i], v + 0.12, f"{v:.1f}", ha="center",
                 fontsize=9, color=C_INK)
    ax1.axhline(8.0, color=C_FREQ, ls="--", lw=1.8)
    ax1.text(5.35, 8.15, "8 GB 台面上限", ha="right", color=C_FREQ, fontsize=9.5)
    ax1.text(2.9, 9.0, "第 4 层就爆了 → OOM", ha="center", color="#7f2d2d",
             fontsize=10, fontweight="bold")
    ax1.set_xlabel("缓存的层数（每层都要存着等反传）")
    ax1.set_ylabel("累计激活显存 / GB")
    ax1.set_title(f"单层 [B,C,F,T]=[{B},{C},{F},{T}] 就 {per_layer_gb:.1f} GB，几层封顶",
                  fontsize=10.5, color=C_INK)
    ax1.set_xticks(layers); ax1.set_ylim(0, 10.5)
    ax1.grid(axis="y", alpha=0.25)

    # 右：杠杆 —— 每个因子翻倍，这层显存跟着翻倍
    ax2.axis("off")
    ax2.set_xlim(0, 10); ax2.set_ylim(0, 10)
    ax2.set_title("激活值 ∝ B × C × F × T：任一因子翻倍，显存翻倍",
                  fontsize=10.5, color=C_INK)
    ax2.text(5, 8.9, r"元素数 = B × C × F × T", ha="center", fontsize=13,
             color=C_INK)
    ax2.text(5, 8.0, "× 4 字节/个 (fp32)", ha="center", fontsize=10.5,
             color="#7f8c8d")
    items = [
        ("batch B", "64 → 128", C_BATCH),
        ("帧数 T（音频时长）", "500 → 1000", C_WAVE),
        ("通道 C", "64 → 128", C_CHAN),
    ]
    for i, (name, chg, col) in enumerate(items):
        y = 6.2 - i * 1.7
        _box(ax2, (2.6, y), 3.6, 1.15, f"{name}\n{chg}", col, fs=10)
        _arrow(ax2, (4.5, y), (6.0, y), color=col)
        _box(ax2, (7.7, y), 3.0, 1.15, "这层显存\n×2", C_ACT, fs=10.5)
    ax2.text(5, 0.7, "省显存第一直觉：把这个乘积压下去（减 batch / 缩音频 / 减通道）",
             ha="center", fontsize=9.5, color=C_INK, style="italic")

    fig.suptitle("算一笔账：OOM 不是写错代码，是 B×C×F×T 再乘层数冲破了台面",
                 fontsize=13.5, color=C_INK, y=1.02)
    fig.subplots_adjust(wspace=0.24, bottom=0.13)
    save(fig, "fig03_memory_math.png")


# ============ 图 3：fp32 vs fp16 vs bf16 —— 位宽拆解与下溢的坑 ============
def fig_bitwidth():
    """按 IEEE 位宽真实比例画符号/指数/尾数格子，点明 fp16 范围窄→下溢。"""
    fig, ax = plt.subplots(figsize=(10.4, 5.0))
    ax.set_xlim(0, 33); ax.set_ylim(0, 6.4); ax.axis("off")
    ax.set_title("fp32 vs fp16 vs bf16：同一个数，用多少「位」去存",
                 fontsize=14, color=C_INK, pad=12)

    # (名字, 总位数, 指数位, 尾数位, y, 字节说明, 颜色)
    specs = [
        ("fp32", 32, 8, 23, 4.6, "4 字节/个 · 范围大、精度高", C_FP32),
        ("fp16", 16, 5, 10, 2.7, "2 字节/个 · 范围窄、精度低（会下溢）", C_FP16),
        ("bf16", 16, 8, 7, 0.8, "2 字节/个 · 范围和 fp32 一样宽、尾数糙", C_BF16),
    ]
    cell_w = 0.85
    x_start = 4.2
    for name, total, ne, nm, y, note, col in specs:
        ax.text(3.7, y + 0.35, name, ha="right", va="center", fontsize=13,
                color=col, fontweight="bold")
        # 符号位(1) + 指数位 + 尾数位
        segs = [(1, C_INK, "符号"), (ne, C_FREQ, "指数(范围)"),
                (nm, C_OK, "尾数(精度)")]
        x = x_start
        for cnt, ccol, _ in segs:
            for _ in range(cnt):
                ax.add_patch(Rectangle((x, y), cell_w, 0.7, fc=ccol,
                             ec="white", lw=0.6, zorder=2, alpha=0.9))
                x += cell_w
        ax.text(x + 0.25, y + 0.35, f"= {total} 位", ha="left", va="center",
                fontsize=10.5, color=C_INK)
        ax.text(x_start, y - 0.28, note, ha="left", va="top",
                fontsize=9, color="#7f8c8d")
        # 指数/尾数分段标注
        ax.text(x_start + cell_w + ne * cell_w / 2, y + 0.95, f"指数 {ne} 位",
                ha="center", fontsize=8.5, color=C_FREQ)
        ax.text(x_start + cell_w + (ne + nm / 2) * cell_w, y + 0.95, f"尾数 {nm} 位",
                ha="center", fontsize=8.5, color=C_OK)

    # 底部：下溢示意
    ax.text(0.3, 0.05, "指数位越少 → 能表示的范围越窄 → 反传里的极小梯度(如 1e-8)在 fp16 里被直接舍成 0，这就是「下溢」，需 GradScaler 救回。",
            ha="left", va="bottom", fontsize=9.5, color=C_INK, style="italic")
    save(fig, "fig03_bitwidth.png")


# ============ 图 4：GradScaler 流程 —— 放大 loss 救回小梯度，更新前再还原 ============
def fig_gradscaler():
    """autocast + GradScaler 一个训练步的数据流：临时借位，不影响最终更新量。"""
    fig, ax = plt.subplots(figsize=(10.6, 5.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.6); ax.axis("off")
    ax.set_title("混合精度一个训练步：autocast 切精度 + GradScaler 救小梯度",
                 fontsize=13.5, color=C_INK, pad=12)

    # 上排流程方框
    y = 4.5
    _box(ax, (1.5, y), 2.3, 1.2,
         "autocast 前向\n矩阵乘/卷积→fp16\nloss/求和→fp32", C_WAVE, fs=9.5)
    _box(ax, (4.4, y), 2.0, 1.2, "scaler.scale(loss)\nloss × 65536\n（放大）", C_GRAD, fs=9.5)
    _box(ax, (7.1, y), 2.0, 1.2, ".backward()\n小梯度被顶进\nfp16 可表示区", C_GRAD, fs=9.5)
    _box(ax, (9.9, y), 2.0, 1.2, "scaler.step(opt)\n梯度÷65536 还原\n查溢出→更新", C_PARAM, fs=9.5)
    for x0, x1 in [(2.65, 3.4), (5.4, 6.1), (8.1, 8.9)]:
        _arrow(ax, (x0, y), (x1, y))

    # 回环：scaler.update 动态调倍数
    _arrow(ax, (9.9, 3.9), (1.5, 3.9), color=C_INK, ls="--", rad=-0.12)
    ax.text(5.7, 3.05, "scaler.update()：动态调整下一步的缩放倍数（遇溢出就调小）",
            ha="center", fontsize=9.5, color="#7f8c8d")

    # 下方：为什么放大不影响更新量（数轴示意）
    ax.add_patch(FancyBboxPatch((0.6, 0.5), 10.8, 1.9,
                 boxstyle="round,pad=0.02,rounding_size=0.06",
                 fc="#fdf6ec", ec=C_GRAD, lw=1.4, zorder=1))
    ax.text(6.0, 2.05, "为什么放大只是「临时借位」，不改变最终更新量：",
            ha="center", fontsize=10, color=C_INK, fontweight="bold")
    ax.text(1.1, 1.35, "小梯度 1e-8", ha="left", va="center", fontsize=10, color=C_FREQ)
    _arrow(ax, (2.9, 1.35), (4.1, 1.35), color=C_GRAD)
    ax.text(3.5, 1.62, "×65536", ha="center", fontsize=8.5, color=C_GRAD)
    ax.text(4.3, 1.35, "6.5e-4（fp16 存得下）", ha="left", va="center", fontsize=10, color=C_OK)
    _arrow(ax, (7.9, 1.35), (9.1, 1.35), color=C_PARAM)
    ax.text(8.5, 1.62, "÷65536", ha="center", fontsize=8.5, color=C_PARAM)
    ax.text(9.3, 1.35, "1e-8（原样还原）", ha="left", va="center", fontsize=10, color=C_INK)
    ax.text(6.0, 0.72, "bf16 范围和 fp32 一样宽，几乎不下溢，很多时候连 GradScaler 都省了",
            ha="center", fontsize=9, color="#7f8c8d", style="italic")
    save(fig, "fig03_gradscaler.png")


# ============ 图 5：fp32 vs 混合精度 —— 峰值显存对比（堆叠条形，真实构成）============
def fig_amp_saving():
    """按四类零件构成堆叠：混合精度砍激活+梯度，参数/优化器 fp32 主副本不省。"""
    # 相对份额（示意量级，非精确实测）：参数1 / 梯度1 / 优化器2 / 激活8
    labels = ["参数", "梯度", "优化器状态", "激活值"]
    colors = [C_PARAM, C_WAVE, C_OPT, C_ACT]
    fp32 = np.array([1.0, 1.0, 2.0, 8.0])          # 各占 4 字节
    # 混合精度：激活+梯度减半；参数/优化器 fp32 主副本保留（不省），另加 fp16 权重副本
    amp = np.array([1.0 + 0.5, 0.5, 2.0, 4.0])     # 参数多一份 fp16 副本

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    xs = [0, 1]
    bottoms = [0.0, 0.0]
    for i in range(4):
        vals = [fp32[i], amp[i]]
        ax.bar(xs, vals, bottom=bottoms, width=0.5, color=colors[i],
               ec="white", lw=1.3, label=labels[i])
        for j in range(2):
            if vals[j] >= 0.6:
                ax.text(xs[j], bottoms[j] + vals[j] / 2, labels[i],
                        ha="center", va="center", color="white", fontsize=9)
        bottoms = [bottoms[0] + fp32[i], bottoms[1] + amp[i]]

    tot32, totamp = fp32.sum(), amp.sum()
    ax.text(0, tot32 + 0.3, f"fp32\n合计 {tot32:.0f}", ha="center", fontsize=11,
            color=C_INK, fontweight="bold")
    ax.text(1, totamp + 0.3, f"混合精度\n合计 {totamp:.1f}", ha="center", fontsize=11,
            color=C_INK, fontweight="bold")
    ax.annotate("", xy=(1, totamp + 0.05), xytext=(1, tot32),
                arrowprops=dict(arrowstyle="-|>", color=C_OK, lw=2))
    save_pct = (1 - totamp / tot32) * 100
    ax.text(1.42, (tot32 + totamp) / 2, f"省约\n{save_pct:.0f}%",
            ha="center", va="center", fontsize=11, color=C_OK, fontweight="bold")

    ax.set_xticks(xs); ax.set_xticklabels(["fp32（全 32 位）", "混合精度（autocast）"])
    ax.set_ylabel("峰值显存（相对份额）")
    ax.set_ylim(0, tot32 + 1.6)
    ax.set_title("混合精度砍掉激活+梯度的一半；参数/优化器 fp32 主副本没省\n"
                 "所以实测通常省 35%~50%，够把 batch 或音频长度翻近一倍",
                 fontsize=11.5, color=C_INK, pad=10)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(axis="y", alpha=0.25)
    save(fig, "fig03_amp_saving.png")


if __name__ == "__main__":
    fig_workbench()
    fig_memory_math()
    fig_bitwidth()
    fig_gradscaler()
    fig_amp_saving()

