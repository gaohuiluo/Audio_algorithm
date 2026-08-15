# -*- coding: utf-8 -*-
"""
生成第 19 篇新增数学推导配图 fig_19_1_1.png：
误差能量面 J(w) 是一个碗，LMS = 在这个碗上做梯度下降。
左：宽带输入 -> 碗接近圆形，梯度直指谷底，收敛快而稳。
右：窄带/病态输入 -> 碗被拉成狭长山谷，某些方向几乎平坦（梯度没信息），
    轨迹在谷里来回横跳、迟迟到不了底——正好对应"持续激励不满足"的坑。
只用 NumPy + Matplotlib。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

# 两抽头玩具问题：真实权重 w* = (1.0, 0.6)
w_star = np.array([1.0, 0.6])

def make_surface(Rxx):
    """给定输入自相关矩阵 Rxx，误差能量面 J(w) = (w-w*)^T Rxx (w-w*) + c。
    这是标准 LMS/维纳滤波的均方误差面：一个以 w* 为底的抛物碗。"""
    w0 = np.linspace(-0.6, 2.4, 240)
    w1 = np.linspace(-1.0, 2.0, 240)
    W0, W1 = np.meshgrid(w0, w1)
    D0 = W0 - w_star[0]
    D1 = W1 - w_star[1]
    J = (Rxx[0, 0]*D0*D0 + 2*Rxx[0, 1]*D0*D1 + Rxx[1, 1]*D1*D1)
    return w0, w1, W0, W1, J

def descend(Rxx, mu, steps=60, start=(-0.4, 1.7)):
    """在 J 上做梯度下降：grad J = 2 Rxx (w - w*)，w <- w - mu*grad。
    这就是 LMS 更新式在期望意义下的形态。"""
    w = np.array(start, dtype=float)
    path = [w.copy()]
    for _ in range(steps):
        g = 2 * Rxx @ (w - w_star)
        w = w - mu * g
        path.append(w.copy())
    return np.array(path)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))

# ---- 左：宽带输入，自相关近似单位阵（各方向能量均衡）----
Rxx_good = np.array([[1.0, 0.1],
                     [0.1, 1.0]])
w0, w1, W0, W1, Jg = make_surface(Rxx_good)
path_g = descend(Rxx_good, mu=0.18)
ax = axes[0]
cs = ax.contour(W0, W1, Jg, levels=18, cmap="viridis", alpha=0.7)
ax.plot(path_g[:, 0], path_g[:, 1], "o-", color="C3", ms=3, lw=1.3,
        label="梯度下降轨迹 (LMS)")
ax.plot(*w_star, "k*", ms=16, label="真实回声路径 w*")
ax.plot(path_g[0, 0], path_g[0, 1], "s", color="C0", ms=9, label="起点 w=0 附近")
ax.set_title("宽带参考：误差能量面接近圆碗\n梯度直指谷底，快而稳", fontsize=11)
ax.set_xlabel("权重 $w_0$"); ax.set_ylabel("权重 $w_1$")
ax.legend(loc="lower right", fontsize=8); ax.grid(alpha=0.25)
ax.set_aspect("equal")

# ---- 右：窄带/单频输入，两抽头强相关 -> 碗被拉成狭长山谷 ----
Rxx_bad = np.array([[1.0, 0.95],
                    [0.95, 1.0]])
w0, w1, W0, W1, Jb = make_surface(Rxx_bad)
path_b = descend(Rxx_bad, mu=0.18)
ax = axes[1]
cs = ax.contour(W0, W1, Jb, levels=18, cmap="viridis", alpha=0.7)
ax.plot(path_b[:, 0], path_b[:, 1], "o-", color="C3", ms=3, lw=1.3,
        label="梯度下降轨迹 (LMS)")
ax.plot(*w_star, "k*", ms=16, label="真实回声路径 w*")
ax.plot(path_b[0, 0], path_b[0, 1], "s", color="C0", ms=9, label="起点 w=0 附近")
ax.set_title("窄带/单频参考：能量面被拉成狭长山谷\n沿谷方向几乎没梯度，横跳、久久到不了底", fontsize=11)
ax.set_xlabel("权重 $w_0$"); ax.set_ylabel("权重 $w_1$")
ax.legend(loc="lower right", fontsize=8); ax.grid(alpha=0.25)
ax.set_aspect("equal")

fig.suptitle("LMS = 在“误差能量面”上做梯度下降；输入的“宽窄”决定这个碗好不好走",
             fontsize=13)
fig.tight_layout()
fig.savefig("E:/Android/Tech-Blog/signals_and_systems/fig_19_1_1.png", dpi=110)
print("图已保存: fig_19_1_1.png")
