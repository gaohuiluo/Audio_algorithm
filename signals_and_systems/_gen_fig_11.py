# -*- coding: utf-8 -*-
"""生成第 11 篇（拉普拉斯与极点零点）全部配图。

fig_11_1_1  第九章 9.1：极零点图 + 沿虚轴扫出的频响（三档 sigma）
fig_11_1_2  第三章 3.1/3.2：增长信号傅里叶积不动 -> 垫一层衰减后收敛
fig_11_1_3  第三章 3.3：|X(s)|=1/|s+a| 在 s=-a 炸掉，极点从积分里长出来
fig_11_1_4  第四章 4.2：沿虚轴走到极点/零点的距离向量 + 对应频响
fig_11_1_5  第五章 5.1：极点位置 <-> 时域积木 e^{pt}（衰减/等幅/增长）
fig_11_1_6  第九章：极点实部三档的冲激响应（衰减振铃/等幅/爆炸）
fig_11_1_7  第十章 10.2：参量 EQ 提升/衰减/平直 = 零极点谁离虚轴更近

只依赖 NumPy + SciPy + Matplotlib。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 110})


def save(fig, name):
    fig.savefig(os.path.join(HERE, name), dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


# ============================================================
# fig_11_1_1  (第九章 9.1，与正文代码一致)
# ============================================================
f0, fz = 500.0, 1500.0
w0, wz = 2 * np.pi * f0, 2 * np.pi * fz
sigmas = [-300.0, -100.0, -30.0]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5))
axL.axvline(0, color="k", lw=1.5)
axL.axvspan(-450, 0, color="#e8f5e9")
axL.axvspan(0, 150, color="#ffebee")
axL.scatter([0, 0], [wz, -wz], marker="o", s=90,
            facecolors="none", edgecolors="k", label="zeros")
for sig, c in zip(sigmas, colors):
    axL.scatter([sig, sig], [w0, -w0], marker="x", s=110, c=c,
                label=f"poles sigma={sig:.0f}")
axL.set_xlim(-450, 150); axL.set_ylim(-wz * 1.4, wz * 1.4)
axL.set_xlabel("Re{s}=sigma"); axL.set_ylabel("Im{s}=omega")
axL.set_title("s-plane pole-zero map"); axL.legend(fontsize=8)

w = 2 * np.pi * np.linspace(1, 3000, 4000)
for sig, c in zip(sigmas, colors):
    zeros = [1j * wz, -1j * wz]
    poles = [sig + 1j * w0, sig - 1j * w0]
    K = (sig**2 + w0**2) / (wz**2)
    b, a = signal.zpk2tf(zeros, poles, K)
    _, h = signal.freqs(b, a, worN=w)
    axR.plot(w / (2 * np.pi), 20 * np.log10(np.abs(h)), color=c)
axR.set_xlabel("Frequency (Hz)"); axR.set_ylabel("|H(jw)| (dB)")
axR.set_title("Frequency response = walk up the jw axis")
axR.set_ylim(-40, 30)
save(fig, "fig_11_1_1.png")


# ============================================================
# fig_11_1_2  (3.1/3.2)：增长信号 -> 垫衰减 -> 收敛
# ============================================================
t = np.linspace(0, 2, 2000)
alpha, f = 3.0, 5.0
x = np.exp(alpha * t) * np.sin(2 * np.pi * f * t)
sigma_damp = 4.0                       # 垫上的衰减强度, sigma > alpha
xd = x * np.exp(-sigma_damp * t)

fig, ax = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
ax[0].plot(t, x, lw=0.9)
ax[0].plot(t, np.exp(alpha * t), "r--", lw=1.2, label="envelope +/-e^{at}")
ax[0].plot(t, -np.exp(alpha * t), "r--", lw=1.2)
ax[0].set_title("(a) Growing oscillation x(t)=e^{at}sin(2*pi*ft):  "
                "envelope never settles -> Fourier integral DIVERGES")
ax[0].set_ylabel("x(t)"); ax[0].legend(fontsize=9)

ax[1].plot(t, xd, lw=0.9)
ax[1].plot(t, np.exp(-(sigma_damp - alpha) * t), "g--", lw=1.2,
           label="envelope +/-e^{-(s-a)t}")
ax[1].plot(t, -np.exp(-(sigma_damp - alpha) * t), "g--", lw=1.2)
ax[1].set_title("(b) Damped first:  x(t)*e^{-st} with s>a:  now decaying -> "
                "integral CONVERGES  (this is the Laplace trick)")
ax[1].set_xlabel("Time (s)"); ax[1].set_ylabel("x(t)*e^{-st}"); ax[1].legend(fontsize=9)
save(fig, "fig_11_1_2.png")


# ============================================================
# fig_11_1_3  (3.3)：1/|s+a| 在 s=-a 处炸掉
# ============================================================
a = 2.0
sig = np.linspace(-1.985, 6, 4000)     # 只在收敛域 sigma > -a 内取值
Xmag = 1.0 / (sig + a)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(sig, Xmag, lw=1.8, color="#1f77b4", label="|X(s)| = 1/|s+a|")
ax.axvline(-a, color="r", lw=2)
ax.axvspan(-a, 7, color="#e8f5e9", alpha=0.6)
ax.axvspan(-7, -a, color="#eeeeee", alpha=0.8)
ax.annotate("pole s = -a\nintegral blows up\n|X| -> infinity",
            xy=(-a, 8), xytext=(0.6, 7.2),
            arrowprops=dict(arrowstyle="->", color="r"), color="r", fontsize=10)
ax.text(-4.6, 2.2, "sigma < -a:\nintegral diverges\n(no transform here)",
        fontsize=10, color="#555555", ha="center")
ax.text(3.6, 3.0, "ROC: sigma > -a\n(converges)", fontsize=10, color="#2e7d32")
ax.set_xlim(-7, 7); ax.set_ylim(0, 9)
ax.set_xlabel("sigma = Re{s}   (evaluated on the real axis, w=0)")
ax.set_ylabel("|X(s)|")
ax.set_title("One honest integral X(s)=1/(s+a): the pole is WHERE it blows up")
ax.legend(fontsize=9, loc="upper right")
save(fig, "fig_11_1_3.png")


# ============================================================
# fig_11_1_4  (4.2)：距离向量几何 + 对应频响
# ============================================================
p1 = -1.0 + 6.0j
p2 = -1.0 - 6.0j
z1 = 9.0j
K = (abs(p1) * abs(p2)) / (abs(z1) ** 2)     # 直流增益归一
b, a_coef = signal.zpk2tf([z1, -z1], [p1, p2], K)
w = np.linspace(0.05, 14, 3000)
_, H = signal.freqs(b, a_coef, worN=w)
magH = np.abs(H)

fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5.2))

# 左：s 平面 + 行走点 + 距离向量
axL.axhline(0, color="k", lw=1)
axL.axvline(0, color="k", lw=1.5)
walk_omegas = [3.0, 6.0, 9.0]
for wq in walk_omegas:
    axL.scatter([0], [wq], s=55, c="#795548", zorder=5)
axL.scatter([-1, -1], [6, -6], marker="x", s=130, c="#d62728", label="poles")
axL.scatter([0, 0], [9, -9], marker="o", s=110, facecolors="none",
            edgecolors="#1f77b4", lw=2, label="zeros")
axL.annotate("", xy=(p1.real, p1.imag), xytext=(0, 6),
             arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.8))
axL.text(-0.75, 6.9, "d_p small\n-> peak", color="#d62728", fontsize=10)
axL.annotate("", xy=(z1.real, z1.imag), xytext=(0, 9),
             arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.8))
axL.text(0.25, 9.7, "d_z = 0 -> null", color="#1f77b4", fontsize=10)
axL.annotate("", xy=(p1.real, p1.imag), xytext=(0, 3),
             arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2, alpha=0.6))
axL.text(-3.6, 3.4, "walker jw", fontsize=9, color="#795548")
axL.set_xlim(-4.5, 2.5); axL.set_ylim(-1, 11.5)
axL.set_xlabel("Re{s}=sigma"); axL.set_ylabel("Im{s}=omega")
axL.set_title("(a) Walking up the jw axis:\neach gain = (dist to zeros)/(dist to poles)")
axL.legend(fontsize=9, loc="lower left")
axL.grid(False)

# 右：对应 |H(jw)|
axR.plot(w, magH, lw=1.8, color="#2ca02c")
for wq, c in zip(walk_omegas, ["#795548"] * 3):
    axR.axvline(wq, color=c, ls=":", lw=1)
axR.annotate("w near pole height:\nsmall d_p -> RESONANCE", xy=(6, magH[np.argmin(abs(w - 6))]),
             xytext=(7.2, 1.55), arrowprops=dict(arrowstyle="->"), fontsize=10)
axR.annotate("w hits zero on axis:\nd_z=0 -> NULL", xy=(9, 0.02),
             xytext=(10.2, 0.6), arrowprops=dict(arrowstyle="->"), fontsize=10)
axR.set_xlabel("frequency omega (rad/s, position on jw axis)")
axR.set_ylabel("|H(jw)|")
axR.set_title("(b) The curve you read while walking")
axR.set_ylim(0, 2)
save(fig, "fig_11_1_4.png")


# ============================================================
# fig_11_1_5  (5.1)：极点位置 <-> 时域积木
# ============================================================
w0m = 6.0
sig_list = [-1.5, 0.0, 1.5]
tt = np.linspace(0, 3, 1500)
fig, axs = plt.subplots(2, 3, figsize=(12.5, 6))
for k, (sg, axp, axt) in enumerate(zip(sig_list, axs[0], axs[1])):
    axp.axhline(0, color="k", lw=1)
    axp.axvline(0, color="k", lw=1.5)
    axp.scatter([sg, sg], [w0m, -w0m], marker="x", s=120, c="#d62728")
    axp.set_xlim(-2.5, 2.5); axp.set_ylim(-8.5, 8.5)
    axp.set_xticks([-2, 0, 2]); axp.set_yticks([-6, 0, 6])
    axp.grid(False)
    ttl = [f"sigma={sg}: LHP\nbuilding block DECAYS (stable)",
           f"sigma=0: on jw axis\nrings FOREVER (marginal)",
           f"sigma=+{sg}: RHP\nbuilding block GROWS (howl!)"][k]
    axp.set_title(ttl, fontsize=10)
    axt.plot(tt, np.exp(sg * tt) * np.cos(w0m * tt), lw=1.0)
    axt.plot(tt, np.exp(sg * tt), "g--", lw=1.1)
    axt.plot(tt, -np.exp(sg * tt), "g--", lw=1.1)
    axt.set_ylim(-60, 60)
    axt.set_xlabel("t"); axt.set_ylabel("e^{pt} mode")
fig.suptitle("One pole p = sigma + j*w0  <->  one time-domain building block "
             "e^{pt} = e^{sigma*t} cos(w0*t)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
save(fig, "fig_11_1_5.png")


# ============================================================
# fig_11_1_6  (第九章)：三档 sigma 的冲激响应
# ============================================================
fs = 4000
tg = np.arange(int(0.5 * fs)) / fs
f0c = 200
w0c = 2 * np.pi * f0c

fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
for i, sg in enumerate([-30.0, 0.0, +30.0]):
    poles = [sg + 1j * w0c, sg - 1j * w0c]
    b, a_coef = signal.zpk2tf([], poles, w0c**2 + sg**2)
    sys = signal.TransferFunction(b, a_coef)
    _, h = signal.impulse(sys, T=tg)
    ax[i].plot(tg, h)
    tag = {-30: "sigma<0  (LEFT half-plane): decays -> STABLE",
           0: "sigma=0  (on jw axis): rings forever",
           30: "sigma>0  (RIGHT half-plane): grows -> HOWL!"}[sg]
    ax[i].set_title(f"pole real part sigma={sg:+.0f}   {tag}")
    ax[i].set_ylabel("Amplitude")
ax[-1].set_xlabel("Time (s)")
fig.tight_layout()
save(fig, "fig_11_1_6.png")


# ============================================================
# fig_11_1_7  (10.2)：参量 EQ 提升/衰减/平直
# ============================================================
w0e = 2 * np.pi * 1000.0
we = 2 * np.pi * np.linspace(20, 4000, 3000)

def peaking_H(zeta_p, zeta_z):
    """零极点对：离虚轴水平距离 = zeta*w0。谁更近谁做主。"""
    num = [1.0, 2 * zeta_z * w0e, w0e**2]
    den = [1.0, 2 * zeta_p * w0e, w0e**2]
    _, H = signal.freqs(num, den, worN=we)
    return np.abs(H)

fig, ax = plt.subplots(1, 3, figsize=(12.5, 4.2), sharey=True)
cases = [
    (0.10, 0.40, "BOOST: poles closer to jw axis\n-> peak rises"),
    (0.40, 0.40, "FLAT: equally distant\n-> effects cancel, 0 dB"),
    (0.40, 0.10, "CUT: zeros closer to jw axis\n-> notch drops"),
]
for (zp, zz, ttl), axx in zip(cases, ax):
    axx.plot(we / (2 * np.pi), 20 * np.log10(peaking_H(zp, zz) + 1e-9), lw=1.8)
    axx.axhline(0, color="k", lw=0.8)
    axx.axvline(1000, color="gray", ls=":", lw=1)
    axx.set_title(ttl, fontsize=10)
    axx.set_xlabel("Frequency (Hz)")
ax[0].set_ylabel("|H| (dB)")
fig.suptitle("One peaking-EQ band = a pole pair + a zero pair at the same height; "
             "the GAIN knob decides which pair sits closer to the jw axis", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.93])
save(fig, "fig_11_1_7.png")

print("all figures done")
