# -*- coding: utf-8 -*-
"""
双讲(double-talk)演示：近端说话时若不冻结更新，LMS 权重会被打飞。
对比两种策略：
  A. 不做双讲检测，一路更新       -> 近端说话段权重发散，路径估计被毁
  B. 简单能量比 DTD，双讲时冻结更新 -> 权重稳住
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 输出与本脚本同目录，避免硬编码绝对路径
base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "")

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(0)
fs, T = 8000, 3.5
n = np.arange(int(fs*T)); t = n/fs
eps = 1e-6

# 远端参考：宽带"人声"，全程都在说
x = (np.sin(2*np.pi*180*t) + 0.6*np.sin(2*np.pi*370*t)
     + rng.standard_normal(len(t))*1.2)
x *= (0.5 + 0.5*np.sin(2*np.pi*3*t))
x /= np.max(np.abs(x))

# 回声路径
M = 64
h = np.zeros(M); h[8]=0.8; h[20]=0.4; h[35]=0.2; h[50]=0.1
echo = np.convolve(x, h)[:len(x)]

# 近端说话：只在 2.0~3.0 秒这段出现（此时滤波器已收敛，对比更干净）
near = np.zeros(len(x))
seg = slice(int(2.0*fs), int(3.0*fs))
near[seg] = 0.6*np.sin(2*np.pi*220*t[seg]) + 0.3*rng.standard_normal(int(1.0*fs))

d = echo + near + 0.005*rng.standard_normal(len(x))   # 麦克风 = 回声 + 近端 + 噪声

def run(use_dtd):
    w = np.zeros(M); e = np.zeros(len(x)); err_norm = np.zeros(len(x))
    delta = 0.1*M*np.mean(x**2)
    L = 200                      # Geigel 观察窗：看远端最近 L 点的最大幅度
    thr = 0.8                    # 门限：实测纯回声段 |d|/far_max 99% 在 0.87 以下，
                                 # 双讲段中位数就有 1.1，卡 0.8 能分开两者
    hang = 240                   # hangover：触发后强制再冻结这么多采样点(30ms@8k)
                                 # 够跨过近端语音过零点，又不会因单点误触发冻结太久
    hold = 0                     # 剩余冻结计数
    froze = 0
    for i in range(M, len(x)):
        xv = x[i-M+1:i+1][::-1]
        y = np.dot(w, xv)
        e[i] = d[i] - y
        # Geigel DTD：麦克风幅度 / 远端最近最大幅度 > 门限 => 混进近端
        # 关键：一旦触发就进入 hangover，强制冻结一段，避免近端过零点时误恢复
        freeze = False
        if use_dtd and i >= L:
            far_max = np.max(np.abs(x[i-L+1:i+1])) + eps
            if abs(d[i]) > thr * far_max:
                hold = hang           # 重新装满迟滞计数
            if hold > 0:
                freeze = True
                hold -= 1
        if not freeze:
            norm = np.dot(xv, xv) + delta
            w = w + (0.5/norm)*e[i]*xv
        else:
            froze += 1
        err_norm[i] = np.linalg.norm(w - h)   # 权重离真实路径多远
    if use_dtd:
        s0, s1 = int(2.0*fs), int(3.0*fs)
        print(f"  [诊断] 双讲前(2.0s)权重误差={err_norm[s0]:.3f}, 总冻结帧={froze}")
    return err_norm

en_no  = run(use_dtd=False)
en_dtd = run(use_dtd=True)

fig, ax = plt.subplots(figsize=(11,4.5))
ax.plot(t, en_no,  "C3", label="不做双讲检测：近端说话段被打飞")
ax.plot(t, en_dtd, "C2", label="带 DTD 冻结：权重稳住")
ax.axvspan(2.0, 3.0, color="orange", alpha=0.15, label="近端说话(双讲)时段")
ax.set_xlabel("时间 (秒)"); ax.set_ylabel("‖w - h‖ (越低=路径估计越准)")
ax.set_title("双讲对自适应滤波器的破坏 & DTD 的作用")
ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(base + "aec_doubletalk.png", dpi=110)

print(f"不做DTD: 双讲结束时权重误差 = {en_no[int(3.0*fs)-1]:.3f}")
print(f"带 DTD : 双讲结束时权重误差 = {en_dtd[int(3.0*fs)-1]:.3f}")
print("图已保存: aec_doubletalk.png")
