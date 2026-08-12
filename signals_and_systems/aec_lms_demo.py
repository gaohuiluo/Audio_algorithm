# -*- coding: utf-8 -*-
"""
自适应滤波器 / LMS 回声消除最小可运行示例
- 远端参考信号 x：模拟一段"人声"（多正弦 + 幅度包络，模拟音节起伏）
- 回声路径 h：延迟若干采样点 + 衰减（多抽头，模拟简单房间反射）
- 麦克风信号 d = 回声(x 经过 h) + 近端小噪声
- LMS 自适应：w <- w + mu * e * x_vec，让 w 逐步逼近真实回声路径 h
- 可视化：第 1 / 10 / 50 / 100 "帧"时，滤波器输出 y 如何逼近真实回声
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(0)

# ---------- 1. 造一段"人声"作为远端参考信号 x ----------
# 真实语音 = 浊音(谐波) + 清音(宽带噪声)。宽带很关键：只有频带足够宽，
# LMS 才能"激励"到回声路径的每个抽头，w 才会真正收敛到真实路径 h。
fs = 8000                      # 采样率 8kHz
T = 2.0                        # 时长 2 秒
n = np.arange(int(fs * T))
t = n / fs

harmonic = (np.sin(2*np.pi*180*t)
            + 0.6*np.sin(2*np.pi*370*t)
            + 0.3*np.sin(2*np.pi*750*t))     # 浊音谐波
broadband = rng.standard_normal(len(t))       # 清音/气声，宽带成分
voice = harmonic + 1.2*broadband              # 二者混合
env = 0.5 + 0.5*np.sin(2*np.pi*3*t)          # 3Hz 的音节起伏
env *= (rng.random(len(t)) > 0.02)           # 偶尔的静音
x = voice * env
x = x / np.max(np.abs(x))                     # 归一化

# ---------- 2. 造一条回声路径 h：延迟 + 衰减（多抽头） ----------
M = 64                                        # 滤波器/回声路径抽头数
h_true = np.zeros(M)
h_true[8]  = 0.8      # 主回声：延迟 8 个采样点，衰减到 0.8
h_true[20] = 0.4      # 一次反射
h_true[35] = 0.2      # 二次反射
h_true[50] = 0.1      # 更晚的反射

# 麦克风收到的回声 = x 卷积 h_true
echo = np.convolve(x, h_true)[:len(x)]

# 近端本底噪声（模拟麦克风环境噪声，不是近端说话）
near_noise = 0.005 * rng.standard_normal(len(x))
d = echo + near_noise                          # 麦克风信号

# ---------- 3. LMS 自适应滤波（按"帧"处理，贴合驱动 Ring Buffer） ----------
FRAME = 128             # 每帧 128 个采样点（16ms@8kHz），和音频驱动一帧概念对齐
mu = 0.5                # 步长（配合归一化）
w = np.zeros(M)         # 待训练权重，初始全 0
y = np.zeros(len(x))    # 滤波器输出（估计出的回声）
e = np.zeros(len(x))    # 误差 = 麦克风 - 估计回声（=送出去的干净信号）

# 记录第 1 / 10 / 50 / 100 帧结束时的权重快照，用于画"逼近过程"
snap_frames = [1, 10, 50, 100]
snap_samples = {f: None for f in snap_frames}

eps = 1e-6
# 正则项 δ：按参考信号平均能量来定，而不是拍一个极小值。
# 这样远端静音帧里 norm 不会塌到 0、mu/norm 不会暴涨——避免"静音时踹飞权重"。
delta = 0.1 * M * np.mean(x**2)
n_frames = len(x) // FRAME
for f in range(n_frames):
    for j in range(FRAME):
        i = f * FRAME + j
        if i < M:                       # 前 M 个点凑不满一个滑窗，跳过
            continue
        x_vec = x[i-M+1:i+1][::-1]      # 最近 M 个参考采样点（倒序对齐 w）
        y[i] = np.dot(w, x_vec)         # 估计回声
        e[i] = d[i] - y[i]             # 误差 = 消完回声后的信号
        norm = np.dot(x_vec, x_vec) + delta  # NLMS 归一化 + 正则，稳定收敛
        w = w + (mu / norm) * e[i] * x_vec  # 核心更新公式
    if (f + 1) in snap_samples:         # 这一帧刚处理完，拍个快照
        snap_samples[f + 1] = w.copy()

# ---------- 4. 可视化 A：权重逼近真实回声路径 ----------
fig, axes = plt.subplots(2, 2, figsize=(12, 7))
for ax, fr in zip(axes.ravel(), snap_frames):
    w_snap = snap_samples[fr]
    ax.stem(h_true, linefmt="C0-", markerfmt="C0o", basefmt=" ", label="真实回声路径 h")
    ax.stem(w_snap, linefmt="C1-", markerfmt="C1x", basefmt=" ", label="滤波器权重 w")
    ax.set_title(f"第 {fr} 帧（{fr*FRAME} 个采样点，≈{fr*FRAME/fs*1000:.0f} ms）")
    ax.set_xlabel("抽头 (延迟采样点)")
    ax.set_ylabel("权重")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
fig.suptitle("LMS 权重逐步逼近真实回声路径", fontsize=14)
fig.tight_layout()
fig.savefig("E:/Android/signals_and_systems/aec_weights.png", dpi=110)

# ---------- 5. 可视化 B：输出波形逼近真实回声 ----------
# 用每个快照的权重，重新算一段输出，和真实回声比
seg = slice(int(0.9*fs), int(0.95*fs))   # 取中间一小段看波形
fig2, axes2 = plt.subplots(2, 2, figsize=(12, 7))
for ax, fr in zip(axes2.ravel(), snap_frames):
    w_snap = snap_samples[fr]
    y_snap = np.convolve(x, w_snap)[:len(x)]
    ax.plot(echo[seg], "C0", lw=2, label="真实回声")
    ax.plot(y_snap[seg], "C1--", lw=1.5, label="滤波器输出")
    ax.set_title(f"第 {fr} 帧的权重还原出的回声")
    ax.set_xlabel("采样点")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
fig2.suptitle("滤波器输出如何一步步逼近真实回声", fontsize=14)
fig2.tight_layout()
fig2.savefig("E:/Android/signals_and_systems/aec_output.png", dpi=110)

# ---------- 6. 收敛曲线：误差能量随时间下降 ----------
win = 200
err_db = 10*np.log10(np.convolve(e**2, np.ones(win)/win, "same") /
                     (np.convolve(d**2, np.ones(win)/win, "same") + eps) + eps)
fig3, ax3 = plt.subplots(figsize=(10, 4))
ax3.plot(t, err_db)
ax3.set_title("残余回声能量（越低=消得越干净）ERLE 视角")
ax3.set_xlabel("时间 (秒)")
ax3.set_ylabel("残余/输入 能量比 (dB)")
ax3.grid(alpha=0.3)
fig3.tight_layout()
fig3.savefig("E:/Android/signals_and_systems/aec_converge.png", dpi=110)

# ---------- 打印数值证据 ----------
def db(a, b):
    return 10*np.log10(np.sum(a**2)/(np.sum(b**2)+eps)+eps)

print("真实回声路径主抽头位置:", np.nonzero(h_true)[0].tolist())
print("第100帧 权重与真实路径的误差范数:", np.linalg.norm(snap_samples[100]-h_true))
print("第1帧   权重与真实路径的误差范数:", np.linalg.norm(snap_samples[1]-h_true))
tail = slice(int(1.5*fs), len(x))
print(f"收敛后 ERLE(回声抑制比): {db(echo[tail], e[tail]):.1f} dB (越大越好)")
print("图已保存: aec_weights.png / aec_output.png / aec_converge.png")

# ---------- 7. 导出 wav：污染前 / 污染后 / 消除后 三段对比听 ----------
import wave

def save_wav(path, sig, fs):
    # 归一化到 int16，留 5% 余量防削顶
    s = sig / (np.max(np.abs(sig)) + eps) * 0.95
    pcm = (s * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)      # 16-bit
        wf.setframerate(fs)
        wf.writeframes(pcm.tobytes())

base = "E:/Android/signals_and_systems/"
save_wav(base + "aec_1_reference.wav", x, fs)   # 远端参考（扬声器放的）
save_wav(base + "aec_2_mic.wav",       d, fs)   # 麦克风收到的（被回声污染）
save_wav(base + "aec_3_cleaned.wav",   e, fs)   # LMS 消除后（送出去的干净信号）
print("wav 已保存: aec_1_reference.wav / aec_2_mic.wav / aec_3_cleaned.wav")

