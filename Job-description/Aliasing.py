import numpy as np
import matplotlib.pyplot as plt

# ---------- 实验一：混叠 (Aliasing) ----------
# 目标：用一个"过低"的采样率去采一个高频正弦，看它伪装成什么频率。

fs = 48000          # 采样率 48 kHz -> 奈奎斯特频率 fN = 24 kHz
f_signal = 30000    # 信号频率 30 kHz，已经【超过】fN=24kHz，必然混叠
duration = 0.002    # 只取 2ms，方便画图看清波形

# 离散时间轴：n = 0,1,2,...  实际时刻 t = n * Ts = n / fs
n = np.arange(0, int(duration * fs))
t = n / fs

# 被采样得到的离散序列 x[n] = sin(2*pi*f*t)
x_sampled = np.sin(2 * np.pi * f_signal * t)

# 按采样定理预测混叠频率： f_alias = |f - k*fs|，取落在 [0, fN] 的那个
# 这里 k=1: |30000 - 48000| = 18000 Hz
f_alias = abs(f_signal - fs)   # = 18000 Hz
print(f"原始频率 = {f_signal} Hz (超过奈奎斯特 {fs/2:.0f} Hz)")
print(f"预测混叠后听起来像 = {f_alias} Hz")

# 用一个"假想的低频信号"来对照：如果直接生成 18kHz，采样点应该重合
x_alias = np.sin(2 * np.pi * f_alias * t)

# 验证：两串采样序列几乎完全相同 —— 这就是"分不清"的铁证
max_diff = np.max(np.abs(x_sampled - x_alias))
print(f"30kHz采样序列 与 18kHz采样序列 的最大差异 = {max_diff:.6e}")
# 输出接近 0，证明采样后 30kHz 与 18kHz 在数字域【完全无法区分】

plt.figure(figsize=(10, 4))
plt.plot(t * 1000, x_sampled, 'o-', label='sampled 30kHz', markersize=4)
plt.plot(t * 1000, x_alias, 'x--', label='true 18kHz (alias)', markersize=8)
plt.xlabel('time (ms)'); plt.ylabel('amplitude')
plt.title('Aliasing: 30kHz sampled at 48kHz looks exactly like 18kHz')
plt.legend(); plt.grid(True); plt.tight_layout()
plt.show()
