"""卷积混响实战：把真实录制的人声/音乐"放进"一个山洞。
对应第 5 篇《冲激响应与卷积》line 150 的动手钩子。
"""
import numpy as np
from scipy.io import wavfile


def make_cave_ir(fs, tail_sec=0.5, seed=0):
    """造一段"山洞"冲激响应：直达声 + 离散早期反射 + 指数衰减稠密尾巴。"""
    rng = np.random.default_rng(seed)
    ir = np.zeros(int(fs * tail_sec))
    ir[0] = 1.0                                   # 直达声
    for delay_ms, gain in [(37, 0.6), (71, 0.45), (113, 0.35), (159, 0.25)]:
        ir[int(fs * delay_ms / 1000)] = gain      # 早期反射
    tail_t = np.arange(len(ir)) / fs
    ir += 0.15 * rng.standard_normal(len(ir)) * np.exp(-tail_t * 8)  # 后期稠密尾巴
    return ir


def apply_reverb(in_path, out_path):
    fs, data = wavfile.read(in_path)

    # 统一转成 float，范围约 [-1, 1]，方便卷积
    x = data.astype(np.float64) / 32768.0
    if x.ndim == 1:
        x = x[:, None]                            # 单声道也当作 (N,1) 处理

    ir = make_cave_ir(fs)

    # 逐声道卷积（每个声道各自"住进"同一个山洞）
    wet_channels = [np.convolve(x[:, c], ir) for c in range(x.shape[1])]
    wet = np.stack(wet_channels, axis=1)

    # 归一化，避免叠加后削顶失真
    peak = np.max(np.abs(wet))
    if peak > 0:
        wet = wet / peak * 0.98

    wet_i16 = np.int16(wet * 32767)
    wavfile.write(out_path, fs, wet_i16)
    print(f"{in_path}: fs={fs}, in={data.shape}, out={wet_i16.shape} -> {out_path}")


if __name__ == "__main__":
    apply_reverb("Test-voice.wav", "Test-voice_reverb.wav")
    apply_reverb("Test_music.wav", "Test_music_reverb.wav")
