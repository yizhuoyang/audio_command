import numpy as np
import sounddevice as sd
from scipy.signal import get_window
from scipy.special import softmax
import matplotlib.pyplot as plt

# ============================================================
# Config
# ============================================================
FS = 16000
BLOCK = 1024
SOUND_SPEED = 343.0
N_CHANNELS = 6                 # respeaker 6-channel firmware
RAW_CH = [1, 2, 3, 4]         # raw mic channels
N_ANGLES = 360

# ============================================================
# reSpeaker geometry
# 4-mic diagonal/square layout
# ============================================================
MIC_HALF_SPAN = 0.02285  # meter

# 假设 ch1~ch4 对应四个角，按顺时针或逆时针环绕
MIC_POS = np.array([
    [ MIC_HALF_SPAN,  MIC_HALF_SPAN],   # ch1
    [-MIC_HALF_SPAN,  MIC_HALF_SPAN],   # ch2
    [-MIC_HALF_SPAN, -MIC_HALF_SPAN],   # ch3
    [ MIC_HALF_SPAN, -MIC_HALF_SPAN],   # ch4
], dtype=np.float64)

MIC_PAIRS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (0, 2),
    (1, 3),
]

# ============================================================
# Hyperparameters
# ============================================================
TEMP = 8.0
EMA_ALPHA = 0.7
EPS = 1e-8
ENERGY_TH = 1e-5
INTERP = 8
ANGLE_OFFSET_DEG = 0   # 若后续标定发现整体偏转，可改这里


# ============================================================
# Utility functions
# ============================================================
def unit_vec_from_deg(theta_deg: np.ndarray) -> np.ndarray:
    theta = np.deg2rad(theta_deg)
    return np.stack([np.cos(theta), np.sin(theta)], axis=-1)


def theoretical_tdoa(mic_positions: np.ndarray, angles_deg: np.ndarray, c: float):
    dirs = unit_vec_from_deg(angles_deg)  # [N, 2]
    tdoa_list = []

    for i, j in MIC_PAIRS:
        delta = mic_positions[j] - mic_positions[i]
        tau = dirs @ delta / c
        tdoa_list.append(tau)

    return np.stack(tdoa_list, axis=1)  # [N_angles, N_pairs]


def gcc_phat(sig, refsig, fs=16000, max_tau=None, interp=8):
    n = sig.shape[0] + refsig.shape[0]
    SIG = np.fft.rfft(sig, n=n)
    REF = np.fft.rfft(refsig, n=n)

    R = SIG * np.conj(REF)
    R /= np.maximum(np.abs(R), EPS)

    cc = np.fft.irfft(R, n=interp * n)
    max_shift = int(interp * n / 2)

    if max_tau is not None:
        max_shift = min(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))
    shift = np.arange(-max_shift, max_shift + 1)
    tau_axis = shift / float(interp * fs)

    return cc, tau_axis


def interp1d_np(x, xp, fp):
    return np.interp(x, xp, fp, left=0.0, right=0.0)


def circular_smooth(x, kernel_size=9):
    assert kernel_size % 2 == 1
    pad = kernel_size // 2
    xp = np.concatenate([x[-pad:], x, x[:pad]])
    kernel = np.ones(kernel_size, dtype=np.float64) / kernel_size
    y = np.convolve(xp, kernel, mode="same")
    return y[pad:-pad]


# ============================================================
# Core estimator
# ============================================================
class DOAProbabilityEstimator:
    def __init__(
        self,
        fs=FS,
        mic_positions=MIC_POS,
        sound_speed=SOUND_SPEED,
        n_angles=N_ANGLES,
        temp=TEMP,
        ema_alpha=EMA_ALPHA
    ):
        self.fs = fs
        self.mic_positions = mic_positions
        self.sound_speed = sound_speed
        self.n_angles = n_angles
        self.temp = temp
        self.ema_alpha = ema_alpha

        self.angles = np.arange(n_angles)
        self.tdoa_table = theoretical_tdoa(
            mic_positions=self.mic_positions,
            angles_deg=self.angles,
            c=self.sound_speed
        )  # [360, n_pairs]

        max_dist = 0.0
        for i, j in MIC_PAIRS:
            d = np.linalg.norm(self.mic_positions[j] - self.mic_positions[i])
            max_dist = max(max_dist, d)
        self.max_tau = max_dist / self.sound_speed

        self.prev_prob = np.ones(n_angles, dtype=np.float64) / n_angles
        self.window = get_window("hann", BLOCK)

    def process_block(self, raw4: np.ndarray) -> np.ndarray:
        assert raw4.ndim == 2 and raw4.shape[1] == 4

        x = raw4.astype(np.float64)
        x = x - np.mean(x, axis=0, keepdims=True)
        x = x * self.window[:, None]

        frame_energy = np.mean(x ** 2)
        if frame_energy < ENERGY_TH:
            return self.prev_prob

        pair_scores = []

        for p_idx, (i, j) in enumerate(MIC_PAIRS):
            cc, tau_axis = gcc_phat(
                x[:, i], x[:, j],
                fs=self.fs,
                max_tau=self.max_tau,
                interp=INTERP
            )

            taus = self.tdoa_table[:, p_idx]
            s = interp1d_np(taus, tau_axis, cc)

            s = s - np.min(s)
            denom = np.max(s) - np.min(s)
            if denom > EPS:
                s = s / denom

            pair_scores.append(s)

        pair_scores = np.stack(pair_scores, axis=0)  # [n_pairs, 360]
        score = np.mean(pair_scores, axis=0)
        score = circular_smooth(score, kernel_size=9)

        prob = softmax(score * self.temp)

        prob = self.ema_alpha * self.prev_prob + (1 - self.ema_alpha) * prob
        prob = prob / np.sum(prob)

        self.prev_prob = prob
        return prob


# ============================================================
# Real-time demo
# ============================================================
class ReSpeakerDOA360Demo:
    def __init__(self, device=None):
        self.estimator = DOAProbabilityEstimator()
        self.device = device
        self.last_prob = np.ones(360) / 360

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 4))
        self.line, = self.ax.plot(np.arange(360), self.last_prob)
        self.ax.set_xlim(0, 359)
        self.ax.set_ylim(0, 1)
        self.ax.set_xlabel("Angle (deg)")
        self.ax.set_ylabel("Probability")
        self.ax.set_title("360-D DOA Probability")

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)

        if indata.shape[1] < max(RAW_CH) + 1:
            print(f"Input channels not enough: got {indata.shape[1]}")
            return

        raw4 = indata[:, RAW_CH]
        prob = self.estimator.process_block(raw4)
        self.last_prob = prob

        peak_raw = int(np.argmax(prob))
        peak_calib = (peak_raw + ANGLE_OFFSET_DEG) % 360
        print(f"DOA peak = {peak_calib:3d} deg | raw peak = {peak_raw:3d} | prob = {prob[peak_raw]:.3f}")

    def update_plot(self):
        self.line.set_ydata(self.last_prob)
        peak_raw = int(np.argmax(self.last_prob))
        peak_calib = (peak_raw + ANGLE_OFFSET_DEG) % 360
        self.ax.set_title(f"360-D DOA Probability | Peak={peak_calib} deg")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def run(self):
        with sd.InputStream(
            device=self.device,
            samplerate=FS,
            blocksize=BLOCK,
            channels=N_CHANNELS,
            dtype="float32",
            callback=self.audio_callback
        ):
            print("Running... Press Ctrl+C to stop.")
            try:
                while True:
                    self.update_plot()
                    plt.pause(0.03)
            except KeyboardInterrupt:
                print("Stopped.")


# ============================================================
# Device helper
# ============================================================
def list_input_devices():
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            print(f"{idx}: {dev['name']} | input_channels={dev['max_input_channels']}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("Available input devices:")
    list_input_devices()

    # 改成你的reSpeaker设备编号
    DEVICE_ID = 17

    demo = ReSpeakerDOA360Demo(device=DEVICE_ID)
    demo.run()
