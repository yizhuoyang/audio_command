import argparse
import wave

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import get_window
from scipy.special import softmax


RATE = 16000
SOUND_SPEED = 343.0
DOA_INTERP = 8
DOA_EPS = 1e-8
MIC_HALF_SPAN = 0.02285

MIC_POS = np.array([
    [MIC_HALF_SPAN, MIC_HALF_SPAN],
    [-MIC_HALF_SPAN, MIC_HALF_SPAN],
    [-MIC_HALF_SPAN, -MIC_HALF_SPAN],
    [MIC_HALF_SPAN, -MIC_HALF_SPAN],
], dtype=np.float64)

MIC_PAIRS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (0, 2),
    (1, 3),
]


def unit_vec_from_deg(theta_deg: np.ndarray) -> np.ndarray:
    theta = np.deg2rad(theta_deg)
    return np.stack([np.cos(theta), np.sin(theta)], axis=-1)


def theoretical_tdoa(mic_positions: np.ndarray, angles_deg: np.ndarray, c: float) -> np.ndarray:
    dirs = unit_vec_from_deg(angles_deg)
    tdoa_list = []
    for i, j in MIC_PAIRS:
        delta = mic_positions[j] - mic_positions[i]
        tau = dirs @ delta / c
        tdoa_list.append(tau)
    return np.stack(tdoa_list, axis=1)


def gcc_phat(sig, refsig, fs=16000, max_tau=None, interp=8, low_freq_hz=300.0, high_freq_hz=3000.0):
    n = sig.shape[0] + refsig.shape[0]
    sig_fft = np.fft.rfft(sig, n=n)
    ref_fft = np.fft.rfft(refsig, n=n)

    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    band_mask = (freqs >= low_freq_hz) & (freqs <= high_freq_hz)
    if np.any(band_mask):
        sig_fft = sig_fft * band_mask
        ref_fft = ref_fft * band_mask

    cross_power = sig_fft * np.conj(ref_fft)
    cross_power /= np.maximum(np.abs(cross_power), DOA_EPS)

    cc = np.fft.irfft(cross_power, n=interp * n)
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


class DOAProbabilityEstimator:
    def __init__(
        self,
        fs=RATE,
        block_size=1024,
        mic_positions=MIC_POS,
        sound_speed=SOUND_SPEED,
        n_angles=360,
        temp=8.0,
        ema_alpha=0.7,
        low_freq_hz=300.0,
        high_freq_hz=3000.0,
    ):
        self.fs = fs
        self.block_size = block_size
        self.mic_positions = mic_positions
        self.sound_speed = sound_speed
        self.n_angles = n_angles
        self.temp = temp
        self.ema_alpha = ema_alpha
        self.low_freq_hz = low_freq_hz
        self.high_freq_hz = high_freq_hz

        self.angles = np.arange(n_angles)
        self.tdoa_table = theoretical_tdoa(
            mic_positions=self.mic_positions,
            angles_deg=self.angles,
            c=self.sound_speed,
        )

        max_dist = 0.0
        for i, j in MIC_PAIRS:
            dist = np.linalg.norm(self.mic_positions[j] - self.mic_positions[i])
            max_dist = max(max_dist, dist)
        self.max_tau = max_dist / self.sound_speed

        self.prev_prob = np.ones(n_angles, dtype=np.float64) / n_angles
        self.window = get_window("hann", self.block_size)

    def process_block(self, raw4: np.ndarray):
        if raw4.ndim != 2 or raw4.shape[1] != 4:
            raise ValueError("Expected block shape [n_samples, 4].")
        if raw4.shape[0] != self.block_size:
            raise ValueError(f"Expected block length {self.block_size}, got {raw4.shape[0]}.")

        x = raw4.astype(np.float64)
        x = x - np.mean(x, axis=0, keepdims=True)
        x = x * self.window[:, None]

        pair_scores = []
        for pair_idx, (i, j) in enumerate(MIC_PAIRS):
            cc, tau_axis = gcc_phat(
                x[:, i],
                x[:, j],
                fs=self.fs,
                max_tau=self.max_tau,
                interp=DOA_INTERP,
                low_freq_hz=self.low_freq_hz,
                high_freq_hz=self.high_freq_hz,
            )

            taus = self.tdoa_table[:, pair_idx]
            score = interp1d_np(taus, tau_axis, cc)
            score = score - np.min(score)
            denom = np.max(score) - np.min(score)
            if denom > DOA_EPS:
                score = score / denom
            pair_scores.append(score)

        score = np.mean(np.stack(pair_scores, axis=0), axis=0)
        score = circular_smooth(score, kernel_size=9)
        prob = softmax(score * self.temp)
        prob = self.ema_alpha * self.prev_prob + (1 - self.ema_alpha) * prob
        prob = prob / np.sum(prob)
        self.prev_prob = prob
        return prob


def save_multichannel_wav(path: str, audio_int16: np.ndarray, sample_rate: int):
    if audio_int16.ndim != 2:
        raise ValueError("Expected 2-D audio array for multichannel wav save.")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(audio_int16.shape[1])
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.astype(np.int16).tobytes())


def read_wav_multichannel(path: str):
    audio, sample_rate = sf.read(path, dtype="int16")
    audio = np.asarray(audio, dtype=np.int16)
    if audio.ndim == 1:
        audio = audio[:, None]
    return audio, sample_rate


def list_input_devices():
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            print(f"{idx}: {dev['name']} | input_channels={dev['max_input_channels']}")


def find_input_device(keyword: str):
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0 and keyword.lower() in dev["name"].lower():
            return idx, dev
    return None, None


def record_multichannel(device, seconds: float, channels: int):
    frames = int(seconds * RATE)
    print(f"[INFO] Recording {seconds:.2f}s, channels={channels}, device={device}")
    audio = sd.rec(
        frames,
        samplerate=RATE,
        channels=channels,
        dtype="int16",
        device=device,
        blocking=True,
    )
    return np.asarray(audio, dtype=np.int16)


def ensure_raw4(audio: np.ndarray, raw_mic_channels):
    if len(raw_mic_channels) != 4:
        raise ValueError(f"Expected exactly 4 raw mic channels, got {raw_mic_channels}")

    if audio.ndim != 2:
        raise ValueError("Expected audio with shape [n_samples, n_channels].")

    if audio.shape[1] == 4:
        return audio.copy()

    if any(ch >= audio.shape[1] for ch in raw_mic_channels):
        raise ValueError(
            f"raw_mic_channels={raw_mic_channels} exceed wav/stream channels {audio.shape[1]}"
        )
    return audio[:, raw_mic_channels].copy()


def estimate_doa_from_multichannel_audio(raw_audio_int16: np.ndarray, estimator: DOAProbabilityEstimator):
    if raw_audio_int16.size == 0:
        return None
    if raw_audio_int16.ndim != 2 or raw_audio_int16.shape[1] != 4:
        raise ValueError("DOA estimation expects shape [n_samples, 4].")

    probs = []
    for start in range(0, raw_audio_int16.shape[0], estimator.block_size):
        block = raw_audio_int16[start:start + estimator.block_size]
        if block.shape[0] < estimator.block_size:
            pad = np.zeros((estimator.block_size - block.shape[0], block.shape[1]), dtype=np.int16)
            block = np.vstack([block, pad])
        probs.append(estimator.process_block(block))

    if not probs:
        return None

    mean_prob = np.mean(np.stack(probs, axis=0), axis=0)
    mean_prob = mean_prob / max(np.sum(mean_prob), DOA_EPS)
    peak_raw = int(np.argmax(mean_prob))
    return {
        "prob": mean_prob,
        "peak_raw": peak_raw,
        "peak_confidence": float(mean_prob[peak_raw]),
        "num_blocks": len(probs),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_path", type=str, default=None)
    parser.add_argument("--record_seconds", type=float, default=None)
    parser.add_argument("--device_keyword", type=str, default="USB Audio")
    parser.add_argument("--device_id", type=int, default=None)
    parser.add_argument("--input_channels", type=int, default=6)
    parser.add_argument("--raw_mic_channels", type=str, default="1,2,3,4")
    parser.add_argument("--doa_block_size", type=int, default=1024)
    parser.add_argument("--doa_n_angles", type=int, default=360)
    parser.add_argument("--doa_temp", type=float, default=8.0)
    parser.add_argument("--doa_ema_alpha", type=float, default=0.7)
    parser.add_argument("--doa_angle_offset_deg", type=int, default=0)
    parser.add_argument("--doa_low_freq_hz", type=float, default=200.0)
    parser.add_argument("--doa_high_freq_hz", type=float, default=5000.0)
    parser.add_argument("--save_recorded_wav", type=str, default=None)
    parser.add_argument("--list_devices", action="store_true")
    args = parser.parse_args()

    if args.list_devices:
        list_input_devices()
        return

    if not args.wav_path and args.record_seconds is None:
        raise SystemExit("Please provide either --wav_path or --record_seconds.")

    raw_mic_channels = [int(x.strip()) for x in args.raw_mic_channels.split(",") if x.strip()]
    estimator = DOAProbabilityEstimator(
        fs=RATE,
        block_size=args.doa_block_size,
        mic_positions=MIC_POS,
        sound_speed=SOUND_SPEED,
        n_angles=args.doa_n_angles,
        temp=args.doa_temp,
        ema_alpha=args.doa_ema_alpha,
        low_freq_hz=args.doa_low_freq_hz,
        high_freq_hz=args.doa_high_freq_hz,
    )

    if args.wav_path:
        audio, sample_rate = read_wav_multichannel(args.wav_path)
        if sample_rate != RATE:
            raise ValueError(f"Expected wav sample rate {RATE}, got {sample_rate}")
        raw4 = ensure_raw4(audio, raw_mic_channels)
        print(f"[INFO] Loaded wav: {args.wav_path}")
    else:
        if args.device_id is not None:
            device_id = args.device_id
        else:
            device_id, device_info = find_input_device(args.device_keyword)
            if device_id is None:
                list_input_devices()
                raise SystemExit(f"Could not find input device with keyword: {args.device_keyword}")
            print(f"[INFO] Using device {device_id}: {device_info['name']}")

        audio = record_multichannel(
            device=device_id,
            seconds=args.record_seconds,
            channels=args.input_channels,
        )
        raw4 = ensure_raw4(audio, raw_mic_channels)
        if args.save_recorded_wav:
            save_multichannel_wav(args.save_recorded_wav, raw4, RATE)
            print(f"[INFO] Saved recorded raw4 wav to {args.save_recorded_wav}")

    result = estimate_doa_from_multichannel_audio(raw4, estimator)
    if result is None:
        print("[DOA] Failed to estimate angle.")
        return

    peak_calib = (result["peak_raw"] + args.doa_angle_offset_deg) % args.doa_n_angles
    print(
        "[DOA] peak="
        f"{peak_calib} deg "
        f"(raw={result['peak_raw']} deg, "
        f"confidence={result['peak_confidence']:.3f}, "
        f"blocks={result['num_blocks']})"
    )


if __name__ == "__main__":
    main()
