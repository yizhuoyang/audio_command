import os
import io
import re
import time
import wave
import base64
import argparse
from collections import deque

import pyaudio
import numpy as np
import soundfile as sf
import torch
from openai import OpenAI
from openwakeword.model import Model
from scipy.signal import get_window
from scipy.special import softmax
from silero_vad import VADIterator, get_speech_timestamps, load_silero_vad

from tts_module import PiperInterruptSpeaker


# =========================
# Parse input arguments
# =========================
parser = argparse.ArgumentParser()

parser.add_argument("--chunk_size", type=int, default=1280)
parser.add_argument("--model_path", type=str,
                    default="/home/kemove/yyz/audio-nav/openWakeWord/hey_jarvis_v0.1.tflite")
parser.add_argument("--inference_framework", type=str, default="tflite")
parser.add_argument("--device_keyword", type=str, default="USB Audio")

parser.add_argument("--input_channels", type=int, default=6)
parser.add_argument("--mono_channel", type=int, default=0)
parser.add_argument("--raw_mic_channels", type=str, default="1,2,3,4")

parser.add_argument("--wake_threshold", type=float, default=0.3)
parser.add_argument("--wake_release_threshold", type=float, default=0.15)
parser.add_argument("--wake_rearm_cooldown_seconds", type=float, default=1.2)
parser.add_argument("--wake_release_frames_required", type=int, default=4)

parser.add_argument("--max_command_seconds", type=float, default=8.0)
parser.add_argument("--post_speech_silence_seconds", type=float, default=0.8)
parser.add_argument("--pre_roll_seconds", type=float, default=0.5)
parser.add_argument("--min_valid_speech_seconds", type=float, default=0.4)
parser.add_argument("--command_start_grace_seconds", type=float, default=1.2)
parser.add_argument("--min_command_record_seconds", type=float, default=1.0)
parser.add_argument("--min_speech_after_start_seconds", type=float, default=1.6)
parser.add_argument("--fast_end_min_speech_seconds", type=float, default=1.2)
parser.add_argument("--min_speech_chunks_required", type=int, default=3)
parser.add_argument("--end_event_required_count", type=int, default=2)
parser.add_argument("--silence_chunks_required", type=int, default=10)
parser.add_argument("--tail_silence_keep_seconds", type=float, default=0.24)
parser.add_argument("--fast_end_silence_seconds", type=float, default=0.28)
parser.add_argument("--fast_end_chunks_required", type=int, default=3)
parser.add_argument("--stream_vad_threshold", type=float, default=0.35)
parser.add_argument("--stream_vad_min_silence_ms", type=int, default=250)
parser.add_argument("--stream_vad_speech_pad_ms", type=int, default=320)
parser.add_argument("--offline_vad_threshold", type=float, default=0.35)
parser.add_argument("--offline_min_speech_ms", type=int, default=200)
parser.add_argument("--offline_min_silence_ms", type=int, default=350)
parser.add_argument("--offline_speech_pad_ms", type=int, default=320)
parser.add_argument("--offline_merge_gap_ms", type=int, default=700)

parser.add_argument("--tts_model", type=str, default="/home/kemove/en_US-lessac-medium.onnx")
parser.add_argument("--tts_volume", type=float, default=0.1)
parser.add_argument("--tts_length_scale", type=float, default=1.0)
parser.add_argument("--tts_reply_text", type=str, default="Hi, how can I help you")
parser.add_argument("--enable_tts", action="store_true")

parser.add_argument("--save_path", type=str, default="captured_command.wav")
parser.add_argument("--save_debug_raw_path", type=str, default="captured_command_raw.wav")
parser.add_argument("--save_multichannel_path", type=str, default="captured_command_multichannel.wav")
parser.add_argument("--save_debug_audio", action="store_true")
parser.add_argument("--enable_visualization", action="store_true")
parser.add_argument("--visual_refresh_seconds", type=float, default=0.08)
parser.add_argument("--visual_wave_seconds", type=float, default=2.0)

parser.add_argument("--doa_block_size", type=int, default=1024)
parser.add_argument("--doa_n_angles", type=int, default=360)
parser.add_argument("--doa_temp", type=float, default=8.0)
parser.add_argument("--doa_ema_alpha", type=float, default=0.7)
parser.add_argument("--doa_angle_offset_deg", type=int, default=0)
parser.add_argument("--doa_low_freq_hz", type=float, default=300.0)
parser.add_argument("--doa_high_freq_hz", type=float, default=3000.0)

# ASR related
parser.add_argument("--asr_model_name", type=str, default="qwen-omni-turbo")
parser.add_argument("--dashscope_base_url", type=str,
                    default="https://dashscope.aliyuncs.com/compatible-mode/v1")
parser.add_argument("--system_prompt_path", type=str, default="/home/kemove/yyz/audio-nav/openWakeWord/robot.txt")
parser.add_argument("--enable_asr", action="store_true")
parser.add_argument("--force_plain_asr_prompt", action="store_true")

args = parser.parse_args()


# =========================
# Constants
# =========================
FORMAT = pyaudio.paInt16
RATE = 16000
CHUNK = args.chunk_size
INPUT_CHANNELS = args.input_channels
MONO_CHANNEL = args.mono_channel
RAW_CH = [int(x.strip()) for x in args.raw_mic_channels.split(",") if x.strip() != ""]
RAW_CH_COUNT = len(RAW_CH)

SAMPLING_RATE = RATE
VAD_WINDOW_SIZE = 512
SOUND_SPEED = 343.0
DOA_INTERP = 8
DOA_EPS = 1e-8
MIC_HALF_SPAN = 0.02285
DOA_LOW_FREQ_HZ = args.doa_low_freq_hz
DOA_HIGH_FREQ_HZ = args.doa_high_freq_hz

WAKE_TRIGGER_THRESHOLD = args.wake_threshold
WAKE_RELEASE_THRESHOLD = args.wake_release_threshold
WAKE_REARM_COOLDOWN_SECONDS = args.wake_rearm_cooldown_seconds
WAKE_RELEASE_FRAMES_REQUIRED = args.wake_release_frames_required
MAX_COMMAND_SECONDS = args.max_command_seconds
POST_SPEECH_SILENCE_SECONDS = args.post_speech_silence_seconds
PRE_ROLL_SECONDS = args.pre_roll_seconds
MIN_VALID_SPEECH_SECONDS = args.min_valid_speech_seconds
COMMAND_START_GRACE_SECONDS = args.command_start_grace_seconds
MIN_COMMAND_RECORD_SECONDS = args.min_command_record_seconds
MIN_SPEECH_AFTER_START_SECONDS = args.min_speech_after_start_seconds
FAST_END_MIN_SPEECH_SECONDS = args.fast_end_min_speech_seconds
MIN_SPEECH_CHUNKS_REQUIRED = args.min_speech_chunks_required
END_EVENT_REQUIRED_COUNT = args.end_event_required_count
SILENCE_CHUNKS_REQUIRED = args.silence_chunks_required
TAIL_SILENCE_KEEP_SECONDS = args.tail_silence_keep_seconds
FAST_END_SILENCE_SECONDS = args.fast_end_silence_seconds
FAST_END_CHUNKS_REQUIRED = args.fast_end_chunks_required

STATE_WAKEWORD = "WAKEWORD"
STATE_TTS_REPLY = "TTS_REPLY"
STATE_COMMAND = "COMMAND_CAPTURE"

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-acc74a8fc8744fb6a4db540d4a1de0a1")
DASHSCOPE_BASE_URL = args.dashscope_base_url
ASR_MODEL_NAME = args.asr_model_name
SYSTEM_PROMPT_PATH = args.system_prompt_path

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


# =========================
# DOA helpers
# =========================
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

        pair_scores = np.stack(pair_scores, axis=0)
        score = np.mean(pair_scores, axis=0)
        score = circular_smooth(score, kernel_size=9)
        prob = softmax(score * self.temp)
        prob = self.ema_alpha * self.prev_prob + (1 - self.ema_alpha) * prob
        prob = prob / np.sum(prob)
        self.prev_prob = prob
        return prob


# =========================
# Helpers
# =========================
def find_input_device(audio_interface, keyword="USB Audio"):
    matched_devices = []
    input_devices = []

    for i in range(audio_interface.get_device_count()):
        info = audio_interface.get_device_info_by_index(i)
        name = info.get("name", "")
        max_input_channels = int(info.get("maxInputChannels", 0))

        if max_input_channels > 0:
            input_devices.append((i, name, max_input_channels))
            if keyword.lower() in name.lower():
                matched_devices.append((i, info))

    if matched_devices:
        return matched_devices[0]

    print(f"[ERROR] No input device found with keyword: '{keyword}'")
    print("[INFO] Available input devices:")
    for idx, name, ch in input_devices:
        print(f"  {idx}: {name} (input channels: {ch})")
    return None, None


def save_wav(path, audio_int16, sample_rate=16000):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


def save_multichannel_wav(path, audio_int16, sample_rate=16000):
    if audio_int16.ndim != 2:
        raise ValueError("Expected 2-D audio array for multichannel wav save.")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(audio_int16.shape[1])
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.astype(np.int16).tobytes())


def flush_mic_stream(stream, chunk_size, num_chunks=8):
    for _ in range(num_chunks):
        try:
            stream.read(chunk_size, exception_on_overflow=False)
        except Exception:
            pass


def load_system_prompt(path: str) -> str:
    default_prompt = (
        'Transcribe the spoken English audio only. '
        'Return exactly one line in the format: ASR: "your transcription". '
        'If the audio has no clear speech, return exactly: ASR: None'
    )
    if args.force_plain_asr_prompt:
        return default_prompt
    if not os.path.exists(path):
        return default_prompt
    with open(path, "r", encoding="utf-8") as f:
        prompt = f.read().strip()
    if not prompt:
        return default_prompt
    return prompt


def build_asr_messages(system_prompt: str, base64_audio: str):
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        'Please only transcribe the speech content from this audio. '
                        'Do not infer actions. '
                        'Output exactly one line: ASR: "..." or ASR: None'
                    ),
                },
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": f"data:;base64,{base64_audio}",
                        "format": "wav",
                    },
                },
            ],
        },
    ]


def parse_asr_text(full_text: str) -> str:
    match = re.search(r'ASR:\s*(?:"([^"]*?)"|“([^”]*?)”|([^\n\r]*))', full_text)
    if match:
        return next(g for g in match.groups() if g is not None).strip()
    return full_text.strip()


def remove_dc_and_normalize(audio_np_int16: np.ndarray) -> np.ndarray:
    if audio_np_int16.size == 0:
        return audio_np_int16
    audio_f32 = audio_np_int16.astype(np.float32)
    audio_f32 = audio_f32 - np.mean(audio_f32)
    peak = np.max(np.abs(audio_f32))
    if peak < 1.0:
        return audio_np_int16
    target_peak = 0.92 * 32767.0
    gain = min(target_peak / peak, 8.0)
    audio_f32 = np.clip(audio_f32 * gain, -32768.0, 32767.0)
    return audio_f32.astype(np.int16)


def denoise_gate(audio_np_int16: np.ndarray) -> np.ndarray:
    if audio_np_int16.size == 0:
        return audio_np_int16
    audio_f32 = audio_np_int16.astype(np.float32)
    noise_floor = max(np.percentile(np.abs(audio_f32), 20), 80.0)
    gated = audio_f32.copy()
    for start in range(0, len(gated), VAD_WINDOW_SIZE):
        end = min(start + VAD_WINDOW_SIZE, len(gated))
        frame = gated[start:end]
        if np.max(np.abs(frame)) < noise_floor * 1.8:
            gated[start:end] *= 0.15
    return np.clip(gated, -32768.0, 32767.0).astype(np.int16)


def preprocess_audio_for_asr(audio_np_int16: np.ndarray) -> np.ndarray:
    processed = remove_dc_and_normalize(audio_np_int16)
    processed = denoise_gate(processed)
    processed = remove_dc_and_normalize(processed)
    return processed


def run_dashscope_asr(audio_np_int16: np.ndarray, sample_rate: int, system_prompt: str) -> str:
    client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)
    tbuffer = io.BytesIO()
    sf.write(tbuffer, audio_np_int16, sample_rate, format="WAV", subtype="PCM_16")
    base64_audio = base64.b64encode(tbuffer.getvalue()).decode("utf-8")

    t0 = time.perf_counter()
    completion = client.chat.completions.create(
        model=ASR_MODEL_NAME,
        messages=build_asr_messages(system_prompt, base64_audio),
        modalities=["text"],
        stream=True,
        stream_options={"include_usage": False},
    )

    full_text = ""
    for c in completion:
        if c.choices:
            delta = c.choices[0].delta
            if getattr(delta, "content", None) is not None:
                full_text += delta.content

    print("[ASR] raw:", full_text, f"t={time.perf_counter() - t0:.2f}s")
    return parse_asr_text(full_text)


def transcribe_audio_with_dashscope(audio_np_int16: np.ndarray, sample_rate: int) -> str:
    if not DASHSCOPE_API_KEY:
        raise RuntimeError("DASHSCOPE_API_KEY is empty. Please set environment variable DASHSCOPE_API_KEY.")
    system_prompt = load_system_prompt(SYSTEM_PROMPT_PATH)
    processed_audio = preprocess_audio_for_asr(audio_np_int16)
    first_pass = run_dashscope_asr(processed_audio, sample_rate, system_prompt)
    if first_pass and first_pass.lower() not in {"none", "asr: none"}:
        return first_pass
    fallback_prompt = (
        'Transcribe the spoken English audio only. '
        'Return exactly one line in the format: ASR: "your transcription". '
        'If the audio has no clear speech, return exactly: ASR: None'
    )
    print("[ASR] First pass was empty or None, retrying with plain ASR prompt...")
    return run_dashscope_asr(processed_audio, sample_rate, fallback_prompt)


def split_interleaved_chunk(audio_chunk: np.ndarray):
    if audio_chunk.size % INPUT_CHANNELS != 0:
        raise ValueError(
            f"Audio chunk size {audio_chunk.size} is not divisible by input channels {INPUT_CHANNELS}."
        )
    multi = audio_chunk.reshape(-1, INPUT_CHANNELS)
    if MONO_CHANNEL >= multi.shape[1]:
        raise ValueError(f"mono_channel={MONO_CHANNEL} is out of range for {multi.shape[1]} channels.")
    if any(ch >= multi.shape[1] for ch in RAW_CH):
        raise ValueError(f"raw_mic_channels={RAW_CH} exceed available input channels {multi.shape[1]}.")
    mono = multi[:, MONO_CHANNEL].copy()
    raw4 = multi[:, RAW_CH].copy()
    return mono, raw4, multi


def estimate_doa_from_multichannel_audio(raw_audio_int16: np.ndarray):
    if raw_audio_int16.size == 0:
        return None
    if raw_audio_int16.ndim != 2 or raw_audio_int16.shape[1] != 4:
        raise ValueError("DOA estimation expects shape [n_samples, 4].")

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

    probs = []
    for start in range(0, raw_audio_int16.shape[0], args.doa_block_size):
        block = raw_audio_int16[start:start + args.doa_block_size]
        if block.shape[0] < args.doa_block_size:
            pad = np.zeros((args.doa_block_size - block.shape[0], block.shape[1]), dtype=np.int16)
            block = np.vstack([block, pad])
        probs.append(estimator.process_block(block))

    if not probs:
        return None

    probs_arr = np.stack(probs, axis=0)
    mean_prob = np.mean(probs_arr, axis=0)
    mean_prob = mean_prob / max(np.sum(mean_prob), DOA_EPS)
    peak_raw = int(np.argmax(mean_prob))
    peak_calib = (peak_raw + args.doa_angle_offset_deg) % args.doa_n_angles
    return {
        "prob": mean_prob,
        "peak_raw": peak_raw,
        "peak_calib": peak_calib,
        "peak_confidence": float(mean_prob[peak_raw]),
        "num_blocks": len(probs),
    }


# =========================
# Init audio / models
# =========================
if RAW_CH_COUNT != 4:
    raise ValueError(f"Expected exactly 4 raw mic channels for DOA, got {RAW_CH_COUNT}: {RAW_CH}")

pa = pyaudio.PyAudio()
device_index, device_info = find_input_device(pa, args.device_keyword)
if device_index is None:
    pa.terminate()
    raise SystemExit(1)

device_input_channels = int(device_info["maxInputChannels"])
if INPUT_CHANNELS > device_input_channels:
    print(
        f"[WARN] Requested input_channels={INPUT_CHANNELS}, but device only reports "
        f"{device_input_channels}. Falling back to {device_input_channels}."
    )
    INPUT_CHANNELS = device_input_channels

print("=" * 100)
print(f"Using input device index: {device_index}")
print(f"Using input device name : {device_info['name']}")
print(f"Input channels         : {INPUT_CHANNELS}")
print(f"Mono channel           : {MONO_CHANNEL}")
print(f"Raw mic channels       : {RAW_CH}")
print("=" * 100)

mic_stream = pa.open(
    format=FORMAT,
    channels=INPUT_CHANNELS,
    rate=RATE,
    input=True,
    input_device_index=device_index,
    frames_per_buffer=CHUNK
)

if args.model_path:
    owwModel = Model(
        wakeword_models=[args.model_path],
        inference_framework=args.inference_framework
    )
else:
    owwModel = Model(inference_framework=args.inference_framework)

vad_model = load_silero_vad()
vad_iterator = VADIterator(
    vad_model,
    threshold=args.stream_vad_threshold,
    sampling_rate=SAMPLING_RATE,
    min_silence_duration_ms=args.stream_vad_min_silence_ms,
    speech_pad_ms=args.stream_vad_speech_pad_ms,
)

tts = None
if args.enable_tts:
    tts = PiperInterruptSpeaker(
        model_path=args.tts_model,
        length_scale=args.tts_length_scale,
        volume=args.tts_volume,
    )


# =========================
# Runtime state
# =========================
state = STATE_WAKEWORD
wakeword_enabled = True
wake_armed = True
wake_cooldown_until = 0.0
wake_release_frame_count = 0

vad_leftover = np.array([], dtype=np.int16)
captured_speech_chunks = []
captured_raw_chunks = []
pre_roll_max_chunks = max(1, int(np.ceil(PRE_ROLL_SECONDS * RATE / CHUNK)))
pre_roll_buffer = deque(maxlen=pre_roll_max_chunks)
pre_roll_raw_buffer = deque(maxlen=pre_roll_max_chunks)
tail_silence_max_chunks = max(1, int(np.ceil(TAIL_SILENCE_KEEP_SECONDS * RATE / CHUNK)))
tail_silence_buffer = deque(maxlen=tail_silence_max_chunks)
tail_silence_raw_buffer = deque(maxlen=tail_silence_max_chunks)

speech_started = False
last_voice_activity_time = None
command_start_time = None
speech_first_detected_time = None
speech_chunk_count = 0
silent_chunk_count = 0
end_event_count = 0
end_candidate_active = False


# =========================
# State helpers
# =========================
def reset_wakeword_detector_state():
    try:
        owwModel.reset()
    except Exception:
        try:
            for mdl in owwModel.prediction_buffer.keys():
                buf = owwModel.prediction_buffer[mdl]
                if hasattr(buf, "clear"):
                    buf.clear()
        except Exception:
            pass


def reset_command_state():
    global vad_leftover, captured_speech_chunks, captured_raw_chunks
    global pre_roll_buffer, pre_roll_raw_buffer, tail_silence_buffer, tail_silence_raw_buffer
    global speech_started, last_voice_activity_time, command_start_time
    global speech_first_detected_time, speech_chunk_count, silent_chunk_count, end_event_count
    global end_candidate_active

    vad_leftover = np.array([], dtype=np.int16)
    captured_speech_chunks = []
    captured_raw_chunks = []
    pre_roll_buffer = deque(maxlen=pre_roll_max_chunks)
    pre_roll_raw_buffer = deque(maxlen=pre_roll_max_chunks)
    tail_silence_buffer = deque(maxlen=tail_silence_max_chunks)
    tail_silence_raw_buffer = deque(maxlen=tail_silence_max_chunks)
    speech_started = False
    last_voice_activity_time = None
    command_start_time = None
    speech_first_detected_time = None
    speech_chunk_count = 0
    silent_chunk_count = 0
    end_event_count = 0
    end_candidate_active = False
    vad_iterator.reset_states()


def process_vad_stream(audio_chunk_int16):
    global vad_leftover
    events = []
    merged = np.concatenate([vad_leftover, audio_chunk_int16])
    num_full = len(merged) // VAD_WINDOW_SIZE
    usable_len = num_full * VAD_WINDOW_SIZE

    if usable_len > 0:
        usable = merged[:usable_len]
        vad_leftover = merged[usable_len:]
    else:
        usable = np.array([], dtype=np.int16)
        vad_leftover = merged

    for i in range(0, len(usable), VAD_WINDOW_SIZE):
        frame = usable[i:i + VAD_WINDOW_SIZE]
        frame_f32 = frame.astype(np.float32) / 32768.0
        speech_dict = vad_iterator(frame_f32, return_seconds=True)
        if speech_dict:
            events.append(speech_dict)
    return events


def refine_command_audio(audio_int16: np.ndarray):
    if audio_int16.size == 0:
        return audio_int16, []

    audio_tensor = torch.from_numpy(audio_int16.astype(np.float32) / 32768.0)
    timestamps = get_speech_timestamps(
        audio_tensor,
        vad_model,
        threshold=args.offline_vad_threshold,
        sampling_rate=SAMPLING_RATE,
        min_speech_duration_ms=args.offline_min_speech_ms,
        min_silence_duration_ms=args.offline_min_silence_ms,
        speech_pad_ms=args.offline_speech_pad_ms,
        return_seconds=False,
        window_size_samples=VAD_WINDOW_SIZE,
    )

    if not timestamps:
        return audio_int16, []

    merged_segments = []
    merge_gap_samples = int(RATE * args.offline_merge_gap_ms / 1000.0)
    for seg in timestamps:
        start = int(seg["start"])
        end = int(seg["end"])
        if not merged_segments:
            merged_segments.append([start, end])
            continue
        prev_start, prev_end = merged_segments[-1]
        if start - prev_end <= merge_gap_samples:
            merged_segments[-1][1] = max(prev_end, end)
        else:
            merged_segments.append([start, end])

    refined_chunks = [audio_int16[start:end] for start, end in merged_segments if end > start]
    if not refined_chunks:
        return audio_int16, []

    refined_audio = np.concatenate(refined_chunks, axis=0)
    return refined_audio.astype(np.int16), merged_segments


def apply_segments_to_multichannel(audio_int16: np.ndarray, segments):
    if audio_int16.size == 0 or audio_int16.ndim != 2:
        return audio_int16
    if not segments:
        return audio_int16

    clipped_chunks = []
    total_samples = audio_int16.shape[0]
    for start, end in segments:
        start = max(0, min(int(start), total_samples))
        end = max(0, min(int(end), total_samples))
        if end > start:
            clipped_chunks.append(audio_int16[start:end])

    if not clipped_chunks:
        return audio_int16
    return np.concatenate(clipped_chunks, axis=0).astype(np.int16)


class VoiceAssistantVisualizer:
    def __init__(self, wave_seconds: float, refresh_seconds: float, wake_trigger: float, wake_release: float):
        import matplotlib.pyplot as plt

        self.plt = plt
        self.refresh_seconds = refresh_seconds
        self.last_refresh_time = 0.0
        self.wave_buffer = deque(maxlen=max(1, int(wave_seconds * RATE / CHUNK)))
        self.wake_history = deque(maxlen=200)
        self.latest_doa_prob = np.ones(args.doa_n_angles, dtype=np.float64) / args.doa_n_angles
        self.latest_doa_peak = None
        self.latest_doa_confidence = 0.0
        self.latest_asr_text = ""
        self.state_text = STATE_WAKEWORD
        self.detail_lines = []

        plt.ion()
        self.fig, axes = plt.subplots(2, 2, figsize=(12, 7))
        self.ax_wave = axes[0, 0]
        self.ax_wake = axes[0, 1]
        self.ax_doa = axes[1, 0]
        self.ax_text = axes[1, 1]

        wave_samples = max(1, int(wave_seconds * RATE))
        self.wave_x = np.linspace(-wave_seconds, 0.0, wave_samples)
        self.wave_line, = self.ax_wave.plot(self.wave_x, np.zeros(wave_samples), color="#0f766e", linewidth=1.2)
        self.ax_wave.set_title("Recent Audio")
        self.ax_wave.set_xlim(-wave_seconds, 0.0)
        self.ax_wave.set_ylim(-1.0, 1.0)
        self.ax_wave.set_xlabel("Seconds")
        self.ax_wave.set_ylabel("Amplitude")
        self.ax_wave.grid(alpha=0.2)

        self.wake_line, = self.ax_wake.plot([], [], color="#ea580c", linewidth=1.6)
        self.ax_wake.axhline(wake_trigger, color="#dc2626", linestyle="--", linewidth=1.0, label="trigger")
        self.ax_wake.axhline(wake_release, color="#2563eb", linestyle="--", linewidth=1.0, label="release")
        self.ax_wake.set_title("Wakeword Score")
        self.ax_wake.set_xlim(0, 200)
        self.ax_wake.set_ylim(0.0, 1.0)
        self.ax_wake.set_xlabel("Recent frames")
        self.ax_wake.set_ylabel("Score")
        self.ax_wake.legend(loc="upper right")
        self.ax_wake.grid(alpha=0.2)

        doa_x = np.arange(args.doa_n_angles)
        self.doa_line, = self.ax_doa.plot(doa_x, self.latest_doa_prob, color="#7c3aed", linewidth=1.4)
        self.ax_doa.set_title("DOA Probability")
        self.ax_doa.set_xlim(0, args.doa_n_angles - 1)
        self.ax_doa.set_ylim(0.0, 1.0)
        self.ax_doa.set_xlabel("Angle (deg)")
        self.ax_doa.set_ylabel("Probability")
        self.ax_doa.grid(alpha=0.2)

        self.ax_text.axis("off")
        self.text_box = self.ax_text.text(
            0.02,
            0.98,
            "",
            va="top",
            ha="left",
            fontsize=11,
            family="monospace",
        )

        self.fig.tight_layout()

    def push_audio(self, mono_chunk: np.ndarray):
        self.wave_buffer.append(mono_chunk.astype(np.float32) / 32768.0)

    def push_wake_score(self, score: float):
        self.wake_history.append(float(score))

    def set_state(self, state_text: str, detail_lines=None):
        self.state_text = state_text
        self.detail_lines = detail_lines or []

    def set_doa_result(self, doa_result):
        if doa_result is None:
            return
        self.latest_doa_prob = doa_result["prob"]
        self.latest_doa_peak = doa_result["peak_calib"]
        self.latest_doa_confidence = doa_result["peak_confidence"]

    def set_asr_text(self, text: str):
        self.latest_asr_text = text or ""

    def refresh(self, force: bool = False):
        now = time.time()
        if (not force) and (now - self.last_refresh_time < self.refresh_seconds):
            return
        self.last_refresh_time = now

        if self.wave_buffer:
            wave = np.concatenate(list(self.wave_buffer), axis=0)
        else:
            wave = np.zeros_like(self.wave_x)
        if wave.size < self.wave_x.size:
            wave = np.pad(wave, (self.wave_x.size - wave.size, 0))
        else:
            wave = wave[-self.wave_x.size:]
        self.wave_line.set_ydata(wave)

        scores = list(self.wake_history)
        if scores:
            x = np.arange(len(scores))
            self.wake_line.set_data(x, scores)
            self.ax_wake.set_xlim(0, max(200, len(scores)))
        else:
            self.wake_line.set_data([], [])
            self.ax_wake.set_xlim(0, 200)

        self.doa_line.set_ydata(self.latest_doa_prob)
        doa_title = "DOA Probability"
        if self.latest_doa_peak is not None:
            doa_title += f" | peak={self.latest_doa_peak} deg | conf={self.latest_doa_confidence:.3f}"
        self.ax_doa.set_title(doa_title)
        peak_val = float(np.max(self.latest_doa_prob)) if self.latest_doa_prob.size else 1.0
        self.ax_doa.set_ylim(0.0, max(0.15, min(1.0, peak_val * 1.15)))

        text_lines = [f"State: {self.state_text}"]
        text_lines.extend(self.detail_lines[:6])
        if self.latest_asr_text:
            text_lines.append("")
            text_lines.append(f"ASR: {self.latest_asr_text[:180]}")
        self.text_box.set_text("\n".join(text_lines))

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


visualizer = None
if args.enable_visualization:
    visualizer = VoiceAssistantVisualizer(
        wave_seconds=args.visual_wave_seconds,
        refresh_seconds=args.visual_refresh_seconds,
        wake_trigger=WAKE_TRIGGER_THRESHOLD,
        wake_release=WAKE_RELEASE_THRESHOLD,
    )


def chunk_has_energy(audio_chunk_int16: np.ndarray) -> bool:
    if audio_chunk_int16.size == 0:
        return False
    rms = float(np.sqrt(np.mean(np.square(audio_chunk_int16.astype(np.float32)))))
    peak = float(np.max(np.abs(audio_chunk_int16.astype(np.float32))))
    return rms >= 180.0 or peak >= 1200.0


def append_speech_chunk(mono_chunk: np.ndarray, raw4_chunk: np.ndarray):
    global captured_speech_chunks, captured_raw_chunks
    global tail_silence_buffer, tail_silence_raw_buffer
    if len(tail_silence_buffer) > 0:
        captured_speech_chunks.extend(list(tail_silence_buffer))
        tail_silence_buffer.clear()
    if len(tail_silence_raw_buffer) > 0:
        captured_raw_chunks.extend(list(tail_silence_raw_buffer))
        tail_silence_raw_buffer.clear()
    captured_speech_chunks.append(mono_chunk.copy())
    captured_raw_chunks.append(raw4_chunk.copy())


def append_tail_silence_chunk(mono_chunk: np.ndarray, raw4_chunk: np.ndarray):
    global tail_silence_buffer, tail_silence_raw_buffer
    tail_silence_buffer.append(mono_chunk.copy())
    tail_silence_raw_buffer.append(raw4_chunk.copy())


def enter_tts_reply_stage():
    global state, wakeword_enabled

    print("\n" + "=" * 100)
    print("[INFO] Wakeword triggered.")
    print("=" * 100)

    wakeword_enabled = False
    state = STATE_TTS_REPLY

    if tts is not None:
        try:
            print("[INFO] Playing TTS reply...")
            tts.speak(args.tts_reply_text)
            tts.wait_idle(timeout=10)
        except Exception as e:
            print(f"[WARN] TTS failed: {e}")
        time.sleep(0.25)

    flush_mic_stream(mic_stream, CHUNK, num_chunks=8)
    reset_wakeword_detector_state()
    reset_command_state()
    state = STATE_COMMAND
    print("[INFO] Entered COMMAND_CAPTURE stage.")


def finalize_command_and_return_to_wakeword():
    global state, wakeword_enabled, wake_armed, wake_cooldown_until, wake_release_frame_count

    if len(captured_speech_chunks) > 0:
        command_audio = np.concatenate(captured_speech_chunks, axis=0)
    else:
        command_audio = np.array([], dtype=np.int16)

    if len(captured_raw_chunks) > 0:
        command_raw_audio = np.concatenate(captured_raw_chunks, axis=0)
    else:
        command_raw_audio = np.empty((0, 4), dtype=np.int16)

    raw_speech_len_sec = len(command_audio) / RATE
    refined_audio, refined_segments = refine_command_audio(command_audio)
    refined_raw_audio = apply_segments_to_multichannel(command_raw_audio, refined_segments)
    processed_audio = preprocess_audio_for_asr(refined_audio)
    speech_len_sec = len(processed_audio) / RATE
    doa_audio = refined_raw_audio
    doa_len_sec = 0.0 if doa_audio.size == 0 else doa_audio.shape[0] / RATE

    if speech_len_sec < MIN_VALID_SPEECH_SECONDS and raw_speech_len_sec >= MIN_VALID_SPEECH_SECONDS:
        print("[INFO] Refined audio became too short, falling back to raw captured command.")
        processed_audio = preprocess_audio_for_asr(command_audio)
        speech_len_sec = len(processed_audio) / RATE
        refined_segments = []
        doa_audio = command_raw_audio
        doa_len_sec = 0.0 if doa_audio.size == 0 else doa_audio.shape[0] / RATE

    print("=" * 100)
    print(f"[INFO] Raw captured length     : {raw_speech_len_sec:.2f} sec")
    print(f"[INFO] Refined command length : {speech_len_sec:.2f} sec")
    print(f"[INFO] DOA audio length       : {doa_len_sec:.2f} sec")
    if refined_segments:
        print(f"[INFO] Offline VAD kept {len(refined_segments)} segment(s): {refined_segments}")
    else:
        print("[INFO] Offline VAD found no extra trim points, using raw capture.")
    print("=" * 100)

    if visualizer is not None:
        visualizer.set_state(
            "FINALIZING",
            [
                f"raw_len={raw_speech_len_sec:.2f}s",
                f"refined_len={speech_len_sec:.2f}s",
                f"doa_len={doa_len_sec:.2f}s",
                f"segments={len(refined_segments)}",
            ],
        )

    if speech_len_sec >= MIN_VALID_SPEECH_SECONDS:
        if args.save_debug_audio:
            save_wav(args.save_debug_raw_path, command_audio, RATE)
            print(f"[INFO] Saved raw command audio to {args.save_debug_raw_path}")
            save_multichannel_wav(args.save_multichannel_path, doa_audio, RATE)
            print(f"[INFO] Saved 4-channel DOA audio to {args.save_multichannel_path}")

        save_wav(args.save_path, processed_audio, RATE)
        print(f"[INFO] Saved speech-only command audio to {args.save_path}")

        doa_result = estimate_doa_from_multichannel_audio(doa_audio)
        if doa_result is not None:
            print(
                "[DOA] peak="
                f"{doa_result['peak_calib']} deg "
                f"(raw={doa_result['peak_raw']} deg, "
                f"confidence={doa_result['peak_confidence']:.3f}, "
                f"blocks={doa_result['num_blocks']})"
            )
            if visualizer is not None:
                visualizer.set_doa_result(doa_result)
        else:
            print("[DOA] No usable multi-channel blocks were found for angle estimation.")

        if args.enable_asr:
            try:
                text = transcribe_audio_with_dashscope(processed_audio, RATE)
                if text:
                    print(f"[ASR] final text: {text}")
                    if visualizer is not None:
                        visualizer.set_asr_text(text)
                else:
                    print("[ASR] empty transcription.")
            except Exception as e:
                print(f"[ASR] failed: {e}")
    else:
        print("[INFO] Captured speech too short. Discarded as invalid command.")

    flush_mic_stream(mic_stream, CHUNK, num_chunks=8)
    reset_wakeword_detector_state()
    reset_command_state()

    wakeword_enabled = True
    wake_armed = False
    wake_release_frame_count = 0
    wake_cooldown_until = time.time() + WAKE_REARM_COOLDOWN_SECONDS
    state = STATE_WAKEWORD
    if visualizer is not None:
        visualizer.set_state("WAKEWORD", ["ready_for_next_trigger=True"])
        visualizer.refresh(force=True)
    print("[INFO] Returning to WAKEWORD stage...\n")


# =========================
# Main loop
# =========================
if __name__ == "__main__":
    print("\n" + "#" * 100)
    print("Listening for wakewords...")
    print("#" * 100)

    try:
        while True:
            raw_bytes = mic_stream.read(CHUNK, exception_on_overflow=False)
            audio_chunk = np.frombuffer(raw_bytes, dtype=np.int16)
            mono_chunk, raw4_chunk, _ = split_interleaved_chunk(audio_chunk)
            if visualizer is not None:
                visualizer.push_audio(mono_chunk)

            if state == STATE_WAKEWORD:
                now = time.time()
                n_spaces = 16
                output_string = """
Model Name         | Score   | Wakeword Status
---------------------------------------------
"""
                triggered = False
                max_curr_score = 0.0

                if wakeword_enabled:
                    _ = owwModel.predict(mono_chunk)

                    for mdl in owwModel.prediction_buffer.keys():
                        scores = list(owwModel.prediction_buffer[mdl])
                        curr_score = scores[-1]
                        max_curr_score = max(max_curr_score, curr_score)
                        in_cooldown = now < wake_cooldown_until

                        if curr_score < WAKE_RELEASE_THRESHOLD:
                            wake_release_frame_count += 1
                        else:
                            wake_release_frame_count = 0

                        if (not in_cooldown) and wake_release_frame_count >= WAKE_RELEASE_FRAMES_REQUIRED:
                            wake_armed = True

                        status = "--"
                        if in_cooldown:
                            status = "Cooldown"
                        elif curr_score > WAKE_TRIGGER_THRESHOLD:
                            status = "Above trigger"
                        elif curr_score > WAKE_RELEASE_THRESHOLD:
                            status = "In between"
                        elif not wake_armed:
                            status = "Waiting re-arm"

                        if (not in_cooldown) and curr_score > WAKE_TRIGGER_THRESHOLD and wake_armed:
                            triggered = True
                            wake_armed = False
                            wake_release_frame_count = 0
                            status = "Wakeword Triggered!"

                        output_string += (
                            f"{mdl}{' ' * max(1, n_spaces - len(mdl))} | "
                            f"{curr_score:.4f} | {status}\n"
                        )
                else:
                    for mdl in owwModel.models.keys():
                        output_string += (
                            f"{mdl}{' ' * max(1, n_spaces - len(mdl))} | "
                            f"{0.0:.4f} | Disabled\n"
                        )

                if visualizer is not None:
                    visualizer.push_wake_score(max_curr_score)
                    visualizer.set_state(
                        "WAKEWORD",
                        [
                            f"wake_enabled={wakeword_enabled}",
                            f"wake_armed={wake_armed}",
                            f"cooldown_active={now < wake_cooldown_until}",
                            f"score={max_curr_score:.4f}",
                        ],
                    )
                    visualizer.refresh()

                print("\r" + output_string, end="")
                if triggered:
                    enter_tts_reply_stage()
                continue

            elif state == STATE_TTS_REPLY:
                if visualizer is not None:
                    visualizer.set_state("TTS_REPLY", ["speaking_reply=True"])
                    visualizer.refresh()
                continue

            elif state == STATE_COMMAND:
                now = time.time()

                if command_start_time is None:
                    command_start_time = now

                if not speech_started:
                    pre_roll_buffer.append(mono_chunk.copy())
                    pre_roll_raw_buffer.append(raw4_chunk.copy())

                events = process_vad_stream(mono_chunk)
                chunk_added_due_to_start = False
                saw_speech_activity = False
                saw_end_event = False
                chunk_energy_active = chunk_has_energy(mono_chunk)

                for ev in events:
                    print(f"[VAD EVENT] {ev}")

                    if "start" in ev:
                        if not speech_started:
                            speech_started = True
                            print("[INFO] Speech started...")
                            speech_first_detected_time = now

                            if len(pre_roll_buffer) > 0:
                                captured_speech_chunks.extend(list(pre_roll_buffer))
                                pre_roll_buffer.clear()
                            if len(pre_roll_raw_buffer) > 0:
                                captured_raw_chunks.extend(list(pre_roll_raw_buffer))
                                pre_roll_raw_buffer.clear()

                            append_speech_chunk(mono_chunk, raw4_chunk)
                            chunk_added_due_to_start = True

                        last_voice_activity_time = now
                        saw_speech_activity = True

                    if "end" in ev and speech_started:
                        print("[INFO] Speech end candidate...")
                        saw_end_event = True

                if speech_started and saw_speech_activity:
                    last_voice_activity_time = now
                    speech_chunk_count += 1
                    silent_chunk_count = 0
                    end_event_count = 0
                    end_candidate_active = False
                    if not chunk_added_due_to_start:
                        append_speech_chunk(mono_chunk, raw4_chunk)
                elif speech_started:
                    if chunk_energy_active:
                        last_voice_activity_time = now
                        speech_chunk_count += 1
                        silent_chunk_count = 0
                        end_event_count = 0
                        end_candidate_active = False
                        append_speech_chunk(mono_chunk, raw4_chunk)
                    else:
                        silent_chunk_count += 1
                        append_tail_silence_chunk(mono_chunk, raw4_chunk)
                        if saw_end_event:
                            end_event_count += 1
                            end_candidate_active = True

                elapsed = now - command_start_time
                should_stop = False
                speech_elapsed = now - last_voice_activity_time if last_voice_activity_time is not None else 0.0
                total_recorded_sec = len(captured_speech_chunks) * CHUNK / RATE
                speech_since_start = (
                    now - speech_first_detected_time
                    if speech_first_detected_time is not None else 0.0
                )
                silence_seconds = silent_chunk_count * CHUNK / RATE
                fast_end_ready = (
                    end_candidate_active
                    and speech_chunk_count >= MIN_SPEECH_CHUNKS_REQUIRED
                    and speech_since_start >= FAST_END_MIN_SPEECH_SECONDS
                    and silent_chunk_count >= FAST_END_CHUNKS_REQUIRED
                    and silence_seconds >= FAST_END_SILENCE_SECONDS
                )

                if visualizer is not None:
                    visualizer.set_state(
                        "COMMAND_CAPTURE",
                        [
                            f"speech_started={speech_started}",
                            f"speech_chunks={speech_chunk_count}",
                            f"silent_chunks={silent_chunk_count}",
                            f"elapsed={elapsed:.2f}s",
                            f"speech_since_start={speech_since_start:.2f}s",
                            f"fast_end_ready={fast_end_ready}",
                        ],
                    )
                    visualizer.refresh()

                if speech_started and last_voice_activity_time is not None:
                    if elapsed < COMMAND_START_GRACE_SECONDS:
                        should_stop = False
                    elif total_recorded_sec < MIN_COMMAND_RECORD_SECONDS:
                        should_stop = False
                    elif speech_chunk_count < MIN_SPEECH_CHUNKS_REQUIRED:
                        should_stop = False
                    elif speech_since_start < MIN_SPEECH_AFTER_START_SECONDS:
                        should_stop = False
                    elif fast_end_ready:
                        print("[INFO] Fast end timeout reached after VAD end event.")
                        should_stop = True
                    elif end_event_count < END_EVENT_REQUIRED_COUNT:
                        should_stop = False
                    elif silent_chunk_count < SILENCE_CHUNKS_REQUIRED:
                        should_stop = False
                    elif silence_seconds < POST_SPEECH_SILENCE_SECONDS:
                        should_stop = False
                    elif speech_elapsed >= POST_SPEECH_SILENCE_SECONDS:
                        print("[INFO] Post-speech silence timeout reached.")
                        should_stop = True

                if (not speech_started) and elapsed >= MAX_COMMAND_SECONDS:
                    print("[INFO] No speech detected within max command duration.")
                    should_stop = True

                if elapsed >= MAX_COMMAND_SECONDS + 3.0:
                    print("[INFO] Hard timeout reached.")
                    should_stop = True

                if should_stop:
                    finalize_command_and_return_to_wakeword()

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        try:
            if tts is not None:
                tts.close()
        except Exception:
            pass

        try:
            mic_stream.stop_stream()
            mic_stream.close()
        except Exception:
            pass

        try:
            if visualizer is not None:
                visualizer.refresh(force=True)
                visualizer.plt.close(visualizer.fig)
        except Exception:
            pass

        pa.terminate()
