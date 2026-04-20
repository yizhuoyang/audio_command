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
parser.add_argument("--save_debug_audio", action="store_true")

# ASR related
parser.add_argument("--asr_model_name", type=str, default="qwen-omni-turbo")
parser.add_argument("--dashscope_base_url", type=str,
                    default="https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
parser.add_argument("--system_prompt_path", type=str, default="/home/kemove/yyz/audio-nav/openWakeWord/robot.txt")
parser.add_argument("--enable_asr", action="store_true")
parser.add_argument("--force_plain_asr_prompt", action="store_true")

args = parser.parse_args()

# =========================
# Constants
# =========================
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = args.chunk_size
SAMPLING_RATE = RATE
VAD_WINDOW_SIZE = 512

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

# DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_API_KEY = 'sk-f39a9ee7479240878f1588601c4a71341'
DASHSCOPE_BASE_URL = args.dashscope_base_url
ASR_MODEL_NAME = args.asr_model_name
SYSTEM_PROMPT_PATH = args.system_prompt_path


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
        {
            "role": "system",
            "content": system_prompt,
        },
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
        asr_text = next(g for g in match.groups() if g is not None).strip()
        return asr_text
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
    frame_size = VAD_WINDOW_SIZE
    noise_floor = max(np.percentile(np.abs(audio_f32), 20), 80.0)
    gated = audio_f32.copy()

    for start in range(0, len(gated), frame_size):
        end = min(start + frame_size, len(gated))
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
    # client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)


    DASHSCOPE_API_KEY = 'sk-f39a9ee7479240878f1588601c4a7134'
    DASHSCOPE_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    client=OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)

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


# =========================
# Init audio / models
# =========================
pa = pyaudio.PyAudio()

device_index, device_info = find_input_device(pa, args.device_keyword)
if device_index is None:
    pa.terminate()
    raise SystemExit(1)

print("=" * 100)
print(f"Using input device index: {device_index}")
print(f"Using input device name : {device_info['name']}")
print("=" * 100)

mic_stream = pa.open(
    format=FORMAT,
    channels=CHANNELS,
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
pre_roll_max_chunks = max(1, int(np.ceil(PRE_ROLL_SECONDS * RATE / CHUNK)))
pre_roll_buffer = deque(maxlen=pre_roll_max_chunks)
tail_silence_max_chunks = max(1, int(np.ceil(TAIL_SILENCE_KEEP_SECONDS * RATE / CHUNK)))
tail_silence_buffer = deque(maxlen=tail_silence_max_chunks)

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
    global vad_leftover, captured_speech_chunks, pre_roll_buffer, tail_silence_buffer
    global speech_started, last_voice_activity_time, command_start_time
    global speech_first_detected_time, speech_chunk_count, silent_chunk_count, end_event_count
    global end_candidate_active

    vad_leftover = np.array([], dtype=np.int16)
    captured_speech_chunks = []
    pre_roll_buffer = deque(maxlen=pre_roll_max_chunks)
    tail_silence_buffer = deque(maxlen=tail_silence_max_chunks)
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


def chunk_has_energy(audio_chunk_int16: np.ndarray) -> bool:
    if audio_chunk_int16.size == 0:
        return False
    rms = float(np.sqrt(np.mean(np.square(audio_chunk_int16.astype(np.float32)))))
    peak = float(np.max(np.abs(audio_chunk_int16.astype(np.float32))))
    return rms >= 180.0 or peak >= 1200.0


def append_speech_chunk(audio_chunk_int16: np.ndarray):
    global captured_speech_chunks, tail_silence_buffer
    if len(tail_silence_buffer) > 0:
        captured_speech_chunks.extend(list(tail_silence_buffer))
        tail_silence_buffer.clear()
    captured_speech_chunks.append(audio_chunk_int16.copy())


def append_tail_silence_chunk(audio_chunk_int16: np.ndarray):
    global tail_silence_buffer
    tail_silence_buffer.append(audio_chunk_int16.copy())


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

    raw_speech_len_sec = len(command_audio) / RATE
    refined_audio, refined_segments = refine_command_audio(command_audio)
    processed_audio = preprocess_audio_for_asr(refined_audio)
    speech_len_sec = len(processed_audio) / RATE

    # If offline refinement trims too aggressively, fall back to the raw captured command.
    if speech_len_sec < MIN_VALID_SPEECH_SECONDS and raw_speech_len_sec >= MIN_VALID_SPEECH_SECONDS:
        print("[INFO] Refined audio became too short, falling back to raw captured command.")
        processed_audio = preprocess_audio_for_asr(command_audio)
        speech_len_sec = len(processed_audio) / RATE
        refined_segments = []

    print("=" * 100)
    print(f"[INFO] Raw captured length     : {raw_speech_len_sec:.2f} sec")
    print(f"[INFO] Refined command length : {speech_len_sec:.2f} sec")
    if refined_segments:
        print(f"[INFO] Offline VAD kept {len(refined_segments)} segment(s): {refined_segments}")
    else:
        print("[INFO] Offline VAD found no extra trim points, using raw capture.")
    print("=" * 100)

    if speech_len_sec >= MIN_VALID_SPEECH_SECONDS:
        if args.save_debug_audio:
            save_wav(args.save_debug_raw_path, command_audio, RATE)
            print(f"[INFO] Saved raw command audio to {args.save_debug_raw_path}")

        save_wav(args.save_path, processed_audio, RATE)
        print(f"[INFO] Saved speech-only command audio to {args.save_path}")

        if args.enable_asr:
            try:
                text = transcribe_audio_with_dashscope(processed_audio, RATE)
                if text:
                    print(f"[ASR] final text: {text}")
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

            if state == STATE_WAKEWORD:
                now = time.time()
                n_spaces = 16
                output_string = """
Model Name         | Score   | Wakeword Status
---------------------------------------------
"""
                triggered = False

                if wakeword_enabled:
                    _ = owwModel.predict(audio_chunk)

                    for mdl in owwModel.prediction_buffer.keys():
                        scores = list(owwModel.prediction_buffer[mdl])
                        curr_score = scores[-1]

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

                print("\r" + output_string, end="")

                if triggered:
                    enter_tts_reply_stage()

                continue

            elif state == STATE_TTS_REPLY:
                continue

            elif state == STATE_COMMAND:
                now = time.time()

                if command_start_time is None:
                    command_start_time = now

                if not speech_started:
                    pre_roll_buffer.append(audio_chunk.copy())

                events = process_vad_stream(audio_chunk)
                chunk_added_due_to_start = False
                saw_speech_activity = False
                saw_end_event = False
                chunk_energy_active = chunk_has_energy(audio_chunk)

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

                            append_speech_chunk(audio_chunk)
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
                        append_speech_chunk(audio_chunk)
                elif speech_started:
                    if chunk_energy_active:
                        last_voice_activity_time = now
                        speech_chunk_count += 1
                        silent_chunk_count = 0
                        end_event_count = 0
                        end_candidate_active = False
                        append_speech_chunk(audio_chunk)
                    else:
                        silent_chunk_count += 1
                        append_tail_silence_chunk(audio_chunk)
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

        pa.terminate()
