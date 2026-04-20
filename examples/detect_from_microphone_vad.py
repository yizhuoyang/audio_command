import pyaudio
import numpy as np
from openwakeword.model import Model
import argparse
import sys
import time
import wave
from collections import deque

# =========================
# VAD imports
# =========================
from silero_vad import VADIterator, load_silero_vad

# =========================
# TTS import
# =========================
from tts_module import PiperInterruptSpeaker


# =========================
# Parse input arguments
# =========================
parser = argparse.ArgumentParser()

parser.add_argument(
    "--chunk_size",
    help="How many samples to read from microphone each time",
    type=int,
    default=1280,
    required=False
)
parser.add_argument(
    "--model_path",
    help="Path to the wakeword model",
    type=str,
    default="/home/kemove/yyz/audio-nav/openWakeWord/hey_jarvis_v0.1.tflite",
    required=False
)
parser.add_argument(
    "--inference_framework",
    help="Wakeword inference framework: 'onnx' or 'tflite'",
    type=str,
    default="tflite",
    required=False
)
parser.add_argument(
    "--device_keyword",
    help="Keyword to auto-select the microphone device",
    type=str,
    default="USB Audio",
    required=False
)
parser.add_argument(
    "--wake_threshold",
    help="Wakeword trigger threshold",
    type=float,
    default=0.55,
    required=False
)
parser.add_argument(
    "--wake_release_threshold",
    help="Wakeword release threshold. Must go below this before re-arming",
    type=float,
    default=0.15,
    required=False
)
parser.add_argument(
    "--max_command_seconds",
    help="Maximum time to wait/capture command after wakeword",
    type=float,
    default=8.0,
    required=False
)
parser.add_argument(
    "--post_speech_silence_seconds",
    help="Stop after this much silence once speech has started",
    type=float,
    default=0.8,
    required=False
)
parser.add_argument(
    "--pre_roll_seconds",
    help="Audio before speech start to keep",
    type=float,
    default=0.5,
    required=False
)
parser.add_argument(
    "--min_valid_speech_seconds",
    help="Minimum captured speech duration to be considered a valid command",
    type=float,
    default=0.4,
    required=False
)
parser.add_argument(
    "--tts_model",
    help="Path to Piper TTS .onnx model",
    type=str,
    default="/home/kemove/en_US-lessac-medium.onnx",
    required=False
)
parser.add_argument(
    "--tts_volume",
    help="TTS volume",
    type=float,
    default=0.1,
    required=False
)
parser.add_argument(
    "--tts_length_scale",
    help="TTS speaking speed control",
    type=float,
    default=1.0,
    required=False
)
parser.add_argument(
    "--tts_reply_text",
    help="Reply text after wakeword",
    type=str,
    default="Hi, how can I help you",
    required=False
)
parser.add_argument(
    "--save_path",
    help="Where to save captured speech wav",
    type=str,
    default="captured_command.wav",
    required=False
)

args = parser.parse_args()

# =========================
# Audio settings
# =========================
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = args.chunk_size

SAMPLING_RATE = RATE
VAD_WINDOW_SIZE = 512

WAKE_TRIGGER_THRESHOLD = args.wake_threshold
WAKE_RELEASE_THRESHOLD = args.wake_release_threshold
MAX_COMMAND_SECONDS = args.max_command_seconds
POST_SPEECH_SILENCE_SECONDS = args.post_speech_silence_seconds
PRE_ROLL_SECONDS = args.pre_roll_seconds
MIN_VALID_SPEECH_SECONDS = args.min_valid_speech_seconds

# =========================
# State machine
# =========================
STATE_WAKEWORD = "WAKEWORD"
STATE_TTS_REPLY = "TTS_REPLY"
STATE_COMMAND = "COMMAND_CAPTURE"

state = STATE_WAKEWORD

# 关键词控制：
# 1) wakeword_enabled: 是否允许当前阶段运行 wakeword 检测
# 2) wake_armed: 只有当分数掉到 release threshold 以下后，才允许下一次触发
wakeword_enabled = True
wake_armed = True

# =========================
# PyAudio init
# =========================
pa = pyaudio.PyAudio()


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
    """
    Read and discard a few chunks to clear buffered old audio.
    """
    for _ in range(num_chunks):
        try:
            stream.read(chunk_size, exception_on_overflow=False)
        except Exception:
            pass


def reset_wakeword_detector_state():
    """
    Try to clear openwakeword prediction buffers so previous high scores
    do not immediately affect the next wakeword round.
    """
    try:
        for mdl in owwModel.prediction_buffer.keys():
            buf = owwModel.prediction_buffer[mdl]
            if hasattr(buf, "clear"):
                buf.clear()
    except Exception:
        pass


# =========================
# Find microphone device
# =========================
device_index, device_info = find_input_device(pa, args.device_keyword)

if device_index is None:
    pa.terminate()
    sys.exit(1)

print("=" * 100)
print(f"Using input device index: {device_index}")
print(f"Using input device name : {device_info['name']}")
print("=" * 100)

# =========================
# Open microphone stream
# =========================
mic_stream = pa.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    input_device_index=device_index,
    frames_per_buffer=CHUNK
)

# =========================
# Load wakeword model
# =========================
if args.model_path != "":
    owwModel = Model(
        wakeword_models=[args.model_path],
        inference_framework=args.inference_framework
    )
else:
    owwModel = Model(inference_framework=args.inference_framework)

n_models = len(owwModel.models.keys())

# =========================
# Load VAD model
# =========================
vad_model = load_silero_vad()
vad_iterator = VADIterator(vad_model, sampling_rate=SAMPLING_RATE)

# =========================
# Load TTS
# =========================
tts = PiperInterruptSpeaker(
    model_path=args.tts_model,
    length_scale=args.tts_length_scale,
    volume=args.tts_volume,
)

# =========================
# VAD / command state
# =========================
vad_leftover = np.array([], dtype=np.int16)

captured_speech_chunks = []
pre_roll_max_chunks = max(1, int(np.ceil(PRE_ROLL_SECONDS * RATE / CHUNK)))
pre_roll_buffer = deque(maxlen=pre_roll_max_chunks)

speech_started = False
last_voice_activity_time = None
command_start_time = None


def reset_command_state():
    global vad_leftover
    global captured_speech_chunks
    global pre_roll_buffer
    global speech_started
    global last_voice_activity_time
    global command_start_time

    vad_leftover = np.array([], dtype=np.int16)
    captured_speech_chunks = []
    pre_roll_buffer = deque(maxlen=pre_roll_max_chunks)
    speech_started = False
    last_voice_activity_time = None
    command_start_time = None
    vad_iterator.reset_states()


def process_vad_stream(audio_chunk_int16):
    """
    Feed int16 streaming audio into VADIterator using fixed 512-sample windows.
    Returns a list of VAD events such as {'start': ...} or {'end': ...}
    """
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


def enter_tts_reply_stage():
    global state, wakeword_enabled

    print("\n" + "=" * 100)
    print("[INFO] Wakeword triggered. Playing TTS reply...")
    print("=" * 100)

    # 一旦触发，立刻关闭 wakeword 检测
    wakeword_enabled = False
    state = STATE_TTS_REPLY

    try:
        tts.speak(args.tts_reply_text)
        tts.wait_idle(timeout=10)
    except Exception as e:
        print(f"[WARN] TTS failed: {e}")

    # 稍微等待尾音结束
    time.sleep(0.25)

    # 清除麦克风缓冲和 wakeword 历史缓冲
    flush_mic_stream(mic_stream, CHUNK, num_chunks=8)
    reset_wakeword_detector_state()

    # TTS 完成后进入命令阶段
    reset_command_state()
    state = STATE_COMMAND

    print("[INFO] TTS finished. Entered COMMAND_CAPTURE stage.")


def finalize_command_and_return_to_wakeword():
    global state, wakeword_enabled, wake_armed

    if len(captured_speech_chunks) > 0:
        command_audio = np.concatenate(captured_speech_chunks, axis=0)
    else:
        command_audio = np.array([], dtype=np.int16)

    speech_len_sec = len(command_audio) / RATE

    print("=" * 100)
    print(f"[INFO] Final captured speech length: {speech_len_sec:.2f} sec")
    print("=" * 100)

    if speech_len_sec >= MIN_VALID_SPEECH_SECONDS:
        save_wav(args.save_path, command_audio, RATE)
        print(f"[INFO] Saved speech-only command audio to {args.save_path}")
    else:
        print("[INFO] Captured speech too short. Discarded as invalid command.")

    # 返回关键词阶段前，清缓冲
    flush_mic_stream(mic_stream, CHUNK, num_chunks=8)
    reset_wakeword_detector_state()
    reset_command_state()

    # 重新回到 wakeword 阶段
    # 但先不要立刻允许触发，必须等分数降到 release threshold 以下
    wakeword_enabled = True
    wake_armed = False
    state = STATE_WAKEWORD

    print("[INFO] Returning to WAKEWORD stage...\n")


# =========================
# Main loop
# =========================
if __name__ == "__main__":
    print("\n")
    print("#" * 100)
    print("Listening for wakewords...")
    print("#" * 100)

    try:
        while True:
            raw_bytes = mic_stream.read(CHUNK, exception_on_overflow=False)
            audio_chunk = np.frombuffer(raw_bytes, dtype=np.int16)

            # =========================================================
            # Stage 1: Wakeword detection
            # =========================================================
            if state == STATE_WAKEWORD:
                n_spaces = 16
                output_string = """
Model Name         | Score   | Wakeword Status
---------------------------------------------
"""
                triggered = False

                # 只有在 wakeword_enabled=True 时才真正做关键词检测
                if wakeword_enabled:
                    _ = owwModel.predict(audio_chunk)

                    for mdl in owwModel.prediction_buffer.keys():
                        scores = list(owwModel.prediction_buffer[mdl])
                        curr_score = scores[-1]

                        # 只有当分数掉得足够低，才重新允许下一次触发
                        if curr_score < WAKE_RELEASE_THRESHOLD:
                            wake_armed = True

                        status = "--"
                        if curr_score > WAKE_TRIGGER_THRESHOLD:
                            status = "Above trigger"
                        elif curr_score > WAKE_RELEASE_THRESHOLD:
                            status = "In between"

                        if curr_score > WAKE_TRIGGER_THRESHOLD and wake_armed:
                            triggered = True
                            wake_armed = False
                            status = "Wakeword Triggered!"

                        output_string += (
                            f"{mdl}{' ' * max(1, n_spaces - len(mdl))} | "
                            f"{curr_score:.4f} | {status}\n"
                        )

                else:
                    # 理论上这里不会长时间停留，因为 TTS 阶段不会走到 WAKEWORD
                    # 但保留这段让逻辑更安全
                    for mdl in owwModel.models.keys():
                        output_string += (
                            f"{mdl}{' ' * max(1, n_spaces - len(mdl))} | "
                            f"{0.0:.4f} | Disabled\n"
                        )

                print("\r" + output_string, end="")

                if triggered:
                    enter_tts_reply_stage()

                continue

            # =========================================================
            # Stage 2: TTS reply
            # TTS is executed synchronously in enter_tts_reply_stage()
            # =========================================================
            elif state == STATE_TTS_REPLY:
                continue

            # =========================================================
            # Stage 3: Command capture using VAD
            # Keep only real speech + pre-roll
            # =========================================================
            elif state == STATE_COMMAND:
                now = time.time()

                if command_start_time is None:
                    command_start_time = now

                # before speech starts, keep a small rolling pre-buffer
                if not speech_started:
                    pre_roll_buffer.append(audio_chunk.copy())

                events = process_vad_stream(audio_chunk)

                chunk_added_due_to_start = False

                for ev in events:
                    print(f"[VAD EVENT] {ev}")

                    if "start" in ev:
                        if not speech_started:
                            speech_started = True
                            print("[INFO] Speech started...")

                            # include short pre-roll so we do not lose the beginning
                            if len(pre_roll_buffer) > 0:
                                captured_speech_chunks.extend(list(pre_roll_buffer))
                                pre_roll_buffer.clear()

                            captured_speech_chunks.append(audio_chunk.copy())
                            chunk_added_due_to_start = True

                        last_voice_activity_time = now

                    if "end" in ev and speech_started:
                        print("[INFO] Speech ended...")
                        last_voice_activity_time = now

                # once speech has started, keep subsequent chunks
                if speech_started and (not chunk_added_due_to_start):
                    captured_speech_chunks.append(audio_chunk.copy())

                elapsed = now - command_start_time
                should_stop = False

                # If speech started and silence lasts long enough -> stop
                if speech_started and last_voice_activity_time is not None:
                    if (now - last_voice_activity_time) >= POST_SPEECH_SILENCE_SECONDS:
                        print("[INFO] Post-speech silence timeout reached.")
                        should_stop = True

                # If user says nothing at all -> timeout
                if (not speech_started) and elapsed >= MAX_COMMAND_SECONDS:
                    print("[INFO] No speech detected within max command duration.")
                    should_stop = True

                # Hard safety timeout
                if elapsed >= MAX_COMMAND_SECONDS + 3.0:
                    print("[INFO] Hard timeout reached.")
                    should_stop = True

                if should_stop:
                    finalize_command_and_return_to_wakeword()

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        try:
            tts.close()
        except Exception:
            pass

        try:
            mic_stream.stop_stream()
            mic_stream.close()
        except Exception:
            pass

        pa.terminate()