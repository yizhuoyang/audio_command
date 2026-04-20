import pyaudio
import numpy as np
from openwakeword.model import Model
import argparse
import sys

# =========================
# Parse input arguments
# =========================
parser = argparse.ArgumentParser()
parser.add_argument(
    "--chunk_size",
    help="How much audio (in number of samples) to predict on at once",
    type=int,
    default=1280,
    required=False
)
parser.add_argument(
    "--model_path",
    help="The path of a specific model to load",
    type=str,
    default="/home/kemove/yyz/audio-nav/openWakeWord/hey_jarvis_v0.1.tflite",
    required=False
)
parser.add_argument(
    "--inference_framework",
    help="The inference framework to use (either 'onnx' or 'tflite')",
    type=str,
    default="tflite",
    required=False
)
parser.add_argument(
    "--device_keyword",
    help="Keyword used to automatically select microphone device",
    type=str,
    default="USB Audio",
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

# =========================
# Initialize PyAudio
# =========================
pa = pyaudio.PyAudio()


def find_input_device(audio_interface, keyword="USB Audio"):
    """
    Automatically find the first input device whose name contains the keyword.
    Returns:
        device_index, device_info
    """
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
        return matched_devices[0]  # pick the first matched USB Audio input device

    print(f"[ERROR] No input device found with keyword: '{keyword}'")
    print("[INFO] Available input devices are:")
    for idx, name, ch in input_devices:
        print(f"  {idx}: {name} (input channels: {ch})")

    return None, None


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
# Main loop
# =========================
if __name__ == "__main__":
    print("\n\n")
    print("#" * 100)
    print("Listening for wakewords...")
    print("#" * 100)
    print("\n" * (n_models * 3))

    try:
        while True:
            # Read audio from selected microphone
            audio_chunk = np.frombuffer(
                mic_stream.read(CHUNK, exception_on_overflow=False),
                dtype=np.int16
            )

            # Predict wakeword
            prediction = owwModel.predict(audio_chunk)

            # Output header
            n_spaces = 16
            output_string = """
Model Name         | Score | Wakeword Status
--------------------------------------
"""

            for mdl in owwModel.prediction_buffer.keys():
                scores = list(owwModel.prediction_buffer[mdl])
                curr_score = format(scores[-1], ".20f").replace("-", "")

                status = "--" if scores[-1] <= 0.5 else "Wakeword Detected!"
                output_string += (
                    f"{mdl}{' ' * max(1, n_spaces - len(mdl))} | "
                    f"{curr_score[0:5]} | {status}\n"
                )

            # Refresh terminal output
            print("\033[F" * (4 * n_models + 1), end="")
            print(output_string, " " * 30, end="\r")

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        mic_stream.stop_stream()
        mic_stream.close()
        pa.terminate()