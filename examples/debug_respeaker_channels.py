import argparse
import os
import wave

import numpy as np
import pyaudio
import sounddevice as sd


FORMAT = pyaudio.paInt16
RATE = 16000


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_multichannel_wav(path: str, audio: np.ndarray, sample_rate: int):
    if audio.ndim != 2:
        raise ValueError("Expected shape [n_samples, n_channels].")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(audio.shape[1])
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.astype(np.int16).tobytes())


def save_single_channel_wavs(prefix: str, audio: np.ndarray, sample_rate: int):
    for ch in range(audio.shape[1]):
        path = f"{prefix}_ch{ch}.wav"
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio[:, ch].astype(np.int16).tobytes())


def summarize_channels(audio: np.ndarray, title: str):
    print(f"\n[{title}]")
    for ch in range(audio.shape[1]):
        x = audio[:, ch].astype(np.float32)
        rms = float(np.sqrt(np.mean(np.square(x))))
        peak = float(np.max(np.abs(x)))
        mean = float(np.mean(x))
        print(f"  ch{ch}: rms={rms:.2f} peak={peak:.2f} mean={mean:.2f}")


def list_pyaudio_input_devices():
    pa = pyaudio.PyAudio()
    try:
        print("\n[PyAudio input devices]")
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if int(info.get("maxInputChannels", 0)) > 0:
                print(f"  {i}: {info['name']} | input_channels={int(info['maxInputChannels'])}")
    finally:
        pa.terminate()


def list_sounddevice_input_devices():
    print("\n[sounddevice input devices]")
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            print(f"  {idx}: {dev['name']} | input_channels={dev['max_input_channels']}")


def find_pyaudio_device(keyword: str):
    pa = pyaudio.PyAudio()
    try:
        matches = []
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if int(info.get("maxInputChannels", 0)) > 0 and keyword.lower() in info["name"].lower():
                matches.append((i, info))
        return matches[0] if matches else (None, None)
    finally:
        pa.terminate()


def find_sounddevice_device(keyword: str):
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        if dev["max_input_channels"] > 0 and keyword.lower() in dev["name"].lower():
            return idx, dev
    return None, None


def record_with_pyaudio(device_index: int, channels: int, seconds: float, chunk_size: int):
    pa = pyaudio.PyAudio()
    try:
        stream = pa.open(
            format=FORMAT,
            channels=channels,
            rate=RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=chunk_size,
        )
        try:
            print(f"[PyAudio] Recording {seconds:.2f}s from device {device_index} ...")
            frames = []
            num_reads = int(np.ceil(seconds * RATE / chunk_size))
            for _ in range(num_reads):
                raw = stream.read(chunk_size, exception_on_overflow=False)
                frames.append(np.frombuffer(raw, dtype=np.int16))
        finally:
            stream.stop_stream()
            stream.close()
    finally:
        pa.terminate()

    audio = np.concatenate(frames, axis=0)
    audio = audio[: int(seconds * RATE) * channels]
    return audio.reshape(-1, channels)


def record_with_sounddevice(device_index: int, channels: int, seconds: float):
    frames = int(seconds * RATE)
    print(f"[sounddevice] Recording {seconds:.2f}s from device {device_index} ...")
    audio = sd.rec(
        frames,
        samplerate=RATE,
        channels=channels,
        dtype="int16",
        device=device_index,
        blocking=True,
    )
    return np.asarray(audio, dtype=np.int16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--channels", type=int, default=6)
    parser.add_argument("--chunk_size", type=int, default=1280)
    parser.add_argument("--device_keyword", type=str, default="USB Audio")
    parser.add_argument("--pyaudio_device", type=int, default=None)
    parser.add_argument("--sounddevice_device", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="debug_respeaker_channels")
    parser.add_argument("--skip_pyaudio", action="store_true")
    parser.add_argument("--skip_sounddevice", action="store_true")
    parser.add_argument("--list_only", action="store_true")
    args = parser.parse_args()

    list_pyaudio_input_devices()
    list_sounddevice_input_devices()

    if args.list_only:
        return

    ensure_dir(args.output_dir)

    pyaudio_device = args.pyaudio_device
    if pyaudio_device is None and not args.skip_pyaudio:
        pyaudio_device, pyaudio_info = find_pyaudio_device(args.device_keyword)
        if pyaudio_device is None:
            raise SystemExit(f"Could not find PyAudio device with keyword: {args.device_keyword}")
        print(f"[PyAudio] Selected device {pyaudio_device}: {pyaudio_info['name']}")

    sounddevice_device = args.sounddevice_device
    if sounddevice_device is None and not args.skip_sounddevice:
        sounddevice_device, sounddevice_info = find_sounddevice_device(args.device_keyword)
        if sounddevice_device is None:
            raise SystemExit(f"Could not find sounddevice device with keyword: {args.device_keyword}")
        print(f"[sounddevice] Selected device {sounddevice_device}: {sounddevice_info['name']}")

    if not args.skip_pyaudio:
        pa_audio = record_with_pyaudio(
            device_index=pyaudio_device,
            channels=args.channels,
            seconds=args.seconds,
            chunk_size=args.chunk_size,
        )
        pa_multi = os.path.join(args.output_dir, "pyaudio_multichannel.wav")
        save_multichannel_wav(pa_multi, pa_audio, RATE)
        save_single_channel_wavs(os.path.join(args.output_dir, "pyaudio"), pa_audio, RATE)
        summarize_channels(pa_audio, "PyAudio channel summary")
        print(f"[PyAudio] Saved multichannel wav: {pa_multi}")

    if not args.skip_sounddevice:
        sd_audio = record_with_sounddevice(
            device_index=sounddevice_device,
            channels=args.channels,
            seconds=args.seconds,
        )
        sd_multi = os.path.join(args.output_dir, "sounddevice_multichannel.wav")
        save_multichannel_wav(sd_multi, sd_audio, RATE)
        save_single_channel_wavs(os.path.join(args.output_dir, "sounddevice"), sd_audio, RATE)
        summarize_channels(sd_audio, "sounddevice channel summary")
        print(f"[sounddevice] Saved multichannel wav: {sd_multi}")

    print("\nDone. Compare the per-channel wav files between PyAudio and sounddevice.")


if __name__ == "__main__":
    main()
