import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import queue
from typing import Optional
import time

class PiperInterruptSpeaker:

    def __init__(
        self,
        model_path: str,
        piper_bin: str = "piper",
        volume: float = 1.0,
        length_scale: float = 1.0,
        player: Optional[str] = None,   # "paplay" or "aplay" or None(auto)
        verbose: bool = False,
    ):
        self.model_path = model_path
        self.piper_bin = piper_bin
        self.verbose = verbose
        self.volume = volume
        self.length_scale = length_scale
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"[PiperInterruptSpeaker] model not found: {self.model_path}")
        if shutil.which(self.piper_bin) is None:
            raise RuntimeError(f"[PiperInterruptSpeaker] '{self.piper_bin}' not found. Did you pip install piper-tts?")

        # choose player
        if player is not None:
            if shutil.which(player) is None:
                raise RuntimeError(f"[PiperInterruptSpeaker] player '{player}' not found.")
            self.player = player
        else:
            self.player = shutil.which("paplay") and "paplay" or (shutil.which("aplay") and "aplay")
            if not self.player:
                raise RuntimeError("[PiperInterruptSpeaker] No audio player found. Install paplay or aplay.")

        self._style = self._detect_piper_style()

        self._q: "queue.Queue[Optional[str]]" = queue.Queue()
        self._closed = False

        self._lock = threading.Lock()
        self._cur_piper: Optional[subprocess.Popen] = None
        self._cur_play: Optional[subprocess.Popen] = None

        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    def _log(self, *a):
        if self.verbose:
            print("[PiperInterruptSpeaker]", *a)

    def _detect_piper_style(self) -> str:
        """
        兼容不同 piper CLI：
          - style="flag_model": piper --model M --output_file out.wav
          - style="flag_m":     piper -m M -o out.wav
          - style="pipe_model": piper M > out.wav
        """
        try:
            help_txt = subprocess.check_output([self.piper_bin, "--help"], stderr=subprocess.STDOUT, text=True).lower()
        except Exception:
            try:
                help_txt = subprocess.check_output([self.piper_bin, "-h"], stderr=subprocess.STDOUT, text=True).lower()
            except Exception:
                return "pipe_model"

        if ("--model" in help_txt) and (("--output_file" in help_txt) or ("--output" in help_txt)):
            return "flag_model"
        if re.search(r"(^|\s)-m(\s|,)", help_txt) and re.search(r"(^|\s)-o(\s|,)", help_txt):
            return "flag_m"
        return "pipe_model"

    def speak(self, text: str):
        if self._closed:
            return
        self._q.put(text)

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._q.put(None)
        self._interrupt_now()

    def _interrupt_now(self):
        with self._lock:
            for p in (self._cur_play, self._cur_piper):
                if p is None:
                    continue
                try:
                    if os.name != "nt":
                        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                    else:
                        p.terminate()
                except Exception:
                    pass
            self._cur_play = None
            self._cur_piper = None

    def _popen(self, cmd, *, text_stdin=False, stdout_to_file=None):
        preexec = os.setsid if os.name != "nt" else None
        if stdout_to_file is not None:
            f = open(stdout_to_file, "wb")
            return subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=f,
                stderr=subprocess.PIPE,
                text=text_stdin,
                preexec_fn=preexec,
            )
        return subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=text_stdin,
            preexec_fn=preexec,
        )

    def _synth_to_wav(self, text: str) -> str:
        fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        if self._style == "flag_model":
            cmd = [self.piper_bin, "--model", self.model_path, "--output_file", wav_path,"--length_scale", str(self.length_scale),"--volume", str(self.volume)]
            p = self._popen(cmd, text_stdin=True)
        elif self._style == "flag_m":
            cmd = [self.piper_bin, "-m", self.model_path, "-o", wav_path,"--length_scale", str(self.length_scale),"--volume", str(self.volume)]
            p = self._popen(cmd, text_stdin=True)
        else:
            cmd = [self.piper_bin, self.model_path]
            p = self._popen(cmd, text_stdin=True, stdout_to_file=wav_path)

        with self._lock:
            self._cur_piper = p

        try:
            p.stdin.write(text + "\n")
            p.stdin.flush()
            p.stdin.close()
        except Exception:
            pass

        ret = p.wait()

        with self._lock:
            if self._cur_piper is p:
                self._cur_piper = None

        if ret != 0:
            stderr = ""
            try:
                stderr = p.stderr.read()
            except Exception:
                pass
            raise RuntimeError(f"piper failed (code {ret}). stderr:\n{stderr}")

        if not os.path.exists(wav_path) or os.path.getsize(wav_path) < 1000:
            raise RuntimeError("wav not generated / too small")

        return wav_path

    def _play_wav_blocking(self, wav_path: str):
        cmd = [self.player, wav_path]
        p = self._popen(cmd, text_stdin=False)

        with self._lock:
            self._cur_play = p

        p.wait()

        with self._lock:
            if self._cur_play is p:
                self._cur_play = None

    def _loop(self):
        while True:
            text = self._q.get()
            if text is None:
                break

            while True:
                try:
                    newer = self._q.get_nowait()
                    if newer is None:
                        return
                    text = newer
                except queue.Empty:
                    break

            text = str(text).strip()
            if not text:
                continue

            self._interrupt_now()

            try:
                wav_path = self._synth_to_wav(text)
                self._play_wav_blocking(wav_path)
            except Exception as e:
                self._log("TTS error:", e)
            finally:
                try:
                    if "wav_path" in locals() and os.path.exists(wav_path):
                        os.remove(wav_path)                                                                                   
                except Exception:
                    pass
    def is_busy(self) -> bool:
        with self._lock:
            return (self._cur_piper is not None) or (self._cur_play is not None) or (not self._q.empty())

    def wait_idle(self, poll=0.05, timeout=None):
        import time
        t0 = time.time()
        while True:
            if not self.is_busy():
                return True
            if timeout is not None and (time.time() - t0) > timeout:
                return False
            time.sleep(poll)

if __name__ == "__main__":
    """
    The used library is from: https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/CLI.md, can follow this link to install this library and select the 
    desired model (voice in .onnx file).

    Input parameters:
    model_path: Path to the Piper .onnx model file.
    length_scale: Controls the speed of speech. Values >1 make speech slower, values <1 make it faster.
    volume: Controls the volume of the speech output. Values >1 increase volume, values <1 decrease it.

    """
    MODEL = "/home/kemove/en_US-lessac-medium.onnx"
    tts = PiperInterruptSpeaker(
        model_path=MODEL,
        length_scale=1,    
        volume=0.1,          
    )
    try:
        tts.speak("Hello. This is the first sentence. The question specifically asks about something hanging from the oven handle, and  is the only object in the list that has a handle and is relevant to the context of hanging items")
        tts.wait_idle(timeout=200)  # Only used for this example to prevent the script from exiting immediately.
    except KeyboardInterrupt:
        print("\nCtrl+C received, stopping TTS...")
    finally:
        tts.close()
