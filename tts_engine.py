import sys
import socket
import threading
import subprocess
import tempfile
import os

try:
    from gtts import gTTS
except ImportError:
    gTTS = None

try:
    import playsound
except ImportError:
    playsound = None


class TTSEngine:
    def __init__(self):
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=2)
            self.online = True
        except OSError:
            self.online = False

        self._speaking = False
        self._last_exercise = None
        self._pending_exercise = None
        self._pending_since = None
        self._incorrect_streak_start = None
        self._last_incorrect_cue_time = None
        self._last_correct_cue_time = None
        self._was_incorrect = False

        self._audio_device = self._probe_audio()

    # ── Audio device detection ──────────────────────────────────────────────

    @staticmethod
    def _probe_audio():
        """Detect available audio output devices. Returns the best device
        ID string for platform-specific playback, or None if none found."""
        if sys.platform == "darwin":
            return TTSEngine._probe_mac()
        elif sys.platform == "linux":
            return TTSEngine._probe_linux()
        return None

    @staticmethod
    def _probe_mac():
        """Probe macOS audio outputs. Prints diagnostic, returns None
        (macOS auto-switches — afplay/say use system default)."""
        try:
            out = subprocess.run(
                ["system_profiler", "SPAudioDataType"],
                capture_output=True, text=True, timeout=5
            )
            outputs = []
            for line in out.stdout.split('\n'):
                if 'Output Source:' in line:
                    name = line.split(':', 1)[1].strip()
                    if name:
                        outputs.append(name)
            if outputs:
                print(f"[TTS] Audio outputs: {', '.join(outputs)}")
                print(f"[TTS] Using system default (macOS auto-switches)")
            else:
                print("[TTS] Audio: using system default (no explicit outputs detected)")
        except Exception:
            print("[TTS] Audio: could not probe outputs, using system default")
        return None

    @staticmethod
    def _probe_linux():
        """Probe Linux audio devices (PulseAudio > ALSA).
        Returns ALSA device ID string or None."""
        sinks = TTSEngine._probe_pulse()
        if sinks:
            display_names = [s["display"] for s in sinks if ".monitor" not in s["name"]]
            print(f"[TTS] PulseAudio sinks: {', '.join(display_names) if display_names else 'none'}")

            # Prefer non-HDMI sink for TTS
            for s in sinks:
                if ".monitor" in s["name"]:
                    continue
                if "hdmi" not in s["name"].lower():
                    print(f"[TTS] Using: {s['display']}")
                    return s["alsa"]
            # Fallback to any non-monitor sink
            for s in sinks:
                if ".monitor" not in s["name"]:
                    print(f"[TTS] Using: {s['display']}")
                    return s["alsa"]

        alsa = TTSEngine._probe_alsa()
        if alsa:
            print(f"[TTS] ALSA device: {alsa}")
            return alsa

        print("[TTS] Audio: no outputs detected — TTS may be silent")
        return None

    @staticmethod
    def _probe_pulse():
        """Return list of dicts {name, display, alsa} from PulseAudio."""
        try:
            out = subprocess.run(
                ["pactl", "list", "short", "sinks"],
                capture_output=True, text=True, timeout=5
            )
            sinks = []
            for line in out.stdout.strip().split('\n'):
                if not line.strip():
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    sinks.append({
                        "name": parts[1].strip(),
                        "display": parts[1].strip(),
                        "alsa": "default",
                    })
            return sinks
        except Exception:
            return []

    @staticmethod
    def _probe_alsa():
        """Return best ALSA PCM device ID (e.g. 'sysdefault') or None.
        Uses aplay -L (PCM listing) instead of aplay -l (hardware listing)
        so that software mixers like sysdefault/default route to any active
        output (HDMI, Bluetooth via PulseAudio, etc.)."""
        try:
            out = subprocess.run(
                ["aplay", "-L"],
                capture_output=True, text=True, timeout=5
            )
            pcm_names = set()
            for line in out.stdout.split('\n'):
                stripped = line.strip()
                if stripped and not stripped.startswith('#'):
                    pcm_names.add(stripped)
            for preferred in ("sysdefault", "default", "front", "dmix"):
                if preferred in pcm_names:
                    return preferred
        except Exception:
            pass
        return None

    # ── Audio playback ──────────────────────────────────────────────────────

    def _play_audio(self, text: str):
        self._speaking = True
        try:
            if self.online and gTTS is not None:
                mp3_path = self._generate_tts(text)
                if mp3_path:
                    self._play_file(mp3_path)
                    try:
                        os.remove(mp3_path)
                    except OSError:
                        pass
                    return
            self._speak_fallback(text)
        finally:
            self._speaking = False

    def _generate_tts(self, text):
        tts = gTTS(text=text, lang="en")
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fp:
            path = fp.name
        tts.save(path)
        return path

    def _play_file(self, filepath):
        """Play an audio file using the best available method."""
        if sys.platform == "darwin":
            subprocess.run(["afplay", filepath])
            return

        # Linux: try playsound first (uses system default PulseAudio sink)
        if playsound is not None:
            try:
                playsound.playsound(filepath)
                return
            except Exception:
                pass

        # Linux: try ffmpeg → wav → aplay with detected device
        if self._audio_device:
            try:
                wav_path = filepath.rsplit('.', 1)[0] + '.wav'
                subprocess.run(
                    ['ffmpeg', '-i', filepath, '-y', '-loglevel', 'error', wav_path],
                    timeout=15
                )
                if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
                    subprocess.run(['aplay', '-D', self._audio_device, wav_path])
                    try:
                        os.remove(wav_path)
                    except OSError:
                        pass
                    return
            except Exception:
                pass

        # Linux: try ffmpeg → wav → aplay (system default, no -D)
        try:
            wav_path = filepath.rsplit('.', 1)[0] + '.wav'
            subprocess.run(
                ['ffmpeg', '-i', filepath, '-y', '-loglevel', 'error', wav_path],
                timeout=15
            )
            if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
                subprocess.run(['aplay', wav_path])
                try:
                    os.remove(wav_path)
                except OSError:
                    pass
                return
        except Exception:
            pass

        # Last resort — try espeak (or mpg123 for mp3 files)
        text = "Audio playback failed"
        self._speak_fallback(text)

    def _speak_fallback(self, text):
        """System TTS fallback. Uses say (macOS) or espeak (Linux).
        Linux: tries aplay -D <device>, then plain aplay, then espeak raw.
        If all fail, prints warning and continues gracefully."""
        if sys.platform == "darwin":
            subprocess.run(["say", text])
            return

        # Linux: try espeak piped through aplay with detected device
        if self._audio_device:
            try:
                proc = subprocess.Popen(
                    ["espeak", "--stdout", text],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
                )
                subprocess.run(
                    ["aplay", "-D", self._audio_device],
                    stdin=proc.stdout, timeout=15
                )
                proc.wait()
                return
            except Exception:
                pass

        # Linux: try espeak piped through aplay without -D (system default)
        try:
            proc = subprocess.Popen(
                ["espeak", "--stdout", text],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
            )
            subprocess.run(
                ["aplay"],
                stdin=proc.stdout, timeout=15
            )
            proc.wait()
            return
        except Exception:
            pass

        # Linux: try raw espeak (no audio device needed for some setups)
        try:
            subprocess.run(["espeak", text], timeout=15)
            return
        except Exception:
            pass

        print("[TTS] Warning: no audio output available")

    # ── Public API ──────────────────────────────────────────────────────────

    def speak(self, text: str):
        if self._speaking:
            return
        t = threading.Thread(target=self._play_audio, args=(text,), daemon=True)
        t.start()

    def update(self, exercise_name: str, quality: int, timestamp: float):
        if exercise_name != self._pending_exercise:
            self._pending_exercise = exercise_name
            self._pending_since = timestamp

        if (
            exercise_name == self._pending_exercise
            and exercise_name != self._last_exercise
            and (timestamp - self._pending_since) >= 1.5
        ):
            self.speak(f"{exercise_name} detected")
            self._last_exercise = exercise_name

        if quality == 0:
            if self._incorrect_streak_start is None:
                self._incorrect_streak_start = timestamp

            streak = timestamp - self._incorrect_streak_start
            if streak >= 3.0:
                cooldown_ok = (
                    self._last_incorrect_cue_time is None
                    or (timestamp - self._last_incorrect_cue_time) >= 5.0
                )
                if cooldown_ok:
                    self.speak("Check your form")
                    self._last_incorrect_cue_time = timestamp
                    self._was_incorrect = True

        if quality == 1:
            if self._incorrect_streak_start is not None:
                streak = timestamp - self._incorrect_streak_start
                if streak >= 3.0 and self._was_incorrect:
                    cooldown_ok = (
                        self._last_correct_cue_time is None
                        or (timestamp - self._last_correct_cue_time) >= 5.0
                    )
                    if cooldown_ok:
                        self.speak("Good job, form restored")
                        self._last_correct_cue_time = timestamp

            self._incorrect_streak_start = None
            self._was_incorrect = False

    def announce_session_start(self):
        self.speak("Session started. Begin your exercise.")

    def announce_session_end(self):
        self.speak("Session complete. Great work.")
