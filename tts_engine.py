import sys
import socket
import threading
import subprocess
import tempfile
import os
import time

# Try-import gTTS and playsound — offline mode still works if unavailable
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
        # Check internet connectivity
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

    def _play_audio(self, text: str):
        """Private method. Called inside a daemon thread by speak()."""
        self._speaking = True
        try:
            if self.online:
                # Online: use gTTS to generate and play MP3
                if gTTS is not None:
                    tts = gTTS(text=text, lang="en")
                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fp:
                        tmp_path = fp.name
                    tts.save(tmp_path)

                    if sys.platform == "darwin":
                        subprocess.run(["afplay", tmp_path])
                    elif sys.platform == "linux":
                        if playsound is not None:
                            playsound.playsound(tmp_path)
                        else:
                            # Fallback if playsound missing on Linux
                            subprocess.run(["espeak", text])

                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
                else:
                    # gTTS unavailable — fallback to system TTS
                    if sys.platform == "darwin":
                        subprocess.run(["say", text])
                    elif sys.platform == "linux":
                        subprocess.run(["espeak", text])
            else:
                # Offline fallback
                if sys.platform == "darwin":
                    subprocess.run(["say", text])
                elif sys.platform == "linux":
                    subprocess.run(["espeak", text])
        finally:
            self._speaking = False

    def speak(self, text: str):
        """Speak the given text. Skips if already speaking (no queue)."""
        if self._speaking:
            return
        t = threading.Thread(target=self._play_audio, args=(text,), daemon=True)
        t.start()

    def update(self, exercise_name: str, quality: int, timestamp: float):
        """Called every frame from gui.py."""
        # --- Exercise announcement ---
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

        # --- INCORRECT streak ---
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
