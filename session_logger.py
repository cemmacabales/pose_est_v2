import os
import json
from datetime import datetime

from joint_map import EXERCISE_NAMES


class SessionLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        self.start_time = datetime.now()
        self.segments = []
        self.current_segment = None

    def log_frame(self, exercise_idx: int, quality: int, confidence: float):
        exercise_name = EXERCISE_NAMES.get(exercise_idx, "Unknown")
        if self.current_segment is None or exercise_name != self.current_segment["name"]:
            if self.current_segment is not None:
                self.segments.append(self.current_segment)
            self.current_segment = {
                "name": exercise_name,
                "segment_start": datetime.now(),
                "frames": []
            }
        self.current_segment["frames"].append({"quality": quality, "confidence": confidence})

    def end_session(self) -> dict:
        if self.current_segment is not None:
            self.segments.append(self.current_segment)

        end_time = datetime.now()
        duration_seconds = int((end_time - self.start_time).total_seconds())

        exercises = []
        for i, segment in enumerate(self.segments):
            frames = segment["frames"]
            frames_correct = sum(1 for f in frames if f["quality"] == 1)
            frames_incorrect = sum(1 for f in frames if f["quality"] == 0)
            total = frames_correct + frames_incorrect
            form_score_pct = int(frames_correct / total * 100) if total > 0 else 0
            avg_confidence = round(sum(f["confidence"] for f in frames) / len(frames), 2) if frames else 0.0

            seg_end = self.segments[i + 1]["segment_start"] if i + 1 < len(self.segments) else end_time
            duration_secs = int((seg_end - segment["segment_start"]).total_seconds())

            exercises.append({
                "name": segment["name"],
                "segment_start": segment["segment_start"].strftime("%H:%M:%S"),
                "duration_seconds": duration_secs,
                "frames_correct": frames_correct,
                "frames_incorrect": frames_incorrect,
                "form_score_pct": form_score_pct,
                "avg_confidence": avg_confidence
            })

        overall_form_score_pct = int(sum(e["form_score_pct"] for e in exercises) / len(exercises)) if exercises else 0

        result = {
            "date": self.start_time.strftime("%Y-%m-%d"),
            "start_time": self.start_time.strftime("%H:%M:%S"),
            "end_time": end_time.strftime("%H:%M:%S"),
            "duration_seconds": duration_seconds,
            "exercises": exercises,
            "overall_form_score_pct": overall_form_score_pct,
            "total_exercises_detected": len(exercises),
            "log_file": ""
        }

        os.makedirs(self.log_dir, exist_ok=True)
        filename = f"session_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.log_dir, filename)
        result["log_file"] = filepath
        with open(filepath, "w") as f:
            json.dump(result, f, indent=2)

        return result
