#!/usr/bin/env python3
"""
preprocess_train_videos.py — Flatten and rename training videos for the pipeline.

Reads videos from eval/train_videos/Person{1..5}/ and outputs
canonically-named .mp4 files to eval/train_videos_flat/.

Canonical format: <exercise>_<quality>_P<person>_<take>.mp4
Example: 01_correct_P1_1.mp4

Converts .mov to .mp4 via ffmpeg if needed.
"""

import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

EXERCISE_ALIASES = {
    "deep squat": "01",
    "01": "01",
    "hurdle step": "02",
    "02": "02",
    "inline lunge": "03",
    "03": "03",
    "side lunge": "04",
    "04": "04",
    "sit to stand": "05",
    "05": "05",
    "standing leg raise": "06",
    "06": "06",
    "shoulder abduction": "07",
    "07": "07",
    "shoulder extension": "08",
    "08": "08",
    "shoulder scaption": "10",
    "10": "10",
}

VIDEO_EXTS = {".mp4", ".mov", ".MOV", ".MP4"}

STRAY_DUPLICATE = "01_correct_P1_6.mp4"


def sanitize(s):
    return s.lower().replace("\u2014", "-").replace("\u2013", "-").replace("_", " ").strip()


def find_exercise_id(path_parts):
    for part in path_parts:
        clean = sanitize(part)
        for alias, ex_id in EXERCISE_ALIASES.items():
            if alias in clean:
                return ex_id
    return None


def find_quality(path_parts, filename):
    for item in [p.lower() for p in path_parts] + [filename.lower()]:
        if "incorrect" in item:
            return "incorrect"
        if "correct" in item:
            return "correct"
    return "unknown"


def extract_person(path_parts):
    for part in path_parts:
        m = re.match(r"^[Pp]erson(\d+)$", part)
        if m:
            return m.group(1)
    return None


def convert_to_mp4(src_path, dst_path):
    try:
        subprocess.run(
            ["ffmpeg", "-i", str(src_path), "-c:v", "libx264",
             "-preset", "fast", "-crf", "23", "-c:a", "aac", "-y",
             str(dst_path)],
            capture_output=True, check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode(errors="replace")[:200]
        print(f"  ffmpeg failed: {err}")
        return False
    except FileNotFoundError:
        print("  ERROR: ffmpeg not found. Install it or convert .mov files manually.")
        return False


def main():
    input_dir = Path("eval/train_videos")
    output_dir = Path("eval/train_videos_flat")

    if not input_dir.is_dir():
        print(f"Error: {input_dir} not found")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Walk and collect all video files ──────────────────────────
    entries = []  # (src_path, person, exercise_id, quality, sort_key)

    person_dirs = sorted([p for p in input_dir.iterdir() if p.is_dir() and p.name.startswith("Person")])
    if not person_dirs:
        print("No Person directories found under", input_dir)
        sys.exit(1)

    for pd in person_dirs:
        parts = pd.parts[-1]  # e.g. "Person1"
        person = extract_person([parts])
        if not person:
            print(f"  Skipping non-person dir: {pd.name}")
            continue

        for root, _dirs, files in os.walk(pd):
            for fname in sorted(files):
                ext = os.path.splitext(fname)[1]
                if ext not in VIDEO_EXTS:
                    continue

                # Skip known stray duplicates
                if fname == STRAY_DUPLICATE and "Incorrect" in root:
                    print(f"  Skipping stray duplicate in Incorrect folder: {root}/{fname}")
                    continue

                src_path = Path(root) / fname
                rel_parts = Path(root).relative_to(input_dir).parts

                exercise_id = find_exercise_id(rel_parts)
                quality = find_quality(rel_parts, fname)
                sort_key = (exercise_id or "ZZ", quality or "zz", fname)

                entries.append((src_path, person, exercise_id, quality, sort_key))

    # ── Phase 2: Group and assign take numbers ─────────────────────────────
    groups = defaultdict(list)
    for src, person, ex_id, quality, sort_key in entries:
        if ex_id is None or quality == "unknown":
            continue
        groups[(person, ex_id, quality)].append((src, sort_key))

    total_processed = 0
    total_skipped = 0
    total_errors = 0
    total_skipped_unknown = 0

    for (person, ex_id, quality), vlist in sorted(groups.items()):
        vlist.sort(key=lambda x: x[1])  # sort by sort_key

        for take_num, (src_path, _sort_key) in enumerate(vlist, start=1):
            stem = f"{ex_id}_{quality}_P{person}_{take_num}"
            dst_path = output_dir / f"{stem}.mp4"

            if dst_path.exists():
                print(f"  SKIP  {stem}.mp4  (already exists)")
                total_skipped += 1
                continue

            ext = os.path.splitext(src_path)[1].lower()
            if ext == ".mp4":
                # Copy as-is
                try:
                    import shutil
                    shutil.copy2(src_path, dst_path)
                    total_processed += 1
                    print(f"  COPY  {src_path.relative_to(input_dir)}  →  {stem}.mp4")
                except OSError as e:
                    print(f"  ERROR copying {src_path.name}: {e}")
                    total_errors += 1
            else:
                # Convert .mov → .mp4
                print(f"  CONV  {src_path.relative_to(input_dir)}  →  {stem}.mp4")
                if convert_to_mp4(src_path, dst_path):
                    total_processed += 1
                else:
                    total_errors += 1

    # ── Report ─────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Preprocessing complete")
    print(f"  Output directory: {output_dir.resolve()}")
    print(f"  Files processed:  {total_processed}")
    print(f"  Files skipped:    {total_skipped}")
    print(f"  Errors:           {total_errors}")

    # Summary by exercise
    flat_files = sorted(output_dir.glob("*.mp4"))
    by_exercise = defaultdict(int)
    by_person = defaultdict(int)
    by_quality = defaultdict(int)
    for f in flat_files:
        m = re.match(r"^(\d{2})_([a-zA-Z]+)_P(\d+)_(\d+)\.mp4$", f.name)
        if m:
            ex, qual, pers, take = m.groups()
            by_exercise[ex] += 1
            by_person[f"P{pers}"] += 1
            by_quality[qual] += 1

    print()
    print("Per exercise:")
    for ex in sorted(by_exercise):
        print(f"  {ex}: {by_exercise[ex]} videos")
    print(f"Per person:")
    for p in sorted(by_person):
        print(f"  {p}: {by_person[p]} videos")
    print(f"Quality:")
    for q in sorted(by_quality):
        print(f"  {q}: {by_quality[q]} videos")
    print(f"  Total: {len(flat_files)}")


if __name__ == "__main__":
    main()
