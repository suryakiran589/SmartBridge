"""
ISL Sign Language Data Collector
=================================
Collects hand landmark data for Indian Sign Language (ISL) alphabet A-Z and digits 0-9.
Each team member runs this script independently. Outputs a CSV file that gets merged later.

Usage:
    python collect_data.py --person your_name
    python collect_data.py --person your_name --samples 100
    python collect_data.py --person your_name --signs A B C   # collect specific signs only

Requirements:
    pip install opencv-python mediapipe numpy
"""

import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time
import argparse
import platform
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────

ALL_SIGNS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [str(i) for i in range(10)]

SAMPLES_PER_SIGN   = 100    # rows saved per sign
HOLD_FRAMES        = 8      # steady frames needed before a sample is saved
COOLDOWN_SECONDS   = 3      # pause between signs
MIN_DETECTION_CONF = 0.7
MIN_TRACKING_CONF  = 0.5

# Colors (BGR)
GREEN   = (0, 200, 80)
RED     = (0, 60, 220)
AMBER   = (0, 165, 255)
WHITE   = (255, 255, 255)
BLACK   = (0, 0, 0)
GRAY    = (160, 160, 160)
DARK    = (30, 30, 30)

OUTPUT_DIR = "data/raw"

# ── MediaPipe setup ────────────────────────────────────────────────────────────

mp_hands    = mp.solutions.hands
mp_drawing  = mp.solutions.drawing_utils
mp_styles   = mp.solutions.drawing_styles

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_camera():
    """Cross-platform camera init — fixes Mac AVFoundation issues."""
    if platform.system() == "Darwin":
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    return cap


def fix_frame(frame):
    """Normalize color space — Mac sometimes delivers BGRA."""
    if platform.system() == "Darwin" and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def extract_features(hand_landmarks, frame_shape):
    """
    Extract normalized 63-dim feature vector from 21 hand landmarks.
    Normalization: subtract wrist (lm 0), scale by hand bounding-box size.
    This makes features invariant to hand position and distance from camera.
    Returns list of 63 floats: [x0,y0,z0, x1,y1,z1, ... x20,y20,z20]
    """
    h, w = frame_shape[:2]
    lm = hand_landmarks.landmark

    # Raw pixel coords
    coords = [(int(p.x * w), int(p.y * h), p.z) for p in lm]

    # Wrist is landmark 0 — use as origin
    wx, wy, wz = coords[0]

    # Bounding box for scale normalization
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    scale = max(max(xs) - min(xs), max(ys) - min(ys), 1)  # avoid div/0

    features = []
    for (x, y, z) in coords:
        features.extend([
            (x - wx) / scale,
            (y - wy) / scale,
            z - wz           # z is already relative in MediaPipe
        ])
    return features


def draw_progress_bar(frame, current, total, x, y, w, h):
    """Draws a filled progress bar."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), GRAY, -1)
    fill = int(w * current / total)
    cv2.rectangle(frame, (x, y), (x + fill, y + h), GREEN, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), WHITE, 1)
    pct = f"{int(100 * current / total)}%"
    cv2.putText(frame, pct, (x + w // 2 - 15, y + h - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)


def draw_overlay(frame, sign, collected, total, state, hold_progress=0, hold_needed=HOLD_FRAMES):
    """Draws the full HUD overlay onto the frame."""
    overlay = frame.copy()

    # Top bar background
    cv2.rectangle(overlay, (0, 0), (640, 70), DARK, -1)
    frame[:] = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)

    # Current sign (large)
    cv2.putText(frame, f"Sign: {sign}", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, WHITE, 2)

    # Sample count
    cv2.putText(frame, f"{collected}/{total}", (500, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, GREEN if collected < total else AMBER, 2)

    # State label
    state_color = {"WAITING": AMBER, "RECORDING": GREEN, "COOLDOWN": RED, "DONE": GREEN}.get(state, WHITE)
    cv2.putText(frame, state, (20, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)

    # Hold stability bar (shows when hand is detected)
    if state == "RECORDING" and hold_progress > 0:
        draw_progress_bar(frame, hold_progress, hold_needed, 200, 450, 220, 16)
        cv2.putText(frame, "Hold steady", (200, 445),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, GRAY, 1)

    # Collection progress bar (bottom right)
    draw_progress_bar(frame, collected, total, 400, 450, 220, 16)

    return frame


def countdown(cap, seconds=3, message=""):
    """Shows a countdown on screen before recording starts."""
    start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = fix_frame(frame)
        remaining = seconds - int(time.time() - start)
        if remaining <= 0:
            break
        cv2.putText(frame, message, (160, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
        cv2.putText(frame, str(remaining), (295, 320),
                    cv2.FONT_HERSHEY_SIMPLEX, 4.0, GREEN, 4)
        cv2.imshow("ISL Data Collector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
    return True


def wait_for_space(cap, sign, collected_so_far, total_signs):
    """Shows a waiting screen until user presses SPACE."""
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = fix_frame(frame)

        # Dark overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (640, 480), DARK, -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # Instructions
        progress_text = f"Sign {collected_so_far + 1} of {total_signs}"
        cv2.putText(frame, progress_text, (220, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, GRAY, 1)
        cv2.putText(frame, f"Next: {sign}", (220, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, GREEN, 3)
        cv2.putText(frame, "Form the sign, then press SPACE", (100, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 1)
        cv2.putText(frame, "Press Q to quit", (240, 360),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, GRAY, 1)

        cv2.imshow("ISL Data Collector", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            return True
        if key == ord('q'):
            return False


# ── Main collector ─────────────────────────────────────────────────────────────

def collect(person_name, signs_to_collect, samples_per_sign, resume=True):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, f"{person_name}_gestures.csv")

    # Resume: find which signs already have enough data
    completed_signs = set()
    if resume and os.path.exists(output_file):
        with open(output_file, "r") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            counts = {}
            for row in reader:
                if row:
                    label = row[0]
                    counts[label] = counts.get(label, 0) + 1
            completed_signs = {s for s, c in counts.items() if c >= samples_per_sign}
        if completed_signs:
            print(f"\nResuming — already completed: {sorted(completed_signs)}")

    remaining = [s for s in signs_to_collect if s not in completed_signs]
    if not remaining:
        print("All signs already collected!")
        return

    print(f"\nWill collect: {remaining}")
    print(f"Output: {output_file}")
    print(f"Samples per sign: {samples_per_sign}")
    print(f"\nPress SPACE at each sign prompt to begin recording.")
    print("Press Q at any time to quit (progress is saved).\n")
    time.sleep(1)

    cap = get_camera()
    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        return

    # Write CSV header if new file
    file_exists = os.path.exists(output_file)
    csv_file = open(output_file, "a", newline="")
    writer = csv.writer(csv_file)
    if not file_exists:
        header = ["label"] + [f"{axis}{i}" for i in range(21) for axis in ["x", "y", "z"]]
        writer.writerow(header)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=MIN_DETECTION_CONF,
        min_tracking_confidence=MIN_TRACKING_CONF
    ) as hands:

        for sign_idx, sign in enumerate(remaining):
            # Wait screen
            if not wait_for_space(cap, sign, sign_idx, len(remaining)):
                print("\nQuit by user.")
                break

            # Countdown
            if not countdown(cap, seconds=3, message=f"Get ready for  {sign}"):
                break

            # Recording loop
            collected  = 0
            hold_count = 0
            last_saved_features = None

            while collected < samples_per_sign:
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = fix_frame(frame)

                # Flip for mirror effect (more natural for user)
                frame = cv2.flip(frame, 1)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = hands.process(rgb)
                rgb.flags.writeable = True

                hand_detected = False
                features = None

                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    features = extract_features(hand_landmarks, frame.shape)
                    hand_detected = True

                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style()
                    )

                # Hold stability logic:
                # Only save a sample if hand has been in similar position for HOLD_FRAMES
                if hand_detected and features:
                    if last_saved_features is None:
                        hold_count += 1
                    else:
                        # Check how different this frame is from last saved
                        diff = np.linalg.norm(
                            np.array(features) - np.array(last_saved_features)
                        )
                        if diff < 0.15:   # threshold — tune if needed
                            hold_count += 1
                        else:
                            hold_count = max(0, hold_count - 2)  # penalize movement

                    if hold_count >= HOLD_FRAMES:
                        writer.writerow([sign] + [f"{v:.6f}" for v in features])
                        csv_file.flush()
                        collected += 1
                        last_saved_features = features
                        hold_count = 0  # reset — require re-stabilization
                else:
                    hold_count = max(0, hold_count - 1)

                # Draw HUD
                frame = draw_overlay(
                    frame, sign, collected, samples_per_sign,
                    "RECORDING" if hand_detected else "NO HAND DETECTED",
                    hold_count, HOLD_FRAMES
                )

                cv2.imshow("ISL Data Collector", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nQuit by user — progress saved.")
                    csv_file.close()
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            # Sign complete — brief cooldown
            print(f"  [{sign}] done — {collected} samples saved.")
            cooldown_end = time.time() + COOLDOWN_SECONDS
            while time.time() < cooldown_end:
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = fix_frame(frame)
                frame = cv2.flip(frame, 1)
                remaining_cd = cooldown_end - time.time()
                draw_overlay(frame, sign, collected, samples_per_sign, "COOLDOWN")
                cv2.putText(frame, f"Next sign in {remaining_cd:.1f}s", (180, 260),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, AMBER, 2)
                cv2.imshow("ISL Data Collector", frame)
                cv2.waitKey(1)

    csv_file.close()
    cap.release()
    cv2.destroyAllWindows()

    print(f"\nCollection complete! Saved to: {output_file}")
    print("Share this file with your team lead to merge into the master dataset.")


# ── Merge utility ──────────────────────────────────────────────────────────────

def merge_csvs(output_path="data/combined_dataset.csv"):
    """Merges all person CSVs in data/raw/ into one combined dataset."""
    import glob
    files = glob.glob(os.path.join(OUTPUT_DIR, "*_gestures.csv"))
    if not files:
        print("No CSV files found in data/raw/")
        return

    all_rows = []
    header   = None
    label_counts = {}

    for f in files:
        with open(f, "r") as fh:
            reader = csv.reader(fh)
            h = next(reader)
            if header is None:
                header = h
            for row in reader:
                if row:
                    all_rows.append(row)
                    label_counts[row[0]] = label_counts.get(row[0], 0) + 1

    with open(output_path, "w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(header)
        writer.writerows(all_rows)

    print(f"\nMerged {len(files)} files → {output_path}")
    print(f"Total samples: {len(all_rows)}")
    print("\nSamples per sign:")
    for sign in sorted(label_counts):
        bar = "█" * (label_counts[sign] // 20)
        print(f"  {sign:>2}: {label_counts[sign]:>5}  {bar}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISL Data Collector")
    parser.add_argument("--person",  required=True,
                        help="Your name (used in output filename, e.g. --person ravi)")
    parser.add_argument("--samples", type=int, default=SAMPLES_PER_SIGN,
                        help=f"Samples per sign (default: {SAMPLES_PER_SIGN})")
    parser.add_argument("--signs",   nargs="+", default=ALL_SIGNS,
                        help="Specific signs to collect, e.g. --signs A B C 1 2")
    parser.add_argument("--merge",   action="store_true",
                        help="Merge all CSVs in data/raw/ into combined_dataset.csv")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start fresh even if output file already exists")

    args = parser.parse_args()

    if args.merge:
        merge_csvs()
    else:
        # Validate signs
        invalid = [s for s in args.signs if s not in ALL_SIGNS]
        if invalid:
            print(f"Unknown signs: {invalid}. Valid: {ALL_SIGNS}")
            exit(1)

        collect(
            person_name      = args.person.lower().replace(" ", "_"),
            signs_to_collect = args.signs,
            samples_per_sign = args.samples,
            resume           = not args.no_resume
        )