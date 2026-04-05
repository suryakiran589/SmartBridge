"""
gesture_model.py  —  ML-powered ISL/ASL Sign Language Classifier
=================================================================
Three modes:
  MODE 1 — Sign Language: ML classifier → sentence builder → TTS
  MODE 2 — Gesture Control: finger counting → mouse / scroll / volume / media
  MODE 3 — Air Keyboard: virtual QWERTY keyboard, pinch to type
"""

import cv2
import numpy as np
import os
import math
import platform
import subprocess
import threading
import joblib
import pyautogui
from utils.hand_tracking import HandTracker
from utils.virtual_keyboard import VirtualKeyboard

# ── Config ──────────────────────────────────────────────────────────────────────

MODEL_PATH     = os.path.join(os.path.dirname(__file__), "gesture_classifier.pkl")
LABEL_MAP_PATH = os.path.join(os.path.dirname(__file__), "label_map.json")

CONFIDENCE_THRESHOLD = 0.75
HOLD_FRAMES          = 20
HOLD_THRESHOLD       = 0.12
COOLDOWN_FRAMES      = 30

MODE_SIGN     = "sign"
MODE_CONTROL  = "control"
MODE_KEYBOARD = "keyboard"

# Colors (BGR)
COLOR_GREEN  = (0, 220, 80)
COLOR_AMBER  = (0, 165, 255)
COLOR_RED    = (0, 60, 220)
COLOR_WHITE  = (255, 255, 255)
COLOR_GRAY   = (160, 160, 160)
COLOR_PURPLE = (200, 80, 200)


# ── TTS engine ─────────────────────────────────────────────────────────────────

class TTSEngine:
    def __init__(self):
        self._lock = threading.Lock()

    def speak(self, text):
        if not text or not text.strip():
            return
        def _run():
            with self._lock:
                try:
                    import pyttsx3
                    engine = pyttsx3.init()
                    engine.setProperty("rate", 150)
                    engine.setProperty("volume", 1.0)
                    engine.say(text)
                    engine.runAndWait()
                    engine.stop()
                except Exception as e:
                    print(f"[TTS] Error: {e}")
        threading.Thread(target=_run, daemon=True).start()


# ── Sentence builder ────────────────────────────────────────────────────────────

class SentenceBuilder:
    def __init__(self):
        self.current_word = []
        self.sentence     = []

    def add_sign(self, sign):
        sign = sign.upper()
        if sign == "SPACE":
            if self.current_word:
                self.sentence.append("".join(self.current_word))
                self.current_word = []
        elif sign == "DELETE":
            if self.current_word:
                self.current_word.pop()
            elif self.sentence:
                self.current_word = list(self.sentence.pop())
        elif sign == "CLEAR":
            self.current_word = []
            self.sentence     = []
        else:
            self.current_word.append(sign)

    @property
    def current_word_str(self):
        return "".join(self.current_word)

    @property
    def sentence_str(self):
        words = self.sentence + ([self.current_word_str] if self.current_word else [])
        return " ".join(words)

    @property
    def completed_sentence(self):
        return " ".join(self.sentence)


# ── Feature extraction ──────────────────────────────────────────────────────────

def extract_features(lm_list):
    if len(lm_list) != 21:
        return None
    coords = [(pt[1], pt[2], 0.0) for pt in lm_list]
    wx, wy, _ = coords[0]
    xs    = [c[0] for c in coords]
    ys    = [c[1] for c in coords]
    scale = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)
    features = []
    for (x, y, z) in coords:
        features.extend([(x - wx) / scale, (y - wy) / scale, z])
    return features


# ── Cross-platform helpers ──────────────────────────────────────────────────────

def screenshot():
    if platform.system() == "Darwin":
        pyautogui.hotkey('command', 'shift', '3')
    else:
        pyautogui.hotkey('win', 'shift', 's')

def lock_screen():
    if platform.system() == "Darwin":
        try:
            subprocess.Popen([
                'osascript', '-e',
                'tell application "System Events" to keystroke "q" '
                'using {command down, control down}'
            ])
        except Exception:
            pass
    else:
        pyautogui.hotkey('win', 'l')


# ── Main classifier ─────────────────────────────────────────────────────────────

class GestureClassifier:

    def __init__(self):
        self.tracker = HandTracker(detection_con=0.8, track_con=0.7)

        # Mode
        self.mode = MODE_SIGN

        # Load trained model
        self._model   = None
        self._classes = []
        self._load_model()

        # Sentence builder + TTS
        self.builder = SentenceBuilder()
        self.tts     = TTSEngine()

        # State
        self.active_gesture = "None"
        self.active_action  = "Waiting..."
        self.confidence     = 0.0
        self.last_signed    = ""

        # Hold-to-confirm
        self._hold_sign     = None
        self._hold_count    = 0
        self._last_features = None
        self._cooldown      = 0

        # Space detection
        self._no_hand_counter  = 0
        self.NO_HAND_THRESHOLD = 50

        # Virtual keyboard
        self.virtual_kb = VirtualKeyboard(frame_w=640, frame_h=480)

        # PyAutoGUI (control mode)
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE    = 0.01
        self.screen_w, self.screen_h = pyautogui.size()
        self.cam_w, self.cam_h       = 640, 480
        self.ploc_x = self.ploc_y    = 0
        self.cloc_x = self.cloc_y    = 0
        self.smoothing               = 5
        self.frame_reduction         = 100
        self.action_cooldown         = 15
        self.cooldown_counter        = 0

    # ── Mode toggle ───────────────────────────────────────────────────────────

    def toggle_mode(self):
        if self.mode == MODE_SIGN:
            self.mode = MODE_CONTROL
            self._hold_count      = 0
            self._hold_sign       = None
            self._no_hand_counter = 0
        elif self.mode == MODE_CONTROL:
            self.mode = MODE_KEYBOARD
            self.cooldown_counter = 0
        else:
            self.mode = MODE_SIGN
            self.virtual_kb.clear()
        return self.mode

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self):
        if not os.path.exists(MODEL_PATH):
            print(f"[GestureClassifier] WARNING: Model not found at {MODEL_PATH}")
            return
        try:
            bundle        = joblib.load(MODEL_PATH)
            self._model   = bundle["model"]
            self._classes = bundle["classes"]
            model_name    = bundle.get("model_name", "unknown")
            accuracy      = bundle.get("accuracy", 0)
            print(f"[GestureClassifier] Model loaded: {model_name} ({accuracy*100:.1f}% accuracy)")
        except Exception as e:
            print(f"[GestureClassifier] ERROR loading model: {e}")

    # ── ML prediction ─────────────────────────────────────────────────────────

    def _predict(self, lm_list):
        if self._model is None:
            return "None", 0.0
        features = extract_features(lm_list)
        if features is None:
            return "None", 0.0
        X     = np.array(features).reshape(1, -1)
        probs = self._model.predict_proba(X)[0]
        idx   = probs.argmax()
        conf  = probs[idx]
        if conf < CONFIDENCE_THRESHOLD:
            return "None", float(conf)
        return self._classes[idx], float(conf)

    # ── Hold-to-confirm ───────────────────────────────────────────────────────

    def _update_hold(self, sign, features):
        if sign == "None" or features is None:
            self._hold_count = max(0, self._hold_count - 2)
            self._hold_sign  = None
            return None
        if sign != self._hold_sign:
            self._hold_sign     = sign
            self._hold_count    = 1
            self._last_features = features
            return None
        if self._last_features is not None:
            drift = np.linalg.norm(np.array(features) - np.array(self._last_features))
            if drift > HOLD_THRESHOLD:
                self._hold_count    = max(0, self._hold_count - 1)
                self._last_features = features
                return None
        self._hold_count   += 1
        self._last_features = features
        if self._hold_count >= HOLD_FRAMES:
            self._hold_count = 0
            self._hold_sign  = None
            return sign
        return None

    # ── Frame processing ───────────────────────────────────────────────────────

    def process_frame(self, frame):
        frame   = cv2.flip(frame, 1)
        frame   = self.tracker.find_hands(frame)
        lm_list = self.tracker.find_position(frame, draw=False)

        self.active_gesture = "None"
        self.active_action  = "Show a sign..." if self.mode == MODE_SIGN else "Show a gesture..."
        self.confidence     = 0.0

        if self.mode == MODE_SIGN:
            frame = self._process_sign_mode(frame, lm_list)
        elif self.mode == MODE_CONTROL:
            frame = self._process_control_mode(frame, lm_list)
        else:
            frame = self._process_keyboard_mode(frame, lm_list)

        frame = self._draw_ui(frame)
        return frame

    # ── Sign language mode ────────────────────────────────────────────────────

    def _process_sign_mode(self, frame, lm_list):
        if self._cooldown > 0:
            self._cooldown -= 1

        if len(lm_list) == 21:
            self._no_hand_counter = 0
            if self._model is not None:
                sign, conf  = self._predict(lm_list)
                features    = extract_features(lm_list)
                self.confidence     = conf
                self.active_gesture = sign if sign != "None" else "..."
                confirmed = None
                if self._cooldown == 0:
                    confirmed = self._update_hold(sign, features)
                if confirmed:
                    self.builder.add_sign(confirmed)
                    self.last_signed   = confirmed
                    self.active_action = f"Signed: {confirmed}"
                    self._cooldown     = COOLDOWN_FRAMES
                else:
                    if self._hold_count > 0 and sign != "None":
                        pct = int(self._hold_count / HOLD_FRAMES * 100)
                        self.active_action = f"Hold... {pct}%"
                    else:
                        self.active_action = "Reading sign..."
            else:
                self.active_action = "No model — run train_model.py"
        else:
            if self.builder.current_word_str:
                self._no_hand_counter += 1
                remaining = self.NO_HAND_THRESHOLD - self._no_hand_counter
                self.active_action = f"Space in {remaining} frames..."
                if self._no_hand_counter >= self.NO_HAND_THRESHOLD:
                    word = self.builder.current_word_str
                    self.builder.add_sign("SPACE")
                    self.last_signed      = "SPACE"
                    self.active_action    = f"Word added: {word}"
                    self._no_hand_counter = 0
                    self.tts.speak(word)
            else:
                self._no_hand_counter = 0
                self.active_action    = "Show a sign..."
        return frame

    # ── Virtual keyboard mode ─────────────────────────────────────────────────

    def _process_keyboard_mode(self, frame, lm_list):
        frame, just_typed = self.virtual_kb.process(frame, lm_list)
        if just_typed:
            if just_typed == 'BACK':
                self.active_gesture = "Delete"
                self.active_action  = "Backspace"
            elif just_typed == ' ':
                self.active_gesture = "SPACE"
                self.active_action  = "Space added"
            else:
                self.active_gesture = just_typed
                self.active_action  = f"Typed: {just_typed}"
            self.last_signed = just_typed
        else:
            self.active_gesture = "Air Keyboard"
            self.active_action  = "Pinch to type"
        self.confidence = 1.0
        return frame

    # ── Gesture control mode ──────────────────────────────────────────────────

    def _process_control_mode(self, frame, lm_list):
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1

        cv2.rectangle(frame,
            (self.frame_reduction, self.frame_reduction),
            (self.cam_w - self.frame_reduction, self.cam_h - self.frame_reduction),
            (255, 0, 255), 2)

        if len(lm_list) == 0:
            self.active_action = "No hand detected"
            return frame

        tip_ids = [4, 8, 12, 16, 20]
        fingers = []
        if lm_list[tip_ids[0]][1] > lm_list[tip_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        for id in range(1, 5):
            fingers.append(1 if lm_list[tip_ids[id]][2] < lm_list[tip_ids[id] - 2][2] else 0)

        total = fingers.count(1)
        self.confidence = 0.95

        if total == 5:
            # Open palm → mouse movement
            self.active_gesture = "Open Palm"
            self.active_action  = "Mouse Movement"
            self._move_mouse(lm_list[8], frame)

        elif total == 0:
            # Fist → scroll down
            self.active_gesture = "Fist"
            self.active_action  = "Scroll Down"
            if self.cooldown_counter == 0:
                pyautogui.scroll(-500)
                self.cooldown_counter = self.action_cooldown

        elif fingers == [0, 1, 0, 0, 0]:
            # Index only → scroll up
            self.active_gesture = "Index Up"
            self.active_action  = "Scroll Up"
            if self.cooldown_counter == 0:
                pyautogui.scroll(500)
                self.cooldown_counter = self.action_cooldown

        elif fingers == [0, 1, 1, 0, 0]:
            # Peace sign → left click
            self.active_gesture = "Peace Sign"
            self.active_action  = "Left Click"
            length, frame, info = self.tracker.find_distance(8, 12, frame)
            if length < 40 and self.cooldown_counter == 0:
                cv2.circle(frame, (info[4], info[5]), 15, (0, 255, 0), cv2.FILLED)
                pyautogui.click()
                self.active_action    = "Clicked!"
                self.cooldown_counter = self.action_cooldown

        elif fingers == [0, 1, 1, 1, 0]:
            # 3 fingers → right click
            self.active_gesture = "3 Fingers"
            self.active_action  = "Right Click"
            if self.cooldown_counter == 0:
                pyautogui.rightClick()
                self.active_action    = "Right Clicked!"
                self.cooldown_counter = self.action_cooldown * 2

        elif fingers == [1, 1, 0, 0, 0]:
            # Thumb + index → volume control
            self.active_gesture = "Volume Control"
            self.active_action  = "Pinch=Vol Down  Spread=Vol Up"
            length, frame, info = self.tracker.find_distance(4, 8, frame)
            if length < 30 and self.cooldown_counter == 0:
                pyautogui.press('volumedown')
                self.active_action    = "Volume Down"
                self.cooldown_counter = self.action_cooldown
            elif length > 120 and self.cooldown_counter == 0:
                pyautogui.press('volumeup')
                self.active_action    = "Volume Up"
                self.cooldown_counter = self.action_cooldown

        elif fingers == [1, 0, 0, 0, 0]:
            # Thumbs up → play/pause
            self.active_gesture = "Thumbs Up"
            self.active_action  = "Play / Pause"
            if self.cooldown_counter == 0:
                pyautogui.press('playpause')
                self.cooldown_counter = self.action_cooldown * 2

        elif fingers == [0, 0, 0, 0, 1]:
            # Pinky only → screenshot
            self.active_gesture = "Pinky"
            self.active_action  = "Screenshot"
            if self.cooldown_counter == 0:
                screenshot()
                self.active_action    = "Screenshot taken!"
                self.cooldown_counter = self.action_cooldown * 3

        elif fingers == [1, 0, 0, 0, 1]:
            # Shaka → lock screen
            self.active_gesture = "Shaka"
            self.active_action  = "Lock Screen"
            if self.cooldown_counter == 0:
                lock_screen()
                self.cooldown_counter = self.action_cooldown * 6

        return frame

    # ── Mouse helper — fixed direction ────────────────────────────────────────

    def _move_mouse(self, index_finger_lm, frame):
        x1, y1 = index_finger_lm[1], index_finger_lm[2]
        # FIX: removed screen_w inversion — now palm left = cursor left
        x3 = np.interp(x1, (self.frame_reduction, self.cam_w - self.frame_reduction), (0, self.screen_w))
        y3 = np.interp(y1, (self.frame_reduction, self.cam_h - self.frame_reduction), (0, self.screen_h))
        self.cloc_x = self.ploc_x + (x3 - self.ploc_x) / self.smoothing
        self.cloc_y = self.ploc_y + (y3 - self.ploc_y) / self.smoothing
        try:
            pyautogui.moveTo(self.cloc_x, self.cloc_y)  # ← removed inversion
            cv2.circle(frame, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            self.ploc_x, self.ploc_y = self.cloc_x, self.cloc_y
        except Exception:
            pass

    # ── UI drawing ─────────────────────────────────────────────────────────────

    def _draw_ui(self, frame):
        h, w = frame.shape[:2]

        if self.mode == MODE_SIGN:
            mode_color = COLOR_GREEN
            mode_label = "SIGN LANG"
        elif self.mode == MODE_CONTROL:
            mode_color = COLOR_PURPLE
            mode_label = "CONTROL"
        else:
            mode_color = COLOR_AMBER
            mode_label = "KEYBOARD"
        cv2.rectangle(frame, (w - 130, 10), (w - 10, 36), mode_color, -1)
        cv2.putText(frame, mode_label, (w - 122, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (20, 20, 20), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        conf_color = COLOR_GREEN if self.confidence >= CONFIDENCE_THRESHOLD else COLOR_AMBER
        cv2.putText(frame, self.active_gesture, (16, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, conf_color, 2)

        bar_w = int(180 * min(self.confidence, 1.0))
        cv2.rectangle(frame, (16, 58), (196, 70), (60, 60, 60), -1)
        cv2.rectangle(frame, (16, 58), (16 + bar_w, 70), conf_color, -1)
        cv2.putText(frame, f"{self.confidence*100:.0f}%", (202, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_GRAY, 1)
        cv2.putText(frame, self.active_action, (16, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 1)

        if self.mode == MODE_SIGN and self._hold_count > 0:
            hold_w = int((w - 32) * self._hold_count / HOLD_FRAMES)
            cv2.rectangle(frame, (16, h - 24), (w - 16, h - 14), (60, 60, 60), -1)
            cv2.rectangle(frame, (16, h - 24), (16 + hold_w, h - 14), COLOR_GREEN, -1)
            cv2.putText(frame, "Hold steady", (16, h - 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_GRAY, 1)

        if self.mode == MODE_SIGN:
            overlay2 = frame.copy()
            cv2.rectangle(overlay2, (0, h - 90), (w, h - 26), (20, 20, 20), -1)
            frame = cv2.addWeighted(overlay2, 0.55, frame, 0.45, 0)
            word_display = f"Word: {self.builder.current_word_str}_" if self.builder.current_word_str else "Word: _"
            sent_display = f"Sentence: {self.builder.sentence_str}" if self.builder.sentence_str else "Sentence: (sign letters)"
            cv2.putText(frame, word_display, (16, h - 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_AMBER, 1)
            cv2.putText(frame, sent_display[:72], (16, h - 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)

        return frame

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_current_state(self):
        return {
            "gesture":     self.active_gesture,
            "action":      self.active_action,
            "confidence":  f"{self.confidence:.2f}",
            "word":        self.builder.current_word_str,
            "sentence":    self.builder.sentence_str,
            "last_signed": self.last_signed,
            "mode":        self.mode
        }

    def clear_sentence(self):
        self.builder.add_sign("CLEAR")

    def speak_now(self):
        text = self.builder.sentence_str
        if text:
            self.tts.speak(text)