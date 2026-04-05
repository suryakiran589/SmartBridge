"""
virtual_keyboard.py
===================
Air-touch QWERTY virtual keyboard rendered on the webcam feed.
- Index fingertip = pointer
- Pinch (index + thumb close) = key press
- 3 second delay on activation so user can click target window
- Types into active window via pyautogui
- Shows typed text on screen

Drop into utils/ folder.
"""

import cv2
import numpy as np
import pyautogui
import math
import time

# ── Layout ─────────────────────────────────────────────────────────────────────

ROWS = [
    ['Q','W','E','R','T','Y','U','I','O','P'],
    ['A','S','D','F','G','H','J','K','L'],
    ['Z','X','C','V','B','N','M'],
    ['SPACE', 'BACK']
]

# ── Colors (BGR) ───────────────────────────────────────────────────────────────

COL_KEY_BG      = (30, 30, 30)
COL_KEY_BORDER  = (70, 70, 70)
COL_KEY_HOVER   = (0, 180, 80)
COL_KEY_PRESS   = (255, 255, 255)
COL_KEY_TEXT    = (220, 220, 220)
COL_KEY_TEXT_H  = (0, 0, 0)
COL_POINTER     = (0, 220, 80)
COL_PINCH       = (0, 200, 255)
COL_TEXT_BG     = (20, 20, 20)
COL_TYPED_TEXT  = (255, 255, 255)
COL_AMBER       = (0, 165, 255)


class VirtualKey:
    """Represents a single key on the keyboard."""

    def __init__(self, label, x, y, w, h):
        self.label       = label
        self.x, self.y   = x, y
        self.w, self.h   = w, h
        self.is_hover    = False
        self.press_flash = 0

    def contains(self, px, py):
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h

    def draw(self, frame):
        if self.press_flash > 0:
            bg = COL_KEY_PRESS
            tc = (0, 0, 0)
            self.press_flash -= 1
        elif self.is_hover:
            bg = COL_KEY_HOVER
            tc = COL_KEY_TEXT_H
        else:
            bg = COL_KEY_BG
            tc = COL_KEY_TEXT

        cv2.rectangle(frame, (self.x, self.y),
                      (self.x + self.w, self.y + self.h), bg, -1)
        cv2.rectangle(frame, (self.x, self.y),
                      (self.x + self.w, self.y + self.h), COL_KEY_BORDER, 1)

        label = self.label
        if label == 'SPACE':
            label = 'SPC'
        elif label == 'BACK':
            label = '<<'

        font_scale = 0.45 if len(label) == 1 else 0.35
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        tx = self.x + (self.w - tw) // 2
        ty = self.y + (self.h + th) // 2
        cv2.putText(frame, label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, tc, 1, cv2.LINE_AA)


class VirtualKeyboard:
    """
    Full QWERTY virtual keyboard overlay.

    Usage:
        kb = VirtualKeyboard()
        frame, typed = kb.process(frame, lm_list)
    """

    def __init__(self, frame_w=640, frame_h=480):
        self.frame_w = frame_w
        self.frame_h = frame_h

        # Keyboard dimensions
        self.kb_x   = 20
        self.kb_y   = 220
        self.key_w  = 52
        self.key_h  = 42
        self.key_gap = 5

        # Typed text
        self.typed_text = ""

        # Pinch state
        self.pinch_active    = False
        self.last_press_time = 0
        self.PRESS_COOLDOWN  = 0.4

        # Focus delay — gives user time to click target window
        self.start_time  = None
        self.START_DELAY = 3.0  # seconds

        # Build key grid
        self.keys = self._build_keys()

        # Pointer smoothing
        self._px, self._py = 0, 0
        self.SMOOTH = 0.4

    # ── Build key layout ──────────────────────────────────────────────────────

    def _build_keys(self):
        keys = []
        row_offsets = [0, 26, 52, 52]

        for row_idx, row in enumerate(ROWS):
            y = self.kb_y + row_idx * (self.key_h + self.key_gap)

            if row_idx == 3:
                space_w = self.key_w * 4 + self.key_gap * 3
                back_w  = self.key_w * 2 + self.key_gap
                x = self.kb_x + row_offsets[row_idx]
                keys.append(VirtualKey('SPACE', x, y, space_w, self.key_h))
                x += space_w + self.key_gap
                keys.append(VirtualKey('BACK', x, y, back_w, self.key_h))
            else:
                indent = row_offsets[row_idx]
                for col_idx, label in enumerate(row):
                    x = self.kb_x + indent + col_idx * (self.key_w + self.key_gap)
                    keys.append(VirtualKey(label, x, y, self.key_w, self.key_h))

        return keys

    # ── Distance helper ───────────────────────────────────────────────────────

    def _distance(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    # ── Main process ──────────────────────────────────────────────────────────

    def process(self, frame, lm_list):
        just_typed = None

        # ── Focus delay countdown ─────────────────────────────────────────────
        if self.start_time is None:
            self.start_time = time.time()

        elapsed   = time.time() - self.start_time
        in_delay  = elapsed < self.START_DELAY

        if in_delay:
            remaining = int(self.START_DELAY - elapsed) + 1

            # Dark overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (self.frame_w, self.frame_h),
                          (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

            # Countdown message
            cv2.putText(frame,
                        "Click your target window now!",
                        (60, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)
            cv2.putText(frame,
                        f"Keyboard starts in  {remaining}",
                        (120, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame,
                        "(Notepad / Word / any text app)",
                        (90, 290),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA)

            # Draw faded keyboard preview
            for key in self.keys:
                key.draw(frame)

            return frame, None

        # ── Normal keyboard processing ────────────────────────────────────────

        # Background panel
        overlay   = frame.copy()
        kb_bottom = self.kb_y + 4 * (self.key_h + self.key_gap)
        cv2.rectangle(overlay,
                      (self.kb_x - 10, self.kb_y - 10),
                      (self.frame_w - 10, kb_bottom + 10),
                      (15, 15, 15), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        # Reset hover
        for key in self.keys:
            key.is_hover = False

        pointer_x, pointer_y = None, None
        is_pinching = False

        if len(lm_list) >= 21:
            raw_px = lm_list[8][1]
            raw_py = lm_list[8][2]

            # Smooth pointer
            self._px = int(self._px + (raw_px - self._px) * self.SMOOTH)
            self._py = int(self._py + (raw_py - self._py) * self.SMOOTH)
            pointer_x, pointer_y = self._px, self._py

            thumb_x = lm_list[4][1]
            thumb_y = lm_list[4][2]

            pinch_dist  = self._distance((pointer_x, pointer_y), (thumb_x, thumb_y))
            is_pinching = pinch_dist < 40

            # Draw pinch line
            pinch_color = COL_PINCH if is_pinching else (80, 80, 80)
            cv2.line(frame, (pointer_x, pointer_y), (thumb_x, thumb_y),
                     pinch_color, 2)
            cv2.circle(frame, (thumb_x, thumb_y), 8,
                       COL_PINCH if is_pinching else (80, 80, 80), -1)

            # Check hover + press
            for key in self.keys:
                if key.contains(pointer_x, pointer_y):
                    key.is_hover = True
                    now = time.time()
                    if (is_pinching and
                            not self.pinch_active and
                            now - self.last_press_time > self.PRESS_COOLDOWN):
                        just_typed           = self._press_key(key)
                        key.press_flash      = 6
                        self.last_press_time = now

            self.pinch_active = is_pinching

        # Draw keys
        for key in self.keys:
            key.draw(frame)

        # Draw pointer
        if pointer_x is not None:
            color = COL_PINCH if is_pinching else COL_POINTER
            cv2.circle(frame, (pointer_x, pointer_y), 10, color, -1)
            cv2.circle(frame, (pointer_x, pointer_y), 12, (255, 255, 255), 1)

        frame = self._draw_text_bar(frame)
        return frame, just_typed

    # ── Key press ─────────────────────────────────────────────────────────────

    def _press_key(self, key):
        label = key.label

        if label == 'SPACE':
            self.typed_text += ' '
            try:
                pyautogui.press('space')
            except Exception:
                pass
            return ' '

        elif label == 'BACK':
            if self.typed_text:
                self.typed_text = self.typed_text[:-1]
            try:
                pyautogui.press('backspace')
            except Exception:
                pass
            return 'BACK'

        else:
            self.typed_text += label
            try:
                pyautogui.typewrite(label.lower(), interval=0.01)
            except Exception:
                pass
            return label

    # ── Text bar ──────────────────────────────────────────────────────────────

    def _draw_text_bar(self, frame):
        bar_y   = self.kb_y - 52
        overlay = frame.copy()
        cv2.rectangle(overlay,
                      (self.kb_x - 10, bar_y),
                      (self.frame_w - 10, bar_y + 40),
                      COL_TEXT_BG, -1)
        frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)

        cv2.putText(frame, "Typed:", (self.kb_x, bar_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (120, 120, 120), 1, cv2.LINE_AA)

        display = self.typed_text[-38:] + "|"
        cv2.putText(frame, display, (self.kb_x + 52, bar_y + 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    COL_TYPED_TEXT, 1, cv2.LINE_AA)

        return frame

    def clear(self):
        """Call this when switching away from keyboard mode."""
        self.typed_text = ""
        self.start_time = None   # reset delay for next activation
        self.pinch_active = False