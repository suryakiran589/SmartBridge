import cv2
import mediapipe as mp
import numpy as np
import math
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class HandTracker:
    def __init__(self, mode=False, max_hands=2, detection_con=0.7, track_con=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con
        
        # Initialize MediaPipe Hands via Tasks API
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'hand_landmarker.task')
        base_options = python.BaseOptions(model_asset_path=model_path)
        
        # Create HandLandmarker
        running_mode = vision.RunningMode.IMAGE
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            num_hands=self.max_hands,
            min_hand_detection_confidence=self.detection_con,
            min_hand_presence_confidence=self.track_con,
            min_tracking_confidence=self.track_con
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        
        # Hand connections for drawing
        self.HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (5, 6), (6, 7), (7, 8),
            (9, 10), (10, 11), (11, 12),
            (13, 14), (14, 15), (15, 16),
            (17, 18), (18, 19), (19, 20),
            (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)
        ]

    def _draw_landmarks(self, frame, landmarks):
        h, w, _ = frame.shape
        # Draw connections
        for connection in self.HAND_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            
            x1 = int(landmarks[start_idx].x * w)
            y1 = int(landmarks[start_idx].y * h)
            x2 = int(landmarks[end_idx].x * w)
            y2 = int(landmarks[end_idx].y * h)
            
            cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            
        # Draw points
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

    def find_hands(self, frame, draw=True):
        """Processes the frame and finds hands, drawing landmarks if requested."""
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        self.results = self.landmarker.detect(mp_image)

        if self.results.hand_landmarks and draw:
            for hand_landmarks in self.results.hand_landmarks:
                self._draw_landmarks(frame, hand_landmarks)
        return frame

    def find_position(self, frame, hand_no=0, draw=True):
        """Returns a list of landmark positions for a specific hand."""
        landmark_list = []
        if hasattr(self, 'results') and self.results.hand_landmarks:
            if len(self.results.hand_landmarks) > hand_no:
                my_hand = self.results.hand_landmarks[hand_no]
                h, w, c = frame.shape
                for id, lm in enumerate(my_hand):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmark_list.append([id, cx, cy])
                    if draw:
                        cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return landmark_list

    def find_distance(self, p1, p2, frame, draw=True, r=15, t=3):
        """Calculates distance between two specific landmarks."""
        if not hasattr(self, 'results') or not self.results.hand_landmarks:
             return 0, frame, [0,0,0,0,0,0]
             
        landmarks = self.find_position(frame, draw=False)
        if len(landmarks) == 0:
            return 0, frame, [0,0,0,0,0,0]

        try:
            x1, y1 = landmarks[p1][1], landmarks[p1][2]
            x2, y2 = landmarks[p2][1], landmarks[p2][2]
        except IndexError:
            return 0, frame, [0,0,0,0,0,0]
            
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(frame, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(frame, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        
        length = math.hypot(x2 - x1, y2 - y1)
        return length, frame, [x1, y1, x2, y2, cx, cy]
