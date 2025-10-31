import cv2
import mediapipe as mp
import numpy as np
import os
import datetime
from PIL import Image

# =============================
# 🖐️ Setup
# =============================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.9, min_tracking_confidence=0.9)
mp_drawing = mp.solutions.drawing_utils

canvas = 255 * np.ones(shape=[720, 1280, 3], dtype=np.uint8)
current_color = (0, 0, 255)
prev_x, prev_y = None, None
pen_down = False
current_stroke = []
strokes = []
just_lifted = False

SAVE_DIR = "saved_drawings"
os.makedirs(SAVE_DIR, exist_ok=True)

buttons = {
    "Red": ((50, 50), (150, 100), (0, 0, 255)),
    "Green": ((180, 50), (280, 100), (0, 255, 0)),
    "Blue": ((310, 50), (410, 100), (255, 0, 0)),
    "Yellow": ((440, 50), (540, 100), (0, 255, 255)),
    "Erase": ((570, 50), (670, 100), (200, 200, 200)),
    "Clear": ((700, 50), (850, 100), (255, 255, 255))
}

erase_mode = False
movement_threshold = 5
alpha = 0.3

# =============================
# 🧩 Helper Functions
# =============================
def check_button_click(x, y):
    global current_color, canvas, erase_mode, strokes, current_stroke
    for label, ((x1, y1), (x2, y2), color) in buttons.items():
        if x1 <= x <= x2 and y1 <= y <= y2:
            if label == "Erase":
                erase_mode = True
            elif label == "Clear":
                canvas[:] = 255
                strokes.clear()
                current_stroke.clear()
                erase_mode = False
            else:
                current_color = color
                erase_mode = False
            return True
    return False


def is_fist(hand_landmarks):
    folded_fingers = 0
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    for tip, pip in zip(finger_tips, finger_pips):
        if hand_landmarks.landmark[tip].y > hand_landmarks.landmark[pip].y:
            folded_fingers += 1
    return folded_fingers >= 3


# =============================
# 🧠 Main Processing
# =============================
def process_frame(frame):
    global canvas, prev_x, prev_y, pen_down, current_color, strokes, current_stroke, just_lifted

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (960, 720))
    height, width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Draw buttons
    for label, ((x1, y1), (x2, y2), color) in buttons.items():
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        cv2.putText(frame, label, (x1 + 5, y1 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Process hand
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cx, cy = int(hand_landmarks.landmark[8].x * width), int(hand_landmarks.landmark[8].y * height)

            if check_button_click(cx, cy):
                continue

            if is_fist(hand_landmarks):
                if pen_down:
                    pen_down = False
                    prev_x, prev_y = None, None
                    if current_stroke:
                        strokes.append(current_stroke)
                        current_stroke = []
                    just_lifted = True
                continue
            else:
                if not pen_down:
                    pen_down = True
                    prev_x, prev_y = None, None

            if just_lifted:
                just_lifted = False
                prev_x, prev_y = cx, cy
                continue

            if pen_down:
                if prev_x is not None and prev_y is not None:
                    cv2.line(canvas, (prev_x, prev_y), (cx, cy), current_color, 5)
                    current_stroke.append(((cx, cy), current_color))
                prev_x, prev_y = cx, cy

    # Redraw strokes
    for stroke in strokes:
        for i in range(1, len(stroke)):
            cv2.line(canvas, stroke[i - 1][0], stroke[i][0], stroke[i][1], 5)
    for i in range(1, len(current_stroke)):
        cv2.line(canvas, current_stroke[i - 1][0], current_stroke[i][0], current_stroke[i][1], 5)

    return frame, canvas
