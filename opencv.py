#Hand gesture control for a game using OpenCV and MediaPipe
import cv2
import mediapipe as mp
import pyautogui
import time

pyautogui.FAILSAFE = False

# ================== MEDIAPIPE SETUP ==================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# ================== CAMERA SETUP ==================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
time.sleep(1)

# Map gestures to game keys
GESTURE_KEY_MAP = {
    "UP": "up",      # Jump
    "DOWN": "down",  # Slide
    "RIGHT": "right",
    "LEFT": "left"
}

active_gesture = None  # Track currently active gesture

# ================== HAND TRACKING ==================
with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
) as hands:

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)  # Mirror image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = hands.process(rgb)
        rgb.flags.writeable = True

        gesture = None

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                h, w, _ = frame.shape
                lm = []
                for id, pt in enumerate(hand_landmarks.landmark):
                    lm.append([id, int(pt.x * w), int(pt.y * h)])

                fingers = []
                hand_label = handedness.classification[0].label  # "Left" or "Right"

                # Thumb logic depends on hand
                if hand_label == "Left":
                    fingers.append(1 if lm[4][1] > lm[3][1] else 0)
                else:  # Right hand
                    fingers.append(1 if lm[4][1] < lm[3][1] else 0)

                # Index, Middle, Ring, Pinky (y-axis)
                for i in [8, 12, 16, 20]:
                    fingers.append(1 if lm[i][2] < lm[i - 2][2] else 0)

                # ================== GESTURE DETECTION ==================
                # Up = all fingers except thumb extended
                if fingers[1:] == [1, 1, 1, 1]:
                    gesture = "UP"
                # Down = all fingers except thumb closed
                elif fingers[1:] == [0, 0, 0, 0]:
                    gesture = "DOWN"
                # Right = only index extended
                elif fingers[1:] == [1, 0, 0, 0]:
                    gesture = "RIGHT"
                # Left = index + middle extended
                elif fingers[1:] == [1, 1, 0, 0]:
                    gesture = "LEFT"

        # ================== PRESS KEY ONCE PER GESTURE ==================
        if gesture != active_gesture and gesture is not None:
            pyautogui.press(GESTURE_KEY_MAP[gesture])
            active_gesture = gesture

        if gesture is None:
            active_gesture = None  # Reset when no hand detected

        # ================== DISPLAY ==================
        if gesture:
            cv2.putText(frame, f"Gesture: {gesture}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        cv2.imshow("Hand Gesture Controller", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# ================== CLEANUP ==================
cap.release()
cv2.destroyAllWindows()
