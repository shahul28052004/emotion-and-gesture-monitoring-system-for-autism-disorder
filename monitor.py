"""
Emotion and Gesture Monitoring System for Autism Disorder
---------------------------------------------------------
Real-time webcam monitoring that:
  * recognises facial emotions (DeepFace)
  * recognises hand gestures / repetitive movements (MediaPipe Hands)
  * raises an alert when distress signs persist
  * logs everything to a CSV file for caregivers / therapists

Press 'q' to quit.

NOTE: This is a research / educational prototype. It is NOT a medical or
diagnostic device.
"""

import argparse
import csv
import os
import time
from collections import deque, Counter
from datetime import datetime

import cv2
import mediapipe as mp
from deepface import DeepFace

NEGATIVE_EMOTIONS = {"angry", "fear", "sad", "disgust"}

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


# --------------------------------------------------------------------------
# Emotion detection
# --------------------------------------------------------------------------
def detect_emotion(frame):
    """Return (emotion, confidence, (x, y, w, h)) or (None, 0, None)."""
    try:
        result = DeepFace.analyze(
            frame,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="opencv",
            silent=True,
        )
        if isinstance(result, list):
            result = result[0]
        emotion = result["dominant_emotion"]
        confidence = result["emotion"][emotion]
        r = result["region"]
        box = (r["x"], r["y"], r["w"], r["h"])
        # DeepFace returns the full frame when no face is found -> ignore it
        if box[2] >= frame.shape[1] * 0.95:
            return None, 0.0, None
        return emotion, confidence, box
    except Exception:
        return None, 0.0, None


# --------------------------------------------------------------------------
# Gesture detection
# --------------------------------------------------------------------------
class GestureAnalyzer:
    """Detects hand flapping, raised hand and hand-near-face."""

    def __init__(self, history=30, flap_amplitude=0.04, flap_turns=5):
        self.history = {}  # hand index -> deque of wrist (x, y)
        self.size = history
        self.flap_amplitude = flap_amplitude
        self.flap_turns = flap_turns

    def _is_flapping(self, points):
        if len(points) < self.size:
            return False
        ys = [p[1] for p in points]
        turns, last_dir = 0, 0
        for a, b in zip(ys, ys[1:]):
            diff = b - a
            if abs(diff) < self.flap_amplitude / 3:
                continue
            direction = 1 if diff > 0 else -1
            if last_dir and direction != last_dir:
                turns += 1
            last_dir = direction
        amplitude = max(ys) - min(ys)
        return turns >= self.flap_turns and amplitude >= self.flap_amplitude

    def analyze(self, hand_landmarks_list, face_box, frame_shape):
        gestures = set()
        h, w = frame_shape[:2]
        for idx, hand in enumerate(hand_landmarks_list):
            wrist = hand.landmark[0]
            self.history.setdefault(idx, deque(maxlen=self.size))
            self.history[idx].append((wrist.x, wrist.y))

            if self._is_flapping(self.history[idx]):
                gestures.add("hand flapping")
            if wrist.y < 0.30:
                gestures.add("hand raised")
            if face_box:
                fx, fy, fw, fh = face_box
                px, py = wrist.x * w, wrist.y * h
                margin = 0.4 * fw
                if (fx - margin < px < fx + fw + margin) and (
                    fy - margin < py < fy + fh + margin
                ):
                    gestures.add("hand near face")
        if not hand_landmarks_list:
            self.history.clear()
        return gestures


# --------------------------------------------------------------------------
# Logging + alerts
# --------------------------------------------------------------------------
class SessionLogger:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        new_file = not os.path.exists(path)
        self.file = open(path, "a", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        if new_file:
            self.writer.writerow(["timestamp", "emotion", "confidence", "gestures", "alert"])

    def log(self, emotion, confidence, gestures, alert):
        self.writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            emotion or "none",
            f"{confidence:.1f}",
            ";".join(sorted(gestures)) or "none",
            alert or "",
        ])
        self.file.flush()

    def close(self):
        self.file.close()


class AlertEngine:
    """Alert if distress persists for `seconds` seconds."""

    def __init__(self, seconds=5):
        self.seconds = seconds
        self.distress_since = None

    def check(self, emotion, gestures):
        distressed = emotion in NEGATIVE_EMOTIONS or "hand flapping" in gestures
        now = time.time()
        if distressed:
            if self.distress_since is None:
                self.distress_since = now
            if now - self.distress_since >= self.seconds:
                return "Possible distress - please check on the child"
        else:
            self.distress_since = None
        return None


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Emotion & Gesture Monitoring System")
    parser.add_argument("--camera", type=int, default=0, help="webcam index")
    parser.add_argument("--log", default="logs/session_log.csv", help="CSV log path")
    parser.add_argument("--alert-seconds", type=int, default=5)
    parser.add_argument("--emotion-every", type=int, default=5,
                        help="run emotion model every N frames (speed vs accuracy)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam.")

    gesture_analyzer = GestureAnalyzer()
    alerts = AlertEngine(args.alert_seconds)
    logger = SessionLogger(args.log)
    emotion_window = deque(maxlen=10)  # smooth emotion over last readings

    emotion, confidence, face_box = None, 0.0, None
    frame_no, last_log = 0, 0.0

    with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6,
                        min_tracking_confidence=0.6) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            frame_no += 1

            if frame_no % args.emotion_every == 0:
                e, c, box = detect_emotion(frame)
                if e:
                    emotion_window.append(e)
                    emotion = Counter(emotion_window).most_common(1)[0][0]
                    confidence, face_box = c, box
                else:
                    face_box = None

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            hand_list = result.multi_hand_landmarks or []
            for hl in hand_list:
                mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

            gestures = gesture_analyzer.analyze(hand_list, face_box, frame.shape)
            alert = alerts.check(emotion, gestures)

            # ---- draw UI ----
            if face_box:
                x, y, w, h = face_box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
            cv2.putText(frame, f"Emotion: {emotion or '-'} ({confidence:.0f}%)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame, f"Gesture: {', '.join(sorted(gestures)) or '-'}",
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
            if alert:
                cv2.putText(frame, "ALERT: " + alert, (10, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # log about once per second
            if time.time() - last_log >= 1.0:
                logger.log(emotion, confidence, gestures, alert)
                last_log = time.time()

            cv2.imshow("Emotion & Gesture Monitoring System", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    logger.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
