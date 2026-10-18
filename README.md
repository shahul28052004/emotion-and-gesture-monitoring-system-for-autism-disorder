# Emotion and Gesture Monitoring System for Autism Disorder

A real-time computer-vision prototype that watches a webcam feed and helps
caregivers and therapists understand a child's emotional state and body
language.

## Features
- **Facial emotion recognition** (happy, sad, angry, fear, surprise, disgust, neutral) using DeepFace
- **Gesture detection** using MediaPipe Hands:
  - hand flapping (repetitive movement)
  - hand raised
  - hand near face
- **Distress alerts** when negative emotion or flapping persists for several seconds
- **CSV session log** (`logs/session_log.csv`) for later review

## Tech stack
Python, OpenCV, MediaPipe, DeepFace (TensorFlow)

## Installation
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m venv venv
# Windows: venv\Scripts\activate     Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```
Python 3.9 - 3.11 is recommended.

## Usage
```bash
python monitor.py
```
Options:
```
--camera 0            webcam index
--alert-seconds 5     seconds of distress before an alert
--emotion-every 5     run the emotion model every N frames (higher = faster)
--log logs/session_log.csv
```
Press **q** to quit.

## How it works
1. Each frame is analysed for the dominant facial emotion (smoothed over the last 10 readings).
2. MediaPipe tracks hand landmarks; wrist motion is analysed to detect flapping.
3. An alert engine raises a warning when distress signs last longer than the threshold.
4. Every second the state is written to a CSV log.

## Future work
- Train a custom emotion model on autism-specific datasets
- Add more gestures (ear covering, rocking, head banging) using MediaPipe Pose
- Dashboard for therapists, sound/SMS alerts

## Disclaimer
This is an educational/research prototype and **not** a medical or diagnostic tool.
Always consult qualified professionals. Obtain consent before recording any child.

## License
MIT
