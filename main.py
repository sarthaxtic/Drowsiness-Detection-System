from flask import Flask, render_template, Response
import cv2
import numpy as np
import dlib
from imutils import face_utils
import pygame
import time
import threading
import atexit

app = Flask(__name__)

# Initialize face detection model
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize pygame mixer
pygame.mixer.init()

# Global state variables
Sleep, Drowsy, Active = 0, 0, 0
status = ""
color = (0, 0, 0)
frame = None
eyes_closed_start_time = None
alarm_playing = False
detection_active = True  # Control start/stop detection

# Compute Euclidean distance
def compute(ptA, ptB):
    return np.linalg.norm(ptA - ptB)

# Blink detection
def blink(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)
    if ratio > 0.25:
        return 2
    elif ratio > 0.21:
        return 1
    else:
        return 0

# Play alarm
def play_alarm():
    global alarm_playing
    if not alarm_playing:
        pygame.mixer.music.load("alarm.mp3")
        pygame.mixer.music.play()
        alarm_playing = True

# Stop alarm
def stop_alarm():
    global alarm_playing
    if alarm_playing:
        pygame.mixer.music.stop()
        alarm_playing = False

# Main drowsiness detection generator
def detect_drowsiness():
    global Sleep, Drowsy, Active, status, color, frame
    global eyes_closed_start_time, detection_active

    while True:
        ret, frame_raw = cap.read()
        if not ret:
            break

        frame = frame_raw.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if detection_active:
            faces = detector(gray)

            for face in faces:
                landmarks = predictor(gray, face)
                landmarks = face_utils.shape_to_np(landmarks)

                left_blink = blink(landmarks[36], landmarks[37], landmarks[38], landmarks[41], landmarks[40], landmarks[39])
                right_blink = blink(landmarks[42], landmarks[43], landmarks[44], landmarks[47], landmarks[46], landmarks[45])

                if left_blink == 0 or right_blink == 0:
                    Sleep += 1
                    Drowsy = 0
                    Active = 0
                    if Sleep > 6:
                        if eyes_closed_start_time is None:
                            eyes_closed_start_time = time.time()
                        if time.time() - eyes_closed_start_time >= 2:
                            play_alarm()
                        status = "SLEEPING ALERT!!!"
                        color = (255, 0, 0)
                    else:
                        stop_alarm()
                else:
                    eyes_closed_start_time = None
                    Sleep = Drowsy = Active = 0
                    stop_alarm()
                    status = "ACTIVE :)"
                    color = (0, 255, 0)

            cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # Encode frame for MJPEG streaming
        ret, jpeg = cv2.imencode('.jpg', frame)
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(detect_drowsiness(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_detection')
def start_detection():
    global detection_active
    detection_active = True
    return "Drowsiness detection started"

@app.route('/stop_detection')
def stop_detection():
    global detection_active
    detection_active = False
    stop_alarm()
    return "Drowsiness detection stopped"

# Graceful shutdown
def release_camera():
    print("Releasing camera and stopping alarm...")
    cap.release()
    stop_alarm()

atexit.register(release_camera)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
