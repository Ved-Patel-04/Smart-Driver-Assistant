import pygame
pygame.init()
pygame.mixer.init()
pygame.mixer.music.load("alarm.mp3")  # Ensure alarm.mp3 is in folder

import cv2
import mediapipe as mp
import time
import math
from datetime import datetime

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)

# Eye landmark indices for EAR calculation
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Mouth landmarks for MAR (mouth aspect ratio)
MOUTH = [61, 291, 81, 311, 13, 14]

def euclidean_dist(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def calculate_ear(landmarks, eye_points):
    p1 = landmarks[eye_points[0]]
    p2 = landmarks[eye_points[1]]
    p3 = landmarks[eye_points[2]]
    p4 = landmarks[eye_points[3]]
    p5 = landmarks[eye_points[4]]
    p6 = landmarks[eye_points[5]]

    vertical_1 = euclidean_dist(p2, p6)
    vertical_2 = euclidean_dist(p3, p5)
    horizontal = euclidean_dist(p1, p4)

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear

def calculate_mar(landmarks, mouth_points):
    vertical = euclidean_dist(landmarks[mouth_points[4]], landmarks[mouth_points[5]])
    horizontal = euclidean_dist(landmarks[mouth_points[0]], landmarks[mouth_points[1]])
    mar = vertical / horizontal
    return mar

# Thresholds
EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 60

MAR_THRESHOLD = 0.6
MAR_CONSEC_FRAMES = 15

cap = cv2.VideoCapture(0)
frame_counter = 0
yawn_counter = 0
drowsy = False
yawning = False

def log_event(event):
    with open("drowsiness_log.txt", "a") as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {event}\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        mesh_points = results.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape

        landmarks = [(int(p.x * w), int(p.y * h)) for p in mesh_points]

        left_ear = calculate_ear(landmarks, LEFT_EYE)
        right_ear = calculate_ear(landmarks, RIGHT_EYE)
        ear_avg = (left_ear + right_ear) / 2.0

        mar = calculate_mar(landmarks, MOUTH)

        # Draw landmarks
        for idx in LEFT_EYE + RIGHT_EYE:
            cv2.circle(frame, landmarks[idx], 2, (0, 255, 0), -1)
        for idx in MOUTH:
            cv2.circle(frame, landmarks[idx], 2, (255, 0, 0), -1)

        if ear_avg < EAR_THRESHOLD:
            frame_counter += 1
            if frame_counter >= CONSEC_FRAMES:
                if not drowsy:
                    log_event("Drowsiness detected")
                drowsy = True
                cv2.putText(frame, "DROWSINESS ALERT!", (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                if not pygame.mixer.music.get_busy():
                    pygame.mixer.music.play()
        else:
            frame_counter = 0
            drowsy = False
            pygame.mixer.music.stop()

        if mar > MAR_THRESHOLD:
            yawn_counter += 1
            if yawn_counter >= MAR_CONSEC_FRAMES:
                if not yawning:
                    log_event("Yawn detected")
                yawning = True
                cv2.putText(frame, "YAWN ALERT!", (30, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 3)
        else:
            yawn_counter = 0
            yawning = False

        cv2.putText(frame, f'EAR: {ear_avg:.2f}', (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f'MAR: {mar:.2f}', (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Driver Assistant - Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()
