import cv2
import mediapipe as mp
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# Indices for left and right eye landmarks on face mesh
LEFT_EYE_IDX = [33, 133, 160, 159, 158, 157, 173, 246]
RIGHT_EYE_IDX = [362, 263, 387, 386, 385, 384, 398, 466]

def calculate_EAR(landmarks, eye_indices):
    # Calculate Eye Aspect Ratio (EAR) to detect eye openness
    # landmarks: list of (x, y) coords normalized
    # eye_indices: indices of eye landmarks
    # returns EAR value

    # You can calculate EAR using distances between vertical and horizontal points
    # For simplicity, we'll use a basic ratio for now
    
    # We'll extract points:
    eye = [landmarks[i] for i in eye_indices]
    
    # Calculate vertical distances
    vertical_1 = ((eye[1][1] - eye[5][1])**2 + (eye[1][0] - eye[5][0])**2) ** 0.5
    vertical_2 = ((eye[2][1] - eye[4][1])**2 + (eye[2][0] - eye[4][0])**2) ** 0.5

    # Calculate horizontal distance
    horizontal = ((eye[0][1] - eye[3][1])**2 + (eye[0][0] - eye[3][0])**2) ** 0.5

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear

cap = cv2.VideoCapture(0)

EAR_THRESHOLD = 0.25
CLOSED_EYES_FRAMES = 20  # Number of consecutive frames to trigger drowsiness alert
counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            landmarks = []
            for lm in face_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                landmarks.append((x, y))

            left_ear = calculate_EAR(landmarks, LEFT_EYE_IDX)
            right_ear = calculate_EAR(landmarks, RIGHT_EYE_IDX)
            avg_ear = (left_ear + right_ear) / 2.0

            # Draw landmarks on eyes
            for idx in LEFT_EYE_IDX + RIGHT_EYE_IDX:
                cv2.circle(frame, landmarks[idx], 2, (0, 255, 0), -1)

            if avg_ear < EAR_THRESHOLD:
                counter += 1
                cv2.putText(frame, "Eyes Closed", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                if counter >= CLOSED_EYES_FRAMES:
                    cv2.putText(frame, "WAKE UP!", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                counter = 0

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
