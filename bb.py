import cv2
import dlib
import time
import numpy as np
import module as m
from threading import Thread
from scipy.spatial import distance as dist
from imutils import face_utils

# Thresholds
EYE_AR_THRESHOLD = 0.3
EYE_AR_CONSEC_FRAMES = 10
YAWN_THRESH = 20
HEAD_TILT_THRESHOLD = 40

# Variables
is_alarm_active = False
closed_eye_start_time = None
yawn_start_time = None

COUNTER = 0
TOTAL_BLINKS = 0
CLOSED_EYES_FRAME = 3
# variables for frame rate.
FRAME_COUNTER = 0
START_TIME = time.time()
FPS = 0


# Dlib models
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def calculate_ear(eye_points): 
    vertical1 = dist.euclidean(eye_points[1], eye_points[5])
    vertical2 = dist.euclidean(eye_points[2], eye_points[4])
    horizontal = dist.euclidean(eye_points[0], eye_points[3])
    return (vertical1 + vertical2) / (2.0 * horizontal)

def lip_distance(shape):
    top_lip = np.concatenate((shape[50:53], shape[61:64]))
    low_lip = np.concatenate((shape[56:59], shape[65:68]))
    return abs(np.mean(top_lip, axis=0)[1] - np.mean(low_lip, axis=0)[1])

def calculate_head_tilt(landmarks):
    left_eye = landmarks[36]
    right_eye = landmarks[45]
    delta_x = right_eye[0] - left_eye[0] 
    delta_y = right_eye[1] - left_eye[1]
    return np.degrees(np.arctan2(delta_y, delta_x))

def play_alarm(message):
    global is_alarm_active
    while is_alarm_active:
        print(message)
        time.sleep(1)

# Initialize camera
time.sleep(2.0)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray_frame)

    for face in faces:
        landmarks = landmark_predictor(gray_frame, face)
        landmarks = face_utils.shape_to_np(landmarks)

        left_eye_points = landmarks[36:42]
        right_eye_points = landmarks[42:48]
        mouth_points = landmarks[48:68]

        left_ear = calculate_ear(left_eye_points)
        right_ear = calculate_ear(right_eye_points)
        average_ear = (left_ear + right_ear) / 2.0
        distance = lip_distance(landmarks)

        cv2.drawContours(frame, [cv2.convexHull(left_eye_points)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(right_eye_points)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(mouth_points)], -1, (0, 255, 0), 1)
        
        image, PointList = m.faceLandmakDetector(frame, gray_frame, face, False)
        # print(PointList)

        RightEyePoint = PointList[36:42]
        LeftEyePoint = PointList[42:48]
        leftRatio, topMid, bottomMid = m.blinkDetector(LeftEyePoint)
        rightRatio, rTop, rBottom = m.blinkDetector(RightEyePoint)
        
        mask, pos, color = m.EyeTracking(frame, gray_frame, RightEyePoint)
        maskleft, leftPos, leftColor = m.EyeTracking(
            frame, gray_frame, LeftEyePoint)
        
        print(pos, leftPos)

        # Eye detection
        if average_ear < EYE_AR_THRESHOLD:
            if closed_eye_start_time is None:
                closed_eye_start_time = time.time()
                # print(time.time() - closed_eye_start_time)
            elif time.time() - closed_eye_start_time >= EYE_AR_CONSEC_FRAMES:
                if not is_alarm_active:
                    is_alarm_active = True
                    alarm_thread = Thread(target=play_alarm, args=("MATA ANDA MENGANTUK!",))
                    alarm_thread.daemon = True
                    alarm_thread.start()
                cv2.putText(frame, "MATA MENGANTUK!", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            closed_eye_start_time = None
            is_alarm_active = False

        # Yawn detection
        if distance > YAWN_THRESH:
            if yawn_start_time is None:
                yawn_start_time = time.time()
                # print(time.time() - yawn_start_time)
            elif time.time() - yawn_start_time > 3:
                cv2.putText(frame, "ANDA MENGUAP !!!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if not is_alarm_active:
                    is_alarm_active = True
                    alarm_thread = Thread(target=play_alarm, args=("ANDA MENGUAP, ISTIRAHAT DULU!",))
                    alarm_thread.daemon = True
                    alarm_thread.start()
        else:
            yawn_start_time = None
            is_alarm_active = False

        # Head tilt detection
        head_tilt_angle = calculate_head_tilt(landmarks)
        cv2.putText(frame, f"Head Tilt: {head_tilt_angle:.2f} deg", (350, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if abs(head_tilt_angle) > HEAD_TILT_THRESHOLD:
            cv2.putText(frame, "KEPALA ANDA MIRING!", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if not is_alarm_active:
                is_alarm_active = True
                alarm_thread = Thread(target=play_alarm, args=("PERHATIAN! KEPALA ANDA MIRING!",))
                alarm_thread.daemon = True
                alarm_thread.start()
        else:
            is_alarm_active = False

        cv2.putText(frame, f"EAR: {average_ear:.2f}", (350, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"MAR: {distance:.2f}", (350, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Deteksi Kantuk dan Menguap", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
