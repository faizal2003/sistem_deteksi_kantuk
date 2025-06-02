import cv2
import dlib
import time
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import pygame  # For sound alerts
import os
import sys

# Import your module if available
try:
    import module as m
except ImportError:
    print("Module not found. Some functionality may be limited.")
    # Define placeholder functions
    class ModulePlaceholder:
        def faceLandmakDetector(self, frame, gray_frame, face, draw=False):
            return frame, []
            
        def blinkDetector(self, eye_points):
            return 0, None, None
            
        def EyeTracking(self, frame, gray_frame, eye_points):
            return None, None, None
    
    m = ModulePlaceholder()

# Initialize pygame for audio
pygame.mixer.init()

class DrowsinessDetector:
    def __init__(self):
        # Drowsiness detection parameters
        self.EYE_AR_THRESHOLD = 0.3  # Will be calibrated
        self.EYE_AR_ADJUSTMENT_FACTOR = 0.75
        self.EYE_AR_CONSEC_FRAMES = 5
        self.YAWN_THRESH = 20
        self.HEAD_TILT_THRESHOLD = 40
        
        # Video output settings - REDUCED SIZE
        self.DISPLAY_WIDTH = 320   # Reduced from 640
        self.DISPLAY_HEIGHT = 240  # Reduced from 480
        self.CAPTURE_WIDTH = 640   # Keep capture resolution higher for better detection
        self.CAPTURE_HEIGHT = 480
        
        # Calibration settings
        self.CALIBRATION_FRAMES = 50
        self.calibration_samples = []
        self.is_calibrated = False
        self.calibration_countdown = self.CALIBRATION_FRAMES
        
        # Detection variables
        self.is_alarm_active = False
        self.closed_eye_start_time = None
        self.yawn_start_time = None
        self.eye_focus_time = None
        
        # Initialize face detection models
        self.initialize_detection_models()
        
        # Load alarm sound
        try:
            pygame.mixer.music.load("alarm.wav")  # Replace with your alarm sound file
        except:
            print("Warning: Alarm sound file not found.")
    
    def initialize_detection_models(self):
        """Initialize face detection and landmark prediction models"""
        try:
            self.face_detector = dlib.get_frontal_face_detector()
            self.landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
            print("Models loaded successfully")
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Make sure shape_predictor_68_face_landmarks.dat is available.")
            sys.exit(1)
    
    def calculate_ear(self, eye_points):
        """Calculate the eye aspect ratio (EAR)"""
        vertical1 = dist.euclidean(eye_points[1], eye_points[5])
        vertical2 = dist.euclidean(eye_points[2], eye_points[4])
        horizontal = dist.euclidean(eye_points[0], eye_points[3])
        return (vertical1 + vertical2) / (2.0 * horizontal) if horizontal > 0 else 0
    
    def lip_distance(self, shape):
        """Calculate the mouth aspect ratio (MAR)"""
        top_lip = np.concatenate((shape[50:53], shape[61:64]))
        low_lip = np.concatenate((shape[56:59], shape[65:68]))
        return abs(np.mean(top_lip, axis=0)[1] - np.mean(low_lip, axis=0)[1])
    
    def calculate_head_tilt(self, landmarks):
        """Calculate head tilt angle in degrees"""
        left_eye = landmarks[36]
        right_eye = landmarks[45]
        delta_x = right_eye[0] - left_eye[0] 
        delta_y = right_eye[1] - left_eye[1]
        return np.degrees(np.arctan2(delta_y, delta_x))
    
    def calibrate_ear_threshold(self, samples):
        """Calculate personalized EAR threshold from calibration samples"""
        if not samples:
            return self.EYE_AR_THRESHOLD
        
        avg_ear = np.mean(samples)
        calibrated_threshold = avg_ear * self.EYE_AR_ADJUSTMENT_FACTOR
        print(f"Calibration complete - Normal EAR: {avg_ear:.3f}, Threshold: {calibrated_threshold:.3f}")
        return calibrated_threshold
    
    def play_alarm(self):
        """Play alarm sound"""
        pygame.mixer.music.play(-1)  # Loop indefinitely
    
    def stop_alarm(self):
        """Stop alarm sound"""
        pygame.mixer.music.stop()
    
    def resize_frame_for_display(self, frame):
        """Resize frame for smaller display while maintaining aspect ratio"""
        return cv2.resize(frame, (self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT))
    
    def scale_coordinates(self, coordinates, scale_x, scale_y):
        """Scale coordinates from original frame to display frame"""
        scaled_coords = []
        for coord in coordinates:
            scaled_x = int(coord[0] * scale_x)
            scaled_y = int(coord[1] * scale_y)
            scaled_coords.append([scaled_x, scaled_y])
        return np.array(scaled_coords)
    
    def run_detection(self):
        """Main detection loop"""
        print("Starting drowsiness detection...")
        print("Press 'q' to quit")
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Cannot open camera!")
            return
        
        # Set camera properties - keep capture resolution for better detection
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.CAPTURE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.CAPTURE_HEIGHT)
        
        # Calculate scaling factors
        scale_x = self.DISPLAY_WIDTH / self.CAPTURE_WIDTH
        scale_y = self.DISPLAY_HEIGHT / self.CAPTURE_HEIGHT
        
        last_alarm_check = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Resize frame for display (smaller output)
            display_frame = self.resize_frame_for_display(frame)
            
            # Convert original frame to grayscale for processing (better accuracy)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector(gray_frame)
            
            # Handle calibration phase
            if not self.is_calibrated:
                progress = 100 - (self.calibration_countdown / self.CALIBRATION_FRAMES * 100)
                
                if len(faces) == 1:  # Only calibrate when exactly one face is detected
                    face = faces[0]
                    landmarks = self.landmark_predictor(gray_frame, face)
                    landmarks = face_utils.shape_to_np(landmarks)
                    
                    left_eye_points = landmarks[36:42]
                    right_eye_points = landmarks[42:48]
                    
                    left_ear = self.calculate_ear(left_eye_points)
                    right_ear = self.calculate_ear(right_eye_points)
                    average_ear = (left_ear + right_ear) / 2.0
                    
                    # Add to calibration samples
                    self.calibration_samples.append(average_ear)
                    self.calibration_countdown -= 1
                    
                    # Scale coordinates for display
                    left_eye_scaled = self.scale_coordinates(left_eye_points, scale_x, scale_y)
                    right_eye_scaled = self.scale_coordinates(right_eye_points, scale_x, scale_y)
                    
                    # Draw eye contours during calibration (on smaller display)
                    cv2.drawContours(display_frame, [cv2.convexHull(left_eye_scaled)], -1, (0, 255, 255), 1)
                    cv2.drawContours(display_frame, [cv2.convexHull(right_eye_scaled)], -1, (0, 255, 255), 1)
                    
                    # Display calibration text (smaller font for smaller display)
                    cv2.putText(display_frame, f"Calibrating... {self.calibration_countdown} left", 
                                (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    
                    print(f"Calibration progress: {progress:.1f}% - EAR: {average_ear:.3f}")
                    
                    # When calibration is complete
                    if self.calibration_countdown <= 0:
                        # Calculate personalized threshold
                        self.EYE_AR_THRESHOLD = self.calibrate_ear_threshold(self.calibration_samples)
                        self.is_calibrated = True
                        print("Calibration complete! Starting detection...")
                else:
                    cv2.putText(display_frame, "Position face for calibration", (5, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    print("Calibrating... Please position your face in the frame")
            
            # Normal operation mode after calibration
            elif faces:
                for face in faces:
                    landmarks = self.landmark_predictor(gray_frame, face)
                    landmarks = face_utils.shape_to_np(landmarks)

                    left_eye_points = landmarks[36:42]
                    right_eye_points = landmarks[42:48]
                    mouth_points = landmarks[48:68]

                    left_ear = self.calculate_ear(left_eye_points)
                    right_ear = self.calculate_ear(right_eye_points)
                    average_ear = (left_ear + right_ear) / 2.0
                    distance = self.lip_distance(landmarks)
                    
                    # Get eye tracking data from module if available
                    image, PointList = m.faceLandmakDetector(frame, gray_frame, face, False)
                    RightEyePoint = PointList[36:42] if len(PointList) > 42 else []
                    LeftEyePoint = PointList[42:48] if len(PointList) > 48 else []
                    
                    if len(LeftEyePoint) > 0 and len(RightEyePoint) > 0:
                        leftRatio, topMid, bottomMid = m.blinkDetector(LeftEyePoint)
                        rightRatio, rTop, rBottom = m.blinkDetector(RightEyePoint)
                            
                        mask, pos, color = m.EyeTracking(frame, gray_frame, RightEyePoint)
                        maskleft, leftPos, leftColor = m.EyeTracking(frame, gray_frame, LeftEyePoint)
                    else:
                        pos, leftPos = "Center", "Center"

                    # Scale coordinates for display
                    left_eye_scaled = self.scale_coordinates(left_eye_points, scale_x, scale_y)
                    right_eye_scaled = self.scale_coordinates(right_eye_points, scale_x, scale_y)
                    mouth_scaled = self.scale_coordinates(mouth_points, scale_x, scale_y)
                    
                    # Draw contours on smaller display
                    cv2.drawContours(display_frame, [cv2.convexHull(left_eye_scaled)], -1, (0, 255, 0), 1)
                    cv2.drawContours(display_frame, [cv2.convexHull(right_eye_scaled)], -1, (0, 255, 0), 1)
                    cv2.drawContours(display_frame, [cv2.convexHull(mouth_scaled)], -1, (0, 255, 0), 1)
                    
                    # Print metrics to console
                    head_tilt_angle = self.calculate_head_tilt(landmarks)
                    print(f"EAR: {average_ear:.2f} | MAR: {distance:.2f} | Head Tilt: {head_tilt_angle:.2f}° | Right Eye: {pos} | Left Eye: {leftPos}")
                    
                    # Alert text positioning and sizing for smaller display
                    y_offset = 20
                    font_scale = 0.4
                    thickness = 1
                    
                    # Eye drowsiness detection
                    if average_ear < self.EYE_AR_THRESHOLD:
                        if self.closed_eye_start_time is None:
                            self.closed_eye_start_time = time.time()
                        elif time.time() - self.closed_eye_start_time >= self.EYE_AR_CONSEC_FRAMES:
                            cv2.putText(display_frame, "MATA MENGANTUK!", (5, y_offset), 
                                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)
                            y_offset += 15
                            
                            # Play alarm only if not already active
                            if not self.is_alarm_active:
                                print("ALERT: Eyes drowsy detected!")
                                self.is_alarm_active = True
                                self.play_alarm()
                    else:
                        self.closed_eye_start_time = None
                        if self.is_alarm_active and not self.yawn_start_time and not self.eye_focus_time:
                            self.is_alarm_active = False
                            self.stop_alarm()

                    # Yawn detection
                    if distance > self.YAWN_THRESH:
                        if self.yawn_start_time is None:
                            self.yawn_start_time = time.time()
                        elif time.time() - self.yawn_start_time > 2:
                            cv2.putText(display_frame, "ANDA MENGUAP!", (5, y_offset), 
                                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)
                            y_offset += 15
                            
                            # Play alarm only if not already active
                            if not self.is_alarm_active:
                                print("ALERT: Yawn detected!")
                                self.is_alarm_active = True
                                self.play_alarm()
                    else:
                        self.yawn_start_time = None
                        if self.is_alarm_active and not self.closed_eye_start_time and not self.eye_focus_time:
                            self.is_alarm_active = False
                            self.stop_alarm()
                            
                    # Eye tracking alarm
                    if pos != "Center" and leftPos != "Center":
                        if self.eye_focus_time is None:
                            self.eye_focus_time = time.time()
                        elif time.time() - self.eye_focus_time > 2:
                            cv2.putText(display_frame, "MATA TIDAK FOKUS!", (5, y_offset), 
                                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)
                            y_offset += 15
                            
                            # Play alarm only if not already active
                            if not self.is_alarm_active:
                                print("ALERT: Eyes not focused!")
                                self.is_alarm_active = True
                                self.play_alarm()
                    else:
                        self.eye_focus_time = None
                        if (self.is_alarm_active and not self.closed_eye_start_time 
                                and not self.yawn_start_time):
                            self.is_alarm_active = False
                            self.stop_alarm()
                    
                    # Head tilt detection
                    if abs(head_tilt_angle) > self.HEAD_TILT_THRESHOLD:
                        cv2.putText(display_frame, "KEPALA MIRING!", (5, y_offset), 
                                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)
                        
                        # Play alarm only if not already active
                        if not self.is_alarm_active:
                            print("ALERT: Head tilted!")
                            self.is_alarm_active = True
                            self.play_alarm()
                    else:
                        if (self.is_alarm_active and not self.closed_eye_start_time 
                                and not self.yawn_start_time and not self.eye_focus_time):
                            self.is_alarm_active = False
                            self.stop_alarm()
                            
            # If calibrated but no face detected
            elif self.is_calibrated:
                cv2.putText(display_frame, "No face detected", (5, 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                
                # Stop alarm if active and no face detected for a while
                if self.is_alarm_active and time.time() - last_alarm_check > 5:
                    self.is_alarm_active = False
                    self.stop_alarm()
            
            # Display the smaller frame
            cv2.imshow('Drowsiness Detection', display_frame)
            
            # Update alarm check time
            last_alarm_check = time.time()
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Control frame rate
            time.sleep(0.05)  # ~20 fps
        
        # Clean up
        if self.is_alarm_active:
            self.stop_alarm()
        
        cap.release()
        cv2.destroyAllWindows()
        print("Detection stopped.")

# Main execution
if __name__ == "__main__":
    detector = DrowsinessDetector()
    try:
        detector.run_detection()
    except KeyboardInterrupt:
        print("\nDetection interrupted by user.")
    except Exception as e:
        print(f"Error occurred: {e}")