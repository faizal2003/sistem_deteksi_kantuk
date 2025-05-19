import cv2
import dlib
import time
import numpy as np
import threading
import tkinter as tk
from tkinter import ttk, font
from PIL import Image, ImageTk
from scipy.spatial import distance as dist
from imutils import face_utils
from collections import deque
import pygame  # For sound alerts

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

class DrowsinessDetectionApp:
    def __init__(self, root):
        # App configuration
        self.root = root
        self.root.title("Sistem Pendeteksi Kantuk")
        self.root.configure(bg="#4169E1")  # Royal blue background
        
        # Window size
        window_width = 1000
        window_height = 700
        self.root.geometry(f"{window_width}x{window_height}")
        
        # UI fonts
        title_font = font.Font(family="Arial", size=24, weight="bold")
        button_font = font.Font(family="Arial", size=14, weight="bold")
        
        # Create frames
        self.title_frame = tk.Frame(root, bg="#8DA9DB", padx=10, pady=10)
        self.title_frame.pack(fill="x", padx=20, pady=20)
        
        self.video_frame = tk.Frame(root, bg="#8DA9DB", padx=10, pady=10)
        self.video_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        self.button_frame = tk.Frame(root, bg="#4169E1", padx=10, pady=10)
        self.button_frame.pack(fill="x", padx=20, pady=20)
        
        # Title label
        self.title_label = tk.Label(self.title_frame, text="SISTEM PENDETEKSI KANTUK", 
                                    font=title_font, bg="#8DA9DB")
        self.title_label.pack()
        
        # Video display
        self.video_label = tk.Label(self.video_frame, bg="black")
        self.video_label.pack(fill="both", expand=True)
        
        # Status frame (for displaying metrics)
        self.status_frame = tk.Frame(self.video_frame, bg="#8DA9DB")
        self.status_frame.pack(fill="x", side="bottom", pady=5)
        
        self.ear_label = tk.Label(self.status_frame, text="EAR: 0.00", font=("Arial", 12), 
                                  bg="#8DA9DB", fg="green")
        self.ear_label.pack(side="left", padx=10)
        
        self.threshold_label = tk.Label(self.status_frame, text="Threshold: 0.00", 
                                       font=("Arial", 12), bg="#8DA9DB", fg="blue")
        self.threshold_label.pack(side="left", padx=10)
        
        self.head_tilt_label = tk.Label(self.status_frame, text="Head Tilt: 0.00°", 
                                       font=("Arial", 12), bg="#8DA9DB", fg="orange")
        self.head_tilt_label.pack(side="left", padx=10)
        
        self.mar_label = tk.Label(self.status_frame, text="MAR: 0.00", 
                                 font=("Arial", 12), bg="#8DA9DB", fg="green")
        self.mar_label.pack(side="left", padx=10)
        
        # Control buttons
        self.start_button = tk.Button(self.button_frame, text="START", font=button_font,
                                     bg="#8DA9DB", fg="black", width=15, height=2,
                                     command=self.start_detection)
        self.start_button.pack(side="left", padx=50)
        
        self.stop_button = tk.Button(self.button_frame, text="STOP", font=button_font,
                                    bg="#8DA9DB", fg="black", width=15, height=2,
                                    command=self.stop_detection)
        self.stop_button.pack(side="right", padx=50)
        
        # Calibration progress
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.video_frame, variable=self.progress_var,
                                            maximum=100, length=200)
        
        self.calibration_label = tk.Label(self.video_frame, text="", 
                                         font=("Arial", 12), bg="#8DA9DB", fg="black")
        
        # System variables
        self.cap = None
        self.is_running = False
        self.current_frame = None
        
        # Drowsiness detection parameters
        self.EYE_AR_THRESHOLD = 0.3  # Will be calibrated
        self.EYE_AR_ADJUSTMENT_FACTOR = 0.75
        self.EYE_AR_CONSEC_FRAMES = 10
        self.YAWN_THRESH = 20
        self.HEAD_TILT_THRESHOLD = 40
        
        # Calibration settings
        self.CALIBRATION_FRAMES = 50
        self.calibration_samples = []
        self.is_calibrated = False
        self.calibration_countdown = self.CALIBRATION_FRAMES
        
        # Detection variables
        self.is_alarm_active = False
        self.closed_eye_start_time = None
        self.yawn_start_time = None
        
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
            tk.messagebox.showerror("Error", "Failed to load detection models!\n"
                                    "Make sure shape_predictor_68_face_landmarks.dat is available.")
            self.root.quit()
    
    def start_detection(self):
        """Start the drowsiness detection process"""
        if self.is_running:
            return
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            tk.messagebox.showerror("Error", "Cannot open camera!")
            return
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.is_running = True
        self.is_calibrated = False
        self.calibration_countdown = self.CALIBRATION_FRAMES
        self.calibration_samples = []
        
        # Show calibration progress bar
        self.calibration_label.config(text="Calibrating... Keep your eyes open")
        self.calibration_label.pack(pady=5)
        self.progress_bar.pack(pady=5)
        
        # Start video processing in a separate thread
        self.video_thread = threading.Thread(target=self.process_video)
        self.video_thread.daemon = True
        self.video_thread.start()
    
    def stop_detection(self):
        """Stop the drowsiness detection process"""
        self.is_running = False
        self.progress_var.set(0)
        self.calibration_label.pack_forget()
        self.progress_bar.pack_forget()
        
        # Reset the video frame
        blank_image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.update_video_display(blank_image)
        
        # Reset status labels
        self.ear_label.config(text="EAR: 0.00")
        self.threshold_label.config(text="Threshold: 0.00")
        self.head_tilt_label.config(text="Head Tilt: 0.00°")
        self.mar_label.config(text="MAR: 0.00")
        
        # Release the camera
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
    
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
    
    def update_video_display(self, frame):
        """Update the video label with the current frame"""
        if frame is not None:
            # Convert the frame to RGB for tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            
            # Resize to fit the display area if needed
            display_width = self.video_label.winfo_width()
            display_height = self.video_label.winfo_height()
            
            if display_width > 1 and display_height > 1:  # Valid dimensions
                img = img.resize((display_width, display_height))
            
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
    
    def process_video(self):
        """Process video frames for drowsiness detection"""
        last_alarm_check = time.time()
        
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Create a copy for display
            display_frame = frame.copy()
            
            # Convert to grayscale for processing
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector(gray_frame)
            
            # Handle calibration phase
            if not self.is_calibrated:
                # Update progress bar
                progress = 100 - (self.calibration_countdown / self.CALIBRATION_FRAMES * 100)
                self.progress_var.set(progress)
                
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
                    
                    # Draw eye contours during calibration
                    cv2.drawContours(display_frame, [cv2.convexHull(left_eye_points)], -1, (0, 255, 255), 1)
                    cv2.drawContours(display_frame, [cv2.convexHull(right_eye_points)], -1, (0, 255, 255), 1)
                    
                    # Display calibration text
                    cv2.putText(display_frame, f"Calibrating... {self.calibration_countdown}", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # When calibration is complete
                    if self.calibration_countdown <= 0:
                        # Calculate personalized threshold
                        self.EYE_AR_THRESHOLD = self.calibrate_ear_threshold(self.calibration_samples)
                        self.is_calibrated = True
                        
                        # Update UI
                        self.root.after(0, lambda: self.threshold_label.config(
                            text=f"Threshold: {self.EYE_AR_THRESHOLD:.2f}"))
                        self.root.after(0, lambda: self.calibration_label.config(
                            text="Calibration complete!"))
                        
                        # Hide progress bar after a delay
                        self.root.after(2000, lambda: self.progress_bar.pack_forget())
                        self.root.after(2000, lambda: self.calibration_label.pack_forget())
                else:
                    cv2.putText(display_frame, "Position one face in frame for calibration", (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
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

                    # Draw contours
                    cv2.drawContours(display_frame, [cv2.convexHull(left_eye_points)], -1, (0, 255, 0), 1)
                    cv2.drawContours(display_frame, [cv2.convexHull(right_eye_points)], -1, (0, 255, 0), 1)
                    cv2.drawContours(display_frame, [cv2.convexHull(mouth_points)], -1, (0, 255, 0), 1)
                    
                    # Update metrics in UI
                    self.root.after(0, lambda ear=average_ear: self.ear_label.config(
                        text=f"EAR: {ear:.2f}"))
                    self.root.after(0, lambda mar=distance: self.mar_label.config(
                        text=f"MAR: {mar:.2f}"))
                    
                    try:
                        # Get eye tracking data from module if available
                        image, PointList = m.faceLandmakDetector(frame, gray_frame, face, False)
                        RightEyePoint = PointList[36:42]
                        LeftEyePoint = PointList[42:48]
                        leftRatio, topMid, bottomMid = m.blinkDetector(LeftEyePoint)
                        rightRatio, rTop, rBottom = m.blinkDetector(RightEyePoint)
                        
                        mask, pos, color = m.EyeTracking(frame, gray_frame, RightEyePoint)
                        maskleft, leftPos, leftColor = m.EyeTracking(frame, gray_frame, LeftEyePoint)
                    except Exception as e:
                        # Module functions might not be available
                        pass
                    
                    # Eye drowsiness detection
                    if average_ear < self.EYE_AR_THRESHOLD:
                        if self.closed_eye_start_time is None:
                            self.closed_eye_start_time = time.time()
                        elif time.time() - self.closed_eye_start_time >= self.EYE_AR_CONSEC_FRAMES:
                            cv2.putText(display_frame, "DROWSINESS DETECTED!", (10, 40), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            
                            # Play alarm only if not already active
                            if not self.is_alarm_active:
                                self.is_alarm_active = True
                                self.play_alarm()
                    else:
                        self.closed_eye_start_time = None
                        if self.is_alarm_active and not self.yawn_start_time:
                            self.is_alarm_active = False
                            self.stop_alarm()

                    # Yawn detection
                    if distance > self.YAWN_THRESH:
                        if self.yawn_start_time is None:
                            self.yawn_start_time = time.time()
                        elif time.time() - self.yawn_start_time > 1:  # Reduced from 3s for responsiveness
                            cv2.putText(display_frame, "YAWNING DETECTED!", (10, 70), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            
                            # Play alarm only if not already active
                            if not self.is_alarm_active:
                                self.is_alarm_active = True
                                self.play_alarm()
                    else:
                        self.yawn_start_time = None
                        if self.is_alarm_active and not self.closed_eye_start_time:
                            self.is_alarm_active = False
                            self.stop_alarm()

                    # Head tilt detection
                    head_tilt_angle = self.calculate_head_tilt(landmarks)
                    self.root.after(0, lambda angle=head_tilt_angle: self.head_tilt_label.config(
                        text=f"Head Tilt: {angle:.2f}°"))
                    
                    if abs(head_tilt_angle) > self.HEAD_TILT_THRESHOLD:
                        cv2.putText(display_frame, "HEAD TILT DETECTED!", (10, 100), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        # Play alarm only if not already active
                        if not self.is_alarm_active:
                            self.is_alarm_active = True
                            self.play_alarm()
                    else:
                        if (self.is_alarm_active and not self.closed_eye_start_time 
                                and not self.yawn_start_time):
                            self.is_alarm_active = False
                            self.stop_alarm()
            
            # If calibrated but no face detected
            elif self.is_calibrated:
                cv2.putText(display_frame, "No face detected", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Stop alarm if active and no face detected for a while
                if self.is_alarm_active and time.time() - last_alarm_check > 5:
                    self.is_alarm_active = False
                    self.stop_alarm()
            
            # Update the video display
            self.update_video_display(display_frame)
            
            # Update alarm check time
            last_alarm_check = time.time()
            
            # Sleep briefly to reduce CPU usage
            time.sleep(0.03)  # ~30 fps
        
        # Clean up when thread ends
        if self.is_alarm_active:
            self.stop_alarm()

# Main execution
if __name__ == "__main__":
    root = tk.Tk()
    app = DrowsinessDetectionApp(root)
    root.mainloop()