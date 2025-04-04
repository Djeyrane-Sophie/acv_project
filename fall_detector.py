import cv2
import numpy as np
import time
import mediapipe as mp
from pygame import mixer
import os

class FallDetector:
    def __init__(self):
        # Initialize MediaPipe Holistic model and drawing utilities
        self.holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.drawing_utils = mp.solutions.drawing_utils
        self.drawing_styles = mp.solutions.drawing_styles
        self.pose = mp.solutions.pose.Pose()

        # Initialize OpenCV and sound mixer
        self.cap = cv2.VideoCapture(0)
        cv2.namedWindow('Fall_detection', cv2.WINDOW_NORMAL)
        mixer.init()
        
        self.mid_shoulder_list = []
        self.mid_heel_list = []
        self.speed_list = []
        self.angle_list = []

    def draw_holistic_results(self, image, results):
        """Draw holistic pose landmarks onto the image."""
        if results.pose_landmarks:
            self.drawing_utils.draw_landmarks(
                image,
                results.pose_landmarks,
                mp.solutions.holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.drawing_styles.get_default_pose_landmarks_style()
            )
        return image

    def process_frame(self, image):
        """Process the image to detect holistic pose landmarks."""
        try:
            results = self.holistic.process(image)
        except Exception:
            results = None
        return results

    def compute_features(self, results):
        """Extract key body landmarks and compute speed and angle."""
        if results.pose_landmarks:
            right_shoulder = results.pose_landmarks.landmark[12]
            left_shoulder = results.pose_landmarks.landmark[11]
            self.mid_shoulder_list.append(((right_shoulder.x + left_shoulder.x)/2, 
                                           (right_shoulder.y + left_shoulder.y)/2, 
                                           (right_shoulder.z + left_shoulder.z)/2))
            
            right_heel = results.pose_landmarks.landmark[30]
            left_heel = results.pose_landmarks.landmark[29]
            self.mid_heel_list.append(((right_heel.x + left_heel.x)/2, 
                                       (right_heel.y + left_heel.y)/2, 
                                       (right_heel.z + left_heel.z)/2))

        if len(self.mid_shoulder_list) > 1:
            (x1, y1, z1) = self.mid_shoulder_list[-2]
            (x2, y2, z2) = self.mid_shoulder_list[-1]
            speed = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2) * 2
            self.speed_list.append(speed)

        if len(self.mid_shoulder_list) > 1 and len(self.mid_heel_list) > 1:
            (xs, ys, zs) = self.mid_shoulder_list[-1]
            (xh, yh, zh) = self.mid_heel_list[-1]
            angle = np.degrees(np.arctan((yh - ys)/(xh - xs)))
            self.angle_list.append(angle)

    def detect_fall(self, result_image):
        """Detect a fall based on speed and angle variations."""
        fall_detected = False
        possible_fall = False
        min_index = 0

        if len(self.speed_list) > 1 and len(self.angle_list) > 1:
            for index in range(min_index, len(self.angle_list)):
                if possible_fall:
                    if 0 < abs(self.angle_list[index]) < 10:
                        print(f"Fall Confirmed.")
                        cv2.putText(result_image, "Game Over!", (10, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        fall_detected = True
                        break  

                    if abs(self.angle_list[index]) > abs(self.angle_list[index - 1]):
                        possible_fall = False

                if abs(self.speed_list[index]) > 0.1 and abs(self.angle_list[index]) < 45:
                    possible_fall = True
                    min_index = index

        return fall_detected

    def play_sound(self):
        """Play sound when a fall is detected."""
        mixer.music.load("game_over.mp3")
        mixer.music.play()

    def start(self):
        """Run the fall detection loop."""
        try:
            while self.cap.isOpened():
                success, image = self.cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue  

                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                results = self.process_frame(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Draw holistic landmarks
                result_image = self.draw_holistic_results(image, results)

                if results.pose_landmarks:
                    self.compute_features(results)

                fall_detected = self.detect_fall(result_image)
                cv2.imshow('Fall_detection', result_image)

                if fall_detected:
                    cv2.imwrite(filename="fall.png", img=image)

                    # Close the window before playing the sound
                    self.cap.release()
                    cv2.destroyAllWindows()
                    time.sleep(0.5)

                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break

        finally:
            self.cap.release()
            cv2.destroyAllWindows()

    def end(self):
        self.play_sound()
        image_path = "fall.png"
        if os.path.exists(image_path):
            image = cv2.imread(image_path)  # Read the image
            if image is not None:
                cv2.imshow("Image", image)  # Show the image
                cv2.waitKey(0)  # Wait for a key press

                if cv2.waitKey(0)  == ord('q'):
                    cv2.destroyAllWindows()
                    # os.remove(image_path)

fall_detector = FallDetector()
fall_detector.start()
fall_detector.end()

