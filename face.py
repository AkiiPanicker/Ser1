import cv2
import mediapipe as mp
import numpy as np
import time
from datetime import datetime
import json
from flask_socketio import emit

# This class is adapted from your 'improve 1.py' file.
# All cv2.imshow, drawing, and UI-related code has been removed.
# It's now designed to process one frame at a time and emit warnings.
class ProctoringService:
    def __init__(self, user_sid):
        self.user_sid = user_sid  # Unique ID for the user's connection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.warnings = {
            'multiple_faces': 0,
            'looking_away': 0,
            'no_face': 0,
            'total': 0
        }
        self.max_warnings = 5  # Configurable number of allowed warnings
        self.session_active = True
        
        self.grace_period_seconds = 5.0  # 5-second grace period for all violations
        self.grace_periods = {
            'looking_away_start': None,
            'no_face_start': None,
            'multiple_faces_start': None
        }

        # Simplified logging
        self.log_file = f"proctoring_log_{self.user_sid}.json"
        self.violations_log = []
        print(f"[*] Proctoring service initialized for user: {self.user_sid}")

    def _add_warning(self, warning_type, message):
        self.warnings[warning_type] += 1
        self.warnings['total'] += 1
        print(f"[!] WARNING for {self.user_sid}: {message} (Total: {self.warnings['total']})")
        
        # Emit a real-time event back to the specific user's browser
        emit('proctoring_violation', {'type': warning_type, 'message': message}, room=self.user_sid)

        # Log the violation
        violation = { 'timestamp': datetime.now().isoformat(), 'type': warning_type, 'message': message }
        self.violations_log.append(violation)

        if self.warnings['total'] >= self.max_warnings:
            self.session_active = False
            emit('proctoring_terminated', {'reason': 'Maximum warnings exceeded.'}, room=self.user_sid)
            print(f"[!!!] Proctoring session TERMINATED for {self.user_sid}.")
            self.save_log()

    def process_frame(self, frame):
        if not self.session_active:
            return

        current_time = time.time()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        # Violation 1: No Face Detected
        if not results.multi_face_landmarks:
            if self.grace_periods['no_face_start'] is None:
                self.grace_periods['no_face_start'] = current_time
            elif current_time - self.grace_periods['no_face_start'] > self.grace_period_seconds:
                self._add_warning('no_face', "No face was detected in the camera view.")
                self.grace_periods['no_face_start'] = None  # Reset after warning
            return # Stop further processing if no face
        else:
            self.grace_periods['no_face_start'] = None

        # Violation 2: Multiple Faces Detected
        num_faces = len(results.multi_face_landmarks)
        if num_faces > 1:
            if self.grace_periods['multiple_faces_start'] is None:
                self.grace_periods['multiple_faces_start'] = current_time
            elif current_time - self.grace_periods['multiple_faces_start'] > self.grace_period_seconds:
                self._add_warning('multiple_faces', f"{num_faces} faces were detected in the view.")
                self.grace_periods['multiple_faces_start'] = None
        else:
            self.grace_periods['multiple_faces_start'] = None

        # Violation 3: Gaze Detection (Looking Away)
        face_landmarks = results.multi_face_landmarks[0].landmark
        
        # Simple but effective head pose estimation
        nose_tip = face_landmarks[1]
        left_eye_inner = face_landmarks[145]
        right_eye_inner = face_landmarks[374]
        gaze_ratio_x = ((nose_tip.x - left_eye_inner.x) + (right_eye_inner.x - nose_tip.x)) / (right_eye_inner.x - left_eye_inner.x)
        
        # A gaze ratio far from 0.5 indicates the head is turned
        if abs(gaze_ratio_x - 0.5) > 0.1: # Adjust this threshold if needed
             if self.grace_periods['looking_away_start'] is None:
                self.grace_periods['looking_away_start'] = current_time
             elif current_time - self.grace_periods['looking_away_start'] > self.grace_period_seconds:
                direction = "right" if gaze_ratio_x < 0.5 else "left"
                self._add_warning('looking_away', f"Candidate appears to be looking to the {direction}.")
                self.grace_periods['looking_away_start'] = None
        else:
            self.grace_periods['looking_away_start'] = None
    
    def save_log(self):
        """Saves the final log of violations."""
        with open(self.log_file, 'w') as f:
            log_data = {
                'user_sid': self.user_sid,
                'session_terminated': not self.session_active,
                'log_end_time': datetime.now().isoformat(),
                'final_warnings': self.warnings,
                'violations_log': self.violations_log
            }
            json.dump(log_data, f, indent=4)
        print(f"[*] Proctoring log saved for {self.user_sid} to {self.log_file}")
