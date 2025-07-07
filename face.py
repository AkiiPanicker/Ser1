import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Initialize OpenCV Haar Cascade for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
if eye_cascade.empty():
    print("Error: Could not load eye cascade classifier.")
    exit()

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

def estimate_gaze_direction(eyes, face_bbox, frame_shape):
    """Estimate gaze direction based on eye centroids relative to face bounding box."""
    if not eyes:
        return "No Eyes Detected"
    
    # Calculate face center
    face_x, face_y, face_w, face_h = face_bbox
    face_center_x = face_x + face_w / 2
    
    # Calculate average eye centroid
    eye_centroids = [(x + w / 2, y + h / 2) for (x, y, w, h) in eyes]
    avg_eye_x = sum([centroid[0] for centroid in eye_centroids]) / len(eye_centroids) if eye_centroids else face_center_x
    
    # Normalize eye centroid position relative to face width
    rel_x = (avg_eye_x - face_x) / face_w if face_w > 0 else 0.5
    
    # Determine gaze direction
    if rel_x < 0.35:
        return "Looking Left"
    elif rel_x > 0.65:
        return "Looking Right"
    return "Looking Center"

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert frame to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        face_results = face_detection.process(frame_rgb)
        frame_rgb.flags.writeable = True
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        face_detected = False
        face_bbox = None
        eyes = []

        # Draw face bounding box and detect eyes
        if face_results.detections:
            face_detected = True
            for detection in face_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = (int(bboxC.xmin * iw), int(bboxC.ymin * ih),
                              int(bboxC.width * iw), int(bboxC.height * ih))
                face_bbox = (x, y, w, h)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Extract face region for eye detection
                face_roi = frame[y:y+h, x:x+w]
                face_roi_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                # Detect eyes in face region
                detected_eyes = eye_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                # Adjust eye coordinates to full frame
                eyes = [(ex + x, ey + y, ew, eh) for (ex, ey, ew, eh) in detected_eyes]
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # Estimate gaze direction
        gaze_status = estimate_gaze_direction(eyes, face_bbox, frame.shape) if face_detected else "No Gaze Detected"

        # Display status
        status_text = "Face Detected" if face_detected else "No Face Detected"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if face_detected else (0, 0, 255), 2)
        cv2.putText(frame, gaze_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if eyes else (0, 0, 255), 2)

        # Show frame
        cv2.imshow("Face and Gaze Detection", frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user.")
finally:
    cap.release()
    cv2.destroyAllWindows()
    face_detection.close()
