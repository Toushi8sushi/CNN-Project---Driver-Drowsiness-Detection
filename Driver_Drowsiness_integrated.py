import cv2
import time
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import pygame

# Initialize pygame mixer
pygame.mixer.init()
alert_sound = pygame.mixer.Sound('T:\TOSHITH\PROGRAMMING\CNN-Project---Driver-Drowsiness-Detection\support_files\Alert.mp3')  # Make sure you have alert.wav in your directory

# Load Haar cascades for face and eyes
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
left_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
right_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')

# Load your trained CNN model
model = load_model("T:\TOSHITH\PROGRAMMING\CNN-Project---Driver-Drowsiness-Detection\driver_drowsiness.keras")

'''#or 
model = load_model("T:\TOSHITH\PROGRAMMING\CNN-Project---Driver-Drowsiness-Detection\driver_drowsiness_MobileNetV2.keras")
'''
rgb_flag=False#-----------------------------keep as True if using the MobileNetV2 model

# Timer tracking
eye_closed_start_time = None
alert_threshold = 3  # seconds
alert_active = False  # To track if we're in alert state


def preprocess_eye(eye_image):
    if rgb_flag:
        # Convert BGR (OpenCV) to RGB for MobileNetV2
        eye = cv2.cvtColor(eye_image, cv2.COLOR_BGR2RGB)
        eye = Image.fromarray(eye).resize((200, 200))
    else:
        eye = Image.fromarray(eye_image).convert("L").resize((200, 200))
    
    eye_array = img_to_array(eye) / 255.0
    return np.expand_dims(eye_array, axis=0)

def is_eye_closed(eye_image):
    processed = preprocess_eye(eye_image)
    prediction = model.predict(processed)
    return prediction[0][0] < 0.5  # Return True if classified as "Closed"

# Start webcam
cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        eyes_closed = {'left': False, 'right': False}

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            left_eyes = left_eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
            right_eyes = right_eye_cascade.detectMultiScale(roi_gray, 1.1, 5)

            # Left eye
            for (ex, ey, ew, eh) in left_eyes[:1]:  # use first detection
                eye_img = roi_color[ey:ey+eh, ex:ex+ew]
                if is_eye_closed(eye_img):
                    eyes_closed['left'] = True
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

            # Right eye
            for (ex, ey, ew, eh) in right_eyes[:1]:
                eye_img = roi_color[ey:ey+eh, ex:ex+ew]
                if is_eye_closed(eye_img):
                    eyes_closed['right'] = True
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)

            break  # process only one face

        # Both eyes closed logic
        if eyes_closed['left'] and eyes_closed['right']:
            if eye_closed_start_time is None:
                eye_closed_start_time = time.time()
            elif time.time() - eye_closed_start_time >= alert_threshold:
                cv2.putText(frame, 'ALERT: Eyes closed!', (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                if not alert_active:
                    alert_active = True
                    alert_sound.play(loops=-1)  # -1 means loop indefinitely
        else:
            eye_closed_start_time = None
            if alert_active:
                alert_active = False
                alert_sound.stop()

        cv2.imshow('Eye State Detection', frame)

        # Break on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Cleanup
    alert_sound.stop()
    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()


