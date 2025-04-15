import cv2
import numpy as np
import mediapipe as mp
import math

def draw_overlay(image):
    overlay = image.copy()
    cv2.circle(overlay, (320, 240), 140, (200, 200, 200), 2)
    return overlay

def analyze_image_quality(results, image):
    if not results.multi_face_landmarks:
        return False, "صورت شناسایی نشد."

    h, w, _ = image.shape
    face = results.multi_face_landmarks[0]
    x = int(face.landmark[1].x * w)
    y = int(face.landmark[1].y * h)

    center_distance = math.sqrt((x - 320)**2 + (y - 240)**2)
    if center_distance > 80:
        return False, "صورت باید در مرکز کادر قرار گیرد."

    brightness = np.mean(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
    if brightness < 80:
        return False, "نور محیط کافی نیست."

    return True, "مناسب برای تشخیص"
