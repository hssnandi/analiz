import cv2
import numpy as np

def draw_overlay(image):
    h, w, _ = image.shape
    overlay = image.copy()
    center = (w // 2, h // 2)
    radius = min(w, h) // 4
    cv2.circle(overlay, center, radius, (128, 128, 128), 2)
    return overlay

def analyze_image_quality(results, image):
    if not results.multi_face_landmarks:
        return False, "چهره‌ای شناسایی نشد. نور کافی یا موقعیت صورت بررسی شود."
    h, w, _ = image.shape
    landmarks = results.multi_face_landmarks[0].landmark
    nose = landmarks[1]
    cx, cy = int(nose.x * w), int(nose.y * h)
    center_x, center_y = w // 2, h // 2
    distance = np.linalg.norm([cx - center_x, cy - center_y])
    if distance > w // 6:
        return False, "صورت باید در مرکز کادر قرار گیرد."
    return True, ""
