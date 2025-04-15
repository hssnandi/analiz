import cv2
import numpy as np
import mediapipe as mp

def detect_face_shape(image):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    if not results.multi_face_landmarks:
        return "ناشناس"

    landmarks = results.multi_face_landmarks[0].landmark
    jaw = [landmarks[i] for i in range(0, 17)]
    jaw_width = np.linalg.norm(
        np.array([jaw[0].x, jaw[0].y]) - np.array([jaw[-1].x, jaw[-1].y])
    )
    forehead_chin = np.linalg.norm(
        np.array([landmarks[10].x, landmarks[10].y]) -
        np.array([landmarks[152].x, landmarks[152].y])
    )

    ratio = forehead_chin / jaw_width

    if ratio > 1.5:
        return "oblong"
    elif ratio < 1.1:
        return "round"
    elif 1.1 <= ratio <= 1.3:
        return "square"
    elif landmarks[162].x < landmarks[127].x:
        return "heart"
    else:
        return "oval"
