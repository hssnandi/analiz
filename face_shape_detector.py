import mediapipe as mp
import numpy as np
import math

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def detect_face_shape(image):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
    results = face_mesh.process(image)
    if not results.multi_face_landmarks:
        return "تشخیص داده نشد"

    face = results.multi_face_landmarks[0]
    h, w, _ = image.shape
    points = lambda idx: (int(face.landmark[idx].x * w), int(face.landmark[idx].y * h))

    jaw = euclidean(points(234), points(454))
    cheek = euclidean(points(93), points(323))
    forehead = euclidean(points(10), points(152))

    ratios = {
        "بیضی": abs(jaw - cheek) < 20 and forehead > cheek,
        "گرد": abs(jaw - cheek) < 20 and abs(forehead - cheek) < 20,
        "مربع": abs(jaw - cheek) < 20 and abs(forehead - cheek) < 20 and jaw > 180,
        "قلبی": forehead > cheek and jaw < cheek,
        "مثلثی": jaw > cheek and cheek > forehead
    }

    for k, v in ratios.items():
        if v:
            return k
    return "نامشخص"
