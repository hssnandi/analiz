import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from utils import analyze_image_quality, draw_overlay
from face_shape_detector import detect_face_shape
from PIL import Image
import os

st.set_page_config(page_title="تشخیص فرم صورت", layout="centered")
st.title("📸 تشخیص فرم صورت و پیشنهاد مدل مو")

FRAME_WINDOW = st.empty()
camera = cv2.VideoCapture(0)

# بارگذاری مدل MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

st.markdown("""
<div style='text-align:center;'>
    صورت خود را در مرکز دایره قرار دهید. نور کافی داشته باشید.
</div>
""", unsafe_allow_html=True)

captured_image = None
capture_triggered = False
start_button = st.button("📷 گرفتن عکس")

while camera.isOpened() and not capture_triggered:
    ret, frame = camera.read()
    if not ret:
        st.error("دوربین پیدا نشد.")
        break

    image = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    overlay_img = draw_overlay(img_rgb.copy())
    quality_ok, reason = analyze_image_quality(results, img_rgb)

    if quality_ok:
        cv2.circle(overlay_img, (320, 240), 140, (0, 255, 0), 3)
    else:
        cv2.circle(overlay_img, (320, 240), 140, (128, 128, 128), 3)
        st.warning(reason)

    FRAME_WINDOW.image(overlay_img)

    if start_button and quality_ok:
        captured_image = img_rgb
        capture_triggered = True

camera.release()

if captured_image is not None:
    st.subheader("📊 آنالیز فرم صورت")
    shape = detect_face_shape(captured_image)
    st.success(f"فرم صورت شما: {shape}")

    # نمایش مدل‌های موی پیشنهادی
    st.subheader("💇 مدل‌های موی پیشنهادی")
    model_path = os.path.join("model_images", shape.lower())
    if os.path.exists(model_path):
        imgs = os.listdir(model_path)
        for img in imgs:
            st.image(os.path.join(model_path, img), use_column_width=True)
    else:
        st.info("تصاویری برای این فرم صورت هنوز بارگذاری نشده‌اند.")
