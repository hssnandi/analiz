import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import os
from face_shape_detector import detect_face_shape
from utils import analyze_image_quality, draw_overlay
from PIL import Image

st.set_page_config(page_title="تشخیص فرم صورت", layout="centered")
st.title("📸 تشخیص فرم صورت و پیشنهاد مدل مو")

st.markdown("""
<div style='text-align:center;'>
    صورت خود را درون دایره نگه دارید تا دایره سبز شود، سپس عکس بگیرید. نور کافی و چهره بدون عینک الزامی است.
</div>
""", unsafe_allow_html=True)

FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

captured_image = None
start_button = st.button("📷 گرفتن عکس")

while camera.isOpened():
    ret, frame = camera.read()
    if not ret:
        st.error("دوربین پیدا نشد.")
        break

    image = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    overlay_img = draw_overlay(img_rgb.copy())
    quality_ok, reason = analyze_image_quality(results, img_rgb)

    h, w, _ = overlay_img.shape
    center = (w // 2, h // 2)
    radius = min(w, h) // 4

    color = (0, 255, 0) if quality_ok else (128, 128, 128)
    cv2.circle(overlay_img, center, radius, color, 3)

    if not quality_ok:
        st.warning(reason)

    FRAME_WINDOW.image(overlay_img)

    if start_button and quality_ok:
        captured_image = img_rgb
        break

camera.release()

if captured_image is not None:
    st.subheader("📊 آنالیز فرم صورت")
    shape = detect_face_shape(captured_image)
    st.success(f"فرم صورت شما: {shape}")

    st.subheader("💇 مدل‌های موی پیشنهادی")
    model_path = os.path.join("model_images", shape.lower())
    if os.path.exists(model_path):
        for img in os.listdir(model_path):
            st.image(os.path.join(model_path, img), use_column_width=True)
    else:
        st.info("تصاویری برای این فرم صورت هنوز بارگذاری نشده‌اند.")
