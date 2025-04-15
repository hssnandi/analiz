import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import cv2
import numpy as np
import mediapipe as mp
import os
from PIL import Image
from face_shape_detector import detect_face_shape
from utils import analyze_image_quality, draw_overlay

st.set_page_config(page_title="تشخیص فرم صورت", layout="centered")
st.title("📸 تشخیص فرم صورت و پیشنهاد مدل مو")

st.markdown("""
<div style='text-align:center;'>
    لطفاً صورت خود را در مرکز دایره قرار دهید. نور کافی داشته باشید. از عینک یا ماسک استفاده نکنید.
</div>
""", unsafe_allow_html=True)

captured_image = None
status_placeholder = st.empty()
captured_img_placeholder = st.empty()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

image_holder = {"image": None}

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    overlay = draw_overlay(img_rgb.copy())
    quality_ok, reason = analyze_image_quality(results, img_rgb)

    center_color = (0, 255, 0) if quality_ok else (128, 128, 128)
    cv2.circle(overlay, (320, 240), 140, center_color, 3)

    if not quality_ok:
        status_placeholder.warning(reason)
    else:
        status_placeholder.info("✅ شرایط مناسب است، برای گرفتن عکس روی دکمه کلیک کنید.")

    image_holder["image"] = img_rgb if quality_ok else None
    return av.VideoFrame.from_ndarray(overlay, format="rgb24")

webrtc_ctx = webrtc_streamer(
    key="face-analyzer",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if st.button("📷 گرفتن عکس"):
    if image_holder["image"] is not None:
        captured_image = image_holder["image"]
        captured_img_placeholder.image(captured_image, caption="تصویر ثبت شده", use_column_width=True)
    else:
        st.error("⛔ شرایط عکس مناسب نیست. لطفاً نور، موقعیت صورت و مرکز تصویر را بررسی کنید.")

if captured_image is not None:
    st.subheader("📊 آنالیز فرم صورت")
    shape = detect_face_shape(captured_image)
    st.success(f"فرم صورت شما: {shape}")

    st.subheader("💇 مدل‌های موی پیشنهادی")
    model_path = os.path.join("model_images", shape.lower())
    if os.path.exists(model_path):
        imgs = os.listdir(model_path)
        for img in imgs:
            st.image(os.path.join(model_path, img), use_column_width=True)
    else:
        st.info("تصاویری برای این فرم صورت هنوز بارگذاری نشده‌اند.")
