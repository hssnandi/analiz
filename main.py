import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import os
import math
from face_shape_detector import detect_face_shape
from utils import analyze_image_quality, draw_overlay

st.set_page_config(page_title="تشخیص فرم صورت", layout="centered")
st.title("📸 تشخیص فرم صورت و پیشنهاد مدل مو")

FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

# بارگذاری مدل MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# رابط کاربری
st.markdown("""
<div style='text-align:center;'>
    لطفاً صورت خود را در مرکز دایره قرار دهید. نور کافی داشته باشید. از عینک یا ماسک استفاده نکنید.
</div>
""", unsafe_allow_html=True)

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

    if quality_ok:
        cv2.circle(overlay_img, (320, 240), 140, (0, 255, 0), 3)  # دایره سبز
    else:
        cv2.circle(overlay_img, (320, 240), 140, (128, 128, 128), 3)  # دایره خاکستری
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

    import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import cv2
import numpy as np

def capture_image():
    # دیکشنری برای ذخیره تصویر گرفته شده
    image = None
    
    def video_frame_callback(frame):
        nonlocal image
        # اینجا می‌تونی تصویر دوربین رو دریافت و پردازش کنی
        image = frame.to_ndarray(format="bgr24")
        return frame

    # استریم دوربین
    webrtc_ctx = webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True},
    )

    if webrtc_ctx.state.playing:
        # نمایش کادر هنگام گرفتن عکس
        if image is not None:
            st.image(image, caption="Current Frame", use_column_width=True)
        
        # گرفتن عکس
        if st.button("Capture Image"):
            if image is not None:
                # ذخیره تصویر گرفته شده
                img_path = "captured_image.png"
                cv2.imwrite(img_path, image)
                st.image(img_path, caption="Captured Image", use_column_width=True)

# نمایش کادر دوربین و امکان گرفتن عکس
capture_image()

    # نمایش مدل‌های پیشنهادی
    st.subheader("💇 مدل‌های موی پیشنهادی")
    model_path = os.path.join("model_images", shape.lower())
    if os.path.exists(model_path):
        imgs = os.listdir(model_path)
        for img in imgs:
            st.image(os.path.join(model_path, img), use_column_width=True)
    else:
        st.info("تصاویری برای این فرم صورت هنوز بارگذاری نشده‌اند.")
