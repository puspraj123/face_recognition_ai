import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os

st.title("🧾 Face Attendance System")
run = st.checkbox("Start Camera", key="camera_toggle")

# Load OpenCV face detector (instead of face_recognition)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# 👇 PUT THIS HERE (right after checkbox)
uploaded_file = st.file_uploader("Upload known face image")

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    known_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    known_gray = cv2.cvtColor(known_img, cv2.COLOR_BGR2GRAY)
else:
    st.warning("Please upload a reference image")
    st.stop()

known_name = "Puspa Raj"


def mark_attendance(name):
    file = "attendance.csv"

    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d %H:%M:%S")

    if not os.path.exists(file):
        df = pd.DataFrame(columns=["Name", "DateTime"])
        df.to_csv(file, index=False)

    df = pd.read_csv(file)

    if name not in df["Name"].values:
        new_row = pd.DataFrame([[name, dt_string]], columns=["Name", "DateTime"])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(file, index=False)

FRAME_WINDOW = st.image([])
cap = cv2.VideoCapture(0)

if run:
    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera error")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            name = known_name

            mark_attendance(name)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    