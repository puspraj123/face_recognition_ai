import streamlit as st
import cv2
import face_recognition
import numpy as np
import pandas as pd
from datetime import datetime
import os

st.title("🧾 Face Attendance System")

run = st.checkbox("Start Camera")

# Load known face (example - add your image in folder)
known_image = face_recognition.load_image_file("known.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

known_names = ["Puspa Raj"]

def mark_attendance(name):
    file = "attendance.csv"

    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d %H:%M:%S")

    if not os.path.exists(file):
        df = pd.DataFrame(columns=["Name", "DateTime"])
        df.to_csv(file, index=False)

    df = pd.read_csv(file)

    if name not in df["Name"].values:
        new_row = {"Name": name, "DateTime": dt_string}
        df = df.append(new_row, ignore_index=True)
        df.to_csv(file, index=False)

FRAME_WINDOW = st.image([])
cap = cv2.VideoCapture(0)

if run:
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera error")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, faces)

        for encoding, face_loc in zip(encodings, faces):
            matches = face_recognition.compare_faces([known_encoding], encoding)
            name = "Unknown"

            if True in matches:
                name = known_names[0]
                mark_attendance(name)

            y1, x2, y2, x1 = face_loc
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, name, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()