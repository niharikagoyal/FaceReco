
import os
import cv2
import numpy as np
import face_recognition
import requests
import base64

from io import BytesIO
from PIL import Image
from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
from numpy import argmin



app = FastAPI(title="Face Recognition Attendance API")

encodeListKnown = []
employee_encodings = []
employee_ids = []
employee_names = []
employee_keys = []


EXTERNAL_API_URL = "https://project.pisofterp.com/pipl/restworld/employees"

def load_employee_data_from_api():
    global employee_encodings, employee_ids
    try:
        resp = requests.get(EXTERNAL_API_URL)
        resp.raise_for_status()
    except Exception:
        return False

    data = resp.json()
    employee_encodings.clear()
    employee_ids.clear()
    for entry in data:
        emp_id = entry.get("id")
        pic_b64 = entry.get("employeePic")
        emp_name = entry.get("employeeName")
        emp_key = entry.get("key")  
        if not pic_b64 or emp_id is None:
            continue

        try:
            img_bytes = base64.b64decode(pic_b64)
            img = Image.open(BytesIO(img_bytes)).convert("RGB")
            img_np = np.array(img)
            enc = face_recognition.face_encodings(img_np)
            if enc:
                employee_encodings.append(enc[0])
                employee_ids.append(emp_id)
                employee_names.append(emp_name)
                employee_keys.append(emp_key)
        except Exception as e:
            print(f"Error loading image for ID {emp_id}: {e}")

    return bool(employee_encodings)

@app.on_event("startup")
def startup_event():
    load_employee_data_from_api()

@app.get("/")
def root():
    return FileResponse("static/index.html")

@app.post("/api/recognize")
async def recognize_face():
    
    cap = cv2.VideoCapture(0)
    success, img = cap.read()
    cap.release()

    if not success:
        return JSONResponse({"match": False, "employeeId": "001"})

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_encodings = face_recognition.face_encodings(img_rgb)

    # for face_encoding in face_encodings:
    #     matches = face_recognition.compare_faces(employee_encodings, face_encoding)
    #     if True in matches:
    #         match_index = matches.index(True)
    #         matched_id = employee_ids[match_index]
    #         matched_name = employee_names[match_index]
    #         # matched_key = employee_keys[match_index]
    #         return JSONResponse({"match": True, "employeeId": matched_id,"employeeName": matched_name})
    #         # return JSONResponse({"match": True, "employeeId": matched_id,"employeeName": matched_name,"employeeKey": matched_key})
    # return JSONResponse({"match": False, "employeeId":"001"})
    for face_encoding in face_encodings:
        face_distances = face_recognition.face_distance(employee_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if face_distances[best_match_index] < 0.5:  # Threshold can be adjusted
            matched_id = employee_ids[best_match_index]
            matched_name = employee_names[best_match_index]
            matched_key = employee_keys[best_match_index]
            return JSONResponse({"match": True, "employeeId": matched_id, "employeeName": matched_name,"employeeKey": matched_key})

    return JSONResponse({"match": False, "employeeId": "001"})

