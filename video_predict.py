import cv2
import numpy as np
import os
import argparse
import face_recognition
import csv
from datetime import datetime
from collections import deque

from src.face_detector import YOLOv5
from src.FaceAntiSpoofing import AntiSpoof
from register_voice import register_user as register_voice_for_user
from recognize_voice import recognize_user as recognize_voice_for_user

COLOR_REAL = (0, 255, 0)
COLOR_FAKE = (0, 0, 255)
COLOR_UNKNOWN = (127, 127, 127)
REGISTER_FOLDER = "registered_faces"
os.makedirs(REGISTER_FOLDER, exist_ok=True)

SCORE_HISTORY_LENGTH = 5
VOICE_THRESHOLD = 0.55

def increased_crop(img, bbox, bbox_inc=1.5):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    w_bbox = int(x2 - x1)
    h_bbox = int(y2 - y1)
    size = int(max(w_bbox, h_bbox) * bbox_inc)
    xc = int((x1 + x2) // 2)
    yc = int((y1 + y2) // 2)
    new_x1 = max(xc - size // 2, 0)
    new_y1 = max(yc - size // 2, 0)
    new_x2 = min(xc + size // 2, w)
    new_y2 = min(yc + size // 2, h)
    crop = img[new_y1:new_y2, new_x1:new_x2]
    return crop if crop.shape[0] > 0 and crop.shape[1] > 0 else None

def make_prediction(img, face_detector, anti_spoof):
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = face_detector([rgb_img])
    if len(faces) == 0 or faces[0].shape[0] == 0:
        return None
    bbox = faces[0][0][:4].astype(int)
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None
    crop = increased_crop(rgb_img, (x1, y1, x2, y2))
    if crop is None:
        return None
    pred = anti_spoof([crop])[0]
    score = float(pred[0][0])
    label = int(np.argmax(pred))  # 0 = real, 1 = fake
    print(f"[DEBUG] Prediction: Score={score:.4f}, Label={'REAL' if label == 0 else 'FAKE'}")
    return (x1, y1, x2, y2), label, score, crop

def encode_known_faces():
    encodings, names = [], []
    for person in os.listdir(REGISTER_FOLDER):
        person_dir = os.path.join(REGISTER_FOLDER, person)
        if os.path.isdir(person_dir):
            for file in os.listdir(person_dir):
                if file.endswith(".jpg"):
                    path = os.path.join(person_dir, file)
                    image = face_recognition.load_image_file(path)
                    face_locations = face_recognition.face_locations(image)
                    if face_locations:
                        enc = face_recognition.face_encodings(image, known_face_locations=face_locations)
                        if enc:
                            encodings.append(enc[0])
                            names.append(person)
    return encodings, names

def register_face(face_detector, anti_spoof, threshold):
    name = input("📝 Enter your name for registration: ").strip()
    person_dir = os.path.join(REGISTER_FOLDER, name)
    os.makedirs(person_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    saved = 0
    score_buffer = deque(maxlen=SCORE_HISTORY_LENGTH)

    print("📸 Press 's' to save face, 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = make_prediction(frame, face_detector, anti_spoof)
        if result:
            (x1, y1, x2, y2), label, score, _ = result
            score_buffer.append(score)

            avg_score = np.mean(score_buffer)
            is_real = (label == 0 and avg_score >= threshold + 0.1)
            color = COLOR_REAL if is_real else COLOR_FAKE
            status = f"{'REAL' if is_real else 'FAKE'} ({avg_score:.2f})"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, status, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Register Face", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("👋 Registration ended.")
            break
        elif key == ord('s') and result and is_real:
            aligned = frame[y1:y2, x1:x2]
            filepath = os.path.join(person_dir, f"{name}_{saved}.jpg")
            cv2.imwrite(filepath, aligned)
            print(f"✅ Saved: {filepath}")
            saved += 1
        elif key == ord('s'):
            print("❌ Cannot save: Not a real face or low confidence.")

    cap.release()
    cv2.destroyAllWindows()

    print("🎙️ Now recording voice...")
    register_voice_for_user(name)

def verify_and_take_attendance(face_detector, anti_spoof, threshold):
    known_encodings, known_names = encode_known_faces()
    session_attendance = []

    cap = cv2.VideoCapture(0)
    print("🎥 Starting verification + attendance. Look at the camera...")

    score_buffer = deque(maxlen=SCORE_HISTORY_LENGTH)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = make_prediction(frame, face_detector, anti_spoof)
        if result:
            (x1, y1, x2, y2), label, score, _ = result
            score_buffer.append(score)
            avg_score = np.mean(score_buffer)

            if label == 0 and avg_score >= threshold + 0.1:
                rgb_face = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_face)
                encs = face_recognition.face_encodings(rgb_face, known_face_locations=face_locations)

                if encs:
                    matches = face_recognition.compare_faces(known_encodings, encs[0], tolerance=0.45)
                    if True in matches:
                        matched_idxs = [i for i, match in enumerate(matches) if match]
                        name = known_names[matched_idxs[0]]

                        print(f"👤 Face match: {name}")
                        similarity = recognize_voice_for_user(name)

                        if similarity >= VOICE_THRESHOLD:
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            session_attendance.append({"Name": name, "Timestamp": timestamp})
                            print(f"✅ {name} verified! Attendance marked.")

                            # Save and exit
                            cap.release()
                            cv2.destroyAllWindows()

                            if session_attendance:
                                csv_file = "attendance_log.csv"
                                file_exists = os.path.isfile(csv_file)
                                with open(csv_file, "a", newline="") as f:
                                    writer = csv.DictWriter(f, fieldnames=["Name", "Timestamp"])
                                    if not file_exists:
                                        writer.writeheader()
                                    writer.writerows(session_attendance)
                                print(f"📁 Attendance saved to {csv_file}")
                            return  # ✅ exit to CLI

                        else:
                            print(f"❌ Voice mismatch for {name} (score={similarity:.2f})")
                            display_name = f"{name} ❌ Voice"
                            color = COLOR_FAKE
                    else:
                        display_name = "Unknown 😶"
                        color = COLOR_UNKNOWN
                else:
                    display_name = "Face Encode Error"
                    color = COLOR_UNKNOWN
            else:
                display_name = f"FAKE ❌ ({avg_score:.2f})"
                color = COLOR_FAKE

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, display_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            cv2.putText(frame, "No face detected", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_UNKNOWN, 2)

        cv2.imshow("Verify + Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("👋 Session ended without successful verification.")

def main():
    parser = argparse.ArgumentParser(description="Face + Anti-Spoofing + Voice Attendance System")
    parser.add_argument("--model_path", "-m", type=str, default="saved_models/AntiSpoofing_bin_1.5_128.onnx", help="Anti-spoofing ONNX model path")
    parser.add_argument("--threshold", "-t", type=float, default=0.65, help="Anti-spoofing threshold")
    args = parser.parse_args()

    face_detector = YOLOv5("saved_models/yolov5s-face.onnx")
    anti_spoof = AntiSpoof(args.model_path)

    while True:
        print("\n=== Menu ===")
        print("1️⃣  Register Face + Voice")
        print("2️⃣  Verify & Take Attendance")
        print("3️⃣  Exit")
        choice = input("Select an option: ")

        if choice == "1":
            register_face(face_detector, anti_spoof, args.threshold)
        elif choice == "2":
            verify_and_take_attendance(face_detector, anti_spoof, args.threshold)
        elif choice == "3":
            print("👋 Exiting.")
            break
        else:
            print("❌ Invalid option.")

if __name__ == "__main__":
    main()
