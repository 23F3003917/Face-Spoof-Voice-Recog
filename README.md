# 🎭 Face Anti-Spoofing + Face Recognition Attendance System
# 🛡️ Real-Time Face Anti-Spoofing + Voice-Verified Attendance System

This is a **real-time facial attendance system** that combines:

- ✅ **YOLOv5-based face detection**
- 🛡️ **ONNX-based face anti-spoofing**
- 🧠 **Face recognition** with `face_recognition`
- 🎤 **Voice recognition** using speaker embeddings
- 📝 **Secure attendance logging**

It ensures that only **real, live, and verified** users are marked **present**, preventing spoofing via printed photos, videos, or impersonation.

---

## 🧾 Project Overview

Traditional face recognition systems are vulnerable to **spoofing attacks** (e.g., showing a photo or video to the webcam). This system prevents that by:

1. Detecting the **liveness** of the face using an ONNX-based anti-spoofing model.
2. Matching the face with registered known encodings.
3. **Verifying the user's voice** to ensure the correct person is physically present.

---

## 🔒 How It Works

1. 🧠 **Face Detection:** Webcam captures frames in real time, and YOLOv5 detects faces.
2. 🛡️ **Anti-Spoofing:** Each face is verified using an ONNX anti-spoof model.
3. 🧍 **Face Recognition:** If real, the system compares the face with registered encodings.
4. 🎤 **Voice Verification:** If matched, the system prompts the user to speak and compares it to the previously registered voice embedding.
5. 📝 **Attendance Logging:** If face and voice both match, the system logs attendance to `attendance_log.csv`.

---

## 🖥️ How to Run

### ✅ Step 1: Install Requirements

```bash
pip install -r requirements.txt

### ✅ Step 2: Run the System

Launch the main script from your terminal:
python video_predict.py
You’ll see a terminal menu like this:


=== Real-Time Face + Voice Attendance System ===
1️⃣  Register Face + Voice
2️⃣  Verify & Take Attendance
3️⃣  Exit

Select an option:
Use your keyboard to select options:

### 🧑‍💻 Step 3: Register Face + Voice (Option 1)

After selecting Option 1, the following actions occur:

🎥 Face Registration (with Anti-Spoofing Check)
The webcam will open automatically.
The system will detect your face and run the anti-spoofing check.
Only REAL faces are allowed to be saved.

✅ How to Save Face:
Press s → Saves the currently detected and verified real face.
Press q → Quits face registration.
The saved face images are stored in:


registered_faces/<your_name>/
You can save multiple real face images for better recognition accuracy. The system will use them to build your facial encoding.

🎙️ Voice Registration
After face registration, you’ll be prompted to record your voice:

🎙️ Please speak clearly for 10 seconds...
Your voice is recorded automatically using your system’s default microphone.

It is saved as a .wav file


### 🛡️ Step 4: Verify & Mark Attendance (Option 2)
After selecting Option 2 from the menu:

📸 Face Verification
The webcam starts and detects your face in real-time.
The anti-spoofing model verifies if the face is real.
If real, your face is matched with the registered faces.

🔊 Voice Verification
If your face is matched, you'll be prompted to speak again:
🎙️ Please speak now...


The system compares your live voice to your stored voice embedding.
If the voice is matched (above threshold), attendance is marked.


✅ Face & Voice verified for Anirudh
📋 Attendance marked at 2025-07-25 12:34:56
The entry is saved to attendance_log.csv like:


Name,Timestamp
Anirudh,2025-07-25 12:34:56

✅ After successful attendance, the program automatically returns to the main menu.


