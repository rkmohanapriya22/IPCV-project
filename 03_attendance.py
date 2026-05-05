# 03_attendance.py
import cv2
import numpy as np
import pickle
import csv
import os
import tempfile
from datetime import datetime
from deepface import DeepFace

# ── Configuration ─────────────────────────────────────────────────────
MODEL_PATH    = "knn_model.pkl"
FACENET_MODEL = "Facenet"
THRESHOLD     = 0.55    
                     
PROCESS_EVERY = 5        
CSV_FILE      = "attendance.csv"
# ─────────────────────────────────────────────────────────────────────

# Load the trained kNN model
with open(MODEL_PATH, "rb") as f:
    data = pickle.load(f)
knn = data["knn"]
print(f"[INFO] Loaded model. Students: {data['students']}")

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


today         = datetime.now().strftime("%Y-%m-%d")
marked_today  = set()

if os.path.exists(CSV_FILE):
    with open(CSV_FILE, newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[1] == today:
                marked_today.add(row[0])

def mark_attendance(name):
    if name not in marked_today:
        with open(CSV_FILE, "a", newline="") as f:
            csv.writer(f).writerow([name, today,
                                    datetime.now().strftime("%H:%M:%S")])
        marked_today.add(name)
        print(f"[ATTENDANCE] Marked: {name}")

def get_embedding(face_bgr):
    """Crop → save temp → FaceNet → L2-normalised 128-D vector."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
        cv2.imwrite(tmp_path, face_bgr)
    try:
        result = DeepFace.represent(
            img_path          = tmp_path,
            model_name        = FACENET_MODEL,
            enforce_detection = False,
            detector_backend  = "skip"    
        )
        emb = np.array(result[0]["embedding"], dtype=np.float32)
        return emb / (np.linalg.norm(emb) + 1e-10)
    except Exception as e:
        print(f"[WARN] Embedding error: {e}")
        return None
    finally:
        os.unlink(tmp_path)

# ── Main camera loop ─────────────────────────────────────────────────
cam          = cv2.VideoCapture(0)
frame_count  = 0
last_results = []   

print("[INFO] Attendance system live — press Q to quit")

while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame_count += 1

    
    if frame_count % PROCESS_EVERY == 0:
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(80, 80))
        last_results = []

        for (x, y, w, h) in faces:
            emb = get_embedding(frame[y:y+h, x:x+w])

            if emb is None:
                last_results.append((x, y, w, h, "Error", 9.9, (0, 165, 255)))
                continue

         
            distances, _ = knn.kneighbors(emb.reshape(1, -1))
            min_dist      = distances[0][0]

            if min_dist < THRESHOLD:
                name  = knn.predict(emb.reshape(1, -1))[0]
                color = (0, 220, 0)       # green = recognised
                mark_attendance(name)
            else:
                name  = "Unknown"
                color = (0, 0, 220)       # red = not recognised

            last_results.append((x, y, w, h, name, min_dist, color))

    # Draw bounding boxes on every frame (smooth display)
    for (x, y, w, h, name, dist, color) in last_results:
        label = f"{name}  [{dist:.3f}]" if name not in ("Unknown", "Error") else name
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.rectangle(frame, (x, y+h-34), (x+w, y+h), color, cv2.FILLED)
        cv2.putText(frame, label, (x+5, y+h-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    cv2.putText(frame,
                f"{today}  |  Marked: {len(marked_today)}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("FaceNet Attendance — Press Q", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
print(f"\n[DONE] Attendance saved → {CSV_FILE}")