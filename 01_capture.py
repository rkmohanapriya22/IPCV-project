# 01_capture.py
import cv2
import os

def capture_faces(student_name, num_samples=40):
    save_dir = f"dataset/{student_name}"
    os.makedirs(save_dir, exist_ok=True)

    cam      = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    count = 0
    print(f"[INFO] Capturing {num_samples} images for '{student_name}' — look at camera.")

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.3,
                                          minNeighbors=5, minSize=(80, 80))

        for (x, y, w, h) in faces:
            count += 1
            cv2.imwrite(f"{save_dir}/{count}.jpg", frame[y:y+h, x:x+w])
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 220, 0), 2)
            cv2.putText(frame, f"Saved {count}/{num_samples}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 0), 2)

        cv2.imshow("Capture — press Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q") or count >= num_samples:
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"[DONE] Saved {count} images → '{save_dir}'")

if __name__ == "__main__":
    name = input("Enter student name: ").strip()
    capture_faces(name)