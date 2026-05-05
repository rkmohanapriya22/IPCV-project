# 02_train.py
import os
import numpy as np
import pickle
from deepface import DeepFace
from sklearn.neighbors import KNeighborsClassifier

DATASET_PATH  = "dataset"
MODEL_PATH    = "knn_model.pkl"
FACENET_MODEL = "Facenet"       

embeddings = []
labels     = []

print("[INFO] Extracting FaceNet embeddings — this may take a few minutes...")

for student_name in os.listdir(DATASET_PATH):
    student_dir = os.path.join(DATASET_PATH, student_name)
    if not os.path.isdir(student_dir):
        continue

    for img_file in os.listdir(student_dir):
        img_path = os.path.join(student_dir, img_file)
        try:
            result = DeepFace.represent(
                img_path          = img_path,
                model_name        = FACENET_MODEL,
                enforce_detection = False,
                detector_backend  = "opencv"
            )
            emb = np.array(result[0]["embedding"], dtype=np.float32)

           
            emb = emb / (np.linalg.norm(emb) + 1e-10)

            embeddings.append(emb)
            labels.append(student_name)
            print(f"  OK  {student_name}/{img_file}")

        except Exception as e:
            print(f"  SKIP {img_file}: {e}")

if not embeddings:
    raise RuntimeError("No embeddings found! Run 01_capture.py first.")

X = np.array(embeddings)   
y = np.array(labels)

print(f"\n[INFO] Training kNN on {len(y)} embeddings, {len(set(y))} student(s)...")

knn = KNeighborsClassifier(
    n_neighbors = min(5, len(y)),   # k=5, safe for small datasets
    metric      = "euclidean",       # works as cosine after L2 norm
    weights     = "distance",        # closer neighbors vote more
    algorithm   = "ball_tree"        # efficient for 128-D data
)
knn.fit(X, y)

with open(MODEL_PATH, "wb") as f:
    pickle.dump({"knn": knn, "students": sorted(set(y))}, f)

print(f"[DONE] Model saved → '{MODEL_PATH}'")
print(f"       Students registered: {sorted(set(y))}")
