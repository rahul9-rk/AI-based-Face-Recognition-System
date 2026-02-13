import cv2
import os
import json
import numpy as np
from datetime import datetime
import imutils
import shutil

if not hasattr(cv2, 'face') or not hasattr(cv2.face, 'LBPHFaceRecognizer_create'):
    raise ImportError(
        "OpenCV installation missing 'cv2.face'.\n"
        "Install the contrib build (pip uninstall opencv-python; "
        "pip install opencv-contrib-python).\n")

IMAGE_DIR = "images"
MODEL_DIR = "models"
LABEL_FILE = "label_info.json"
MODEL_PATH = "face_recognizer_model.yml"

CAPTURE_COUNT = 25            # per person (effective samples after augmentation)
FRAME_SKIP = 3
MIN_FACE_SIZE = (60, 60)
LBPH_PARAMS = dict(radius=3, neighbors=8, grid_x=10, grid_y=10)
CONFIDENCE_THRESHOLD = 110    # lower -> stricter; increase to be more permissive
DNN_CONF_THRESHOLD = 0.5      # face detector confidence
AUTO_CLEAN_BAD_FILES = True   # removes non-image or badly-named files before training

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------
# Helper: label file handling
# -------------------------
if not os.path.exists(LABEL_FILE):
    with open(LABEL_FILE, "w") as f:
        json.dump({}, f)

with open(LABEL_FILE, "r") as f:
    label_info_map = json.load(f)   # keys will be strings

def save_labels():
    with open(LABEL_FILE, "w") as f:
        json.dump(label_info_map, f, indent=4)

# -------------------------
# Face detector: DNN (preferred) with Haar fallback
# -------------------------
dnn_proto = os.path.join(MODEL_DIR, "deploy.prototxt.txt")
dnn_model = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

use_dnn = False
net = None
if os.path.exists(dnn_proto) and os.path.exists(dnn_model):
    try:
        net = cv2.dnn.readNetFromCaffe(dnn_proto, dnn_model)
        use_dnn = True
        print("[INFO] Using DNN face detector.")
    except Exception as e:
        print("[WARN] Could not load DNN detector, falling back to Haar. Error:", e)

if not use_dnn:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    print("[INFO] Using Haar Cascade face detector (fallback).")

def detect_face_dnn(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0), swapRB=False, crop=False)
    net.setInput(blob)
    detections = net.forward()
    best_box = None
    best_conf = 0.0
    for i in range(0, detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf > DNN_CONF_THRESHOLD and conf > best_conf:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            # clamp
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            best_box = (x1, y1, x2 - x1, y2 - y1)
            best_conf = conf
    if best_box is None:
        return None, None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    x, y, ww, hh = best_box
    # guard size
    if ww < MIN_FACE_SIZE[0] or hh < MIN_FACE_SIZE[1]:
        return None, None
    return gray[y:y+hh, x:x+ww], best_box

def detect_faces_haar(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=6, minSize=MIN_FACE_SIZE)
    return gray, faces

def detect_face(frame):
    if use_dnn:
        return detect_face_dnn(frame)
    else:
        gray, faces = detect_faces_haar(frame)
        if len(faces) == 0:
            return None, None
        (x, y, w, h) = max(faces, key=lambda r: r[2] * r[3])
        return gray[y:y+h, x:x+w], (x, y, w, h)

# -------------------------
# Augmentation
# -------------------------
def augment_image(face_gray):
    out = []
    # ensure face_gray is grayscale numpy array
    if face_gray is None:
        return out
    # base (resized)
    base = cv2.resize(face_gray, (200, 200))
    out.append(base)
    # mirrored
    out.append(cv2.flip(base, 1))
    # slight rotations
    out.append(imutils.rotate_bound(base, 10))
    out.append(imutils.rotate_bound(base, -10))
    # small zooms/crops
    h, w = base.shape
    zx = int(w * 0.9)
    zy = int(h * 0.9)
    crop = base[(h-zy)//2:(h+zy)//2, (w-zx)//2:(w+zx)//2]
    out.append(cv2.resize(crop, (200, 200)))
    # brightness variations
    bright = cv2.convertScaleAbs(base, alpha=1.1, beta=15)
    dark = cv2.convertScaleAbs(base, alpha=0.9, beta=-10)
    out.append(bright)
    out.append(dark)
    # gaussian blur mild
    out.append(cv2.GaussianBlur(base, (3,3), 0))
    # ensure unique sizes
    final = [cv2.resize(x, (200,200)) for x in out]
    return final

# -------------------------
# Utility: robust label extraction (supports old dot and new underscore formats)
# -------------------------
def extract_label_from_filename(filename):
    # filename examples:
    # new format: "6_0_2.jpg" -> "6"
    # older: "6.0.20241011.jpg" -> "6"
    base = os.path.basename(filename)
    name, ext = os.path.splitext(base)
    # try underscore
    part = name.split("_")[0]
    if part.isdigit():
        return int(part)
    # else fallback to dot
    part = name.split(".")[0]
    if part.isdigit():
        return int(part)
    # if still bad, raise
    raise ValueError("Cannot extract label from filename: " + filename)

# -------------------------
# TRAINING PREP functions
# -------------------------
def clean_non_jpg_files():
    removed = 0
    for f in os.listdir(IMAGE_DIR):
        if not f.lower().endswith(".jpg") and not f.lower().endswith(".jpeg") and not f.lower().endswith(".png"):
            try:
                os.remove(os.path.join(IMAGE_DIR, f))
                removed += 1
            except:
                pass
    if removed > 0:
        print(f"[CLEAN] Removed {removed} non-image files from {IMAGE_DIR}")

def gather_training_data():
    faces = []
    labels = []
    files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if len(files) == 0:
        return faces, labels
    for file in files:
        full = os.path.join(IMAGE_DIR, file)
        try:
            lbl = extract_label_from_filename(file)
        except Exception as e:
            # skip badly-named file
            print("[WARN] Skipping file with bad name:", file)
            continue
        img = cv2.imread(full, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("[WARN] Could not read image:", file)
            continue
        try:
            img = cv2.resize(img, (200, 200))
        except Exception:
            continue
        faces.append(img)
        labels.append(int(lbl))
    return faces, np.array(labels, dtype=np.int32)

# -------------------------
# TRAIN function
# -------------------------
def train_model():
    if AUTO_CLEAN_BAD_FILES:
        clean_non_jpg_files()

    faces, labels = gather_training_data()
    print(f"[INFO] Collected {len(faces)} images across {len(set(labels))} labels.")
    if len(faces) < 10:
        print("[ERROR] Not enough images to train. Capture more samples (recommended 60+ with augmentation).")
        return False

    # create LBPH recognizer
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create(**LBPH_PARAMS)
    except Exception as e:
        # compatibility fallback
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, labels)
    recognizer.save(MODEL_PATH)
    print("[OK] Model trained and saved to", MODEL_PATH)
    # Print label summary
    print("Trained labels:", sorted(set(labels)))
    return True

# -------------------------
# CAPTURE + AUGMENT workflow
# -------------------------
def capture_and_augment():
    name = input("Enter your name: ").strip()
    roll = input("Enter your roll number: ").strip()
    # assign a numeric label (next integer)
    # ensure we use string keys in label_info_map
    existing_labels = [int(k) for k in label_info_map.keys()] if label_info_map else []
    next_label = max(existing_labels) + 1 if existing_labels else 1
    label_info_map[str(next_label)] = f"{name} ({roll})"
    save_labels()
    print(f"[INFO] Assigned label {next_label} to {name} ({roll})")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    print("\nLook at the camera and slowly move head left / right / up / down.")
    print("Press 'c' to capture manually. Press ESC to stop early.\n")

    count = 0
    frame_no = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Camera read failed.")
            break
        face_gray, rect = detect_face(frame)
        if rect:
            (x, y, w, h) = rect
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Captured sets: {count}/{CAPTURE_COUNT}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        else:
            cv2.putText(frame, "No Face Detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.imshow("Capture (move head)", frame)
        key = cv2.waitKey(1) & 0xFF
        frame_no += 1

        if (key == ord('c') or (frame_no % FRAME_SKIP == 0)) and face_gray is not None:
            # augment and store multiple images per detection
            aug = augment_image(face_gray)
            for idx, img in enumerate(aug):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{next_label}_{count}_{idx}_{timestamp}.jpg"
                cv2.imwrite(os.path.join(IMAGE_DIR, filename), img)
            count += 1
            print(f"[INFO] Captured set {count}/{CAPTURE_COUNT}")
        if key == 27 or count >= CAPTURE_COUNT:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Capture finished. Now you may re-run option 1 to train based on images, or choose option 2 to train+recognize.")

# -------------------------
# RECOGNITION function
# -------------------------
def recognize_loop():
    if not os.path.exists(MODEL_PATH):
        print("[ERROR] Model not found. Train first (choose option 1).")
        return

    # load model
    recognizer = cv2.face.LBPHFaceRecognizer_create(**LBPH_PARAMS)
    recognizer.read(MODEL_PATH)

    # load labels (ensure keys are strings)
    with open(LABEL_FILE, "r") as f:
        labels_map = json.load(f)

    # invert to int->name
    labels_map_int = {}
    for k, v in labels_map.items():
        try:
            labels_map_int[int(k)] = v
        except:
            pass

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    print("[INFO] Starting recognition. Press 'e' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame.")
            break

        # detect all faces (if using haar fallback multiple faces)
        if use_dnn:
            face_gray, rect = detect_face(frame)
            faces_to_check = []
            if rect:
                faces_to_check.append(rect)
        else:
            gray, faces = detect_faces_haar(frame)
            faces_to_check = faces

        # work on each detected face
        for rect in faces_to_check:
            x, y, w, h = rect
            if use_dnn:
                face_region = face_gray
            else:
                face_region = gray[y:y+h, x:x+w]

            try:
                face_resized = cv2.resize(face_region, (200, 200))
            except Exception:
                continue

            label_pred, confidence = recognizer.predict(face_resized)
            # debug print
            print("[DEBUG] Predicted:", label_pred, "Confidence:", confidence)

            if confidence < CONFIDENCE_THRESHOLD:
                name = labels_map_int.get(int(label_pred), f"ID:{label_pred}")
                label_text = f"{name} ({int(confidence)})"
                color = (0, 255, 0)
            else:
                label_text = f"Unknown ({int(confidence)})"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label_text, (x, y - 10),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)

        cv2.imshow("Recognition", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('e'):
            break

    cap.release()
    cv2.destroyAllWindows()

# -------------------------
# Small utility to standardize/rename older files (optional)
# -------------------------
def normalize_old_filenames():
    changed = 0
    for f in os.listdir(IMAGE_DIR):
        if not f.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        try:
            _ = extract_label_from_filename(f)
        except:
            # attempt to map "6.0.2024.jpg" -> "6_0_0_timestamp.jpg"
            parts = f.split(".")
            if parts and parts[0].isdigit():
                src = os.path.join(IMAGE_DIR, f)
                newname = f"{parts[0]}_0_0_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
                dst = os.path.join(IMAGE_DIR, newname)
                shutil.move(src, dst)
                changed += 1
    if changed:
        print(f"[NORMALIZE] Renamed {changed} old-style files to new format.")

# -------------------------
# Main menu
# -------------------------
def main():
    while True:
        print("\n========== Face System ==========")
        print("1) Capture images (multi-angle) and Train model")
        print("2) Train model from images (if already captured)")
        print("3) Run recognition (live)")
        print("4) Normalize old filenames (optional)")
        print("5) Exit")
        choice = input("Choose option: ").strip()
        if choice == "1":
            capture_and_augment()
            trained = train_model()
            if trained:
                print("[INFO] Training finished. You can now run option 3 to test recognition.")
        elif choice == "2":
            trained = train_model()
            if trained:
                print("[INFO] Training finished. You can now run option 3.")
        elif choice == "3":
            recognize_loop()
        elif choice == "4":
            normalize_old_filenames()
        elif choice == "5":
            break
        else:
            print("Invalid option. Try again.")

if __name__ == "__main__":
    main()
