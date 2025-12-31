from flask import Flask, request, jsonify, send_from_directory
from ultralytics import YOLO
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS

from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from datetime import datetime
from dotenv import load_dotenv
from bson import ObjectId
import uuid
import re

# =========================
# Load env
# =========================
import os
from pymongo import MongoClient

MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME", "medicalai_db")  # اختياري

mongo_client = MongoClient(
    MONGODB_URI,
    tls=True,
    serverSelectionTimeoutMS=20000
)

db = mongo_client[DB_NAME]   #  لازم قبل أي collection

patients_col = db["patients"]
pred_col = db["predictions"]


# Indexes (safe)
try:
    users_col.create_index("pid", unique=True)
    pred_col.create_index([("patient.pid", 1), ("created_at", -1)])
except Exception as e:
    print("⚠️ Mongo index creation skipped:", e)

# بعد كذا تبدأ routes
@app.route("/outputs/<path:filename>")
def serve_output(filename):
    ...


# =========================
# Paths
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# Flask app
# =========================
app = Flask(__name__)

# Restrict CORS (keep * only for local testing)
CORS(app, resources={r"/predict": {"origins": FRONTEND_ORIGIN}})

# Limit upload size (10MB)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR

ALLOWED_EXT = {"png", "jpg", "jpeg", "webp"}

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def clean_id(x: str, min_len=2, max_len=40) -> str:
    x = (x or "").strip()
    x = re.sub(r"\s+", "", x)  # remove spaces
    if len(x) < min_len or len(x) > max_len:
        return ""
    # allow letters, digits, underscore, dash
    if not re.match(r"^[A-Za-z0-9_\-]+$", x):
        return ""
    return x

def parse_age(form):
    """
    Accepts:
      - age_value + age_unit (years/months)
    Returns: (value:int, unit:str) or (None, None)
    """
    age_unit = (form.get("age_unit", "") or "").strip().lower()
    age_value_raw = (form.get("age_value", "") or "").strip()

    if not age_value_raw:
        return None, None

    try:
        v = int(age_value_raw)
    except ValueError:
        return None, None

    if age_unit not in {"years", "months"}:
        return None, None

    if age_unit == "years" and v < 1:
        return None, None
    if age_unit == "months" and v < 0:
        return None, None

    return v, age_unit

def build_fracture_text(preds):
    """
    preds: list of tuples (class_name, conf)
    returns: (fracture_type, description)
    """
    if not preds:
        return "None", "No detections were found in the image."

    # pick best detection
    best = max(preds, key=lambda x: x[1])
    best_name, best_conf = best[0], float(best[1])

    # if only Healthy or no fracture
    non_healthy = [(c, cf) for c, cf in preds if c.lower() != "healthy"]
    if not non_healthy:
        return "Healthy", "The AI did not detect a fracture. The image appears healthy based on the model output."

    # fracture type = best non-healthy
    best_fr = max(non_healthy, key=lambda x: x[1])
    fr_name, fr_conf = best_fr[0], float(best_fr[1])

    desc = (
        f"The model detected a possible fracture type: '{fr_name}' "
        f"with confidence {fr_conf:.3f}. "
        "Please confirm clinically and review the annotated image for the highlighted region."
    )
    return fr_name, desc

# =========================
# Load YOLO once
# =========================
model = YOLO(MODEL_PATH)
names = model.names

# =========================
# Serve annotated outputs
# =========================
@app.route("/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)

# (Optional) serve original uploads (not necessary for UI now)
@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)

# =========================
# Predict endpoint
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    # Required IDs
    staff_id = clean_id(request.form.get("staff_id", ""), min_len=3, max_len=30)
    pid = clean_id(request.form.get("pid", ""), min_len=2, max_len=40)

    if not staff_id:
        return jsonify({"error": "Invalid Staff ID. Use letters/numbers/_/- and length 3-30."}), 400
    if not pid:
        return jsonify({"error": "Invalid Patient ID. Use letters/numbers/_/- and length 2-40."}), 400

    full_name = (request.form.get("name", "") or "").strip()
    if len(full_name) < 2 or len(full_name) > 80:
        return jsonify({"error": "Invalid patient name length."}), 400

    scan = (request.form.get("scanType", "X-Ray") or "").strip()
    notes = (request.form.get("notes", "") or "").strip()
    if len(notes) > 500:
        return jsonify({"error": "Notes too long (max 500 characters)."}), 400

    age_value, age_unit = parse_age(request.form)
    if age_value is None:
        return jsonify({"error": "Invalid age. Use age_value + age_unit. Years >= 1, Months >= 0."}), 400

    # Validate image
    if "xray" not in request.files:
        return jsonify({"error": "No file part 'xray'"}), 400

    file = request.files["xray"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Use PNG/JPG/JPEG/WebP"}), 400

    # Save original image with unique name
    original_name = secure_filename(file.filename)
    ext = os.path.splitext(original_name)[1].lower()
    unique_name = f"{uuid.uuid4().hex}{ext}"

    save_path = os.path.join(UPLOAD_DIR, unique_name)
    file.save(save_path)

    # YOLO prediction + save annotated
    results = model.predict(
        source=save_path,
        imgsz=640,
        conf=0.20,
        iou=0.55,
        save=True,
        project=OUTPUT_DIR,
        name="pred",
        exist_ok=True,
        verbose=False
    )

    r = results[0]
    preds = []
    if r.boxes is not None and len(r.boxes) > 0:
        for cls_id, confv in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            preds.append((names[int(cls_id)], float(confv)))

    has_fracture = any(c.lower() != "healthy" for c, _ in preds)
    best_conf = max([c for _, c in preds], default=0.0)
    result_label = "Fracture Detected" if has_fracture else "No Fracture"

    fracture_type, fracture_description = build_fracture_text(preds)

    # Annotated image file path (Ultralytics)
    pred_folder = os.path.join(OUTPUT_DIR, "pred")
    annotated_path = os.path.join(pred_folder, unique_name)

    annotated_file = None
    if os.path.exists(annotated_path):
        annotated_file = unique_name
    else:
        stem = os.path.splitext(unique_name)[0]
        if os.path.exists(pred_folder):
            for f in os.listdir(pred_folder):
                if os.path.splitext(f)[0] == stem:
                    annotated_file = f
                    break

    annotated_url = f"/outputs/pred/{annotated_file}" if annotated_file else None

    # =========================
    # Save / Upsert staff
    # =========================
    staff_col.update_one(
        {"staff_id": staff_id},
        {
            "$set": {"updated_at": datetime.utcnow()},
            "$setOnInsert": {"created_at": datetime.utcnow()}
        },
        upsert=True
    )

    # =========================
    # Save / Upsert patient (PID unique)
    # =========================
    # Unique PID means one profile per patient
    try:
        patients_col.update_one(
            {"pid": pid},
            {
                "$set": {
                    "pid": pid,
                    "name": full_name,
                    "scanType": scan,
                    "age_value": age_value,
                    "age_unit": age_unit,
                    "updated_at": datetime.utcnow()
                },
                "$setOnInsert": {"created_at": datetime.utcnow()}
            },
            upsert=True
        )
    except DuplicateKeyError:
        # extremely rare race, but handle safely
        return jsonify({"error": "Patient ID already exists. Please use a unique ID."}), 409

    # =========================
    # Save prediction record
    # =========================
    doc = {
        "created_at": datetime.utcnow(),
        "staff": {"staff_id": staff_id},
        "patient": {
            "pid": pid,
            "name": full_name,
            "scanType": scan,
            "age_value": age_value,
            "age_unit": age_unit
        },
        "result": result_label,
        "confidence": round(float(best_conf), 3),
        "fracture_type": fracture_type,
        "fracture_description": fracture_description,
        "clinician_notes": notes,
        "detections": [{"class": c, "conf": float(cf)} for c, cf in preds],
        "original_image_file": unique_name,
        "original_image_url": f"/uploads/{unique_name}",
        "annotated_image_url": annotated_url
    }

    inserted = pred_col.insert_one(doc)

    return jsonify({
        "record_id": str(inserted.inserted_id),
        "result": result_label,
        "confidence": round(float(best_conf), 3),
        "fracture_type": fracture_type,
        "fracture_description": fracture_description,
        "saved_notes": notes,
        "annotated_image_url": annotated_url
    })

# =========================
# History endpoints (optional)
# =========================
@app.route("/history", methods=["GET"])
def history():
    items = list(pred_col.find().sort("created_at", -1).limit(20))
    for it in items:
        it["_id"] = str(it["_id"])
    return jsonify(items)

@app.route("/history/<record_id>", methods=["GET"])
def history_one(record_id):
    try:
        doc = pred_col.find_one({"_id": ObjectId(record_id)})
        if not doc:
            return jsonify({"error": "Not found"}), 404
        doc["_id"] = str(doc["_id"])
        return jsonify(doc)
    except Exception:
        return jsonify({"error": "Invalid record id"}), 400

if __name__ == "__main__":
    # Keep debug=True locally فقط
    debug_mode = os.getenv("FLASK_DEBUG", "1") == "1"
    app.run(host="0.0.0.0", port=5000, debug=debug_mode)
