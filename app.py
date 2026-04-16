import os
import re
import logging
from functools import wraps
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from detector.water_level_detector import WaterLevelDetector
from detector.image_processor import ImageProcessor
from auth.user_model import UserModel
from logger_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Keep this secret for sessions to work
app.secret_key = "aqualevel-secret-key-change-in-production"

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize our components
detector   = WaterLevelDetector()
processor  = ImageProcessor()
user_model = UserModel()

def is_valid_email(email: str) -> bool:
    pattern = r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.method == "POST":
                return jsonify({"session_expired": True, "error": "Please log in."}), 401
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return decorated

@app.route("/", methods=["GET"])
def login_page():
    if "user_id" in session:
        return redirect(url_for("detect_page"))
    return render_template("login.html")

@app.route("/login", methods=["POST"])
def login():
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    result = user_model.login(username, password)
    if not result["success"]:
        flash(result["error"], "error")
        return redirect(url_for("login_page"))
    session["user_id"] = result["user"]["id"]
    session["username"] = result["user"]["username"]
    return redirect(url_for("detect_page"))

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("register.html")
    
    # Handle Registration Logic
    username = request.form.get("username", "").strip()
    email = request.form.get("email", "").strip()
    password = request.form.get("password", "")
    
    result = user_model.register(username, email, password)
    if result["success"]:
        flash("Account created!", "success")
        return redirect(url_for("login_page"))
    flash(result["error"], "error")
    return redirect(url_for("register"))

@app.route("/detect", methods=["GET"])
@login_required
def detect_page():
    return render_template("index.html", username=session.get("username"))

@app.route("/detect", methods=["POST"])
@login_required
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # 1. Save File
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # 2. Process/Load Image
    image = processor.load_and_fit(filepath)
    if image is None:
        return jsonify({"error": "Could not read image"}), 422

    # 3. Analyze Water Level
    result, err_reason = detector.detect(image)
    
    if result is None:
        return jsonify({"error": err_reason or "Analysis failed"}), 200

    # 4. Annotate (Draw the red line and level text)
    annotated = processor.annotate(image, result)
    processor.save(annotated, filepath)

    # 5. Return JSON with correct URL
    return jsonify({
        "level_meters": result["level_meters"],
        "level_percent": result["level_percent"],
        "status": result["status"],
        # Add a version query to bypass browser cache
        "image_url": url_for('static', filename='uploads/' + file.filename) + "?v=" + str(os.path.getmtime(filepath))
    })

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login_page"))

if __name__ == "__main__":
    app.run(debug=True)