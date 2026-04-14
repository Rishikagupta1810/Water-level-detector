"""
Flask app — now includes authentication.
Routes:
  GET  /           → login page (redirects to /detect if already logged in)
  POST /login      → handle login form
  GET  /register   → register page
  POST /register   → handle register form
  GET  /logout     → clears session, back to login
  GET  /detect     → main water level page (login required)
  POST /detect     → run detection (login required)
"""

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
app.secret_key = "aqualevel-secret-key-change-in-production"

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

detector   = WaterLevelDetector()
processor  = ImageProcessor()
user_model = UserModel()


def is_valid_email(email: str) -> bool:
    """Validates email against standard RFC-style pattern."""
    pattern = r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            logger.warning("Unauthenticated access to: %s", request.path)
            if request.method == "POST" or request.headers.get("X-Requested-With") == "XMLHttpRequest":
                return jsonify({"session_expired": True, "error": "Session expired. Please log in again."}), 401
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
    logger.info("Login attempt | username=%s", username)
    result = user_model.login(username, password)
    if not result["success"]:
        flash(result["error"], "error")
        return redirect(url_for("login_page"))
    session["user_id"]  = result["user"]["id"]
    session["username"] = result["user"]["username"]
    logger.info("Session created | username=%s", username)
    return redirect(url_for("detect_page"))


@app.route("/register", methods=["GET"])
def register_page():
    if "user_id" in session:
        return redirect(url_for("detect_page"))
    return render_template("register.html")


@app.route("/register", methods=["POST"])
def register():
    username = request.form.get("username", "").strip()
    email    = request.form.get("email", "").strip()
    password = request.form.get("password", "")
    confirm  = request.form.get("confirm_password", "")

    # ── Username length validation ──
    if len(username) < 3:
        flash("Username must be at least 3 characters.", "error")
        return redirect(url_for("register_page"))

    # ── Email format validation ──
    if not is_valid_email(email):
        flash("Please enter a valid email address (e.g. user@example.com).", "error")
        return redirect(url_for("register_page"))

    # ── Password length validation ──
    if len(password) < 6:
        flash("Password must be at least 6 characters.", "error")
        return redirect(url_for("register_page"))

    # ── Password match validation ──
    if password != confirm:
        flash("Passwords do not match.", "error")
        return redirect(url_for("register_page"))

    result = user_model.register(username, email, password)
    if not result["success"]:
        flash(result["error"], "error")
        return redirect(url_for("register_page"))
    flash("Account created! Please log in.", "success")
    return redirect(url_for("login_page"))


@app.route("/logout")
def logout():
    username = session.get("username", "unknown")
    session.clear()
    logger.info("User logged out | username=%s", username)
    return redirect(url_for("login_page"))


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
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)
    image = processor.load_and_fit(filepath)
    if image is None:
        return jsonify({"error": "Could not read image"}), 422
    result, err_reason = detector.detect(image)
    if result is None:
        user_msg = err_reason or "No water detected."
        logger.warning("Detection failed | reason=%s | user=%s", err_reason, session.get("username"))
        return jsonify({"error": user_msg}), 200
    annotated = processor.annotate(image, result)
    processor.save(annotated, filepath)
    logger.info("Detection done | file=%s | level=%.2fm | status=%s | user=%s",
                file.filename, result["level_meters"], result["status"], session.get("username"))
    return jsonify({
        "level_meters":  result["level_meters"],
        "level_percent": result["level_percent"],
        "status":        result["status"],
        "image_url":     "/" + filepath,
    })


if __name__ == "__main__":
    logger.info("Starting AquaLevel Flask app")
    app.run(debug=True)