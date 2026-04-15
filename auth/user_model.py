"""
UserModel: Handles all user-related database operations.
- PostgreSQL (persistent across Render deploys)
- Passwords are hashed using werkzeug (never stored as plain text)
- Falls back to SQLite if DATABASE_URL env var is not set (local dev)
"""

import os
import logging
from werkzeug.security import generate_password_hash, check_password_hash

logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL")  # set by Render automatically

# ── Use PostgreSQL on Render, SQLite locally ──
if DATABASE_URL:
    import psycopg2
    import psycopg2.extras

    def _get_connection():
        # Render gives postgres:// but psycopg2 needs postgresql://
        url = DATABASE_URL.replace("postgres://", "postgresql://", 1)
        return psycopg2.connect(url)

    PLACEHOLDER = "%s"
    USE_POSTGRES = True
else:
    try:
        import sqlite3
    except ImportError:
        raise RuntimeError(
            "sqlite3 is not available. Please set the DATABASE_URL environment "
            "variable to use PostgreSQL, or use a Python build with SQLite support."
        )

    DB_PATH = os.path.join("database", "users.db")
    os.makedirs("database", exist_ok=True)

    def _get_connection():
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row  # allows column access by name
        return conn

    PLACEHOLDER = "?"
    USE_POSTGRES = False


class UserModel:

    def __init__(self):
        self._create_table()

    def _create_table(self):
        """Creates users table if it doesn't already exist."""
        if USE_POSTGRES:
            sql = """
                CREATE TABLE IF NOT EXISTS users (
                    id         SERIAL PRIMARY KEY,
                    username   TEXT UNIQUE NOT NULL,
                    email      TEXT UNIQUE NOT NULL,
                    password   TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
        else:
            sql = """
                CREATE TABLE IF NOT EXISTS users (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    username   TEXT UNIQUE NOT NULL,
                    email      TEXT UNIQUE NOT NULL,
                    password   TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
        with _get_connection() as conn:
            cur = conn.cursor()
            cur.execute(sql)
            conn.commit()
        logger.info("Database ready | postgres=%s", USE_POSTGRES)

    def register(self, username: str, email: str, password: str) -> dict:
        """
        Registers a new user.
        Returns {"success": True} or {"success": False, "error": "..."}
        """
        if len(username) < 3:
            return {"success": False, "error": "Username must be at least 3 characters."}
        if len(password) < 6:
            return {"success": False, "error": "Password must be at least 6 characters."}

        # Check for duplicates explicitly before inserting
        existing = self._find_existing(username, email)
        if existing == "username":
            return {"success": False, "error": "Username already taken."}
        if existing == "email":
            return {"success": False, "error": "Email already registered."}

        hashed = generate_password_hash(password)
        p = PLACEHOLDER
        try:
            with _get_connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    f"INSERT INTO users (username, email, password) VALUES ({p}, {p}, {p})",
                    (username.strip(), email.strip().lower(), hashed)
                )
                conn.commit()
            logger.info("New user registered | username=%s | email=%s", username, email)
            return {"success": True}
        except Exception as e:
            logger.error("Registration error: %s", e)
            return {"success": False, "error": "Registration failed. Please try again."}

    def _find_existing(self, username: str, email: str):
        """Returns 'username', 'email', or None depending on what already exists."""
        p = PLACEHOLDER
        with _get_connection() as conn:
            cur = conn.cursor()
            cur.execute(f"SELECT username, email FROM users WHERE username = {p} OR email = {p}",
                        (username.strip(), email.strip().lower()))
            row = cur.fetchone()
        if row is None:
            return None
        if row[0].lower() == username.strip().lower():
            return "username"
        return "email"

    def login(self, username: str, password: str) -> dict:
        """
        Validates login credentials.
        Returns {"success": True, "user": {...}} or {"success": False, "error": "..."}
        """
        p = PLACEHOLDER
        with _get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                f"SELECT id, username, email, password FROM users WHERE username = {p}",
                (username.strip(),)
            )
            row = cur.fetchone()

        if row is None:
            logger.warning("Login failed — username not found: %s", username)
            return {"success": False, "error": "Invalid username or password."}

        user_id, db_username, db_email, db_password = row[0], row[1], row[2], row[3]

        if not check_password_hash(db_password, password):
            logger.warning("Login failed — wrong password for: %s", username)
            return {"success": False, "error": "Invalid username or password."}

        logger.info("Login successful | username=%s", username)
        return {
            "success": True,
            "user": {"id": user_id, "username": db_username, "email": db_email}
        }

    def get_user_by_id(self, user_id: int) -> dict:
        p = PLACEHOLDER
        with _get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                f"SELECT id, username, email FROM users WHERE id = {p}", (user_id,)
            )
            row = cur.fetchone()
        if row:
            return {"id": row[0], "username": row[1], "email": row[2]}
        return None