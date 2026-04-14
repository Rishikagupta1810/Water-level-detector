"""
UserModel: Handles all user-related database operations.
- SQLite database (no external server needed)
- Passwords are hashed using werkzeug (never stored as plain text)
- Single Responsibility: only knows about users, nothing about Flask/routes
"""

import sqlite3
import logging
import os
from werkzeug.security import generate_password_hash, check_password_hash

logger = logging.getLogger(__name__)

DB_PATH = os.path.join("database", "users.db")


class UserModel:

    def __init__(self):
        os.makedirs("database", exist_ok=True)
        self._create_table()

    def _get_connection(self):
        return sqlite3.connect(DB_PATH)

    def _create_table(self):
        """Creates users table if it doesn't already exist."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id       INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT    UNIQUE NOT NULL,
                    email    TEXT    UNIQUE NOT NULL,
                    password TEXT    NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
        logger.info("Database ready | path=%s", DB_PATH)

    def register(self, username: str, email: str, password: str) -> dict:
        """
        Registers a new user.
        Returns {"success": True} or {"success": False, "error": "..."}
        """
        if len(username) < 3:
            return {"success": False, "error": "Username must be at least 3 characters."}
        if len(password) < 6:
            return {"success": False, "error": "Password must be at least 6 characters."}

        hashed = generate_password_hash(password)
        try:
            with self._get_connection() as conn:
                conn.execute(
                    "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                    (username.strip(), email.strip().lower(), hashed)
                )
                conn.commit()
            logger.info("New user registered | username=%s | email=%s", username, email)
            return {"success": True}
        except sqlite3.IntegrityError as e:
            if "username" in str(e):
                logger.warning("Registration failed — username taken: %s", username)
                return {"success": False, "error": "Username already taken."}
            if "email" in str(e):
                logger.warning("Registration failed — email taken: %s", email)
                return {"success": False, "error": "Email already registered."}
            return {"success": False, "error": "Registration failed."}

    def login(self, username: str, password: str) -> dict:
        """
        Validates login credentials.
        Returns {"success": True, "user": {...}} or {"success": False, "error": "..."}
        """
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT id, username, email, password FROM users WHERE username = ?",
                (username.strip(),)
            ).fetchone()

        if row is None:
            logger.warning("Login failed — username not found: %s", username)
            return {"success": False, "error": "Invalid username or password."}

        user_id, db_username, db_email, db_password = row

        if not check_password_hash(db_password, password):
            logger.warning("Login failed — wrong password for: %s", username)
            return {"success": False, "error": "Invalid username or password."}

        logger.info("Login successful | username=%s", username)
        return {
            "success": True,
            "user": {"id": user_id, "username": db_username, "email": db_email}
        }

    def get_user_by_id(self, user_id: int) -> dict | None:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT id, username, email FROM users WHERE id = ?", (user_id,)
            ).fetchone()
        if row:
            return {"id": row[0], "username": row[1], "email": row[2]}
        return None