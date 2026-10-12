"""Local SQLite storage for scan history (standard-library `sqlite3`, no extra packages).

Only the RESULT of each scan is saved (label, category, confidence, advice, time).
The photo itself is never saved.

An older ECHO SCAN prototype used a table called `scan` filled by a general MobileNet
model. We leave that table untouched and use a new table, `scans`, so old guesses are
never mixed with real model results.
"""
import sqlite3
from datetime import datetime, timezone

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    predicted_label  TEXT    NOT NULL,
    item_name        TEXT    NOT NULL,
    category         TEXT    NOT NULL,
    route            TEXT    NOT NULL,
    recyclable       TEXT    NOT NULL,
    confidence       REAL    NOT NULL,
    status           TEXT    NOT NULL,
    guidance         TEXT    NOT NULL,
    warning          TEXT    NOT NULL,
    model_id         TEXT    NOT NULL,
    created_at       TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_scans_created ON scans (created_at);
"""


def connect(path):
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def init_db(path):
    """Create tables on a fresh install; safe to run on an existing database."""
    connection = connect(path)
    try:
        connection.executescript(SCHEMA)
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        connection.commit()
    finally:
        connection.close()


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def save_scan(connection, result, model_id):
    created_at = now_iso()
    cursor = connection.execute(
        """INSERT INTO scans (predicted_label, item_name, category, route, recyclable,
                              confidence, status, guidance, warning, model_id, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (result["predicted_label"], result["item"], result["category"], result["route"],
         result["recyclable"], result["confidence"], result["status"], result["guidance"],
         result["warning"], model_id, created_at),
    )
    connection.commit()
    return cursor.lastrowid, created_at


def list_scans(connection, limit=100):
    rows = connection.execute(
        "SELECT * FROM scans ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(row) for row in rows]


def stats(connection):
    total = connection.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
    count = lambda where: connection.execute(
        f"SELECT COUNT(*) FROM scans WHERE {where}").fetchone()[0]
    recyclable = count("status = 'classified' AND route = 'Recycle'")
    needs_review = count("status != 'classified'")
    other = total - recyclable - needs_review
    average = connection.execute(
        "SELECT AVG(confidence) FROM scans WHERE status = 'classified'").fetchone()[0]
    by_category = [
        {"category": row["category"], "count": row["n"]}
        for row in connection.execute(
            "SELECT category, COUNT(*) AS n FROM scans WHERE status = 'classified' "
            "GROUP BY category ORDER BY n DESC, category")
    ]
    return {
        "total": total,
        "recyclable": recyclable,
        "other_routes": other,
        "needs_review": needs_review,
        "average_confidence": average,
        "by_category": by_category,
    }


def clear_scans(connection):
    deleted = connection.execute("DELETE FROM scans").rowcount
    connection.commit()
    return deleted
