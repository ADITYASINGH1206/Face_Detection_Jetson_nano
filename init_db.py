import sqlite3
import numpy as np
import os
import sys

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attendance.db")

def create_tables(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS known_faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            embedding BLOB NOT NULL,
            enrolled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_name TEXT NOT NULL,
            camera_id INTEGER NOT NULL,
            tracker_id INTEGER NOT NULL,
            confidence REAL NOT NULL,
            logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (person_name) REFERENCES known_faces(name)
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_attendance_person
        ON attendance_logs(person_name)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_attendance_time
        ON attendance_logs(logged_at)
    """)

    conn.commit()
    conn.close()
    print("[init_db] Tables created successfully at: {}".format(db_path))


def enroll_face(name, embedding_vector, db_path=DB_PATH):
    if not isinstance(embedding_vector, np.ndarray):
        embedding_vector = np.array(embedding_vector, dtype=np.float32)

    norm = np.linalg.norm(embedding_vector)
    if norm > 0:
        embedding_vector = embedding_vector / norm

    blob = embedding_vector.astype(np.float32).tobytes()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO known_faces (name, embedding) VALUES (?, ?)",
            (name, blob)
        )
        conn.commit()
        print("[init_db] Enrolled '{}' with {}-dim embedding.".format(name, len(embedding_vector)))
    except sqlite3.IntegrityError:
        cursor.execute(
            "UPDATE known_faces SET embedding = ? WHERE name = ?",
            (blob, name)
        )
        conn.commit()
        print("[init_db] Updated embedding for '{}'.".format(name))
    finally:
        conn.close()


def list_enrolled(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, enrolled_at FROM known_faces")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("[init_db] No faces enrolled.")
        return

    print("[init_db] Enrolled faces:")
    print("{:<5} {:<20} {}".format("ID", "Name", "Enrolled At"))
    print("-" * 50)
    for row in rows:
        print("{:<5} {:<20} {}".format(row[0], row[1], row[2]))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        create_tables()
        list_enrolled()
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "create":
        create_tables()

    elif cmd == "enroll":
        if len(sys.argv) < 4:
            print("Usage: python3 init_db.py enroll <name> <embedding_npy_path>")
            sys.exit(1)
        name = sys.argv[2]
        npy_path = sys.argv[3]
        create_tables()
        emb = np.load(npy_path)
        enroll_face(name, emb)

    elif cmd == "list":
        list_enrolled()

    elif cmd == "demo":
        create_tables()
        for i, demo_name in enumerate(["Alice", "Bob", "Charlie"]):
            fake_emb = np.random.randn(512).astype(np.float32)
            enroll_face(demo_name, fake_emb)
        list_enrolled()

    else:
        print("Unknown command: {}".format(cmd))
        print("Usage: python3 init_db.py [create|enroll|list|demo]")
