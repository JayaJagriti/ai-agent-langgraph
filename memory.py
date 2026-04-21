import sqlite3

conn = sqlite3.connect("memory.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS chat (
    user_id TEXT,
    role TEXT,
    content TEXT
)
""")

def save_message(user_id, role, content):
    cursor.execute(
        "INSERT INTO chat VALUES (?, ?, ?)",
        (user_id, role, content)
    )
    conn.commit()


def load_history(user_id):
    cursor.execute(
        "SELECT role, content FROM chat WHERE user_id=?",
        (user_id,)
    )
    rows = cursor.fetchall()

    return [{"role": r, "content": c} for r, c in rows]