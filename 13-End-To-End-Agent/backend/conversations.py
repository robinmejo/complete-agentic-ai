from .database import (
    checkpoint,
    conn,
    database_lock,
)


# ------------------------------------------------------------
# Initialize conversation title table
# ------------------------------------------------------------

def initialize_database():

    with database_lock:

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_titles (
                thread_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        conn.commit()


# ------------------------------------------------------------
# Save conversation title
# ------------------------------------------------------------

def save_thread_title(thread_id, title):

    thread_id = str(thread_id).strip()
    title = str(title).strip()

    if not thread_id:
        raise ValueError(
            "thread_id cannot be empty"
        )

    if not title:
        title = "New Conversation"

    with database_lock:

        conn.execute(
            """
            INSERT INTO conversation_titles (
                thread_id,
                title,
                created_at,
                updated_at
            )
            VALUES (
                ?,
                ?,
                CURRENT_TIMESTAMP,
                CURRENT_TIMESTAMP
            )

            ON CONFLICT(thread_id)
            DO UPDATE SET
                title = excluded.title,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                thread_id,
                title,
            ),
        )

        conn.commit()


# ------------------------------------------------------------
# Get saved titles
# ------------------------------------------------------------

def get_saved_thread_titles():

    with database_lock:

        rows = conn.execute(
            """
            SELECT
                thread_id,
                title
            FROM conversation_titles
            """
        ).fetchall()

    return {
        row[0]: row[1]
        for row in rows
    }


# ------------------------------------------------------------
# Get all conversations
# ------------------------------------------------------------

def get_all_threads_with_titles():

    saved_titles = get_saved_thread_titles()

    conversations = {}

    for ckpt in checkpoint.list(None):

        configurable = ckpt.config.get(
            "configurable",
            {}
        )

        thread_id = configurable.get(
            "thread_id"
        )

        if not thread_id:
            continue

        thread_id = str(thread_id)

        if thread_id not in conversations:

            conversations[thread_id] = (
                saved_titles.get(
                    thread_id,
                    "New Conversation",
                )
            )

    return conversations


# ------------------------------------------------------------
# Delete conversation
# ------------------------------------------------------------

def delete_thread(thread_id):

    thread_id = str(thread_id).strip()

    if not thread_id:
        raise ValueError(
            "thread_id cannot be empty"
        )

    with database_lock:

        # Delete LangGraph checkpoints
        checkpoint.delete_thread(
            thread_id
        )

        # Delete saved title
        conn.execute(
            """
            DELETE FROM conversation_titles
            WHERE thread_id = ?
            """,
            (thread_id,),
        )

        conn.commit()

    return True


# ------------------------------------------------------------
# Initialize database when module loads
# ------------------------------------------------------------

initialize_database()