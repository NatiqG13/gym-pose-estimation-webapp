import sqlite3


DB_PATH = "gym_pose.db"


def get_connection():
    """
    Opens a SQLite connection and returns rows like dictionaries.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """
    Creates the database tables if they do not already exist.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Stores one row per workout session so we can track history and trends.
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS analyses (
        id INTEGER PRIMARY KEY,
        created_at TEXT,
        exercise TEXT,
        original_filename TEXT,
        rep_count INTEGER,
        pass_count INTEGER,
        fail_count INTEGER,
        avg_rom REAL,
        avg_duration REAL,
        uploaded_file_path TEXT,
        output_dir TEXT,
        summary_json_path TEXT,
        reps_csv_path TEXT
    )
    """)

    # Stores one row per rep so a single workout can be inspected in detail.
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS rep_results (
        id INTEGER PRIMARY KEY,
        analysis_id INTEGER,
        rep_index INTEGER,
        start_idx INTEGER,
        end_idx INTEGER,
        duration REAL,
        rom REAL,
        label TEXT,
        reason TEXT
    )
    """)

    conn.commit()
    conn.close()


def insert_analysis(
    created_at,
    exercise,
    original_filename,
    rep_count,
    pass_count,
    fail_count,
    avg_rom,
    avg_duration,
    uploaded_file_path,
    output_dir,
    summary_json_path,
    reps_csv_path
):
    """
    Inserts one workout/session row into the analyses table and returns its id.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO analyses (
        created_at,
        exercise,
        original_filename,
        rep_count,
        pass_count,
        fail_count,
        avg_rom,
        avg_duration,
        uploaded_file_path,
        output_dir,
        summary_json_path,
        reps_csv_path
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        created_at,
        exercise,
        original_filename,
        rep_count,
        pass_count,
        fail_count,
        avg_rom,
        avg_duration,
        uploaded_file_path,
        output_dir,
        summary_json_path,
        reps_csv_path
    ))

    analysis_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return analysis_id


def insert_rep_results(analysis_id, reps):
    """
    Inserts all rep rows for a single workout.
    """
    conn = get_connection()
    cursor = conn.cursor()

    for i, rep in enumerate(reps, start=1):
        cursor.execute("""
        INSERT INTO rep_results (
            analysis_id,
            rep_index,
            start_idx,
            end_idx,
            duration,
            rom,
            label,
            reason
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            analysis_id,
            i,
            rep.get("start_idx"),
            rep.get("end_idx"),
            rep.get("duration"),
            rep.get("rom"),
            rep.get("label"),
            rep.get("reason")
        ))

    conn.commit()
    conn.close()


def get_all_analyses():
    """
    Returns all saved workout sessions, newest first.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT *
    FROM analyses
    ORDER BY id DESC
    """)

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def get_analysis_by_id(analysis_id):
    """
    Returns a single saved workout session.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT *
    FROM analyses
    WHERE id = ?
    """, (analysis_id,))

    row = cursor.fetchone()
    conn.close()

    return dict(row) if row else None


def get_rep_results_by_analysis_id(analysis_id):
    """
    Returns all rep rows for one saved workout.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT *
    FROM rep_results
    WHERE analysis_id = ?
    ORDER BY rep_index ASC
    """, (analysis_id,))

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]