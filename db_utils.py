# db_utils.py
import os
import psycopg2
import psycopg2.extras

# --- FIX: Changed password to 'seri' for consistency with other files ---
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/db_name")

def get_connection():
    """
    Establishes and returns a new connection to the PostgreSQL database.
    """
    try:
        return psycopg2.connect(DATABASE_URL)
    except psycopg2.OperationalError as e:
        print(f"Error: Could not connect to the database.")
        print(f"Please ensure the database server is running and the connection string is correct.")
        print(f"Details: {e}")
        exit()

def get_dict_cursor(conn):
    """
    Returns a dictionary cursor for the given connection.
    This allows accessing results by column name (e.g., row['username']).
    """
    return conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
