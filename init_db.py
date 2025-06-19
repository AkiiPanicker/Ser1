# init_db.py
import sqlite3
from werkzeug.security import generate_password_hash

# Establish connection to the database
connection = sqlite3.connect('database.db')
cursor = connection.cursor()

# --- Create 'users' table for admin login ---
cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL
);
''')
print("Table 'users' created or already exists.")

# --- Create 'roles' table for interview customization ---
cursor.execute('''
CREATE TABLE IF NOT EXISTS roles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    system_prompt TEXT NOT NULL
);
''')
print("Table 'roles' created or already exists.")

# --- Check if the default admin user exists ---
cursor.execute("SELECT id FROM users WHERE username = ?", ('admin',))
if cursor.fetchone() is None:
    # --- Add a default admin user (change the password in a real app) ---
    # Password is "admin"
    hashed_password = generate_password_hash('admin')
    cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", ('admin', hashed_password))
    print("Default admin user 'admin' with password 'admin' created.")
else:
    print("Default admin user already exists.")

# --- Check if the default role exists ---
cursor.execute("SELECT id FROM roles WHERE name = ?", ('Software Engineer',))
if cursor.fetchone() is None:
    # --- Add the initial 'Software Engineer' role ---
    software_engineer_prompt = """
    You are an AI interviewer for a Software Engineer position. Based on the resume and interview history, ask the next question.
    Your questions should cover technical skills, problem-solving, and past project experience relevant to software engineering.
    Avoid using markdown formatting and acronyms in your responses. 
    Provide clear, simple language and keep your responses concise, as if you're speaking to someone.
    The results should be suitable for text-to-speech conversion.
    """
    cursor.execute("INSERT INTO roles (name, system_prompt) VALUES (?, ?)", ('Software Engineer', software_engineer_prompt.strip()))
    print("Default 'Software Engineer' role created.")
else:
    print("Default 'Software Engineer' role already exists.")

# Commit the changes and close the connection
connection.commit()
connection.close()

print("\nDatabase initialization complete.")
