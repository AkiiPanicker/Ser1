# init_db.py
import os
import psycopg2
from werkzeug.security import generate_password_hash

# It's good practice to get the connection string from an environment variable
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:uatsa%40123@103.15.67.132:5432/interview_db") 

# Establish connection to the database
try:
    connection = psycopg2.connect(DATABASE_URL)
    cursor = connection.cursor()
    print("Successfully connected to PostgreSQL database.")
except psycopg2.OperationalError as e:
    print(f"Could not connect to the database: {e}")
    print("Please ensure your PostgreSQL server is running and the DATABASE_URL is correct.")
    exit()

# Use SERIAL PRIMARY KEY for auto-incrementing integer keys in PostgreSQL
# --- Create 'users' table for admin login ---
cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL
);
''')
print("Table 'users' created or already exists.")

# --- Create 'roles' table for interview customization ---
cursor.execute('''
CREATE TABLE IF NOT EXISTS roles (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    system_prompt TEXT NOT NULL
);
''')
print("Table 'roles' created or already exists.")

# --- Create 'interviews' table for storing results ---
cursor.execute('''
DROP TABLE IF EXISTS interviews; 
CREATE TABLE interviews (
    id SERIAL PRIMARY KEY,
    candidate_name TEXT NOT NULL,
    candidate_email TEXT NOT NULL,         -- NEW: Candidate's email
    candidate_phone TEXT NOT NULL,         -- NEW: Candidate's phone number
    role_name TEXT NOT NULL,
    final_verdict TEXT NOT NULL,
    average_score REAL NOT NULL,
    percentage_score REAL NOT NULL,
    interview_data TEXT NOT NULL,         -- Stores structured Q&A, feedback, etc.
    resume_filename VARCHAR(255),         -- To store the filename of the original uploaded resume
    report_filename VARCHAR(255),         -- Stores the filename of the generated PDF report
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
''')
print("Table 'interviews' was dropped and recreated to include candidate email and phone.")


# --- Check if the default admin user exists ---
# Use %s for placeholders in psycopg2
cursor.execute("SELECT id FROM users WHERE username = %s", ('admin',))
if cursor.fetchone() is None:
    # --- Add a default admin user (change the password in a real app) ---
    # Password is "admin"
    hashed_password = generate_password_hash('admin')
    cursor.execute("INSERT INTO users (username, password_hash) VALUES (%s, %s)", ('admin', hashed_password))
    print("Default admin user 'admin' with password 'admin' created.")
else:
    print("Default admin user already exists.")

# --- Check if the default role exists ---
cursor.execute("SELECT id FROM roles WHERE name = %s", ('Software Engineer',))
if cursor.fetchone() is None:
    # --- Add the initial 'Software Engineer' role ---
    software_engineer_prompt = """
    You are an AI interviewer for a Software Engineer position. Based on the resume and interview history, ask the next question.
    Your questions should cover technical skills, problem-solving, and past project experience relevant to software engineering.
    Avoid using markdown formatting and acronyms in your responses. 
    Provide clear, simple language and keep your responses concise, as if you're speaking to someone.
    The results should be suitable for text-to-speech conversion.
    """
    cursor.execute("INSERT INTO roles (name, system_prompt) VALUES (%s, %s)", ('Software Engineer', software_engineer_prompt.strip()))
    print("Default 'Software Engineer' role created.")
else:
    print("Default 'Software Engineer' role already exists.")

# Commit the changes and close the connection
connection.commit()
cursor.close()
connection.close()

print("\nDatabase initialization complete.")
