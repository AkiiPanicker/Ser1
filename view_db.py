import sqlite3

# Connect to the database
conn = sqlite3.connect('database.db')
cursor = conn.cursor()

# View all users
print("\n--- Users Table ---")
cursor.execute("SELECT * FROM users")
for row in cursor.fetchall():
    print(row)

# View all roles
print("\n--- Roles Table ---")
cursor.execute("SELECT * FROM roles")
for row in cursor.fetchall():
    print(row)

conn.close()
