# view_tables.py
from db_utils import get_connection, get_dict_cursor

def view_tables():
    """
    Connects to the database and prints the content of all major tables.
    """
    try:
        with get_connection() as conn:
            with get_dict_cursor(conn) as cursor:

                # View all users
                print("\n--- Users Table ---")
                cursor.execute("SELECT id, username FROM users") # Hide the password hash for cleaner viewing
                users = cursor.fetchall()
                if not users:
                    print("No users found.")
                else:
                    for user in users:
                        print(dict(user))

                # View all roles
                print("\n--- Roles Table ---")
                cursor.execute("SELECT id, name FROM roles") # Hiding long prompt for summary view
                roles = cursor.fetchall()
                if not roles:
                    print("No roles found.")
                else:
                    for role in roles:
                        print(dict(role))
                
                # --- IMPROVEMENT: Add a section to view the interviews table ---
                print("\n--- Interviews Table (Summary) ---")
                # Order by most recent first
                cursor.execute("""
                    SELECT id, candidate_name, final_verdict, average_score, created_at 
                    FROM interviews 
                    ORDER BY created_at DESC
                """)
                interviews = cursor.fetchall()
                if not interviews:
                    print("No completed interviews found.")
                else:
                    for interview in interviews:
                        # Format the timestamp for better readability
                        interview_dict = dict(interview)
                        interview_dict['created_at'] = interview_dict['created_at'].strftime('%Y-%m-%d %H:%M')
                        print(interview_dict)
                # --- END OF IMPROVEMENT ---

    except Exception as e:
        print(f"\nAn error occurred while trying to view the database: {e}")

if __name__ == "__main__":
    view_tables()
