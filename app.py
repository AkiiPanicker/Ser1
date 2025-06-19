import os
import uuid
import spacy
import PyPDF2
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_from_directory, g
from functools import wraps
import google.generativeai as genai
from gtts import gTTS
from fpdf import FPDF
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

# --- START: NEW IMPORTS FOR ADMIN/DB ---
import sqlite3
from werkzeug.security import check_password_hash, generate_password_hash
# --- END: NEW IMPORTS ---


app = Flask(__name__)
app.secret_key = os.urandom(24)

# --- Define all folders and DB ---
UPLOAD_FOLDER = 'uploads'
AUDIO_FOLDER = 'static/audio'
REPORTS_FOLDER = 'reports'
DATABASE = 'database.db' # Path to our SQLite database

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Email Configuration
SENDER_EMAIL = "akshataki7905@gmail.com"
SENDER_APP_PASSWORD = "dwct esbd fucy nybp"
RECIPIENT_EMAIL = "sakshamsaxenamoreyeahs@gmail.com"

# --- REMOVED THE STATIC MODEL DEFINITION HERE ---
# We will now create the model dynamically in the interview step.

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# --- START: DATABASE HELPER FUNCTIONS ---
def get_db():
    """Opens a new database connection if there is none yet for the current application context."""
    if 'db' not in g:
        g.db = sqlite3.connect(DATABASE)
        g.db.row_factory = sqlite3.Row  # This allows us to access columns by name
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    """Closes the database again at the end of the request."""
    db = g.pop('db', None)
    if db is not None:
        db.close()
# --- END: DATABASE HELPER FUNCTIONS ---


# --- START: ADMIN AUTH DECORATOR ---
def admin_login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'admin_logged_in' not in session:
            return redirect(url_for('admin_login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function
# --- END: ADMIN AUTH DECORATOR ---


# --- Functions from your original file (cleanup, pdf, email, extract, tts) ---
# --- (No changes needed in these functions) ---
def cleanup_session_files():
    """Safely deletes temporary files (resume, audio) associated with the current user's session."""
    resume_file = session.get('resume_file')
    audio_files = session.get('session_audio_files', [])
    if resume_file:
        try:
            resume_path = os.path.join(UPLOAD_FOLDER, resume_file)
            if os.path.exists(resume_path):
                os.remove(resume_path)
        except Exception as e:
            print(f"Error cleaning up resume file {resume_file}: {e}")
    for audio_file in audio_files:
        try:
            audio_path = os.path.join(app.root_path, AUDIO_FOLDER, audio_file)
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception as e:
            print(f"Error cleaning up audio file {audio_file}: {e}")
    session.pop('resume_file', None)
    session.pop('session_audio_files', None)
    session.pop('audio_filename', None)

def create_summary_pdf(candidate_name, avg_score, perc_score, final_message, interview_details):
    pdf_filename = f"Summary_{candidate_name.replace(' ', '_')}_{uuid.uuid4().hex[:8]}.pdf"
    filepath = os.path.join(REPORTS_FOLDER, pdf_filename)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(0, 10, f"Interview Summary for {candidate_name}", 0, 1, 'C')
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Overall Results", 0, 1, 'L')
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 8, f"Average Score: {avg_score:.1f}/10", 0, 1, 'L')
    pdf.cell(0, 8, f"Percentage Score: {perc_score:.1f}%", 0, 1, 'L')
    pdf.multi_cell(0, 8, f"Final Verdict: {final_message}", 0, 'L')
    pdf.ln(5)
    pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + 190, pdf.get_y())
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Detailed Interview Breakdown", 0, 1, 'L')
    pdf.ln(2)
    interview_list = list(interview_details)
    if not interview_list:
        pdf.set_font("Arial", 'I', 12)
        pdf.cell(0, 10, "No questions were answered to provide detailed feedback.", 0, 1, 'L')
    else:
        for i, (question, answer, score, feedback) in enumerate(interview_list, 1):
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 8, f"Question {i} - Score: {score:.1f}/10", 0, 1, 'L')
            pdf.set_font("Arial", 'B', 11)
            clean_question = question.split(":", 1)[-1].strip() if ":" in question else question
            pdf.multi_cell(0, 6, f'Question Asked: "{clean_question}"', 0, 'L')
            pdf.set_font("Arial", '', 11)
            pdf.multi_cell(0, 6, f'Candidate Answer: "{answer}"', 0, 'L')
            pdf.set_font("Arial", 'I', 11)
            feedback_text = feedback if feedback else "No detailed feedback provided."
            pdf.multi_cell(0, 6, f'Feedback Provided: "{feedback_text}"', 0, 'L')
            pdf.ln(8)
    try:
        pdf.output(filepath)
        return filepath
    except Exception as e:
        print(f"Error occurred while generating PDF report: {e}")
        return None

def send_email_with_pdf(recipient_email, subject, body, pdf_filepath):
    message = MIMEMultipart()
    message["From"] = SENDER_EMAIL
    message["To"] = recipient_email
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))
    try:
        with open(pdf_filepath, "rb") as pdf_file:
            attach = MIMEApplication(pdf_file.read(), _subtype="pdf")
            attach.add_header("Content-Disposition", f"attachment; filename= {os.path.basename(pdf_filepath)}")
            message.attach(attach)
    except FileNotFoundError:
        return False
    except Exception:
        return False
    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(SENDER_EMAIL, SENDER_APP_PASSWORD)
            server.sendmail(SENDER_EMAIL, recipient_email, message.as_string())
            if os.path.exists(pdf_filepath):
                os.remove(pdf_filepath)
            return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False

def extract_info_with_gemini(text, model):
    prompt = f"Extract the following information from the text below: Full Name, Education, Previous Experience, Useful Skills. Present the extracted information clearly. Text:\n{text}"
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred during information extraction: {e}"

def text_to_speech(text, filename):
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        filepath = os.path.join(app.root_path, AUDIO_FOLDER, filename)
        tts.save(filepath)
        return True
    except Exception as e:
        print(f"Error converting text to speech: {e}")
        return False

# --- MODIFIED FUNCTION ---
def conduct_interview_step(role_system_prompt, resume_text, interview_history, candidate_answer, question_num):
    """Now takes role_system_prompt to customize the AI's behavior."""
    score, feedback = None, None

    # This model is for evaluation and question generation, so we initialize it with the role's instructions.
    model = genai.GenerativeModel(
        'gemini-1.5-flash',
        system_instruction=role_system_prompt
    )

    if candidate_answer.strip() != "":
        last_question = interview_history[-1] if interview_history else "No previous question found."
        eval_prompt = f"""
You are an interviewer evaluating a candidate's answer. Provide a score out of 10 and short constructive feedback.
Format the output as: a single decimal number for the score (e.g., 8.5) on the first line, and feedback on subsequent lines.
Interview Question: {last_question}
Candidate Answer: {candidate_answer}
"""
        try:
            eval_response = model.generate_content(eval_prompt)
            lines = eval_response.text.strip().splitlines()
            score = float(lines[0])
            feedback = "\n".join(lines[1:]).strip() if len(lines) > 1 else "Good answer."
        except (ValueError, IndexError, Exception):
            score, feedback = 0, "Could not evaluate the answer."

    next_question_prompt = f"""
Based on the resume and history, ask the next question.
Resume: {resume_text}
History: {chr(10).join(interview_history)}
Ask question number {question_num}:
"""
    try:
        next_question = model.generate_content(next_question_prompt).text.strip()
    except Exception:
        next_question = "An error occurred. Please refresh."
    return next_question, score, feedback


@app.route('/', methods=['GET', 'POST'])
def page1():
    if request.method == 'POST':
        candidate_name = request.form.get('candidate_name', '').strip()
        if not candidate_name:
            flash("Please enter a candidate name.")
            return redirect(url_for('page1'))
        session.clear()
        session['candidate_name'] = candidate_name
        return redirect(url_for('upload_resume'))
    return render_template('page1.html')

# --- MODIFIED ROUTE ---
@app.route('/upload', methods=['GET', 'POST'])
def upload_resume():
    if 'candidate_name' not in session:
        flash("Please enter candidate name first.")
        return redirect(url_for('page1'))

    db = get_db()
    
    if request.method == 'POST':
        # Get the role_id from the form
        role_id = request.form.get('role_id')
        if not role_id:
            flash("Please select an interview role.", "warning")
            # Refetch roles for the template on failed submission
            roles = db.execute('SELECT id, name FROM roles ORDER BY name').fetchall()
            return render_template('page2.html', roles=roles)
        
        uploaded_file = request.files.get('resume_file')
        job_description = request.form.get('job_description', '').strip()
        if not uploaded_file and not job_description:
            flash("Please upload a resume or provide a job description.", "warning")
            roles = db.execute('SELECT id, name FROM roles ORDER BY name').fetchall()
            return render_template('page2.html', roles=roles)

        file_content = ""
        if uploaded_file and uploaded_file.filename:
            # ... (resume reading logic - no change)
            if uploaded_file.filename.lower().endswith('.pdf'):
                try:
                    pdf_reader = PyPDF2.PdfReader(uploaded_file)
                    for page in pdf_reader.pages:
                        text = page.extract_text()
                        if text: file_content += text
                except Exception as e:
                    flash(f"Error reading PDF: {e}")
                    return redirect(url_for('upload_resume'))
            else:
                file_content = uploaded_file.read().decode('utf-8')
        else:
            file_content = job_description
        
        # We need a generic model just for the initial resume extraction
        extraction_model = genai.GenerativeModel('gemini-1.5-flash')
        extracted_text = extract_info_with_gemini(file_content, extraction_model)
        
        unique_id = str(uuid.uuid4())
        save_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}.txt")
        with open(save_path, 'w', encoding='utf-8') as f: f.write(extracted_text)
        
        session['resume_file'] = f"{unique_id}.txt"
        session['selected_role_id'] = role_id # Store the selected role ID

        # Reset interview session
        session['interview_history'] = []
        session['scores'] = []
        session['feedbacks'] = []
        session['full_questions'] = [] 
        session['candidate_answers'] = []
        session['question_num'] = 1
        session['interview_finished'] = False
        session['session_audio_files'] = []

        return redirect(url_for('interview'))
    
    # GET Request: Fetch roles from DB and pass to the template
    roles = db.execute('SELECT id, name FROM roles ORDER BY name').fetchall()
    return render_template('page2.html', roles=roles)


# --- MODIFIED ROUTE ---
@app.route('/interview', methods=['GET', 'POST'])
def interview():
    if 'candidate_name' not in session or 'selected_role_id' not in session:
        flash("Interview session expired or invalid. Please start again.", "warning")
        return redirect(url_for('page1'))

    db = get_db()
    role = db.execute('SELECT system_prompt FROM roles WHERE id = ?', (session['selected_role_id'],)).fetchone()
    if not role:
        flash("The selected interview role no longer exists. Please start over.", "danger")
        cleanup_session_files()
        session.clear()
        return redirect(url_for('page1'))
    
    # This is the specific instruction set for the AI for this interview
    role_system_prompt = role['system_prompt']

    # --- THE REST OF YOUR interview() ROUTE LOGIC FOLLOWS ---
    # It just needs to pass `role_system_prompt` to `conduct_interview_step`

    if session.get('interview_finished') and request.method == 'GET':
        summary_data_zip = zip(session.get('scores', []), session.get('feedbacks', []))
        return render_template('page3.html', # ... same as before
            interview_finished=True,
            average_score=session.get('final_avg_score', 0),
            percentage_score=session.get('final_perc_score', 0),
            final_message=session.get('final_message', 'Interview complete.'),
            summary_data=summary_data_zip,
            candidate_name=session.get('candidate_name', 'Candidate'))

    resume_text = ""
    if 'resume_file' in session:
        resume_path = os.path.join(UPLOAD_FOLDER, session['resume_file'])
        if os.path.exists(resume_path):
            with open(resume_path, 'r', encoding='utf-8') as f:
                resume_text = f.read()

    def render_summary_and_cleanup():
        scores = session.get('scores', [])
        avg_score = sum(scores) / len(scores) if scores else 0
        perc_score = (avg_score / 10) * 100
        # ... rest of the function is the same ...
        final_message = "Interview ended."
        if scores:
            if avg_score > 8: final_message = "Excellent performance!"
            elif avg_score > 5: final_message = "Good job, with room for improvement."
            else: final_message = "Needs significant improvement."
        session['final_avg_score'] = avg_score
        session['final_perc_score'] = perc_score
        session['final_message'] = final_message
        interview_details_for_pdf = zip(session.get('full_questions',[]), session.get('candidate_answers',[]), scores, session.get('feedbacks',[]))
        generated_pdf_filepath = create_summary_pdf(session['candidate_name'], avg_score, perc_score, final_message, interview_details_for_pdf)
        if generated_pdf_filepath:
            email_body = f"Please find the interview summary for {session['candidate_name']} attached."
            send_email_with_pdf(RECIPIENT_EMAIL, f"Interview Summary: {session['candidate_name']}", email_body, generated_pdf_filepath)
        
        cleanup_session_files()
        session.modified = True
        return render_template('page3.html', interview_finished=True, average_score=avg_score, percentage_score=perc_score, final_message=final_message, summary_data=zip(scores, session.get('feedbacks', [])), candidate_name=session['candidate_name'])


    if request.method == 'POST':
        if 'end_interview' in request.form or session.get('question_num', 1) > 10:
            session['interview_finished'] = True
            return render_summary_and_cleanup()

        candidate_answer = request.form.get('answer', '').strip()
        question_num = session.get('question_num', 1)
        interview_history = session.get('interview_history', [])
        last_question_text = interview_history[-1] if interview_history else "N/A"

        next_question, score, feedback = conduct_interview_step(
            role_system_prompt, resume_text, interview_history, candidate_answer, question_num
        )
        
        if candidate_answer:
            session['interview_history'].append(f"Answer {question_num-1}: {candidate_answer}")
            if score is not None:
                session.setdefault('full_questions', []).append(last_question_text)
                session.setdefault('candidate_answers', []).append(candidate_answer)
                session['scores'].append(score)
                session['feedbacks'].append(feedback)
        
        session['interview_history'].append(f"Question {question_num}: {next_question}")
        session['question_num'] = question_num + 1

        if session.get('question_num') > 10:
             session['interview_finished'] = True
             return render_summary_and_cleanup()
        else:
             filename = f"q_{session['question_num']-1}_{uuid.uuid4().hex}.mp3"
             if text_to_speech(next_question, filename):
                 session['audio_filename'] = filename
                 session.setdefault('session_audio_files', []).append(filename)
             current_question = next_question

    else: # GET request
        session['question_num'] = 1
        first_question, _, _ = conduct_interview_step(role_system_prompt, resume_text, [], "", 1)
        session['interview_history'] = [f"Question 1: {first_question}"]
        filename = f"q_1_{uuid.uuid4().hex}.mp3"
        if text_to_speech(first_question, filename):
            session['audio_filename'] = filename
            session.setdefault('session_audio_files', []).append(filename)
        current_question = first_question
            
    session.modified = True
    return render_template('page3.html',
                           question=current_question,
                           last_score=session.get('scores', [])[-1] if session.get('scores') and request.method == 'POST' else None,
                           last_feedback=session.get('feedbacks', [])[-1] if session.get('feedbacks') and request.method == 'POST' else None,
                           question_number=session.get('question_num', 1) - 1 if request.method == 'POST' else 1,
                           candidate_name=session.get('candidate_name', 'Candidate'),
                           interview_finished=session.get('interview_finished', False),
                           audio_file_url=url_for('static', filename=f'audio/{session.get("audio_filename")}') if session.get('audio_filename') else None)


# --- START: NEW ADMIN ROUTES ---
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        db = get_db()
        user = db.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()

        if user and check_password_hash(user['password_hash'], password):
            session['admin_logged_in'] = True
            session['admin_username'] = user['username']
            flash('Login successful!', 'success')
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Invalid username or password', 'danger')

    return render_template('admin_login.html')


@app.route('/admin/dashboard', methods=['GET', 'POST'])
@admin_login_required
def admin_dashboard():
    db = get_db()
    if request.method == 'POST':
        role_name = request.form['name'].strip()
        system_prompt = request.form['system_prompt'].strip()
        if role_name and system_prompt:
            try:
                db.execute('INSERT INTO roles (name, system_prompt) VALUES (?, ?)', (role_name, system_prompt))
                db.commit()
                flash(f'Role "{role_name}" added successfully.', 'success')
            except sqlite3.IntegrityError:
                flash(f'Error: Role "{role_name}" already exists.', 'danger')
        else:
            flash('Both fields are required.', 'warning')
        return redirect(url_for('admin_dashboard'))

    # GET request: fetch and display existing roles
    roles = db.execute('SELECT * FROM roles ORDER BY name').fetchall()
    return render_template('admin_dashboard.html', roles=roles)


@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    session.pop('admin_username', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('admin_login'))
# --- END: NEW ADMIN ROUTES ---


if __name__ == '__main__':
    app.run(debug=True)
