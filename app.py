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

# --- NEW IMPORTS ---
import json
import sqlite3
from werkzeug.security import check_password_hash, generate_password_hash
# --- END NEW IMPORTS ---


app = Flask(__name__)
app.secret_key = os.urandom(24)

# --- Define all folders, DB, and constants --- # MODIFIED
UPLOAD_FOLDER = 'uploads'
AUDIO_FOLDER = 'static/audio'
REPORTS_FOLDER = 'reports'
DATABASE = 'database.db'
MAX_QUESTIONS = 10

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Email Configuration
SENDER_EMAIL = "akshataki7905@gmail.com"
SENDER_APP_PASSWORD = "dwct esbd fucy nybp"
RECIPIENT_EMAIL = "sakshamsaxenamoreyeahs@gmail.com"

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# --- DATABASE HELPER FUNCTIONS (Unchanged) ---
def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DATABASE)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    db = g.pop('db', None)
    if db is not None:
        db.close()

# --- ADMIN AUTH DECORATOR (Unchanged) ---
def admin_login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'admin_logged_in' not in session:
            return redirect(url_for('admin_login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

# --- HELPER FUNCTIONS (MODIFIED) ---
def cleanup_session_files():
    resume_file = session.get('resume_file')
    audio_files = session.get('session_audio_files', [])
    report_filename = session.get('report_filename') # Get the report filename

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

    if report_filename: # Add cleanup logic for the PDF report
        try:
            report_path = os.path.join(REPORTS_FOLDER, report_filename)
            if os.path.exists(report_path):
                os.remove(report_path)
        except Exception as e:
            print(f"Error cleaning up report file {report_filename}: {e}")
            
    # Don't pop from session here, let session.clear() handle it

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
    if not interview_details:
        pdf.set_font("Arial", 'I', 12)
        pdf.cell(0, 10, "No questions were answered to provide detailed feedback.", 0, 1, 'L')
    else:
        for i, detail in enumerate(interview_details, 1):
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 8, f"Question {i} - Score: {detail['score']:.1f}/10", 0, 1, 'L')
            pdf.set_font("Arial", 'B', 11)
            clean_question = detail['question'].split(":", 1)[-1].strip() if ":" in detail['question'] else detail['question']
            pdf.multi_cell(0, 6, f'Question Asked: "{clean_question}"', 0, 'L')
            pdf.set_font("Arial", '', 11)
            pdf.multi_cell(0, 6, f'Candidate Answer: "{detail["answer"]}"', 0, 'L')
            pdf.set_font("Arial", 'I', 11)
            feedback_text = detail['feedback'] if detail['feedback'] else "No detailed feedback provided."
            pdf.multi_cell(0, 6, f'Feedback Provided: "{feedback_text}"', 0, 'L')
            pdf.ln(8)
    try:
        pdf.output(filepath)
        return filepath
    except Exception as e:
        print(f"Error occurred while generating PDF report: {e}")
        return None

def send_email_with_pdf(recipient_email, subject, body, pdf_filepath):
    # This function is MODIFIED to no longer delete the PDF
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
    except FileNotFoundError: return False
    except Exception: return False
    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(SENDER_EMAIL, SENDER_APP_PASSWORD)
            server.sendmail(SENDER_EMAIL, recipient_email, message.as_string())
            # THE FILE IS NO LONGER DELETED HERE. It will be cleaned up on the next session.
            return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False

def extract_info_with_gemini(text, model):
    # This function is unchanged
    prompt = f"Extract the following information from the text below: Full Name, Education, Previous Experience, Useful Skills. Present the extracted information clearly. Text:\n{text}"
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e: return f"An error occurred during information extraction: {e}"

def text_to_speech(text, filename):
    # This function is unchanged
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        filepath = os.path.join(app.root_path, AUDIO_FOLDER, filename)
        tts.save(filepath)
        return True
    except Exception as e:
        print(f"Error converting text to speech: {e}")
        return False
        
def generate_dashboard_analytics(interview_details):
    # This function is unchanged
    interview_transcript = ""
    for detail in interview_details:
        interview_transcript += f"Q: {detail['question']}\nA: {detail['answer']}\n\n"
    SKILLS_TO_EVALUATE = ["Communication", "Technical Knowledge", "Problem-Solving", "Relevant Experience", "Proactivity"]
    skills_prompt = f"""
    Based on the interview transcript below, evaluate the candidate's skills on a scale of 1 to 10 for each category: {', '.join(SKILLS_TO_EVALUATE)}.
    Your output MUST BE a single, valid JSON object. Do not include explanations, notes, or markdown formatting like ```json.
    Example:
    {{
      "Communication": 8, "Technical Knowledge": 6.5, "Problem-Solving": 9, "Relevant Experience": 7, "Proactivity": 8
    }}
    Interview Transcript:
    ---
    {interview_transcript}
    ---
    """
    try:
        analysis_model = genai.GenerativeModel('gemini-pro')
        response = analysis_model.generate_content(skills_prompt)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()
        skills_data = json.loads(cleaned_response)
        for skill in SKILLS_TO_EVALUATE:
            if skill not in skills_data:
                skills_data[skill] = 0
        return skills_data
    except (json.JSONDecodeError, Exception) as e:
        print(f"Error generating or parsing skills analysis from Gemini: {e}")
        return {skill: 0 for skill in SKILLS_TO_EVALUATE}

def conduct_interview_step(role_system_prompt, resume_text, interview_history, candidate_answer, question_num):
    # This function is unchanged
    score, feedback = None, None
    model = genai.GenerativeModel('gemini-1.5-flash', system_instruction=role_system_prompt)
    if candidate_answer.strip() != "":
        last_question = interview_history[-1] if interview_history else "No previous question found."
        eval_prompt = f"You are an interviewer evaluating a candidate's answer. Provide a score out of 10 and short constructive feedback. Format the output as: a single decimal number for the score (e.g., 8.5) on the first line, and feedback on subsequent lines.\nInterview Question: {last_question}\nCandidate Answer: {candidate_answer}"
        try:
            eval_response = model.generate_content(eval_prompt)
            lines = eval_response.text.strip().splitlines()
            score = float(lines[0])
            feedback = "\n".join(lines[1:]).strip() if len(lines) > 1 else "Good answer."
        except (ValueError, IndexError, Exception): score, feedback = 0, "Could not evaluate the answer."
    next_question_prompt = f"Based on the resume and history, ask the next question.\nResume: {resume_text}\nHistory: {''.join(interview_history)}\nAsk question number {question_num}:"
    try:
        next_question = model.generate_content(next_question_prompt).text.strip()
    except Exception: next_question = "An error occurred. Please refresh."
    return next_question, score, feedback

@app.route('/', methods=['GET', 'POST'])
def page1():
    if request.method == 'POST':
        # Clean up files and session data from any *previous* interview.
        cleanup_session_files()
        session.clear()
        candidate_name = request.form.get('candidate_name', '').strip()
        if not candidate_name:
            flash("Please enter a candidate name.")
            return redirect(url_for('page1'))
        session['candidate_name'] = candidate_name
        return redirect(url_for('upload_resume'))
    return render_template('page1.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_resume():
    if 'candidate_name' not in session:
        flash("Please enter candidate name first.")
        return redirect(url_for('page1'))
    db = get_db()
    if request.method == 'POST':
        role_id = request.form.get('role_id')
        if not role_id:
            flash("Please select an interview role.", "warning")
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
        else: file_content = job_description
        extraction_model = genai.GenerativeModel('gemini-1.5-flash')
        extracted_text = extract_info_with_gemini(file_content, extraction_model)
        unique_id = str(uuid.uuid4())
        save_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}.txt")
        with open(save_path, 'w', encoding='utf-8') as f: f.write(extracted_text)
        session['resume_file'] = f"{unique_id}.txt"
        session['selected_role_id'] = role_id
        session['interview_history'] = []
        session['scores'] = []
        session['feedbacks'] = []
        session['full_questions'] = []
        session['candidate_answers'] = []
        session['question_num'] = 1
        session['interview_finished'] = False
        session['session_audio_files'] = []
        return redirect(url_for('interview'))
    roles = db.execute('SELECT id, name FROM roles ORDER BY name').fetchall()
    return render_template('page2.html', roles=roles)

# --- MAJORLY MODIFIED ROUTE ---
@app.route('/interview', methods=['GET', 'POST'])
def interview():
    if 'candidate_name' not in session or 'selected_role_id' not in session:
        flash("Interview session expired or invalid. Please start again.", "warning")
        return redirect(url_for('page1'))

    if session.get('interview_finished') and request.method == 'GET':
        summary_data_zip = zip(session.get('scores', []), session.get('feedbacks', []))
        return render_template('page3.html',
            interview_finished=True,
            average_score=session.get('final_avg_score', 0),
            percentage_score=session.get('final_perc_score', 0),
            final_message=session.get('final_message', 'Interview complete.'),
            summary_data=summary_data_zip,
            candidate_name=session.get('candidate_name', 'Candidate'),
            report_filename=session.get('report_filename') # Pass filename to template
        )

    db = get_db()
    role = db.execute('SELECT system_prompt FROM roles WHERE id = ?', (session['selected_role_id'],)).fetchone()
    if not role:
        flash("The selected interview role no longer exists. Please start over.", "danger")
        session.clear()
        return redirect(url_for('page1'))
    role_system_prompt = role['system_prompt']

    resume_text = ""
    if 'resume_file' in session:
        resume_path = os.path.join(UPLOAD_FOLDER, session['resume_file'])
        if os.path.exists(resume_path):
            with open(resume_path, 'r', encoding='utf-8') as f: resume_text = f.read()

    if request.method == 'POST':
        is_finished = 'end_interview' in request.form or session.get('question_num', 1) > MAX_QUESTIONS
        candidate_answer = request.form.get('answer', '').strip()
        question_num = session.get('question_num', 1)
        interview_history = session.get('interview_history', [])
        
        if candidate_answer:
            _, score, feedback = conduct_interview_step(role_system_prompt, resume_text, interview_history, candidate_answer, question_num)
            last_question_text = interview_history[-1] if interview_history else "N/A"
            session['interview_history'].append(f"Answer {question_num-1}: {candidate_answer}")
            if score is not None:
                session.setdefault('full_questions', []).append(last_question_text)
                session.setdefault('candidate_answers', []).append(candidate_answer)
                session['scores'].append(score)
                session['feedbacks'].append(feedback)

        if is_finished:
            session['interview_finished'] = True
            
            scores = session.get('scores', [])
            avg_score = sum(scores) / len(scores) if scores else 0
            perc_score = (avg_score / 10) * 100
            final_message = "Interview ended."
            if scores:
                if avg_score > 8: final_message = "Excellent performance! Highly recommended."
                elif avg_score > 6: final_message = "Good performance, a solid candidate."
                elif avg_score > 4: final_message = "Fair performance, with room for improvement."
                else: final_message = "Needs significant improvement."

            structured_interview_details = []
            for q, a, s, f in zip(session.get('full_questions',[]), session.get('candidate_answers',[]), scores, session.get('feedbacks',[])):
                 structured_interview_details.append({'question': q, 'answer': a, 'score': s, 'feedback': f})

            skills_data = generate_dashboard_analytics(structured_interview_details)
            
            session['final_avg_score'] = avg_score
            session['final_perc_score'] = perc_score
            session['final_message'] = final_message
            session['structured_interview_details'] = structured_interview_details
            session['skills_data'] = skills_data
            session['line_chart_scores'] = scores

            generated_pdf_filepath = create_summary_pdf(session['candidate_name'], avg_score, perc_score, final_message, structured_interview_details)
            if generated_pdf_filepath:
                session['report_filename'] = os.path.basename(generated_pdf_filepath)
                email_body = f"Please find the interview summary for {session['candidate_name']} attached."
                send_email_with_pdf(RECIPIENT_EMAIL, f"Interview Summary: {session['candidate_name']}", email_body, generated_pdf_filepath)

            session.modified = True
            
            return render_template('page3.html',
                interview_finished=True,
                average_score=avg_score,
                percentage_score=perc_score,
                final_message=final_message,
                summary_data=zip(scores, session.get('feedbacks', [])),
                candidate_name=session['candidate_name'],
                report_filename=session.get('report_filename') # Pass filename to template
            )

        next_question, _, _ = conduct_interview_step(role_system_prompt, resume_text, session['interview_history'], "", session['question_num'])
        session['interview_history'].append(f"Question {session['question_num']}: {next_question}")
        session['question_num'] += 1
        
        filename = f"q_{session['question_num']-1}_{uuid.uuid4().hex}.mp3"
        if text_to_speech(next_question, filename):
            session['audio_filename'] = filename
            session.setdefault('session_audio_files', []).append(filename)
        current_question = next_question
        
    else:  # GET request for the first question
        session['question_num'] = 1
        first_question, _, _ = conduct_interview_step(role_system_prompt, resume_text, [], "", 1)
        session['interview_history'] = [f"Question 1: {first_question}"]
        filename = f"q_1_{uuid.uuid4().hex}.mp3"
        if text_to_speech(first_question, filename):
            session['audio_filename'] = filename
            session.setdefault('session_audio_files', []).append(filename)
        current_question = first_question
            
    session.modified = True
    display_question_num = session.get('question_num') if request.method == 'POST' else 1
    
    return render_template('page3.html',
                           question=current_question,
                           last_score=session.get('scores', [])[-1] if session.get('scores') and request.method == 'POST' else None,
                           last_feedback=session.get('feedbacks', [])[-1] if session.get('feedbacks') and request.method == 'POST' else None,
                           question_number=display_question_num,
                           max_questions=MAX_QUESTIONS,
                           candidate_name=session.get('candidate_name', 'Candidate'),
                           interview_finished=False,
                           audio_file_url=url_for('static', filename=f'audio/{session.get("audio_filename")}') if session.get('audio_filename') else None)


# --- START: NEW DOWNLOAD ROUTE ---
@app.route('/download_report/<path:filename>')
def download_report(filename):
    # Security check: only allow download if the filename matches the one in the user's session.
    # This prevents users from guessing filenames and downloading other reports.
    if 'report_filename' not in session or session['report_filename'] != filename:
        flash("No report available for download or permission denied.", "danger")
        return redirect(url_for('page1'))
    
    # Use send_from_directory for secure file serving
    try:
        return send_from_directory(REPORTS_FOLDER, filename, as_attachment=True)
    except FileNotFoundError:
        flash("Report file not found. It may have been cleaned up.", "warning")
        return redirect(url_for('page1'))

# --- All other routes below are unchanged ---
@app.route('/dashboard')
def show_dashboard():
    # Protect this route: only accessible after an interview is finished
    if not session.get('interview_finished'):
        flash("You must complete an interview to view the dashboard.", "warning")
        return redirect(url_for('page1'))

    return render_template('dashboard.html',
        candidate_name=session.get('candidate_name', 'N/A'),
        average_score=session.get('final_avg_score', 0),
        percentage_score=session.get('final_perc_score', 0),
        final_verdict=session.get('final_message', 'N/A'),
        skills_data=session.get('skills_data', {}),
        line_chart_scores=session.get('line_chart_scores', []),
        interview_details=session.get('structured_interview_details', [])
    )

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
    roles = db.execute('SELECT * FROM roles ORDER BY name').fetchall()
    return render_template('admin_dashboard.html', roles=roles)

@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    session.pop('admin_username', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('admin_login'))

if __name__ == '__main__':
    app.run(debug=True)
