# ats_calculator.py

import os
import re
import nltk
import spacy
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
import textstat
from datetime import datetime
import warnings
import html

from groq import Groq
import PyPDF2
from docx import Document

warnings.filterwarnings('ignore')

def setup_nlp_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    model_name = 'en_core_web_md'
    try:
        nlp = spacy.load(model_name)
    except OSError:
        print(f"Downloading '{model_name}' for spaCy...")
        from spacy.cli import download
        download(model_name)
        nlp = spacy.load(model_name)
    return nlp

nlp = setup_nlp_resources()


class AdvancedATSCalculator:
    def __init__(self):
        self.groq_client = None
        if os.getenv("GROQ_API_KEY"):
            self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            self.groq_model = "llama3-8b-8192"

        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.action_verbs = [
            'achieved', 'accelerated', 'administered', 'advised', 'advocated', 'analyzed', 'architected', 'automated', 'authored',
            'built', 'budgeted', 'calculated', 'centralized', 'chaired', 'coached', 'collaborated', 'conceived', 'conducted', 'consolidated', 'constructed', 'consulted', 'created',
            'decreased', 'delivered', 'designed', 'developed', 'directed', 'documented', 'drove', 'engineered', 'enhanced', 'established', 'evaluated', 'executed',
            'facilitated', 'founded', 'generated', 'guided', 'headed', 'identified', 'implemented', 'improved', 'increased', 'influenced', 'innovated', 'instituted', 'instructed', 'integrated', 'invented',
            'launched', 'led', 'lectured', 'mentored', 'managed', 'modernized', 'motivated', 'negotiated',
            'operated', 'orchestrated', 'organized', 'overhauled', 'oversaw', 'pioneered', 'planned', 'prioritized', 'produced', 'proposed',
            'quantified', 'redesigned', 'reduced', 're-engineered', 'resolved', 'restructured', 'revamped', 'scaled', 'scheduled', 'secured', 'simplified', 'solved', 'spearheaded', 'standardized', 'streamlined', 'strengthened',
            'trained', 'transformed', 'troubleshot', 'unified', 'upgraded', 'validated', 'won'
        ]
        self.passive_phrases = [
            'responsible for', 'duties included', 'tasked with', 'worked on',
            'assisted with', 'involved in', 'in charge of'
        ]
        self.technical_skills = {
            'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust', 'scala', 'kotlin', 'swift', 'r', 'matlab', 'sql', 'html', 'css', 'typescript'],
            'frameworks': ['react', 'angular', 'vue', 'node.js', 'django', 'flask', 'spring', 'laravel', 'express', 'bootstrap', 'jquery', 'tensorflow', 'pytorch', 'keras', 'next.js'],
            'databases': ['mysql', 'postgresql', 'mongodb', 'redis', 'cassandra', 'oracle', 'sqlite', 'dynamodb', 'elasticsearch'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'terraform', 'ansible', 'ci/cd', 'circleci', 'gitlab'],
            'tools': ['git', 'jira', 'confluence', 'slack', 'trello', 'postman', 'swagger', 'tableau', 'power bi'],
            'methodologies': ['agile', 'scrum', 'kanban', 'devops', 'tdd', 'bdd', 'ci/cd']
        }
        self.soft_skills = [
            'leadership', 'communication', 'teamwork', 'problem-solving', 'analytical',
            'creative', 'adaptable', 'organized', 'detail-oriented', 'time management',
            'critical thinking', 'collaboration', 'innovation', 'strategic thinking',
            'project management', 'mentoring', 'coaching', 'negotiation'
        ]

    def extract_text(self, filepath, filename):
        if filename.lower().endswith('.pdf'): return self.extract_text_from_pdf(filepath)
        elif filename.lower().endswith('.docx'): return self.extract_text_from_docx(filepath)
        elif filename.lower().endswith('.txt'): return self.extract_text_from_txt(filepath)
        else: return ""

    def extract_text_from_pdf(self, file_path):
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = "".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
                return text
        except Exception: return ""

    def extract_text_from_docx(self, file_path):
        try:
            doc = Document(file_path)
            return "\n".join(para.text for para in doc.paragraphs)
        except Exception: return ""

    def extract_text_from_txt(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file: return file.read()
        except Exception: return ""
    
    def extract_skills_from_jd(self, job_description):
        if not job_description:
            return set()
        doc = nlp(job_description.lower())
        jd_skills = set()
        for token in doc:
            if not token.is_stop and not token.is_punct and token.pos_ in ['NOUN', 'PROPN']:
                jd_skills.add(token.lemma_)
        all_skills = [skill for cat in self.technical_skills.values() for skill in cat] + self.soft_skills
        for skill in all_skills:
            if skill.lower() in job_description.lower():
                jd_skills.add(skill.lower())
        return jd_skills

    def analyze_work_experience(self, text):
        analysis = {
            'quantified_metrics': [], 'action_verbs_found': [], 'passive_phrases_found': [] }
        quant_pattern = re.compile(r'\b(\d+%|\$\d[,\d\.]*|\d+ years|\d+-\d+)\b')
        lines = text.split('\n')
        found_metrics = set(); found_verbs = set(); found_passive = set()
        for line in lines:
            for match in re.finditer(quant_pattern, line):
                found_metrics.add(match.group(0))
            stripped_line = line.strip()
            for verb in self.action_verbs:
                if re.match(r'^[•\*\-\s]*' + verb + r'\b', stripped_line, re.IGNORECASE):
                    verb_match = re.search(r'\b' + verb + r'\b', stripped_line, re.IGNORECASE)
                    if verb_match: found_verbs.add(verb_match.group(0))
                    break
            for phrase in self.passive_phrases:
                if phrase.lower() in line.lower():
                    phrase_match = re.search(re.escape(phrase), line, re.IGNORECASE)
                    if phrase_match: found_passive.add(phrase_match.group(0))
        analysis['quantified_metrics'] = list(found_metrics)
        analysis['action_verbs_found'] = list(found_verbs)
        analysis['passive_phrases_found'] = list(found_passive)
        impact_score = min(len(analysis['quantified_metrics']) * 10 + len(analysis['action_verbs_found']) * 5 - len(analysis['passive_phrases_found']) * 10, 100)
        analysis['score'] = max(0, impact_score)
        return analysis

    def analyze_keyword_match(self, resume_text, job_description):
        if not job_description:
            return {'score': 75, 'matched_keywords': [], 'missing_keywords': [], 'feedback': "No job description provided. Score is based on general resume quality."}
        
        jd_keywords = self.extract_skills_from_jd(job_description)
        if not jd_keywords: 
            return {'score': 70, 'matched_keywords': [], 'missing_keywords': [], 'feedback': "Could not extract scannable skills from job description. Score is based on general resume quality."}
        
        resume_lower = resume_text.lower()
        matched_keywords = [kw for kw in jd_keywords if kw in resume_lower]
        missing_keywords = list(jd_keywords - set(matched_keywords))
        
        # Calculate direct keyword match score
        direct_match_score = (len(matched_keywords) / len(jd_keywords)) * 100 if jd_keywords else 0

        # --- OPTIMIZED SEMANTIC SIMILARITY ---
        # Instead of comparing two large documents, we compare the resume to a
        # compact string of the JD's most important keywords. This is much faster.
        resume_doc = nlp(resume_text)
        jd_keywords_text = " ".join(jd_keywords)
        jd_doc = nlp(jd_keywords_text)
        
        similarity_score = 0
        if resume_doc.has_vector and jd_doc.has_vector:
            similarity_score = resume_doc.similarity(jd_doc)

        # Combine direct match and semantic similarity for a more robust score
        final_score = (direct_match_score * 0.7) + (similarity_score * 100 * 0.3)
        
        return {'score': min(100, final_score), 'matched_keywords': matched_keywords, 'missing_keywords': sorted(missing_keywords[:10])}

    def preprocess_text(self, text):
        text = text.lower(); text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\-\.\@\+\#]', ' ', text)
        return text.strip()

    def extract_contact_info(self, text):
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})'
        linkedin_pattern = r'linkedin\.com/in/[\w\-]+'
        return {
            'emails': re.findall(email_pattern, text),
            'phones': ['-'.join(filter(None, phone)) for phone in re.findall(phone_pattern, text)],
            'linkedin': re.findall(linkedin_pattern, text)
        }

    def extract_skills(self, text):
        text_lower = self.preprocess_text(text)
        found_skills = {'technical': {}, 'soft': []}
        for category, skills in self.technical_skills.items():
            found_in_category = [skill for skill in skills if f' {skill.replace("."," ")} ' in f' {text_lower} ']
            if found_in_category: found_skills['technical'][category] = found_in_category
        found_skills['soft'] = [skill for skill in self.soft_skills if f' {skill} ' in f' {text_lower} ']
        found_skills['total_technical'] = sum(len(v) for v in found_skills['technical'].values())
        found_skills['total_soft'] = len(found_skills['soft'])
        return found_skills

    def analyze_readability(self, text):
        try:
            flesch_score = textstat.flesch_reading_ease(text)
            if 60 <= flesch_score <= 80: readability_score = 100
            elif 50 <= flesch_score < 60 or 80 < flesch_score <= 90: readability_score = 80
            else: readability_score = max(0, 100 - abs(70 - flesch_score) * 2)
            return {'score': min(readability_score, 100), 'flesch_score': flesch_score, 'grade_level': textstat.flesch_kincaid_grade(text)}
        except: return {'score': 50, 'flesch_score': 0, 'grade_level': 0}
        
    def calculate_overall_score(self, text, filename, job_description):
        results = {'word_count': len(text.split()), 'raw_text': text}
        
        # Calculate all component scores
        contact_info = self.extract_contact_info(text)
        results['contact_score'] = 100 if contact_info['emails'] and contact_info['phones'] else 50
        results['contact_info'] = contact_info
        
        keyword_results = self.analyze_keyword_match(text, job_description)
        results['keyword_score'] = keyword_results['score']
        results.update(keyword_results)

        experience_analysis = self.analyze_work_experience(text)
        results['impact_score'] = experience_analysis['score']
        results['experience_analysis'] = experience_analysis
        
        skills = self.extract_skills(text)
        results['skills_score'] = min(skills['total_technical'] * 4 + skills['total_soft'] * 1.5, 100)
        results['skills'] = skills
        
        readability = self.analyze_readability(text)
        length_score = 100 if 400 <= results['word_count'] <= 800 else max(0, 100 - (abs(600 - results['word_count']) / 6))
        results['readability_score'] = (readability['score'] * 0.6) + (length_score * 0.4)
        results['readability_details'] = readability

        results['format_score'] = 100 if filename.lower().endswith(('.pdf', '.docx')) else 70

        # --- DYNAMIC SCORE WEIGHTING ---
        if job_description:
            # If JD is provided, focus on the match score.
            overall_score = (
                results['keyword_score'] * 0.40 +
                results['impact_score'] * 0.25 +
                results['skills_score'] * 0.10 +
                results['readability_score'] * 0.10 +
                results['contact_score'] * 0.05 +
                results['format_score'] * 0.10
            )
        else:
            # If NO JD, focus on general resume quality by redistributing the keyword weight.
            overall_score = (
                results['impact_score'] * 0.50 +      # Increased from 0.25
                results['skills_score'] * 0.25 +      # Increased from 0.10
                results['readability_score'] * 0.10 +
                results['contact_score'] * 0.05 +
                results['format_score'] * 0.10
            )

        results['overall_score'] = round(overall_score, 2)
        return results

    def generate_recommendations(self, results):
        recs = []
        score = results['overall_score']
        if score >= 90: benchmark = f"A score of {score} is exceptional! Your resume is highly optimized and competitive."
        elif score >= 80: benchmark = f"A score of {score} is excellent. Your resume should perform very well with ATS systems."
        elif score >= 70: benchmark = f"A score of {score} is good, but has room for targeted improvements to be more competitive."
        else: benchmark = f"A score of {score} indicates your resume needs significant optimization to pass initial ATS screenings effectively."
        recs.append(benchmark)
        if results['contact_score'] < 100: recs.append("Ensure both a professional email and a phone number are clearly listed.")
        if results.get('missing_keywords'):
            if results['keyword_score'] < 70: recs.append("Incorporate more relevant keywords and skills from the job description to improve your semantic match.")
            recs.append(f"Consider including these top missing keywords if relevant: {', '.join(results['missing_keywords'])}.")
        if results['impact_score'] < 60: recs.append("Strengthen your work experience section by adding more quantified results (using numbers, $, %) and starting bullet points with strong action verbs.")
        if results['experience_analysis']['passive_phrases_found']: recs.append("Rephrase passive sentences (e.g., 'Responsible for...') to be more active and results-oriented (e.g., 'Managed...').")
        if not (400 <= results['word_count'] <= 800): recs.append(f"Your resume's word count is {results['word_count']}. The ideal range is 400-800 words for most roles.")
        return recs

    def generate_ai_suggestions(self, results, job_description):
        if not self.groq_client or not job_description or not results.get('experience_analysis'):
            return ""
        
        passive_phrases_str = "\n- ".join(results['experience_analysis']['passive_phrases_found'][:2])
        missing_keywords_str = ", ".join(results['missing_keywords'][:5])

        prompt = f"""You are a helpful career coach. A user has an ATS score of {results['overall_score']}/100 for a job. The Job Description includes keywords like: {missing_keywords_str}. The resume has some passive phrases, such as:\n- {passive_phrases_str if passive_phrases_str else 'None found'}\n\nBased on this, provide 2-3 specific, actionable suggestions for bullet points they could add or rephrase in their resume to make it stronger for THIS job. Be concise and encouraging. Do not use markdown. Start with a phrase like "Here are a few AI-powered suggestions:".\n\nExample Suggestion:\nInstead of 'Responsible for a project', try 'Spearheaded a 6-month project, increasing team productivity by 15% by implementing agile methodologies.'"""
        
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt,}], model=self.groq_model)
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"Error calling Groq for suggestions: {e}")
            return "An error occurred while generating AI suggestions."

    def generate_highlighted_resume_html(self, results):
        text = results.get('raw_text', '')
        if not text: return ""
        highlights = results.get('experience_analysis', {})
        phrases_to_highlight = []
        for phrase in set(highlights.get('passive_phrases_found', [])): phrases_to_highlight.append((phrase, "highlight-red", "This phrase is passive. Try starting with an action verb."))
        for metric in set(highlights.get('quantified_metrics', [])): phrases_to_highlight.append((metric, "highlight-green", "Excellent! Quantified results are highly impactful."))
        for verb in set(highlights.get('action_verbs_found', [])): phrases_to_highlight.append((verb, "highlight-blue", "Great use of an action verb!"))
        phrases_to_highlight.sort(key=lambda x: len(x[0]), reverse=True)
        safe_text = html.escape(text)
        for phrase, class_name, title_text in phrases_to_highlight:
            safe_phrase = html.escape(phrase)
            replacement_html = f'<span class="{class_name}" title="{title_text}">{safe_phrase}</span>'
            try:
                safe_text = re.sub(r'\b' + re.escape(safe_phrase) + r'\b', replacement_html, safe_text, flags=re.IGNORECASE)
            except re.error:
                safe_text = safe_text.replace(safe_phrase, replacement_html)
        return safe_text.replace('\n', '<br>')

    def create_visualizations(self, results):
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number", value = results['overall_score'], title = {'text': "Overall ATS Score", 'font': {'size': 24}},
            gauge = {'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"}, 'bar': {'color': "#2575fc"},
                'bgcolor': "white", 'borderwidth': 2, 'bordercolor': "#ccc",
                'steps': [{'range': [0, 50], 'color': '#ea4335'}, {'range': [50, 80], 'color': '#fbbc05'}, {'range': [80, 100], 'color': '#34a853'}]} ))
        fig_gauge.update_layout(paper_bgcolor = "rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=300, margin=dict(l=20, r=20, t=50, b=20))
        categories = ['Keyword Match', 'Experience Impact', 'Listed Skills', 'Readability', 'Contact Info']
        scores = [results['keyword_score'], results['impact_score'], results['skills_score'], results['readability_score'], results['contact_score']]
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=scores, theta=categories, fill='toself', name='Score Breakdown', line=dict(color='#6a11cb')))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])), title="Score Breakdown by Category", height=400,
            paper_bgcolor = "rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        return fig_gauge, fig_radar
