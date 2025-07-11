import os
import re
import nltk
import spacy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
import textstat
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Document processing imports
import PyPDF2
from docx import Document
from google.colab import files
import io

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

class AdvancedATSCalculator:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

        # Comprehensive ATS-friendly keywords database
        self.technical_skills = {
            'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust', 'scala', 'kotlin', 'swift', 'r', 'matlab', 'sql', 'html', 'css', 'typescript'],
            'frameworks': ['react', 'angular', 'vue', 'node.js', 'django', 'flask', 'spring', 'laravel', 'express', 'bootstrap', 'jquery', 'tensorflow', 'pytorch', 'keras'],
            'databases': ['mysql', 'postgresql', 'mongodb', 'redis', 'cassandra', 'oracle', 'sqlite', 'dynamodb', 'elasticsearch'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'terraform', 'ansible'],
            'tools': ['git', 'jira', 'confluence', 'slack', 'trello', 'postman', 'swagger', 'tableau', 'power bi'],
            'methodologies': ['agile', 'scrum', 'kanban', 'devops', 'ci/cd', 'tdd', 'bdd']
        }

        self.soft_skills = [
            'leadership', 'communication', 'teamwork', 'problem-solving', 'analytical',
            'creative', 'adaptable', 'organized', 'detail-oriented', 'time management',
            'critical thinking', 'collaboration', 'innovation', 'strategic thinking',
            'project management', 'mentoring', 'coaching', 'negotiation'
        ]

        self.industry_keywords = {
            'technology': ['software', 'development', 'programming', 'coding', 'algorithm', 'architecture', 'api', 'microservices'],
            'data_science': ['machine learning', 'deep learning', 'data analysis', 'statistics', 'visualization', 'big data', 'analytics'],
            'marketing': ['seo', 'sem', 'social media', 'content marketing', 'digital marketing', 'brand management', 'campaign'],
            'finance': ['financial analysis', 'accounting', 'budgeting', 'forecasting', 'risk management', 'compliance'],
            'hr': ['recruitment', 'talent acquisition', 'employee relations', 'performance management', 'compensation']
        }

        # ATS-unfriendly elements
        self.ats_unfriendly = [
            'graphics', 'images', 'tables', 'columns', 'headers', 'footers',
            'text boxes', 'special characters', 'fancy fonts', 'colors'
        ]

    def extract_text_from_pdf(self, file_path):
        """Extract text from PDF file"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                return text
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return ""

    def extract_text_from_docx(self, file_path):
        """Extract text from DOCX file"""
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            print(f"Error reading DOCX: {e}")
            return ""

    def extract_text_from_txt(self, file_path):
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"Error reading TXT: {e}")
            return ""

    def preprocess_text(self, text):
        """Clean and preprocess text"""
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s\-\.\@\+\#]', ' ', text)
        return text.strip()

    def extract_contact_info(self, text):
        """Extract contact information"""
        contact_info = {}

        # Email extraction
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        contact_info['emails'] = emails

        # Phone number extraction
        phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})'
        phones = re.findall(phone_pattern, text)
        contact_info['phones'] = ['-'.join(filter(None, phone)) for phone in phones]

        # LinkedIn profile extraction
        linkedin_pattern = r'linkedin\.com/in/[\w\-]+'
        linkedin = re.findall(linkedin_pattern, text)
        contact_info['linkedin'] = linkedin

        return contact_info

    def extract_skills(self, text):
        """Extract technical and soft skills"""
        text_lower = text.lower()
        found_skills = {
            'technical': {},
            'soft': [],
            'total_technical': 0,
            'total_soft': 0
        }

        # Extract technical skills by category
        for category, skills in self.technical_skills.items():
            found_in_category = []
            for skill in skills:
                if skill in text_lower:
                    found_in_category.append(skill)
            found_skills['technical'][category] = found_in_category
            found_skills['total_technical'] += len(found_in_category)

        # Extract soft skills
        for skill in self.soft_skills:
            if skill in text_lower:
                found_skills['soft'].append(skill)
        found_skills['total_soft'] = len(found_skills['soft'])

        return found_skills

    def analyze_structure(self, text):
        """Analyze resume structure and format"""
        structure_score = 0
        feedback = []

        # Check for common sections
        sections = {
            'summary/objective': ['summary', 'objective', 'profile'],
            'experience': ['experience', 'work history', 'employment'],
            'education': ['education', 'qualification', 'degree'],
            'skills': ['skills', 'technical skills', 'competencies'],
            'projects': ['projects', 'portfolio'],
            'certifications': ['certification', 'certificate', 'licensed']
        }

        found_sections = 0
        text_lower = text.lower()

        for section_name, keywords in sections.items():
            if any(keyword in text_lower for keyword in keywords):
                found_sections += 1
                structure_score += 15
            else:
                feedback.append(f"Missing {section_name} section")

        # Check for consistent formatting indicators
        if len(re.findall(r'\n\s*•', text)) > 3:  # Bullet points
            structure_score += 10
            feedback.append("Good use of bullet points")

        # Check for dates (experience timeline)
        date_patterns = [
            r'\d{4}\s*-\s*\d{4}',
            r'\d{4}\s*-\s*present',
            r'\w+\s+\d{4}\s*-\s*\w+\s+\d{4}'
        ]
        dates_found = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in date_patterns)
        if dates_found >= 2:
            structure_score += 15
            feedback.append("Good chronological structure")

        return min(structure_score, 100), feedback, found_sections

    def calculate_keyword_density(self, text, job_description=""):
        """Calculate keyword density and relevance"""
        if not job_description:
            # Use general keywords if no job description provided
            all_keywords = []
            for skills in self.technical_skills.values():
                all_keywords.extend(skills)
            all_keywords.extend(self.soft_skills)
            job_keywords = all_keywords
        else:
            # Extract keywords from job description
            job_doc = nlp(job_description.lower())
            job_keywords = [token.lemma_ for token in job_doc
                          if not token.is_stop and not token.is_punct and len(token.text) > 2]

        text_lower = text.lower()
        keyword_matches = 0
        total_keywords = len(job_keywords)
        matched_keywords = []

        for keyword in job_keywords:
            if keyword.lower() in text_lower:
                keyword_matches += 1
                matched_keywords.append(keyword)

        density_score = (keyword_matches / max(total_keywords, 1)) * 100
        return min(density_score, 100), matched_keywords

    def analyze_readability(self, text):
        """Analyze text readability and complexity"""
        try:
            # Flesch Reading Ease
            flesch_score = textstat.flesch_reading_ease(text)

            # Flesch-Kincaid Grade Level
            fk_grade = textstat.flesch_kincaid_grade(text)

            # Average sentence length
            sentences = sent_tokenize(text)
            words = word_tokenize(text)
            avg_sentence_length = len(words) / max(len(sentences), 1)

            # Readability score (optimal range: 60-70 for professional documents)
            if 60 <= flesch_score <= 70:
                readability_score = 100
            elif 50 <= flesch_score < 60 or 70 < flesch_score <= 80:
                readability_score = 80
            else:
                readability_score = max(0, 100 - abs(65 - flesch_score) * 2)

            return {
                'score': min(readability_score, 100),
                'flesch_score': flesch_score,
                'grade_level': fk_grade,
                'avg_sentence_length': avg_sentence_length
            }
        except:
            return {'score': 50, 'flesch_score': 50, 'grade_level': 10, 'avg_sentence_length': 15}

    def check_ats_compatibility(self, text, filename=""):
        """Check ATS compatibility"""
        compatibility_score = 100
        issues = []

        # File format check
        if filename.lower().endswith('.pdf'):
            compatibility_score -= 0  # PDF is generally good
        elif filename.lower().endswith('.docx'):
            compatibility_score -= 5  # DOCX might have formatting issues
        elif filename.lower().endswith('.doc'):
            compatibility_score -= 15  # Older format
        else:
            compatibility_score -= 10

        # Check for potential formatting issues
        if len(re.findall(r'[^\x00-\x7F]', text)) > 10:  # Non-ASCII characters
            compatibility_score -= 10
            issues.append("Contains special characters that might not parse correctly")

        # Check text length
        word_count = len(text.split())
        if word_count < 200:
            compatibility_score -= 20
            issues.append("Resume might be too short")
        elif word_count > 1000:
            compatibility_score -= 10
            issues.append("Resume might be too long for optimal ATS parsing")

        # Check for common ATS-friendly elements
        if not re.search(r'\b(email|@)\b', text.lower()):
            compatibility_score -= 15
            issues.append("Email address not clearly identifiable")

        if not re.search(r'\b(phone|tel|\d{3}[-.\s]?\d{3}[-.\s]?\d{4})\b', text.lower()):
            compatibility_score -= 10
            issues.append("Phone number not clearly identifiable")

        return max(compatibility_score, 0), issues

    def calculate_overall_score(self, text, filename="", job_description=""):
        """Calculate comprehensive ATS score"""
        results = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'filename': filename,
            'word_count': len(text.split()),
            'character_count': len(text)
        }

        # 1. Contact Information (10%)
        contact_info = self.extract_contact_info(text)
        contact_score = 0
        if contact_info['emails']:
            contact_score += 40
        if contact_info['phones']:
            contact_score += 30
        if contact_info['linkedin']:
            contact_score += 30
        results['contact_score'] = min(contact_score, 100)
        results['contact_info'] = contact_info

        # 2. Skills Analysis (25%)
        skills = self.extract_skills(text)
        skills_score = min(skills['total_technical'] * 3 + skills['total_soft'] * 2, 100)
        results['skills_score'] = skills_score
        results['skills'] = skills

        # 3. Structure and Format (20%)
        structure_score, structure_feedback, sections_found = self.analyze_structure(text)
        results['structure_score'] = structure_score
        results['structure_feedback'] = structure_feedback
        results['sections_found'] = sections_found

        # 4. Keyword Density (20%)
        keyword_score, matched_keywords = self.calculate_keyword_density(text, job_description)
        results['keyword_score'] = keyword_score
        results['matched_keywords'] = matched_keywords

        # 5. ATS Compatibility (15%)
        ats_score, ats_issues = self.check_ats_compatibility(text, filename)
        results['ats_score'] = ats_score
        results['ats_issues'] = ats_issues

        # 6. Readability (10%)
        readability = self.analyze_readability(text)
        results['readability_score'] = readability['score']
        results['readability_details'] = readability

        # Calculate weighted overall score
        overall_score = (
            results['contact_score'] * 0.10 +
            results['skills_score'] * 0.25 +
            results['structure_score'] * 0.20 +
            results['keyword_score'] * 0.20 +
            results['ats_score'] * 0.15 +
            results['readability_score'] * 0.10
        )

        results['overall_score'] = round(overall_score, 2)
        results['grade'] = self.get_grade(overall_score)

        return results

    def get_grade(self, score):
        """Convert score to letter grade"""
        if score >= 90:
            return "A+ (Excellent)"
        elif score >= 80:
            return "A (Very Good)"
        elif score >= 70:
            return "B (Good)"
        elif score >= 60:
            return "C (Fair)"
        elif score >= 50:
            return "D (Poor)"
        else:
            return "F (Needs Major Improvement)"

    def generate_recommendations(self, results):
        """Generate personalized recommendations"""
        recommendations = []

        if results['contact_score'] < 80:
            recommendations.append("✅ Add complete contact information (email, phone, LinkedIn)")

        if results['skills_score'] < 70:
            recommendations.append("✅ Include more relevant technical and soft skills")

        if results['structure_score'] < 70:
            recommendations.append("✅ Improve resume structure with clear sections")

        if results['keyword_score'] < 60:
            recommendations.append("✅ Include more industry-relevant keywords")

        if results['ats_score'] < 80:
            recommendations.append("✅ Improve ATS compatibility by using standard formatting")

        if results['readability_score'] < 70:
            recommendations.append("✅ Improve readability with shorter sentences and clearer language")

        if results['word_count'] < 200:
            recommendations.append("✅ Expand content - resume appears too brief")
        elif results['word_count'] > 1000:
            recommendations.append("✅ Consider condensing content for better readability")

        return recommendations

    def create_visualizations(self, results):
        """Create comprehensive visualizations"""
        # 1. Overall Score Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = results['overall_score'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Overall ATS Score"},
            delta = {'reference': 80},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig_gauge.update_layout(height=400, title_text="ATS Score Dashboard")

        # 2. Category Breakdown
        categories = ['Contact Info', 'Skills', 'Structure', 'Keywords', 'ATS Compatibility', 'Readability']
        scores = [
            results['contact_score'],
            results['skills_score'],
            results['structure_score'],
            results['keyword_score'],
            results['ats_score'],
            results['readability_score']
        ]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=scores,
            theta=categories,
            fill='toself',
            name='ATS Scores'
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100])
            ),
            showlegend=False,
            title="ATS Score Breakdown by Category"
        )

        # 3. Skills Distribution
        if results['skills']['total_technical'] > 0:
            tech_skills_data = []
            for category, skills in results['skills']['technical'].items():
                if skills:
                    tech_skills_data.extend([(skill, category) for skill in skills])

            if tech_skills_data:
                skills_df = pd.DataFrame(tech_skills_data, columns=['Skill', 'Category'])
                fig_skills = px.sunburst(
                    skills_df,
                    path=['Category', 'Skill'],
                    title='Technical Skills Distribution'
                )

        return fig_gauge, fig_radar

    def generate_report(self, results):
        """Generate comprehensive ATS report"""
        report = f"""
# 📊 ATS RESUME ANALYSIS REPORT
**Generated on:** {results['timestamp']}
**File:** {results['filename']}

## 🎯 OVERALL SCORE: {results['overall_score']}/100
**Grade:** {results['grade']}

---

## 📈 SCORE BREAKDOWN

### Contact Information: {results['contact_score']}/100
- Emails found: {len(results['contact_info']['emails'])}
- Phone numbers found: {len(results['contact_info']['phones'])}
- LinkedIn profiles: {len(results['contact_info']['linkedin'])}

### Skills Analysis: {results['skills_score']}/100
- Technical skills: {results['skills']['total_technical']}
- Soft skills: {results['skills']['total_soft']}

### Structure & Format: {results['structure_score']}/100
- Resume sections found: {results['sections_found']}/6

### Keyword Optimization: {results['keyword_score']}/100
- Relevant keywords matched: {len(results['matched_keywords'])}

### ATS Compatibility: {results['ats_score']}/100
- Compatibility issues: {len(results['ats_issues'])}

### Readability: {results['readability_score']}/100
- Flesch Reading Score: {results['readability_details']['flesch_score']:.1f}
- Grade Level: {results['readability_details']['grade_level']:.1f}

---

## 📋 DOCUMENT STATISTICS
- Word Count: {results['word_count']}
- Character Count: {results['character_count']}

---

## 🔍 DETAILED FINDINGS

### Technical Skills Found:
"""

        for category, skills in results['skills']['technical'].items():
            if skills:
                report += f"**{category.title()}:** {', '.join(skills)}\n"

        report += f"\n### Soft Skills Found:\n{', '.join(results['skills']['soft'])}\n"

        if results['ats_issues']:
            report += f"\n### ⚠️ ATS Issues:\n"
            for issue in results['ats_issues']:
                report += f"- {issue}\n"

        if results['structure_feedback']:
            report += f"\n### 📝 Structure Feedback:\n"
            for feedback in results['structure_feedback']:
                report += f"- {feedback}\n"

        return report

def main():
    """Main function to run the ATS calculator"""
    print("🚀 Advanced ATS Resume Score Calculator")
    print("=" * 50)

    # Initialize calculator
    calculator = AdvancedATSCalculator()

    # Upload file
    print("📁 Please upload your resume file (PDF, DOCX, or TXT)")
    uploaded = files.upload()

    if not uploaded:
        print("No file uploaded. Please try again.")
        return

    filename = list(uploaded.keys())[0]
    print(f"✅ File uploaded: {filename}")

    # Extract text based on file type
    if filename.lower().endswith('.pdf'):
        text = calculator.extract_text_from_pdf(filename)
    elif filename.lower().endswith('.docx'):
        text = calculator.extract_text_from_docx(filename)
    elif filename.lower().endswith('.txt'):
        text = calculator.extract_text_from_txt(filename)
    else:
        print("❌ Unsupported file format. Please upload PDF, DOCX, or TXT file.")
        return

    if not text.strip():
        print("❌ Could not extract text from the file. Please check the file format.")
        return

    print(f"📄 Text extracted successfully! ({len(text)} characters)")

    # Optional: Job description input
    job_desc = input("\n📝 Enter job description for keyword matching (optional, press Enter to skip): ")

    # Calculate ATS score
    print("\n🔄 Analyzing resume...")
    results = calculator.calculate_overall_score(text, filename, job_desc)

    # Generate recommendations
    recommendations = calculator.generate_recommendations(results)

    # Display results
    print("\n" + "="*60)
    print(f"🎯 OVERALL ATS SCORE: {results['overall_score']}/100")
    print(f"📊 GRADE: {results['grade']}")
    print("="*60)

    # Display score breakdown
    print(f"\n📈 SCORE BREAKDOWN:")
    print(f"   Contact Information: {results['contact_score']}/100")
    print(f"   Skills Analysis:     {results['skills_score']}/100")
    print(f"   Structure & Format:  {results['structure_score']}/100")
    print(f"   Keyword Optimization: {results['keyword_score']}/100")
    print(f"   ATS Compatibility:   {results['ats_score']}/100")
    print(f"   Readability:         {results['readability_score']}/100")

    # Display recommendations
    if recommendations:
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in recommendations:
            print(f"   {rec}")

    # Create visualizations
    print("\n📊 Generating visualizations...")
    fig_gauge, fig_radar = calculator.create_visualizations(results)

    # Display charts
    fig_gauge.show()
    fig_radar.show()

    # Generate and display full report
    report = calculator.generate_report(results)
    print("\n📋 DETAILED REPORT:")
    print(report)

    # Save results to file
    save_option = input("\n💾 Save detailed report to file? (y/n): ")
    if save_option.lower() == 'y':
        report_filename = f"ATS_Report_{filename.split('.')[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"✅ Report saved as: {report_filename}")
        files.download(report_filename)

    print("\n🎉 Analysis complete! Thank you for using the ATS Calculator.")

# Run the application
if __name__ == "__main__":
    main()
