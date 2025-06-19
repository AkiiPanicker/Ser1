import spacy
from google.colab import files
import PyPDF2
import google.generativeai as genai
from google.colab import userdata

# Ensure you have authenticated with Google AI
genai.configure(api_key=userdata.get('GOOGLE_API_KEY'))

# Initialize the Gemini model
model = genai.GenerativeModel('gemini-2.0-flash',
                              system_instruction="Avoid using markdown formatting and acronyms in your responses. Provide clear, simple language and keep your responses concise, as if you're speaking to someone. The results should be suitable for text-to-speech conversion.")

# Load the English language model for spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    nlp = spacy.load("en_core_web_sm")

def extract_info_with_gemini(text, model):
    """
    Extracts specific information (skills, name, education, experience)
    using the Gemini model.

    Args:
        text (str): The input text (job description or resume content).
        model: The initialized Gemini model.

    Returns:
        str: A formatted string containing the extracted information.
    """
    prompt = f"""
Extract the following information from the text below:
- Full Name
- Education (degrees, institutions, years)
- Previous Experience (job titles, companies, dates, responsibilities)
- Useful Skills (technical skills, soft skills)

Present the extracted information in a clear and organized format, suitable for understanding key details for an interview. Use bullet points or numbered lists for clarity.

Text:
{text}
"""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred during information extraction: {e}"

print("Upload your job description or resume file (e.g., .txt or .pdf):")
uploaded = files.upload()

for filename in uploaded.keys():
    print(f'User uploaded file "{filename}"')
    file_content = ""

    # Check if the uploaded file is a PDF
    if filename.lower().endswith('.pdf'):
        try:
            with open(filename, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    file_content += page.extract_text()
        except Exception as e:
            print(f"Error reading PDF file {filename}: {e}")
            continue
    else:
        # Assume it's a text file
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                file_content = f.read()
        except UnicodeDecodeError:
            print(f"Could not decode text file {filename} with UTF-8. Please ensure it's a plain text file.")
            continue

    if file_content: # Process only if content was successfully extracted
        print("\n--- Extracted Information for Interview ---")
        extracted_info_text = extract_info_with_gemini(file_content, model) # Pass the model
        print(extracted_info_text)
        print("-------------------------------------------")
