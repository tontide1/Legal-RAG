from PyPDF2 import PdfReader

def read_pdf(file_path):
    """Reads a PDF file and extracts text from it."""
    text = ""
    try:
        with open(file_path, "rb") as file:
            reader = PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error reading PDF file: {e}")
    return text

def extract_pdf_data(file_path):
    """Extracts relevant data from a PDF file."""
    pdf_text = read_pdf(file_path)
    # Further processing can be done here to extract specific data
    return pdf_text.strip() if pdf_text else None