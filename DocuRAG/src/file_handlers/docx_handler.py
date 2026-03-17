from docx import Document

def read_docx(file_path):
    """
    Reads the content of a DOCX file and returns it as a string.
    
    Args:
        file_path (str): The path to the DOCX file.
        
    Returns:
        str: The text content of the DOCX file.
    """
    doc = Document(file_path)
    text = []
    for paragraph in doc.paragraphs:
        text.append(paragraph.text)
    return '\n'.join(text)

def extract_text_from_docx(file_path):
    """
    Extracts text from a DOCX file for further processing.
    
    Args:
        file_path (str): The path to the DOCX file.
        
    Returns:
        str: The extracted text from the DOCX file.
    """
    return read_docx(file_path)