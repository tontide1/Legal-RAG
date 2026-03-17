from flask import Flask, request, jsonify
import streamlit as st
from file_handlers.csv_handler import handle_csv
from file_handlers.json_handler import handle_json
from file_handlers.pdf_handler import handle_pdf
from file_handlers.docx_handler import handle_docx
from chatbot import Chatbot
from chroma_utils import ChromaUtils

app = Flask(__name__)

# Initialize Chroma vector store
chroma_utils = ChromaUtils()

# Initialize chatbot
chatbot = Chatbot(chroma_utils)

def main():
    st.title("Drag and Drop RAG")
    
    # File upload section
    uploaded_file = st.file_uploader("Upload a file (CSV, JSON, PDF, DOCX)", type=["csv", "json", "pdf", "docx"])
    
    if uploaded_file is not None:
        if uploaded_file.type == "text/csv":
            data = handle_csv(uploaded_file)
        elif uploaded_file.type == "application/json":
            data = handle_json(uploaded_file)
        elif uploaded_file.type == "application/pdf":
            data = handle_pdf(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            data = handle_docx(uploaded_file)
        
        # Store data in Chroma
        chroma_utils.store_data(data)
        st.success("File uploaded and data stored successfully!")
    
    # Chatbot interaction
    user_input = st.text_input("Ask a question:")
    if user_input:
        response = chatbot.get_response(user_input)
        st.write(response)

if __name__ == "__main__":
    main()