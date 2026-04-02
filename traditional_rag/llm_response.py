import os
import argparse
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_google_genai import GoogleGenerativeAI

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser



# Load environment variables from .env file
load_dotenv()

# Constants
EMBEDDING_MODEL = "keepitreal/vietnamese-sbert"
VECTOR_STORE_PATH = "traditional_rag/vector_store/faiss_index_traditional"
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def format_docs(docs):
    """Helper function to format retrieved documents for the prompt."""
    return "\n\n".join(doc.page_content for doc in docs)


def make_response(question: str):
    """
    Main function for the traditional RAG pipeline.
    """
    if not GEMINI_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")

    if not os.path.exists(VECTOR_STORE_PATH):
        print(f"Vector store not found at '{VECTOR_STORE_PATH}'.")
        print("Please run 'python src/traditional_rag/embedding.py' first to create it.")
        return

    # 1. Load Vector Store
    print("Loading vector store...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5}) # Retrieve top 5 chunks

    # 2. Define LLM and Prompt
    llm = GoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=GEMINI_API_KEY)

    template = """
    Bạn là một trợ lý AI chuyên về pháp luật Việt Nam.
    Sử dụng các thông tin được cung cấp dưới đây để trả lời câu hỏi của người dùng.
    Nếu bạn không biết câu trả lời dựa trên thông tin này, hãy nói rằng bạn không biết. Đừng cố bịa ra câu trả lời.
    Hãy trả lời bằng tiếng Việt.

    Ngữ cảnh:
    {context}

    Câu hỏi:
    {question}

    Câu trả lời:
    """
    prompt = PromptTemplate.from_template(template)

    # 3. Create RAG Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 4. Invoke Chain and Print Answer
    print("\nĐang tạo câu trả lời...")
    answer = rag_chain.invoke(question)
    print("\n--- Câu trả lời ---")
    print(answer)
    print("-------------------\n")