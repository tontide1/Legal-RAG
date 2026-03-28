import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from NER import ner
from retrive import multi_retr

load_dotenv()

CODE_ROOT = Path(__file__).resolve().parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from pipeline_utils import get_configured_gemini_model

ANSWER_PROMPT = PromptTemplate.from_template(
    """Hãy trở thành chuyên gia tư vấn luật tại Việt Nam.
Câu hỏi của người dùng: {query}
Trả lời dựa vào thông tin sau:
{source_information}
Yêu cầu:
1. Trả lời ngắn gọn, rõ ràng
2. Nếu không có thông tin, trả lời "Tôi không tìm thấy căn cứ pháp lý cho trường hợp này" và gợi ý cách tìm kiếm ở nguồn khác
3. Kèm điều luật liên quan"""
)


def build_answer_chain():
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in .env file")

    gemini_model = get_configured_gemini_model()
    answer_model = ChatGoogleGenerativeAI(
        model=gemini_model,
        google_api_key=google_api_key,
        temperature=0.3,
        max_output_tokens=500,
    )
    return ANSWER_PROMPT | answer_model


def main():
    query = input("Nhập query của bạn: ").strip()
    if not query:
        print("Query không được để trống.")
        return

    _, _, ner_entities = ner.infer(
        query,
        model_path=str(CODE_ROOT / "NER" / "bilstm_ner.pt"),
    )
    print("\nThực thể được NER nhận diện:", ner_entities)

    if not ner_entities:
        print("\nKhông nhận diện được thực thể nào. Truy vấn toàn câu:")

    results = multi_retr.retrieve_entity(query, ner_entities if ner_entities else None)
    top_results = results[:5]
    if not top_results:
        print("Không tìm thấy căn cứ pháp lý phù hợp.")
        return

    source_info = ""
    for entity in top_results:
        source_info += f"- {entity['label']} ({entity['name']}): {entity['value']}\n"

    try:
        answer_chain = build_answer_chain()
        response = answer_chain.invoke({"query": query, "source_information": source_info})
    except Exception as exc:
        error_text = str(exc)
        if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text:
            print("Lỗi Gemini: vượt quota/rate limit (429 RESOURCE_EXHAUSTED).")
        else:
            print(f"Lỗi khi gọi Gemini: {exc}")
        print("Tôi không thể tạo câu trả lời lúc này. Vui lòng thử lại sau hoặc đổi API key.")
        return

    print("\n=== Câu trả lời từ LLM ===")
    print(response.content if hasattr(response, "content") else response)


if __name__ == "__main__":
    main()
