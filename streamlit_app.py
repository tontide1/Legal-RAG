import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import streamlit as st

from ui_runtime import check_env_status, run_legal_qa_for_ui

st.set_page_config(
    page_title="Legal-RAG Demo",
    page_icon="⚖️",
    layout="wide",
)

st.title("⚖️ Legal-RAG: Hỏi đáp pháp luật tiếng Việt")
st.caption("Pipeline: NER -> Hybrid Retrieval (BM25 + SBERT) -> Graph Rerank -> Gemini")

env_status = check_env_status()

with st.sidebar:
    st.header("Trạng thái môi trường")
    for key, value in env_status.items():
        if value == "MISSING":
            st.error(f"{key}: {value}")
        elif value == "SET":
            st.success(f"{key}: {value}")
        else:
            st.info(f"{key}: {value}")

    st.divider()
    st.caption(
        "Giới hạn hiện tại: NER nhận diện tốt các tham chiếu dạng 'Điều <số>'. "
        "Tên luật/văn bản có thể chưa được nhận diện đầy đủ."
    )

query = st.text_area(
    "Nhập câu hỏi pháp luật của bạn:",
    placeholder="Ví dụ: Điều 33 quy định gì về quyền sở hữu?",
    height=100,
)

col1, col2, col3 = st.columns([1, 1, 3])
with col1:
    submit = st.button("Tra cứu", type="primary", use_container_width=True)

if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "last_result" not in st.session_state:
    st.session_state.last_result = None

if submit and query.strip():
    with st.spinner("Đang xử lý truy vấn..."):
        try:
            result = run_legal_qa_for_ui(query.strip())
            st.session_state.last_query = query.strip()
            st.session_state.last_result = result
        except Exception as e:
            st.error(f"Lỗi khi xử lý truy vấn: {e}")
            st.session_state.last_result = None
elif submit and not query.strip():
    st.warning("Vui lòng nhập câu hỏi trước khi tra cứu.")

result = st.session_state.last_result

if result:
    st.divider()
    st.subheader("Câu trả lời")
    if result.get("errors") and not result.get("retrieved_nodes"):
        st.warning(result.get("answer_text", "Không có câu trả lời."))
    else:
        st.info(result.get("answer_text", "Không có câu trả lời."))

    st.divider()
    st.subheader("Thực thể NER nhận diện")
    ner_entities = result.get("ner_entities", [])
    if ner_entities:
        st.write(", ".join(ner_entities))
    else:
        st.caption("Không nhận diện được thực thể nào.")

    st.divider()
    st.subheader("Căn cứ truy xuất")
    retrieved_nodes = result.get("retrieved_nodes", [])
    if retrieved_nodes:
        df_data = []
        for node in retrieved_nodes:
            df_data.append({
                "Label": node.get("label", ""),
                "Tên": node.get("name", ""),
                "Nội dung": node.get("value", ""),
                "BM25": round(node.get("bm25", 0.0), 4),
                "Cosine": round(node.get("cosine", 0.0), 4),
                "Graph Sum": round(node.get("graph_sum", 0.0), 4),
                "Final Score": round(node.get("final_score", node.get("combined_score", 0.0)), 4),
            })
        st.dataframe(df_data, use_container_width=True)
    else:
        st.caption("Không tìm thấy căn cứ pháp lý phù hợp.")

    st.divider()
    st.subheader("Citation")
    citations = result.get("citations", [])
    if citations:
        for citation in citations:
            st.markdown(f"- **{citation.get('label', '')}** ({citation.get('name', '')})")
    else:
        st.caption("Không có citation.")

    st.divider()
    with st.expander("Chẩn đoán"):
        scores = result.get("scores", {})
        timings = result.get("timings", {})
        errors = result.get("errors", [])

        if scores:
            st.write("**Scores**")
            st.json(scores)

        if timings:
            st.write("**Timings (ms)**")
            st.json(timings)

        if errors:
            st.write("**Errors**")
            for error in errors:
                st.error(error)
