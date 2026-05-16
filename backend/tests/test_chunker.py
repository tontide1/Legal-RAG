import sys
sys.path.insert(0, ".")
from backend.core.legal_chunker import chunk_markdown

# Test 1: Vietnamese with proper diacritics (expected: structure-aware chunking)
SAMPLE_PROPER = """
CHƯƠNG I
QUY ĐỊNH CHUNG

Điều 1. Phạm vi điều chỉnh
Bộ luật này quy định về xử phạt vi phạm hành chính trong lĩnh vực giao thông đường bộ.

Điều 2. Đối tượng áp dụng
1. Bộ luật này áp dụng với:
a) Cá nhân;
b) Tổ chức.
2. Người nước ngoài có thể bị xử phạt theo quy định này.

CHƯƠNG II
QUY ĐỊNH CỤ THỂ

MỤC 1. XỬ PHẠT VI PHẠM

Điều 3. Xử phạt vi phạm tốc độ
1. Phạt cảnh cáo đối với hành vi lần đầu.
2. Phạt tiền từ 100.000 đến 500.000 đồng.
"""

print("=== Test 1: With proper diacritics ===")
chunks = chunk_markdown(SAMPLE_PROPER, source_file="test_law.pdf")
print(f"Total chunks: {len(chunks)}")
for i, c in enumerate(chunks, 1):
    meta = c.metadata
    t = meta.get("type")
    d = meta.get("dieu", "-")
    ch = str(meta.get("chuong", "-"))[:25]
    mu = str(meta.get("muc", "-"))[:20]
    text_preview = c.text[:100].replace("\n", " ")
    print(f"  [{i}] type={t} | dieu={d} | chuong={ch} | muc={mu}")
    print(f"       {text_preview}")

# Test 2: No structure (expected: paragraph fallback)
SAMPLE_NOSTRUCTURE = """
Đây là văn bản không có cấu trúc pháp lý rõ ràng.

Chỉ có các đoạn văn thông thường.

Không có điều khoản gì cả.
"""

print("\n=== Test 2: Fallback paragraph mode ===")
chunks2 = chunk_markdown(SAMPLE_NOSTRUCTURE, source_file="freetext.pdf")
print(f"Total chunks: {len(chunks2)}")
for i, c in enumerate(chunks2, 1):
    print(f"  [{i}] type={c.metadata.get('type')} | text[:60]: {c.text[:60].replace(chr(10),' ')}")
