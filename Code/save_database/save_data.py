from neo4j import GraphDatabase
import json
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent

# Kết nối tới Neo4j
uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
user = os.getenv("NEO4J_USER", "neo4j")
password = os.getenv("NEO4J_PASSWORD")
if not password:
    raise ValueError("NEO4J_PASSWORD not found in environment variables. Please set it in .env file")
driver = GraphDatabase.driver(uri, auth=(user, password))

# Đọc dữ liệu từ 4 file JSON
file_paths = [
    PROJECT_ROOT / "dataset" / "Luật" / "Luật Xử lý vi phạm hành chính" / "merged_data.json",
    PROJECT_ROOT / "dataset" / "Luật" / "Luật sửa đổi VPHC" / "sdbs.json",
    PROJECT_ROOT / "dataset" / "Luật" / "Luật Hải quan" / "luathaiquan.json",
    PROJECT_ROOT / "dataset" / "Nghị_Định" / "nghidinh.json"
]

combined_entities = []
combined_relationships = []

for path in file_paths:
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        combined_entities.extend(data.get("thực_thể", []))
        combined_relationships.extend(data.get("mối_quan_hệ", []))

# Hàm xóa toàn bộ dữ liệu trong Neo4j
def clear_database(tx):
    tx.run("MATCH (n) DETACH DELETE n")

# Hàm tạo node trong Neo4j
def create_node(tx, entity):
    if "Tên" not in entity:
        print(f"Bỏ qua thực thể thiếu 'Tên': {entity}")
        return

    label = entity.get("Label", "Unknown").replace("`", "")  # tránh lỗi injection
    ten = entity["Tên"]
    properties = {k: v for k, v in entity.items() if k != "Tên" and v}

    query = (
        f"MERGE (n:`{label}` {{ten: $ten}}) "
        "SET n += $properties"
    )

    tx.run(query, ten=ten, properties=properties)

# Hàm tạo relationship trong Neo4j
def create_relationship(tx, source_name, rel_type, target_name):
    rel_type_safe = rel_type.replace("`", "")

    query = (
        "MATCH (source {ten: $source_name}), (target {ten: $target_name}) "
        f"MERGE (source)-[r:`{rel_type_safe}`]->(target)"
    )

    tx.run(query, source_name=source_name, target_name=target_name)

# Thực thi
with driver.session() as session:
    print("Đang xóa dữ liệu cũ...")
    session.execute_write(clear_database)

    print("Bắt đầu tạo các node...")
    for entity in combined_entities:
        session.execute_write(create_node, entity)

    print("Hoàn thành tạo node. Bắt đầu tạo relationship...")
    for source_name, rel_type, target_name in combined_relationships:
        session.execute_write(create_relationship, source_name, rel_type, target_name)

# Đóng kết nối
driver.close()
print("Hoàn tất quá trình lưu dữ liệu lên Neo4j.")