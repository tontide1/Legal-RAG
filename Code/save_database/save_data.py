import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from pipeline_utils import make_node_id

REPO_ROOT = Path(__file__).resolve().parents[2]
FILE_PATHS = [
    REPO_ROOT / "dataset" / "Luật" / "Luật Xử lý vi phạm hành chính" / "merged_data.json",
    REPO_ROOT / "dataset" / "Luật" / "Luật sửa đổi VPHC" / "sdbs.json",
    REPO_ROOT / "dataset" / "Luật" / "Luật Hải quan" / "luathaiquan.json",
    REPO_ROOT / "dataset" / "Nghị_Định" / "nghidinh.json",
]


def get_driver():
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    if not password:
        raise ValueError("NEO4J_PASSWORD not found in environment variables. Please set it in .env file")
    return GraphDatabase.driver(uri, auth=(user, password))


def load_dataset_payloads(file_paths):
    payloads = []
    for path in file_paths:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        payloads.append((data.get("thực_thể", []), data.get("mối_quan_hệ", [])))
    return payloads


def clear_database(tx):
    tx.run("MATCH (n:LegalRAG) DETACH DELETE n")


def create_node(tx, entity):
    if "Tên" not in entity:
        print(f"Bỏ qua thực thể thiếu 'Tên': {entity}")
        return

    label = entity.get("Label", "Unknown").replace("`", "")
    ten = entity["Tên"]
    node_id = make_node_id(label, ten)
    properties = {k: v for k, v in entity.items() if k != "Tên" and v is not None}
    properties["Value"] = entity.get("Value") or ""
    properties["node_id"] = node_id

    query = (
        f"MERGE (n:LegalRAG:`{label}` {{node_id: $node_id}}) "
        "SET n.ten = $ten "
        "SET n += $properties"
    )
    tx.run(query, node_id=node_id, ten=ten, properties=properties)


def create_relationship(tx, source_node_id, rel_type, target_node_id):
    rel_type_safe = rel_type.replace("`", "")
    query = (
        "MATCH (source:LegalRAG {node_id: $source_node_id}), "
        "(target:LegalRAG {node_id: $target_node_id}) "
        f"MERGE (source)-[r:`{rel_type_safe}`]->(target)"
    )
    tx.run(query, source_node_id=source_node_id, target_node_id=target_node_id)


def save_dataset(session, entities, relationships):
    entity_index = {}
    for entity in entities:
        if "Tên" not in entity:
            continue
        entity_index[entity["Tên"]] = entity
        session.execute_write(create_node, entity)

    for source_name, rel_type, target_name in relationships:
        source_entity = entity_index.get(source_name)
        target_entity = entity_index.get(target_name)
        if not source_entity or not target_entity:
            print(f"Bỏ qua relationship thiếu node nguồn/đích: {(source_name, rel_type, target_name)}")
            continue

        source_id = make_node_id(source_entity.get("Label", "Unknown").replace("`", ""), source_name)
        target_id = make_node_id(target_entity.get("Label", "Unknown").replace("`", ""), target_name)
        session.execute_write(create_relationship, source_id, rel_type, target_id)


def main():
    dataset_payloads = load_dataset_payloads(FILE_PATHS)
    driver = get_driver()
    try:
        with driver.session() as session:
            print("Đang xóa dữ liệu LegalRAG cũ...")
            session.execute_write(clear_database)

            print("Bắt đầu tạo các node và relationship...")
            for entities, relationships in dataset_payloads:
                save_dataset(session, entities, relationships)
    finally:
        driver.close()

    print("Hoàn tất quá trình lưu dữ liệu lên Neo4j.")


if __name__ == "__main__":
    main()
