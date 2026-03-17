from typing import Any, Dict, List
import json

def load_json(file_path: str) -> Dict[str, Any]:
    """Load a JSON file and return its content as a dictionary."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def extract_data(json_content: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    """Extract specific keys from the JSON content."""
    return {key: json_content.get(key) for key in keys if key in json_content}

def save_json(data: Dict[str, Any], file_path: str) -> None:
    """Save a dictionary as a JSON file."""
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)