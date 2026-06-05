import json
import pickle
from typing import Any

def read_json(file_path: str) -> Any:
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

def save_to_json(data: Any, file_path: str) -> None:
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)
    print(f"Data successfully saved to {file_path}")

def load_pickle(file_path: str) -> Any:
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def save_to_pickle(items_to_save: Any, file_path: str) -> None:
    with open(file_path, 'wb') as f:
        pickle.dump(items_to_save, f)
    print(f"Data successfully saved to {file_path}")
