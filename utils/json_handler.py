"""JSON serialization and validation utilities."""

import json
from pathlib import Path
from typing import Dict, Any, Union


class JSONHandler:
    @staticmethod
    def load(file_path: Union[str, Path]) -> Dict[str, Any]:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def save(data: Dict[str, Any], file_path: Union[str, Path], indent: int = 2) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
