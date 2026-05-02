import json
from pathlib import Path
from typing import Any, Dict

from src.utils.paths import get_project_layout

class StateManager:
    def __init__(self, state_dir: str | Path = "outputs/checkpoints"):
        self.state_dir = get_project_layout().resolve(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_path(self, task_name: str) -> Path:
        return self.state_dir / f"{task_name}.json"
        
    def save_state(self, task_name: str, state: Dict[str, Any]):
        """Save a state dictionary atomically."""
        path = self._get_path(task_name)
        tmp_path = path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        tmp_path.replace(path)
        
    def load_state(self, task_name: str) -> Dict[str, Any] | None:
        """Load a state dictionary. Returns None if it doesn't exist."""
        path = self._get_path(task_name)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
            
    def mark_completed(self, task_name: str, result: Dict[str, Any] = None):
        """Mark a task as completed and optionally store its result."""
        state = self.load_state(task_name) or {}
        state["status"] = "COMPLETED"
        if result:
            state["result"] = result
        self.save_state(task_name, state)
        
    def is_completed(self, task_name: str) -> bool:
        """Check if a task was successfully completed."""
        state = self.load_state(task_name)
        return state is not None and state.get("status") == "COMPLETED"
