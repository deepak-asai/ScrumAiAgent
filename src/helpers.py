import json
from typing import cast
from models import SystemCommand

def deserialize_system_command(json_str: str) -> SystemCommand:
    """
    Deserialize a JSON string into a SystemCommand TypedDict.
    Raises ValueError if the JSON is invalid or missing required keys.
    """
    data = json.loads(json_str)
    if not isinstance(data, dict) or "command" not in data:
        raise ValueError("Invalid SystemCommand format")
    # If 'args' is missing, add it as None or an empty dict (as you prefer)
    if "args" not in data:
        data["args"] = None  # or data["args"] = {}
    return cast(SystemCommand, data)