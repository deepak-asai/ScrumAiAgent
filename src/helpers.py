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

def is_json(string: str) -> bool:
    """
    Check if a string is valid JSON.
    Returns True if valid, False otherwise.
    """
    try:
        json.loads(string)
        return True
    except ValueError:
        return False
    
def print_ai_response(response_content: str):
    """
    Print the AI response in a formatted way.
    """
    if is_json(response_content) or response_content == "":
        # If the response is JSON or empty, we don't print it directly
        return
    print(f"\n🤖 AI: {response_content}")