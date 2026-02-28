import argparse
import inspect
import os
import subprocess
import sys
import json
from typing import Any, Callable, get_type_hints

from openai import OpenAI

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL",
                     default="https://openrouter.ai/api/v1")

TOOLS: dict[str, dict[str, Any]] = {}

PYTHON_TO_JSON_TYPES = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def build_tool_definition(func: Callable) -> dict[str, Any]:
    """Build a tool definition from a function's type hints and docstring"""
    hints = get_type_hints(func)
    hints.pop("return", None)
    sig = inspect.signature(func)
    properties = {}
    for param_name, param_type in hints.items():
        json_type = PYTHON_TO_JSON_TYPES.get(param_type, "string")
        # use inline param docs if present (e.g. ":param foo: description")
        param_doc = ""
        if func.__doc__:
            for line in func.__doc__.splitlines():
                if f":param {param_name}:" in line:
                    param_doc = line.split(f":param {param_name}")[1].strip()
                    break
        properties[param_name] = {"type": json_type, "description": param_doc}

    required = [
        name for name, param in sig.parameters.items() if param.default is inspect.Parameter.empty
    ]

    # use first line of docstring as the tool description
    description = ""
    if func.__doc__:
        description = func.__doc__.strip().splitlines()[0].strip()

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            },
        }
    }


def tool(func: Callable) -> Callable:
    TOOLS[func.__name__] = {
        "definition": build_tool_definition(func),
        "handler": func
    }
    return func


@tool
def Read(file_path: str) -> str:
    """Read and return the contents of a file. 
    :param file_path: The path to the file to read.
    """
    try:
        with open(file_path, 'r') as f:
            return f.read()

    except Exception as e:
        error_msg = f"Error reading file: {e}"
        print(error_msg, file=sys.stderr)
        return error_msg


@tool
def Write(file_path: str, content: str) -> str:
    """Write content to a file.
    :param file_path: The path of the file to write to.
    :param content: The content to write to the file.
    """
    try:
        with open(file_path, "w") as f:
            f.write(content)
        return f"Successfully wrote to {file_path}"
    except Exception as e:
        error_msg = f"Error writing to file: {e}"
        print(error_msg, file=sys.stderr)
        return error_msg
@tool
def Bash(command: str) -> str:
    """Execute a shell command
    :param command: The command to execute
    """
    result = subprocess.run(command.split(),capture_output=True, text=True)
    if result.stderr:
        return result.stderr
    return result.stdout

def execute_tool_call(tool_call) -> dict[str, Any]:
    """Execute a tool call and return the result as a message."""
    if tool_call.type != "function":
        return {
            "role":"tool",
            "tool_call_id": tool_call.id,
            "content":f"Unknown tool call type: {tool_call.type}"
        }

    function_name = tool_call.function.name
    function_args = json.loads(tool_call.function.arguments)

    if function_name not in TOOLS:
        return {
            "role":"tool",
            "tool_call_id": tool_call.id,
            "content":f"Unknown function: {function_name}"
        }

    result = TOOLS[function_name]["handler"](**function_args)
    return {
        "role":"tool",
        "tool_call_id": tool_call.id,
        "content": result
    }

def run_conversation(client: OpenAI, initial_prompt: str) -> None:
    """Run the conversation loop with the AI model."""
    conversation_log = [{"role":"user", "content":initial_prompt}]
    tool_definitions = [t["definition"] for t in TOOLS.values()]

    while True:
        chat = client.chat.completions.create(
            model="anthropic/claude-haiku-4.5",
            messages=conversation_log,
            tools=tool_definitions,
        )

        if not chat.choices:
            raise RuntimeError("No choices in response")

        response = chat.choices[0].message
        conversation_log.append(response)

        if not response.tool_calls:
            print(response.content)
            break

        for tool_call in response.tool_calls:
            tool_result = execute_tool_call(tool_call)
            conversation_log.append(tool_result)

def main():
    p = argparse.ArgumentParser(description="AI assistant with file operations")
    p.add_argument("-p", required=True, help="The prompt to send to the AI")
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    run_conversation(client, args.p)

if __name__ == "__main__":
    main()
