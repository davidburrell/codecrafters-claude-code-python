import argparse
import os
import sys
import json

from openai import OpenAI

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL",
                     default="https://openrouter.ai/api/v1")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    conversationLog = [{"role": "user", "content": args.p}]
    tools = [{
        "type": "function",
        "function": {
                "name": "Read",
                "description": "Read and return the contents of a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The path to the file to read",
                        }
                    },
                    "required": ["file_path"],
                },
        }

    }]

    while True:
        chat = client.chat.completions.create(
            model="anthropic/claude-haiku-4.5",
            messages=conversationLog,
            tools=tools,
        )

        if not chat.choices or len(chat.choices) == 0:
            raise RuntimeError("no choices in response")

        response = chat.choices[0].message
        conversationLog.append(response)

        if not response.tool_calls:
            print(response.content)
            break

        for tool_call in response.tool_calls:

            if tool_call.type == "function":
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                match function_name:
                    case "Read":
                        file_path = function_args['file_path']
                        try:
                            with open(file_path, 'r') as f:
                                file_content = f.read()
                                conversationLog.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": file_content
                                })
                        except Exception as e:
                            print(f"Error reading file: {e}", file=sys.stderr)

                    case _:
                        print(f"Custom tool call type: {tool_call.type}")


if __name__ == "__main__":
    main()
