"""
Gemini language model client.

Usage:
    Set the GEMINI_API_KEY environment variable, then run:
        python gemini_client.py "Your prompt here"
    Or run interactively (no arguments):
        python gemini_client.py
"""

import os
import sys

from google import genai


def call_gemini(prompt: str, model_name: str = "gemini-2.0-flash") -> str:
    """Call the Gemini language model with the given prompt.

    Args:
        prompt: The text prompt to send to the model.
        model_name: The Gemini model to use. Defaults to "gemini-2.0-flash".

    Returns:
        The model's text response.

    Raises:
        ValueError: If the GEMINI_API_KEY environment variable is not set.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable is not set. "
            "Get your API key from https://aistudio.google.com/app/apikey"
        )

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=model_name, contents=prompt)
    return response.text


def main() -> None:
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    else:
        print("Enter your prompt (press Enter twice to submit):")
        lines = []
        while True:
            line = input()
            if line == "" and lines:
                break
            lines.append(line)
        prompt = "\n".join(lines)

    if not prompt.strip():
        print("No prompt provided. Exiting.")
        sys.exit(1)

    print("\nCalling Gemini...\n")
    try:
        response = call_gemini(prompt)
        print("Response:")
        print(response)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
