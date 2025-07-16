from dotenv import load_dotenv
import os
from openai import OpenAI

# Load .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Init OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

def run_agent(prompt: str, mode: str):
    if mode not in ["info", "research"]:
        raise ValueError("Invalid mode. Use 'info' or 'research'.")

    if mode == "info":
        model = "gpt-4.1-nano"
    else:
        model = "gpt-4o"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content, model

if __name__ == "__main__":
    # Example usage:
    user_prompt = input("Enter your prompt: ")
    mode = input("Enter mode (info or research): ")

    answer, model_used = run_agent(user_prompt, mode)
    print(f"Model: {model_used}")
    print(f"Answer: {answer}")
