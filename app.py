from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from openai import OpenAI

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()

class Query(BaseModel):
    prompt: str
    mode: str  # "info" or "research"

@app.get("/")
def read_root():
    return {"status": "✅ ResearchInfoAgent is running!"}

@app.post("/ask")
def ask(query: Query):
    if query.mode not in ["info", "research"]:
        raise HTTPException(status_code=400, detail="Invalid mode. Use 'info' or 'research'.")

    # Pick the model
    if query.mode == "info":
        model = "gpt-4.1-nano"
    else:
        model = "gpt-4o"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": query.prompt}
            ]
        )
        answer = response.choices[0].message.content
        return {"answer": answer, "model_used": model}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
