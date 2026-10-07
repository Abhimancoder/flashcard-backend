import json
import os
from typing import List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from huggingface_hub import InferenceClient
from pydantic import BaseModel, Field

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN missing in .env file!")

client = InferenceClient(api_key=HF_TOKEN)
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

app = FastAPI(title="Flashcard Generator API")


# --- Schemas ---

class Flashcard(BaseModel):
    front: str
    back: str
    tag: Optional[str] = "General"


class FlashcardDeck(BaseModel):
    title: str
    flashcards: List[Flashcard]


class FlashcardRequest(BaseModel):
    content: str
    count: int = 3
    model: Optional[str] = DEFAULT_MODEL


# --- Routes ---

@app.get("/")
def home():
    return {"message": "API is online! Visit /docs to generate flashcards."}


@app.post("/api/flashcards", response_model=FlashcardDeck)
def generate_flashcards(payload: FlashcardRequest):
    system_prompt = (
        "You are an expert study aid generator. Output ONLY a valid JSON object matching this schema:\n"
        "{\n"
        '  "title": "Deck Title",\n'
        '  "flashcards": [\n'
        '    {"front": "Question or prompt", "back": "Short answer", "tag": "Subject"}\n'
        "  ]\n"
        "}\n"
        "Do not include markdown fences (like ```json), commentary, or extra text."
    )

    user_prompt = (
        f"Create exactly {payload.count} flashcards based on this content:\n"
        f"{payload.content}"
    )

    try:
        completion = client.chat_completion(
            model=payload.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=1200,
            temperature=0.2,
        )

        raw = completion.choices[0].message.content.strip()

        # Clean any accidental markdown wrappers if the model includes them
        if raw.startswith("```json"):
            raw = raw[7:]
        elif raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

        data = json.loads(raw)
        return FlashcardDeck(**data)

    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Model did not return valid JSON. Output was: {raw}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during generation: {str(exc)}",
        ) from exc