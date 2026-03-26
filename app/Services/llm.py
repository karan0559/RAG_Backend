import os
import httpx
from dotenv import load_dotenv
from typing import Optional
from app.Memory import memory_db

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Shared async HTTP client (connection pool reuse across requests)
_http_client: httpx.AsyncClient | None = None


def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(timeout=30.0)
    return _http_client


async def call_groq_llm(payload: dict) -> str:
    """Async call to Groq API — does NOT block the FastAPI event loop."""
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set in .env")

    client = get_http_client()
    try:
        response = await client.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json=payload,
        )

        if response.status_code != 200:
            error_details = response.text
            print(f"🚨 Groq API error ({response.status_code}): {error_details}")
            raise RuntimeError(f"LLM request failed: {response.status_code} - {error_details}")

        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    except httpx.TimeoutException:
        raise RuntimeError("LLM request timed out after 30s")
    except httpx.RequestError as e:
        raise RuntimeError(f"LLM network error: {str(e)}")


async def answer_question(
    context: str,
    question: str,
    session_id: Optional[str] = None,
    use_documents: bool = True,
) -> str:
    """
    Answer a user question, injecting memory from previous interactions.
    Saves the turn to memory after answering (only once — callers must NOT save again).
    """
    memory_context = ""
    if session_id:
        memory_context = memory_db.get_recent_history(session_id, limit=10)

    if use_documents:
        system_prompt = (
            "You are a helpful assistant. "
            "Answer questions using the provided document context. "
            "If the context contains the answer, use it. "
            "If the context is missing or insufficient, say that clearly and ask a short follow-up. "
            "Be concise and accurate."
        )
        user_prompt = (
            f"Conversation History:\n{memory_context or 'None'}\n\n"
            f"Document Context:\n{context or 'None'}\n\n"
            f"Question:\n{question}"
        )
    else:
        system_prompt = (
            "You are a friendly and concise AI assistant. "
            "For greetings or small talk, reply naturally in 1-2 short sentences."
        )
        user_prompt = (
            f"Conversation History:\n{memory_context or 'None'}\n\n"
            f"User message:\n{question}"
        )

    prompt = {
        "model": GROQ_MODEL,
        "temperature": 0.7,
        "max_tokens": 1024,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    answer = await call_groq_llm(prompt)

    # Save this turn to memory (only called here — NOT repeated in query.py)
    if session_id:
        memory_db.add_to_memory(session_id, user_input=question, bot_output=answer)

    return answer


async def summarize_text(text: str) -> str:
    prompt = {
        "model": GROQ_MODEL,
        "temperature": 0.5,
        "max_tokens": 1024,
        "messages": [
            {"role": "system", "content": "You are a helpful summarizer."},
            {"role": "user", "content": f"Summarize the following content:\n\n{text}"},
        ],
    }
    return await call_groq_llm(prompt)


async def compare_documents(doc_map: dict, question: str = "") -> str:
    context = ""
    for doc_id, chunks in doc_map.items():
        sample = "\n".join(chunks[:10])
        context += f"\nDocument [{doc_id}]:\n{sample}\n"

    full_question = question or "Compare these documents and highlight key similarities and differences."

    prompt = {
        "model": GROQ_MODEL,
        "temperature": 0.5,
        "max_tokens": 1024,
        "messages": [
            {"role": "system", "content": "You are an expert in comparing documents."},
            {"role": "user", "content": f"{full_question}\n\n{context}"},
        ],
    }
    return await call_groq_llm(prompt)
