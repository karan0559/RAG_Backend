import os
import requests
from dotenv import load_dotenv
from typing import Optional
from app.Memory import memory_db  

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


def call_groq_llm(payload: dict) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set in .env")

    print(" Diagnostic Payload Sent to Groq API:")
    print(payload)

    try:
        response = requests.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json=payload,
            timeout=20
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    except requests.exceptions.HTTPError as e:
        print(" Groq API error:", response.text)
        raise RuntimeError(f"LLM request failed: {e}")
    except Exception as e:
        raise RuntimeError(f"LLM unexpected error: {e}")


def answer_question(context: str, question: str, session_id: Optional[str] = None) -> str:
    """
    Answer a user question, injecting memory from previous interactions.
    """
    memory_context = ""

    if session_id:
        memory_context = memory_db.get_recent_history(session_id, limit=30)

    #context for LLM
    prompt = {
        "model": GROQ_MODEL,
        "temperature": 0.7,
        "max_tokens": 512,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": (
                    f"Use the conversation history and document context below to answer the question.\n\n"
                    f"Conversation History:\n{memory_context if memory_context else 'No history available.'}\n\n"
                    f"Document Context:\n{context}\n\n"
                    f"Question:\n{question}"
                )
            }
        ]
    }

    # Call Groq 
    answer = call_groq_llm(prompt)

    # Store to memory
    if session_id:
        memory_db.add_to_memory(session_id, user_input=question, bot_output=answer)

    return answer


def summarize_text(text: str) -> str:
    prompt = {
        "model": GROQ_MODEL,
        "temperature": 0.5,
        "max_tokens": 1024,
        "messages": [
            {"role": "system", "content": "You are a helpful summarizer."},
            {"role": "user", "content": f"Summarize the following content:\n\n{text}"}
        ]
    }
    return call_groq_llm(prompt)


def compare_documents(doc_map: dict, question: str = "") -> str:
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
            {"role": "user", "content": f"{full_question}\n\n{context}"}
        ]
    }
    return call_groq_llm(prompt)
