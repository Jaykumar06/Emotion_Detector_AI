"""
main.py

Mood-aware chatbot (Approach B: separate HuggingFace emotion classifier
feeding into a Groq LLM via LangChain). No UI — runs as a terminal chat loop.

Behavior:
    1. On startup, asks the user a fixed set of onboarding questions
       (how they're feeling, how their day went, etc.).
    2. Combines the answers and runs emotion detection on them to
       predict an overall mood for the session.
    3. Uses that predicted mood to shape the tone of every response
       for the rest of the conversation (re-checked each turn too).
    4. Persists the full chat history (questions, answers, detected
       emotion, and every message) to a local JSON file so it survives
       between runs.

Run:
    python main.py
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

from emotion_detector import detect_emotion

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

load_dotenv()

HISTORY_FILE = "chat_history.json"

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

# ---------------------------------------------------------------------------
# Onboarding questions
# ---------------------------------------------------------------------------
# Asked once at the start of every session. Answers are combined and run
# through the emotion classifier to establish a baseline mood.

ONBOARDING_QUESTIONS = [
    "How are you feeling right now, in your own words?",
    "How would you describe your day so far?",
    "Is there anything on your mind that's been bothering you or exciting you?",
    "How well did you sleep last night?",
    "On a scale of stressed to relaxed, where would you place yourself today?",
]

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

SYSTEM_TEMPLATE = """You are a warm, emotionally intelligent assistant.

The user's overall detected emotional state for this session is: {emotion} \
(confidence: {confidence}).
This was predicted from a short set of onboarding questions and answers:
{onboarding_summary}

Adjust your tone to fit this emotion:
- joy -> be upbeat and share in their enthusiasm
- sadness -> be gentle, empathetic, and supportive
- anger -> stay calm, validate their frustration, avoid being defensive
- fear -> be reassuring and grounding
- surprise -> be engaged and curious
- disgust -> be understanding and non-judgmental
- neutral -> be friendly and straightforward

Still answer the user's actual question or message clearly and helpfully.
Do not explicitly mention the emotion label or confidence score unless the
user asks how you knew how they were feeling."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_TEMPLATE),
        ("placeholder", "{chat_history}"),
        ("human", "{user_input}"),
    ]
)

chain = prompt | llm | StrOutputParser()

# ---------------------------------------------------------------------------
# Chat history persistence
# ---------------------------------------------------------------------------

def load_history() -> list:
    """Loads past sessions from disk, or returns an empty list."""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []


def save_history(all_sessions: list) -> None:
    """Writes the full history (all sessions) back to disk."""
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(all_sessions, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Onboarding flow
# ---------------------------------------------------------------------------

def run_onboarding() -> dict:
    """
    Asks the fixed onboarding questions, collects answers, and predicts
    an overall emotion from the combined answers.

    Returns a dict with questions/answers, predicted emotion, and a
    human-readable summary used inside the system prompt.
    """
    print("Before we start, a few quick questions:\n")

    qa_pairs = []
    combined_text = []

    for question in ONBOARDING_QUESTIONS:
        answer = input(f"{question}\nYou: ").strip()
        qa_pairs.append({"question": question, "answer": answer})
        if answer:
            combined_text.append(answer)

    combined_answers = " ".join(combined_text)
    emotion_result = detect_emotion(combined_answers)

    onboarding_summary = "\n".join(
        f"- Q: {qa['question']}\n  A: {qa['answer']}" for qa in qa_pairs
    )

    print(f"\n[Predicted overall mood: {emotion_result['label']} "
          f"({emotion_result['score']*100:.1f}%)]\n")

    return {
        "qa_pairs": qa_pairs,
        "emotion": emotion_result["label"],
        "confidence": emotion_result["score"],
        "onboarding_summary": onboarding_summary,
    }


# ---------------------------------------------------------------------------
# Chat turn
# ---------------------------------------------------------------------------

def chat(user_input: str, session_state: dict, chat_history_messages: list):
    """
    Runs one turn: re-check emotion on the new message, build prompt,
    call Groq, return response text plus the emotion used for this turn.
    """
    # Re-check emotion on each message too, but fall back to the session
    # baseline emotion if the new message is too short/neutral to read.
    turn_emotion = detect_emotion(user_input)
    if turn_emotion["score"] >= 0.5:
        active_emotion = turn_emotion["label"]
        active_confidence = turn_emotion["score"]
    else:
        active_emotion = session_state["emotion"]
        active_confidence = session_state["confidence"]

    response = chain.invoke(
        {
            "emotion": active_emotion,
            "confidence": active_confidence,
            "onboarding_summary": session_state["onboarding_summary"],
            "chat_history": chat_history_messages,
            "user_input": user_input,
        }
    )

    chat_history_messages.append(HumanMessage(content=user_input))
    chat_history_messages.append(AIMessage(content=response))

    print(f"[turn emotion: {active_emotion} ({active_confidence*100:.1f}%)]")

    return response, active_emotion, active_confidence


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    print("Mood-aware chatbot (Groq + LangChain). Type 'exit' to quit.\n")

    all_sessions = load_history()

    session_state = run_onboarding()
    chat_history_messages = []

    session_record = {
        "session_started": datetime.now().isoformat(timespec="seconds"),
        "onboarding": session_state["qa_pairs"],
        "predicted_emotion": session_state["emotion"],
        "predicted_confidence": session_state["confidence"],
        "turns": [],
    }

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        if not user_input:
            continue

        reply, turn_emotion, turn_confidence = chat(
            user_input, session_state, chat_history_messages
        )
        print(f"Bot: {reply}\n")

        session_record["turns"].append(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "user": user_input,
                "bot": reply,
                "detected_emotion": turn_emotion,
                "confidence": turn_confidence,
            }
        )

        # Save after every turn so nothing is lost if the session ends abruptly
        save_history(all_sessions + [session_record])

    # Final save on clean exit
    all_sessions.append(session_record)
    save_history(all_sessions)


if __name__ == "__main__":
    main()