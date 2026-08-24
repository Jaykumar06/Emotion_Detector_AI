"""
app.py

Streamlit UI for the mood-aware chatbot. Wraps the exact same logic as
main.py (onboarding questions -> emotion prediction -> Groq/LangChain
chain -> JSON persistence) — only the input/output layer changes from
terminal input() to a Streamlit chat UI.

Run:
    streamlit run app.py
"""

import os
import json
from datetime import datetime

import streamlit as st

# set_page_config MUST be the very first Streamlit command in the script,
# before any other st.* call (including ones triggered indirectly by
# @st.cache_resource) or Streamlit will raise
# StreamlitSetPageConfigMustBeFirstCommandError.
st.set_page_config(
    page_title="MoodMate — Emotion-Aware Chat",
    page_icon="💬",
    layout="centered",
)

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

from emotion_detector import detect_emotion

# ---------------------------------------------------------------------------
# Setup (identical to main.py)
# ---------------------------------------------------------------------------

load_dotenv()

HISTORY_FILE = "chat_history.json"

EMOTION_EMOJI = {
    "joy": "😄",
    "sadness": "😔",
    "anger": "😠",
    "fear": "😟",
    "surprise": "😲",
    "disgust": "😖",
    "neutral": "🙂",
}

ONBOARDING_QUESTIONS = [
    "How are you feeling right now, in your own words?",
    "How would you describe your day so far?",
    "Is there anything on your mind that's been bothering you or exciting you?",
    "How well did you sleep last night?",
    "On a scale of stressed to relaxed, where would you place yourself today?",
]

# Icon shown next to each onboarding question (purely cosmetic, matches
# the question at the same index).
ONBOARDING_ICONS = ["🙂", "🌤️", "💭", "🌙", "📊"]

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


@st.cache_resource
def get_chain():
    """Builds the LLM + prompt + chain once and caches it across reruns."""
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_TEMPLATE),
            ("placeholder", "{chat_history}"),
            ("human", "{user_input}"),
        ]
    )
    return prompt | llm | StrOutputParser()


chain = get_chain()

# ---------------------------------------------------------------------------
# Persistence helpers (identical logic to main.py)
# ---------------------------------------------------------------------------

def load_history() -> list:
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []


def save_history(all_sessions: list) -> None:
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(all_sessions, f, indent=2, ensure_ascii=False)


def persist_current_session():
    """Writes all past sessions + the in-progress session to disk."""
    save_history(st.session_state.all_sessions + [st.session_state.session_record])


# ---------------------------------------------------------------------------
# Core chat function (identical logic to main.py's chat())
# ---------------------------------------------------------------------------

def run_chat_turn(user_input: str):
    turn_emotion = detect_emotion(user_input)
    if turn_emotion["score"] >= 0.5:
        active_emotion = turn_emotion["label"]
        active_confidence = turn_emotion["score"]
    else:
        active_emotion = st.session_state.baseline_emotion
        active_confidence = st.session_state.baseline_confidence

    response = chain.invoke(
        {
            "emotion": active_emotion,
            "confidence": active_confidence,
            "onboarding_summary": st.session_state.onboarding_summary,
            "chat_history": st.session_state.lc_history,
            "user_input": user_input,
        }
    )

    st.session_state.lc_history.append(HumanMessage(content=user_input))
    st.session_state.lc_history.append(AIMessage(content=response))

    st.session_state.session_record["turns"].append(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "user": user_input,
            "bot": response,
            "detected_emotion": active_emotion,
            "confidence": active_confidence,
        }
    )
    persist_current_session()

    return response, active_emotion, active_confidence


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at 20% 0%, #1a1230 0%, #0b0b14 45%, #0a0a10 100%);
    }
    #MainMenu, footer, header[data-testid="stHeader"] {visibility: hidden;}
    .block-container { padding-top: 2rem; max-width: 900px; }

    * { color: #e5e5f0; }

    /* ---- Header ---- */
    .mm-header {
        text-align: center;
        padding: 0.6rem 0 1.6rem 0;
    }
    .mm-header h1 {
        font-size: 2.6rem;
        margin-bottom: 0.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .mm-header h1 .grad {
        background: linear-gradient(90deg, #a78bfa, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .mm-header p {
        color: #9ca3af;
        font-size: 1rem;
        margin-top: 0;
    }

    /* ---- Card ---- */
    .mm-card {
        background: linear-gradient(180deg, rgba(30,27,50,0.75), rgba(20,18,35,0.75));
        border-radius: 20px;
        padding: 2rem 2.2rem;
        box-shadow: 0 8px 40px rgba(124, 58, 237, 0.15);
        border: 1px solid rgba(139, 92, 246, 0.25);
        margin-bottom: 1.2rem;
    }

    .mm-welcome-title {
        font-size: 1.6rem;
        font-weight: 800;
        color: #c4b5fd;
        margin-bottom: 0.2rem;
    }
    .mm-welcome-sub {
        color: #9ca3af;
        margin-bottom: 1.4rem;
        font-size: 0.95rem;
    }

    /* ---- Question rows ---- */
    .mm-qrow {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(139,92,246,0.15);
        border-radius: 12px;
        padding: 0.7rem 1rem;
        margin-bottom: 0.6rem;
    }
    .mm-qnum {
        min-width: 30px;
        height: 30px;
        border-radius: 50%;
        background: linear-gradient(135deg, #a78bfa, #6366f1);
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.85rem;
        flex-shrink: 0;
    }
    .mm-qicon { font-size: 1.2rem; flex-shrink: 0; }
    .mm-qtext { font-size: 0.95rem; color: #e5e5f0; }

    div[data-testid="stTextInput"] input {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(139,92,246,0.25) !important;
        border-radius: 10px !important;
        color: #e5e5f0 !important;
    }
    div[data-testid="stTextInput"] input::placeholder { color: #6b7280 !important; }
    div[data-testid="stTextInput"] label { color: #d1d5db !important; font-size: 0.9rem; }

    /* ---- Mood badge ---- */
    .mm-mood-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: linear-gradient(90deg, rgba(167,139,250,0.18), rgba(99,102,241,0.18));
        color: #c4b5fd;
        padding: 0.5rem 1.1rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.9rem;
        border: 1px solid rgba(139,92,246,0.3);
    }

    div[data-testid="stChatMessage"] {
        border-radius: 14px;
        padding: 0.3rem 0.2rem;
        background: rgba(255,255,255,0.02);
    }

    /* ---- Buttons ---- */
    .stButton>button, .stFormSubmitButton>button {
        background: linear-gradient(90deg, #a855f7, #6366f1) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.7rem 1.4rem !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        width: 100%;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        box-shadow: 0 4px 20px rgba(139, 92, 246, 0.35);
    }
    .stButton>button:hover, .stFormSubmitButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 26px rgba(139, 92, 246, 0.5);
    }

    .mm-footer {
        text-align: center;
        color: #6b7280;
        font-size: 0.85rem;
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="mm-header">
        <h1>🧠 MoodMate <span class="grad">AI</span></h1>
        <p>An emotionally aware AI chatbot powered by Groq + LangChain + HuggingFace</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

if "stage" not in st.session_state:
    st.session_state.stage = "onboarding"  # -> "chat"
    st.session_state.all_sessions = load_history()
    st.session_state.qa_pairs = []
    st.session_state.lc_history = []

# ---------------------------------------------------------------------------
# Stage 1: Onboarding
# ---------------------------------------------------------------------------

if st.session_state.stage == "onboarding":
    st.markdown('<div class="mm-card">', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="mm-welcome-title">👋 Welcome!</div>
        <div class="mm-welcome-sub">
            Before we begin, I'd like to understand how you're feeling.
            Answer these five quick questions.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("onboarding_form"):
        answers = []
        for i, question in enumerate(ONBOARDING_QUESTIONS):
            st.markdown(
                f"""
                <div class="mm-qrow">
                    <div class="mm-qnum">{i+1}</div>
                    <div class="mm-qicon">{ONBOARDING_ICONS[i]}</div>
                    <div class="mm-qtext">{question}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            answers.append(
                st.text_input(
                    "",
                    key=f"onboard_{i}",
                    placeholder="Type your answer here...",
                    label_visibility="collapsed",
                )
            )
        submitted = st.form_submit_button("✨ Analyze My Mood")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="mm-footer">🔒 Your responses are private and secure.</div>',
        unsafe_allow_html=True,
    )

    if submitted:
        qa_pairs = [
            {"question": q, "answer": a}
            for q, a in zip(ONBOARDING_QUESTIONS, answers)
        ]
        combined_answers = " ".join(a for a in answers if a and a.strip())
        emotion_result = detect_emotion(combined_answers)

        onboarding_summary = "\n".join(
            f"- Q: {qa['question']}\n  A: {qa['answer']}" for qa in qa_pairs
        )

        st.session_state.qa_pairs = qa_pairs
        st.session_state.baseline_emotion = emotion_result["label"]
        st.session_state.baseline_confidence = emotion_result["score"]
        st.session_state.onboarding_summary = onboarding_summary

        st.session_state.session_record = {
            "session_started": datetime.now().isoformat(timespec="seconds"),
            "onboarding": qa_pairs,
            "predicted_emotion": emotion_result["label"],
            "predicted_confidence": emotion_result["score"],
            "turns": [],
        }

        st.session_state.stage = "chat"
        st.rerun()

# ---------------------------------------------------------------------------
# Stage 2: Chat
# ---------------------------------------------------------------------------

else:
    emoji = EMOTION_EMOJI.get(st.session_state.baseline_emotion, "🙂")
    st.markdown(
        f"""
        <div class="mm-card" style="display:flex; align-items:center; justify-content:space-between;">
            <div class="mm-mood-badge">{emoji} Mood: {st.session_state.baseline_emotion.capitalize()}
                ({st.session_state.baseline_confidence*100:.0f}% confidence)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([5, 1.4])
    with col2:
        if st.button("🔄 New session"):
            st.session_state.all_sessions.append(st.session_state.session_record)
            save_history(st.session_state.all_sessions)
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # Render past turns of this session
    for turn in st.session_state.session_record["turns"]:
        with st.chat_message("user"):
            st.write(turn["user"])
        with st.chat_message("assistant"):
            st.write(turn["bot"])
            e = turn["detected_emotion"]
            st.caption(f"{EMOTION_EMOJI.get(e, '🙂')} detected: {e} "
                       f"({turn['confidence']*100:.0f}%)")

    user_input = st.chat_input("Type your message...")

    if user_input:
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response, active_emotion, active_confidence = run_chat_turn(user_input)
            st.write(response)
            st.caption(
                f"{EMOTION_EMOJI.get(active_emotion, '🙂')} detected: "
                f"{active_emotion} ({active_confidence*100:.0f}%)"
            )

    with st.expander("📜 Onboarding answers"):
        for qa in st.session_state.qa_pairs:
            st.markdown(f"**{qa['question']}**")
            st.write(qa["answer"] or "_(no answer)_")