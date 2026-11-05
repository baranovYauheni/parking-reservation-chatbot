from __future__ import annotations

import logging
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

_PATTERNS = {
    "greeting": re.compile(
        r"^\s*(hi|hello|hey|hiya|good (morning|afternoon|evening)|greetings|"
        r"привет|здравствуй(те)?|добрый (день|вечер)|салют)\b[\s!.,]*\w{0,12}[\s!.]*$",
        re.IGNORECASE,
    ),
    "thanks": re.compile(
        r"^\s*(thanks|thank you|thx|cheers|great|perfect|awesome|спасибо)\b[\s\w!.,]{0,20}$",
        re.IGNORECASE,
    ),
    "bye": re.compile(
        r"^\s*(bye|goodbye|see you|have a nice day|пока|до свидания)\b[\s!.,]*$",
        re.IGNORECASE,
    ),
    "how_are_you": re.compile(
        r"^\s*(how are you|how's it going|how are things|как дела)\b[\s?!.]*$",
        re.IGNORECASE,
    ),
    "help": re.compile(
        r"^\s*(help|what can you do|what do you do|who are you|menu|options|"
        r"what can i ask)\b[\s?!.]*$",
        re.IGNORECASE,
    ),
}

CAPABILITIES = (
    "I can tell you where we are, prices, opening hours and free spaces, explain "
    "the parking rules, or book a space for you."
)

TEMPLATES = {
    "greeting": f"Hi there! Welcome to Demo Central Parking. {CAPABILITIES} "
    "What would you like to do?",
    "thanks": "You're welcome! Anything else I can do for you?",
    "bye": "Thanks for stopping by, have a great day!",
    "how_are_you": "I'm doing well, thanks for asking! How can I help you with "
    "parking today?",
    "help": f"{CAPABILITIES} Just ask a question, or say “I'd like to book a space”.",
}

SYSTEM_PROMPT = (
    "You are the friendly chat assistant of Demo Central Parking, a parking garage "
    "in Tbilisi. Reply to the visitor's small-talk message in one or two short, "
    "warm sentences, then offer help. You can help with: location, prices, "
    "opening hours, free spaces, parking rules and booking a space. Do not state "
    "any facts, prices or numbers. Reply in the visitor's language."
)


def detect(text: str) -> str | None:
    for kind, pattern in _PATTERNS.items():
        if pattern.match(text):
            return kind
    return None


def reply(kind: str, text: str, llm: BaseChatModel | None = None) -> str:
    if llm is not None:
        try:
            message = llm.invoke(
                [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=text)]
            )
            answer = str(message.content).strip()
            if answer and not re.search(r"\d", answer):
                return answer
        except Exception as exc:
            logger.warning("small-talk LLM call failed: %s", exc)
    return TEMPLATES[kind]
