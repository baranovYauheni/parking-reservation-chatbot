from __future__ import annotations

import json
import logging
import re
from datetime import datetime

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from parking_assistant.timeutils import to_local

logger = logging.getLogger(__name__)

FIELDS = ("first_name", "last_name", "car_number", "start", "end")

SYSTEM_PROMPT = """You extract parking reservation details from one visitor message.
Return ONLY a JSON object with these keys: first_name, last_name, car_number, start, end.
Use null for anything the message does not clearly contain. Never invent values.
Dates: convert to local Tbilisi time in the format YYYY-MM-DD HH:MM.
The current local date and time is {now} ({weekday}).
Details already collected (do not repeat them unless the message changes them): {draft}
The assistant is currently asking for: {awaiting}."""

_JSON = re.compile(r"\{.*\}", re.DOTALL)


def build_extractor(llm: BaseChatModel):
    def extract(text: str, draft: dict, now: datetime) -> dict:
        local_now = to_local(now)
        known = {
            k: v for k, v in draft.items() if k in FIELDS and k not in ("start", "end")
        }
        awaiting = next((f for f in FIELDS if not draft.get(f)), "nothing")
        prompt = SYSTEM_PROMPT.format(
            now=local_now.strftime("%Y-%m-%d %H:%M"),
            weekday=local_now.strftime("%A"),
            draft=json.dumps(known),
            awaiting=awaiting,
        )
        message = llm.invoke(
            [SystemMessage(content=prompt), HumanMessage(content=text)]
        )
        match = _JSON.search(str(message.content))
        if not match:
            return {}
        data = json.loads(match.group(0))
        if not isinstance(data, dict):
            return {}
        return {k: data[k] for k in FIELDS if isinstance(data.get(k), str)}

    return extract
