from datetime import UTC, datetime

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from parking_assistant import smalltalk
from parking_assistant.booking_dialog import (
    handle_booking_message,
    new_booking_state,
    start_booking,
)
from parking_assistant.extraction import build_extractor
from parking_assistant.timeutils import parse_user_datetime, to_local

NOW = datetime(2026, 10, 13, 8, 0, tzinfo=UTC)


def run(service, *messages, extractor=None):
    step = start_booking(new_booking_state(), messages[0], service, extractor)
    for text in messages[1:]:
        step = handle_booking_message(step.state, text, service, extractor)
    return step


def local(value: str) -> str:
    return to_local(datetime.fromisoformat(value)).strftime("%Y-%m-%d %H:%M")


def test_greeting_gets_a_friendly_reply_not_unknown(container):
    state = container.chat.send("talk-1", "hello")
    assert state["route"] == "smalltalk"
    assert "Hi there" in state["reply"]
    assert "don't have anything" not in state["reply"]


def test_smalltalk_detection():
    assert smalltalk.detect("Hello!") == "greeting"
    assert smalltalk.detect("thanks a lot") == "thanks"
    assert smalltalk.detect("what can you do?") == "help"
    assert smalltalk.detect("where is the parking?") is None


def test_smalltalk_rejects_llm_reply_with_facts():
    llm = FakeListChatModel(responses=["We have 10 spaces at GEL 5."])
    assert smalltalk.reply("greeting", "hi", llm) == smalltalk.TEMPLATES["greeting"]


def test_typo_tomorrow_and_time_only_end():
    start = parse_user_datetime("tomorow 10:00", now=NOW)
    assert to_local(start).strftime("%Y-%m-%d %H:%M") == "2026-10-14 10:00"
    end = parse_user_datetime("18:00", now=NOW, base=start)
    assert to_local(end).strftime("%Y-%m-%d %H:%M") == "2026-10-14 18:00"


def test_range_in_one_message(service):
    step = run(
        service, "book", "John Smith", "AB123CD", "2026-10-20 10:00 to tomorow 10:00"
    )
    assert step.state["mode"] == "collecting"
    assert "is before the start" in step.reply

    step = run(service, "book", "John Smith", "AB123CD", "tomorrow 9:00 to 18:00")
    assert step.state["mode"] == "confirming"
    assert local(step.state["draft"]["start"]) == "2026-10-14 09:00"
    assert local(step.state["draft"]["end"]) == "2026-10-14 18:00"


def test_past_start_is_explained(service):
    step = run(service, "book", "John Smith", "AB123CD", "2026-10-01 10:00")
    assert "has already passed" in step.reply
    assert step.state["awaiting"] == "start"


def test_personal_prompts(service):
    step = run(service, "book", "John")
    assert "Nice to meet you, John" in step.reply


def test_llm_extractor_fills_fields(service):
    llm = FakeListChatModel(
        responses=[
            '{"first_name": "Anna", "last_name": "Lee", "car_number": "XY777ZZ",'
            ' "start": "2026-10-21 08:00", "end": "2026-10-21 17:30"}'
        ]
    )
    step = run(
        service,
        "book a spot for Anna Lee, plate XY777ZZ, next Wednesday 8am till half five",
        extractor=build_extractor(llm),
    )
    assert step.state["mode"] == "confirming"
    assert step.state["draft"]["car_number"] == "XY777ZZ"


def test_llm_extractor_failure_falls_back_to_questions(service):
    def broken(text, draft, now):
        raise RuntimeError("model down")

    step = run(service, "please book something nice", extractor=broken)
    assert step.state["awaiting"] == "first_name"
    assert "first name" in step.reply
