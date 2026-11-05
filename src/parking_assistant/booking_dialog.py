from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from parking_assistant.reservations import (
    ReservationRequest,
    ReservationService,
    validate_car_number,
    validate_name,
    validate_period,
)
from parking_assistant.timeutils import (
    EXAMPLE,
    DateParseError,
    friendly_local,
    from_utc_text,
    parse_user_datetime,
    to_utc_text,
)

Extractor = Callable[[str, dict, datetime], dict]

FIELDS: tuple[str, ...] = ("first_name", "last_name", "car_number", "start", "end")

LABELS = {
    "first_name": "first name",
    "last_name": "last name",
    "car_number": "car number",
    "start": "start",
    "end": "end",
}


def question(field_name: str, draft: dict) -> str:
    first = draft.get("first_name")
    if field_name == "first_name":
        return "What's your first name?"
    if field_name == "last_name":
        return f"Nice to meet you, {first}! And your last name?"
    if field_name == "car_number":
        return "Thanks! What's the car's plate number?"
    if field_name == "start":
        return (
            "When would you like to arrive? A date and time like "
            f"{EXAMPLE} works, and you can give both at once, e.g. "
            "\u201ctomorrow 9:00 to 18:00\u201d."
        )
    start = from_utc_text(draft["start"])
    return f"Arriving {friendly_local(start)}. Until when do you need the space?"


_ALIASES = {
    "first name": "first_name",
    "firstname": "first_name",
    "given name": "first_name",
    "last name": "last_name",
    "lastname": "last_name",
    "surname": "last_name",
    "family name": "last_name",
    "full name": "name",
    "name": "name",
    "car number": "car_number",
    "license plate": "car_number",
    "plate number": "car_number",
    "plate": "car_number",
    "registration": "car_number",
    "car": "car_number",
    "start": "start",
    "from": "start",
    "arrival": "start",
    "end": "end",
    "until": "end",
    "till": "end",
    "to": "end",
    "departure": "end",
}
_KEY_ALT = "|".join(sorted((re.escape(k) for k in _ALIASES), key=len, reverse=True))
_PAIR = re.compile(rf"\b({_KEY_ALT})\s*[:=]\s*([^,;\n]+)", re.IGNORECASE)
_RANGE = re.compile(
    r"^\s*(?:from\s+)?(.+?)\s+(?:to|until|till|through|\-|\u2013|\u2014)\s+(.+?)\s*[.!]?\s*$",
    re.IGNORECASE,
)
_FROM_RANGE = re.compile(
    r"\bfrom\s+(.+?)\s+(?:to|until|till|through|\-|\u2013|\u2014)\s+(.+?)\s*[.!]?\s*$",
    re.IGNORECASE,
)
_EDIT = re.compile(
    r"^\s*(?:edit|change|update|fix|correct)\s+(?:the\s+|my\s+)?"
    r"(first name|last name|surname|name|car number|car|plate|start|end|period|dates?)",
    re.IGNORECASE,
)
_CANCEL = re.compile(r"^\s*(cancel|stop|abort|never ?mind|discard)\b", re.IGNORECASE)
_CONFIRM = re.compile(
    r"^\s*(yes|y|yep|confirm|confirmed|correct|submit|send( it)?|ok|okay|sure)\b[.! ]*$",
    re.IGNORECASE,
)
_DECLINE = re.compile(r"^\s*(no|n|nope|not correct|wrong)\b", re.IGNORECASE)

_BOOKING_INTENT = re.compile(
    r"\b(book|booking|reserve|reservation|make a reservation|"
    r"бронь|брониров\w*|забронир\w*)\b",
    re.IGNORECASE,
)
_STATUS_INTENT = re.compile(
    r"\b(status|my (reservation|booking|request)s?|approved|rejected|"
    r"was it (approved|rejected))\b",
    re.IGNORECASE,
)


def new_booking_state() -> dict:
    return {"mode": "idle", "draft": {}, "awaiting": None}


def wants_booking(text: str) -> bool:
    return bool(_BOOKING_INTENT.search(text)) and not wants_status(text)


def wants_status(text: str) -> bool:
    return bool(_STATUS_INTENT.search(text)) and not re.search(
        r"\b(make|new|want to|would like to|i'd like to)\b", text, re.IGNORECASE
    )


def is_confirmation(text: str) -> bool:
    return bool(_CONFIRM.match(text))


def is_question(text: str) -> bool:
    return text.strip().endswith("?")


@dataclass
class BookingStep:
    reply: str
    state: dict
    submit: ReservationRequest | None = None
    errors: list[str] = field(default_factory=list)


def missing(draft: dict) -> list[str]:
    return [f for f in FIELDS if not draft.get(f)]


def _apply(draft: dict, key: str, value: str, now: datetime) -> str | None:
    value = value.strip().strip(".")
    try:
        if key == "name":
            parts = value.split()
            if len(parts) < 2:
                return "Could you give me both your first and last name?"
            draft["first_name"] = validate_name(parts[0], "First name")
            draft["last_name"] = validate_name(" ".join(parts[1:]), "Last name")
        elif key in ("first_name", "last_name"):
            draft[key] = validate_name(value, LABELS[key].capitalize())
        elif key == "car_number":
            draft[key] = validate_car_number(value)[0]
        elif key == "start":
            draft[key] = to_utc_text(parse_user_datetime(value, now))
        elif key == "end":
            base = from_utc_text(draft["start"]) if draft.get("start") else None
            draft[key] = to_utc_text(parse_user_datetime(value, now, base=base))
    except (ValueError, DateParseError) as exc:
        return str(exc)
    return None


def _check_period(draft: dict, now: datetime) -> str | None:
    if not draft.get("start"):
        return None
    start = from_utc_text(draft["start"])
    if start <= now:
        draft.pop("start", None)
        draft.pop("end", None)
        return f"{friendly_local(start)} has already passed, so I can't book it."
    if not draft.get("end"):
        return None
    end = from_utc_text(draft["end"])
    try:
        validate_period(start, end, now)
    except ValueError as exc:
        draft.pop("end", None)
        message = str(exc)
        if "after the start" in message:
            return (
                f"The end ({friendly_local(end)}) is before the start "
                f"({friendly_local(start)})."
            )
        return message
    return None


def summary(draft: dict, service: ReservationService) -> str:
    start, end = from_utc_text(draft["start"]), from_utc_text(draft["end"])
    free = service.free_spaces(start, end)
    price = service.quote(start, end)
    availability = (
        f"{free} free {'space' if free == 1 else 'spaces'} for that time right now"
        if free > 0
        else "the parking looks full for that time, so the administrator may "
        "have to decline"
    )
    return (
        "Here's your booking, please take a look:\n"
        f"- Name: {draft['first_name']} {draft['last_name']}\n"
        f"- Car: {draft['car_number']}\n"
        f"- From: {friendly_local(start)}\n"
        f"- To: {friendly_local(end)}\n"
        f"- Estimated price: GEL {price:.2f}\n"
        f"- Availability: {availability}\n"
        "Shall I send it to the administrator? Say \u201cyes\u201d, tell me "
        "what to change (e.g. \u201cchange the car number\u201d), or "
        "\u201ccancel\u201d."
    )


def _next_prompt(state: dict, service: ReservationService) -> str:
    draft = state["draft"]
    todo = missing(draft)
    if todo:
        state["mode"] = "collecting"
        state["awaiting"] = todo[0]
        return question(todo[0], draft)
    state["mode"] = "confirming"
    state["awaiting"] = "confirmation"
    return summary(draft, service)


def current_prompt(state: dict, service: ReservationService) -> str:
    return _next_prompt(state, service)


def start_booking(
    state: dict,
    text: str,
    service: ReservationService,
    extractor: Extractor | None = None,
) -> BookingStep:
    state = _copy(state)
    now = service.clock()
    state["mode"] = "collecting"
    errors, applied = _extract(state["draft"], text, now)
    if not applied and extractor is not None:
        errors, applied = _llm_extract(state["draft"], text, now, extractor)
    period_error = _check_period(state["draft"], now)
    if period_error:
        errors.append(period_error)
    intro = "Happy to help you book a space! "
    if applied and not errors:
        intro += "Got it. "
    reply = intro + _format_errors(errors) + _next_prompt(state, service)
    return BookingStep(reply, state, errors=errors)


def handle_booking_message(
    state: dict,
    text: str,
    service: ReservationService,
    extractor: Extractor | None = None,
) -> BookingStep:
    state = _copy(state)
    now = service.clock()
    draft = state["draft"]

    if _CANCEL.match(text):
        return BookingStep(
            "No problem, I've dropped that booking. Anything else I can help with?",
            new_booking_state(),
        )

    edit = _EDIT.match(text)
    if edit:
        target = edit.group(1).lower()
        keys = {
            "first name": ["first_name"],
            "last name": ["last_name"],
            "surname": ["last_name"],
            "name": ["first_name", "last_name"],
            "car number": ["car_number"],
            "car": ["car_number"],
            "plate": ["car_number"],
            "start": ["start"],
            "end": ["end"],
            "period": ["start", "end"],
            "date": ["start", "end"],
            "dates": ["start", "end"],
        }[target]
        rest = re.sub(r"^\s*(?:to\b|[:=])?\s*", "", text[edit.end() :]).strip()
        for key in keys:
            draft.pop(key, None)
        errors: list[str] = []
        if rest and len(keys) == 1:
            error = _apply(draft, keys[0], rest, now)
            if error:
                errors.append(error)
        period_error = _check_period(draft, now)
        if period_error:
            errors.append(period_error)
        return BookingStep(
            _format_errors(errors) + _next_prompt(state, service), state, errors=errors
        )

    if state["mode"] == "confirming":
        if _CONFIRM.match(text):
            period_error = _check_period(draft, now)
            if period_error:
                return BookingStep(
                    _format_errors([period_error]) + _next_prompt(state, service),
                    state,
                    errors=[period_error],
                )
            request = ReservationRequest(
                first_name=draft["first_name"],
                last_name=draft["last_name"],
                car_number=draft["car_number"],
                start=from_utc_text(draft["start"]),
                end=from_utc_text(draft["end"]),
            )
            return BookingStep("", state, submit=request)
        if _DECLINE.match(text):
            return BookingStep(
                "Sure, what should I change: the name, the car, the start or the "
                "end? Or say \u201ccancel\u201d to drop it.",
                state,
            )

    errors, applied = _extract(draft, text, now)
    if not applied:
        awaiting = state.get("awaiting")
        if awaiting in FIELDS:
            key = awaiting
            if (
                awaiting == "first_name"
                and len(text.split()) >= 2
                and not draft.get("last_name")
            ):
                key = "name"
            error = _apply(draft, key, text, now)
            if error and extractor is not None:
                llm_errors, llm_applied = _llm_extract(draft, text, now, extractor)
                if llm_applied:
                    error = None
                    errors.extend(llm_errors)
            if error:
                errors.append(error)
        elif state["mode"] == "confirming":
            errors.append(
                "Just say \u201cyes\u201d to send it, tell me what to change, or "
                "\u201ccancel\u201d."
            )
    period_error = _check_period(draft, now)
    if period_error:
        errors.append(period_error)
    return BookingStep(
        _format_errors(errors) + _next_prompt(state, service), state, errors=errors
    )


def _extract(draft: dict, text: str, now: datetime) -> tuple[list[str], bool]:
    errors: list[str] = []
    pairs = _PAIR.findall(text)
    if pairs:
        for raw_key, value in pairs:
            error = _apply(draft, _ALIASES[raw_key.lower()], value, now)
            if error:
                errors.append(error)
        return errors, True
    for match in (_RANGE.match(text), _FROM_RANGE.search(text)):
        if match is None:
            continue
        trial = dict(draft)
        if _apply(trial, "start", match.group(1), now) is None and (
            _apply(trial, "end", match.group(2), now) is None
        ):
            draft.update(trial)
            return errors, True
    return errors, False


def _llm_extract(
    draft: dict, text: str, now: datetime, extractor: Extractor
) -> tuple[list[str], bool]:
    try:
        found = extractor(text, dict(draft), now) or {}
    except Exception:
        return [], False
    errors: list[str] = []
    applied = False
    for key in ("first_name", "last_name", "car_number", "start", "end"):
        value = found.get(key)
        if isinstance(value, str) and value.strip():
            error = _apply(draft, key, value, now)
            if error:
                errors.append(error)
            else:
                applied = True
    return errors, applied


def _format_errors(errors: list[str]) -> str:
    return "".join(f"{e} " for e in errors)


def _copy(state: dict) -> dict:
    return {
        "mode": state.get("mode", "idle"),
        "draft": dict(state.get("draft", {})),
        "awaiting": state.get("awaiting"),
    }
