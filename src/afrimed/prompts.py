"""Prompt format shared by training, evaluation and the app.

MCQ: options shown as A-E; target is "Answer: B" (or "Answer: B, D" for multi-answer).
SAQ: target is the reference answer text.
"""
import re

SYSTEM = ("You are a medical question-answering assistant focused on healthcare in African contexts. "
          "Answer accurately and concisely. You do not replace a qualified health professional.")
LETTERS = "ABCDE"


def option_letter(key: str) -> str:
    return LETTERS[int(key.removeprefix("option")) - 1]


def mcq_user(rec: dict) -> str:
    opts = "\n".join(f"{option_letter(k)}. {v}" for k, v in sorted(rec["options"].items()))
    return (f"{rec['question']}\n\nOptions:\n{opts}\n\n"
            "Select the correct option(s). Reply in the form 'Answer: <letter(s)>'.")


def gold_letters(rec: dict) -> frozenset:
    return frozenset(option_letter(k) for k in rec["answer"])


def mcq_target(rec: dict) -> str:
    return "Answer: " + ", ".join(sorted(gold_letters(rec)))


def to_messages(rec: dict, with_answer: bool = True) -> list[dict]:
    """rec needs question_type and question (+ options for MCQ); answer only if with_answer."""
    user = mcq_user(rec) if rec["question_type"] == "mcq" else rec["question"]
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    if with_answer:
        target = mcq_target(rec) if rec["question_type"] == "mcq" else rec["answer"]
        msgs.append({"role": "assistant", "content": target})
    return msgs


# A leading run of letters such as "B", "(B)", "B, D", "B and D", "B/D".
_LETTER_RUN = re.compile(r"\(?([A-E])\)?((?:\s*(?:,\s*and|,|and|&|/|or)\s*\(?[A-E]\)?)*)(?![A-Za-z])")
_ANSWER_CUE = re.compile(r"(?i)answer\s*(?:is)?\s*[:\-]?\s*")


def _leading_letters(text: str, require_boundary: bool) -> frozenset | None:
    m = _LETTER_RUN.match(text)
    if not m:
        return None
    rest = text[m.end():]
    # Without an "Answer:" cue, "A 35-year-old..." must not parse as option A.
    if require_boundary and rest and not re.match(r"\s*(?:$|[.):\n])", rest):
        return None
    return frozenset(re.findall(r"[A-E]", m.group(0)))


def parse_mcq(text: str) -> frozenset | None:
    """Extract the chosen option letters from model output; None if unparseable."""
    text = text.replace("*", "").strip()
    for cue in _ANSWER_CUE.finditer(text):
        letters = _leading_letters(text[cue.end():], require_boundary=False)
        if letters:
            return letters
    letters = _leading_letters(text, require_boundary=True)
    if letters:
        return letters
    mentioned = set(re.findall(r"\b[Oo]ption\s+([A-E])\b", text))
    return frozenset(mentioned) if len(mentioned) == 1 else None
