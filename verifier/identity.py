"""Candidate identity check - do these two documents belong to the same person?

Every other check in this tool silently assumes they do. Without this gate, a
master for one candidate and a single-pager for another produce a high score:
the metric matcher just sees "some numbers are unsupported", which reads as a
tidy resume with a few unverified claims rather than the wrong document.

The check only ever fires on a *positive contradiction* - two identities that
disagree. A missing name or email means "cannot verify", never "mismatch",
because plenty of legitimate single-pagers drop the contact block.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_ROLL = re.compile(r"\b(?:roll(?:\s*(?:no|number))?\.?\s*[:\-]?\s*)(\d{6,9})\b", re.I)

# Words that rule a line out as a person's name.
_NOT_A_NAME = re.compile(
    r"(?i)\b(resume|curriculum|vitae|education|experience|skills|projects?|"
    r"objective|summary|achievements?|honou?rs|profile|contact|undergraduate|"
    r"institute|university|college|engineering|technology|department)\b"
)


@dataclass
class Identity:
    name: str = ""
    emails: set[str] = field(default_factory=set)
    rolls: set[str] = field(default_factory=set)

    @property
    def is_empty(self) -> bool:
        return not (self.name or self.emails or self.rolls)


@dataclass
class IdentityVerdict:
    same_person: bool | None      # True / False / None = could not determine
    reason: str
    master: Identity
    tailored: Identity


def _looks_like_a_name(line: str) -> bool:
    line = line.strip()
    if not (3 <= len(line) <= 60) or _NOT_A_NAME.search(line):
        return False
    if any(ch.isdigit() for ch in line) or "@" in line or "|" in line:
        return False
    words = line.split()
    if not (1 < len(words) <= 5):
        return False
    # Real names are alphabetic and normally capitalised.
    return all(re.fullmatch(r"[A-Za-z][A-Za-z.'\-]*", w) for w in words) and line[0].isupper()


def extract_identity(text: str) -> Identity:
    """Pull name / emails / roll number out of a resume's text."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    name = ""
    for line in lines[:6]:          # the name is at the very top on any resume
        if _looks_like_a_name(line):
            name = line
            break

    emails = {m.group(0).lower() for m in _EMAIL.finditer(text)}
    rolls = {m.group(1) for m in _ROLL.finditer(text)}
    return Identity(name=name, emails=emails, rolls=rolls)


def _name_tokens(name: str) -> set[str]:
    return {t for t in re.split(r"[^A-Za-z]+", name.lower()) if len(t) >= 3}


def _local_parts(emails: set[str]) -> set[str]:
    # Strip trailing digits: aradhyag24@ and aradhyag@ are the same person.
    return {re.sub(r"\d+$", "", e.split("@", 1)[0]) for e in emails}


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def compare_identities(master: Identity, tailored: Identity) -> IdentityVerdict:
    """Decide whether two documents describe the same candidate.

    Ordered strongest signal first. Agreement on any strong signal wins
    outright - a candidate may well use a personal address on one resume and
    an institute address on another, so a single disagreement is not enough
    when something else matches.
    """
    def verdict(same, reason):
        return IdentityVerdict(same, reason, master, tailored)

    if master.is_empty or tailored.is_empty:
        return verdict(None, "Not enough identifying information to compare.")

    # --- strong agreement --------------------------------------------------
    if master.emails & tailored.emails:
        return verdict(True, f"Same email address ({sorted(master.emails & tailored.emails)[0]}).")
    if master.rolls & tailored.rolls:
        return verdict(True, f"Same roll number ({sorted(master.rolls & tailored.rolls)[0]}).")

    m_tokens, t_tokens = _name_tokens(master.name), _name_tokens(tailored.name)
    if m_tokens and t_tokens and (m_tokens & t_tokens):
        return verdict(True, f"Name matches ('{master.name}' / '{tailored.name}').")

    # --- contradiction -----------------------------------------------------
    if m_tokens and t_tokens and not (m_tokens & t_tokens):
        # No shared name token at all. Guard against a pure spelling variant.
        if _similar(master.name.lower(), tailored.name.lower()) < 0.6:
            return verdict(
                False,
                f"Different candidate: master is '{master.name}', "
                f"tailored resume is '{tailored.name}'.",
            )

    if master.emails and tailored.emails:
        m_local, t_local = _local_parts(master.emails), _local_parts(tailored.emails)
        if not (m_local & t_local) and max(
            (_similar(a, b) for a in m_local for b in t_local), default=0.0
        ) < 0.7:
            return verdict(
                False,
                f"Different candidate: master email {sorted(master.emails)[0]}, "
                f"tailored email {sorted(tailored.emails)[0]}.",
            )

    if master.rolls and tailored.rolls and not (master.rolls & tailored.rolls):
        return verdict(False, f"Different roll number: {sorted(master.rolls)[0]} "
                              f"vs {sorted(tailored.rolls)[0]}.")

    return verdict(None, "Could not confirm the two documents are the same person.")
