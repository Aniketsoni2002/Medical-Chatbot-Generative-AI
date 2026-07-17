"""Safety layer: emergency triage and crisis detection.

These checks run BEFORE the LLM so red-flag situations always surface a clear,
deterministic escalation message rather than relying on the model to do so.
"""

from __future__ import annotations

import re

# Red-flag phrases suggesting a medical emergency. Kept broad but specific
# enough to avoid false positives on general questions.
_EMERGENCY_PATTERNS = [
    r"\bchest pain\b",
    r"\b(can't|cannot|difficulty|trouble) breath",
    r"\bshortness of breath\b",
    r"\bstroke\b",
    r"\b(face|arm) (droop|numb)",
    r"\bslurred speech\b",
    r"\bsevere bleeding\b",
    r"\bwon'?t stop bleeding\b",
    r"\bunconscious\b",
    r"\bloss of consciousness\b",
    r"\bpassed out\b|\bfaint(ed|ing)?\b",
    r"\bseizure\b",
    r"\bchoking\b",
    r"\banaphyla",
    r"\boverdose\b",
    r"\bpoison(ed|ing)?\b",
    r"\bsevere allergic\b",
    r"\bheart attack\b",
]

# Mental-health crisis phrases. Handled separately with supportive resources.
_CRISIS_PATTERNS = [
    r"\bkill (myself|my self)\b",
    r"\bsuicid",
    r"\bend my life\b",
    r"\bwant to die\b",
    r"\bharm (myself|my self)\b",
    r"\bself[- ]harm\b",
    r"\bno reason to live\b",
]

EMERGENCY_MESSAGE = (
    "🚨 **This may be a medical emergency.**\n\n"
    "Please **call your local emergency number now** (for example **911** in the US, "
    "**112** in the EU, **999** in the UK, or **108** in India) or go to the nearest "
    "emergency department. If someone is unconscious, not breathing, or has severe "
    "bleeding, get emergency help immediately.\n\n"
    "I'm an information tool and can't handle emergencies — please reach a real "
    "professional right away."
)

CRISIS_MESSAGE = (
    "💙 **You're not alone, and help is available right now.**\n\n"
    "If you're thinking about harming yourself, please reach out immediately:\n\n"
    "- **US:** call or text **988** (Suicide & Crisis Lifeline)\n"
    "- **UK & Ireland:** call **116 123** (Samaritans)\n"
    "- **India:** call **9152987821** (iCall) or **112**\n"
    "- **Elsewhere:** find a helpline at **https://findahelpline.com**\n\n"
    "If you feel you might act on these thoughts, please call your local emergency "
    "number or reach out to someone you trust right now. You matter."
)


def _matches(text: str, patterns: list[str]) -> bool:
    lowered = text.lower()
    return any(re.search(p, lowered) for p in patterns)


def check_emergency(text: str) -> bool:
    return _matches(text, _EMERGENCY_PATTERNS)


def check_crisis(text: str) -> bool:
    return _matches(text, _CRISIS_PATTERNS)


def screen(text: str) -> str | None:
    """Return an escalation message if the input is a red flag, else ``None``.

    Crisis (self-harm) is checked first so it takes priority.
    """
    if check_crisis(text):
        return CRISIS_MESSAGE
    if check_emergency(text):
        return EMERGENCY_MESSAGE
    return None


DISCLAIMER = (
    "⚕️ *This assistant provides general health information from your documents and "
    "is **not** a substitute for professional medical advice, diagnosis, or treatment. "
    "Always consult a qualified healthcare provider.*"
)
