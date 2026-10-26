"""Tiny text utilities shared by retrieval, grading and extractive answering."""
import re

STOP = set("""a an the of to in on at by with from for and or if it its is are am was were be been
i my me we our you your they their this that these those there any some
how what which who when where why many much long often quickly
can could should will would must do does did get gets per after before s
under about into between than""".split())

# Domain synonyms: query word -> words the corpus may use instead.
_SYN_RAW = {
    "vacation": ["leave"], "holiday": ["leave"], "pto": ["leave"],
    "home": ["remote"], "wfh": ["remote"],
    "stolen": ["lost"], "notify": ["report"], "tell": ["report"],
    "deadline": ["within"], "daily": ["day"], "limit": ["allowance", "cap"],
    "cap": ["limit"], "respond": ["response"], "fast": ["within"],
    "reimburse": ["reimbursed", "payment"], "course": ["learning"],
}


def stem(w: str) -> str:
    if len(w) > 4:
        if w.endswith("ies"):
            return w[:-3] + "y"
        if w.endswith("ing"):
            w = w[:-3]
            return w[:-1] if len(w) > 2 and w[-1] == w[-2] else w
        if w.endswith("ed"):
            return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def tokens(text: str) -> list[str]:
    return [stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]


SYN = {stem(k): [stem(v) for v in vs] for k, vs in _SYN_RAW.items()}


def expand_terms(question: str) -> list[str]:
    out = []
    for t in tokens(question):
        out.extend(SYN.get(t, []))
    return out


def expand_query(question: str) -> str:
    return " ".join([question] + expand_terms(question))


def coverage(question: str, text: str) -> float:
    """Share of the question's content terms (or a synonym) present in the text."""
    q = list(dict.fromkeys(tokens(question)))
    if not q:
        return 0.0
    have = set(tokens(text))
    hit = sum(1 for t in q if t in have or any(s in have for s in SYN.get(t, [])))
    return hit / len(q)
