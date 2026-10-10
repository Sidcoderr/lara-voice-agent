"""
A small retrieval layer over the clinic FAQ data.

This uses simple keyword overlap scoring rather than embeddings, on purpose:
it's fast, has zero extra dependencies, and is easy to explain in an
interview. If you want to upgrade this later, swap `search()` for a real
vector search (e.g. chromadb + sentence-transformers, or an embeddings API)
without changing anything else in the pipeline -- the tool interface
(`search()` returning a list of {question, answer}) stays identical.
"""

import json
import re
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data" / "faq.json"

# Common words that shouldn't count toward relevance matching.
_STOPWORDS = {
    "a", "an", "the", "is", "are", "do", "does", "what", "how", "i", "my",
    "to", "for", "of", "in", "on", "and", "or", "can", "you", "your", "at",
    "it", "me", "please", "with", "have", "need",
    # generic filler that would otherwise cause spurious matches
    "about", "tell", "know", "when", "should", "would", "could", "will", "any",
    "some", "there", "this", "that", "just", "also", "like", "want", "whats",
    "long", "last", "lasts", "much", "many", "get", "got", "go", "been", "am",
    "be", "was", "were", "if", "so", "very", "really", "lot", "hurt", "hurts",
    "ve", "ll", "my", "all", "going", "feel", "feeling",
}


def _singular(word: str) -> str:
    """Crude plural handling so 'colds' matches 'cold' and 'headaches' matches 'headache'.

    Applied to both the entries and the query, so it only has to be consistent.
    """
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-z]+", text.lower())
    return {_singular(w) for w in words if w not in _STOPWORDS}


class KnowledgeBase:
    def __init__(self, data_path: Path = DATA_PATH):
        with open(data_path, "r") as f:
            self.entries = json.load(f)
        # Pre-tokenize once at startup: the question line (topic + aliases) and
        # the whole entry (question + answer).
        for entry in self.entries:
            entry["_q_tokens"] = _tokenize(entry["question"])
            entry["_tokens"] = _tokenize(entry["question"] + " " + entry["answer"])

    def search(self, query: str, top_k: int = 2, min_score: int = 1,
               relative_cutoff: float = 0.5) -> list[dict]:
        """Return the top_k FAQ entries most relevant to the query.

        Scored by overlapping keyword count. Returns an empty list if
        nothing clears min_score, so the caller can tell the LLM there
        was no good match (important -- we don't want the LLM inventing
        an answer when the knowledge base has nothing relevant).
        """
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scored = []
        for entry in self.entries:
            # A hit on the question line (the topic and its aliases) is worth 3
            # points; a hit anywhere else in the answer text is worth 1. This
            # stops a stray common word in an answer outranking the right topic.
            question_hits = len(query_tokens & entry["_q_tokens"])
            other_hits = len(query_tokens & entry["_tokens"]) - question_hits
            score = 3 * question_hits + other_hits
            if score >= min_score:
                scored.append((score, entry))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        if scored:
            # Drop weak matches that trail well behind the best one.
            best = scored[0][0]
            scored = [pair for pair in scored if pair[0] > relative_cutoff * best]
        return [
            {"question": e["question"], "answer": e["answer"]}
            for _, e in scored[:top_k]
        ]


# Singleton instances used by tools.py
kb = KnowledgeBase()  # surgery FAQs (opening hours, booking, prescriptions...)
health_kb = KnowledgeBase(Path(__file__).parent / "data" / "health_info.json")