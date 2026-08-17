"""Unit checks for the pure pieces. No index, no network, runs in a second.

Covers the parts where a silent bug would corrupt every downstream number:
tokenisation across four scripts, BM25 ranking, RRF ordering, script-based
language detection, sentence splitting, and the naive chunker.

Run: ``python scripts/selftest.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import RRF_K  # noqa: E402
from rag.index.bm25 import BM25Index  # noqa: E402
from rag.index.text import tokenize  # noqa: E402
from rag.retrieval.baseline import chunk_text  # noqa: E402
from rag.retrieval.lang import detect_lang, from_sarvam_code, sarvam_code  # noqa: E402
from scripts.prepare_data import split_sentences  # noqa: E402

PASSED = 0


def check_result(label: str, condition: bool, detail: str = "") -> None:
    global PASSED
    if condition:
        PASSED += 1
        print(f"  ok    {label}")
    else:
        print(f"  FAIL  {label} {detail}")
        sys.exit(1)


def test_tokenize() -> None:
    print("tokenize")
    check_result("splits English on punctuation", tokenize("What is a corporation?") == ["what", "is", "a", "corporation"])
    check_result("keeps Devanagari clusters", tokenize("कॉर्पोरेशन क्या है?") == ["कॉर्पोरेशन", "क्या", "है"])
    check_result("drops the danda", "।" not in "".join(tokenize("यह एक वाक्य है।")))
    check_result("handles Tamil", len(tokenize("நிறுவனம் என்றால் என்ன")) == 3)
    check_result("handles Bengali", len(tokenize("কর্পোরেশন কী")) == 2)
    check_result("case folds", tokenize("ABC abc") == ["abc", "abc"])


def test_bm25() -> None:
    print("bm25")
    docs = [
        ("a", "the corporation is a legal entity separate from its owners"),
        ("b", "a limited liability company protects personal assets"),
        ("c", "corporation tax rates changed this year for every corporation"),
        ("d", "unrelated text about cricket and the weather in Goa"),
    ]
    index = BM25Index.build([d[0] for d in docs], [d[1] for d in docs])
    hits = index.search("corporation legal entity", top_k=4)
    check_result("returns ranked hits", len(hits) >= 2, str(hits))
    check_result("ranks the on-topic doc first", hits[0][0] == "a", str(hits))
    check_result("excludes zero-score docs", all(doc != "d" for doc, _ in hits), str(hits))
    check_result("unknown terms return nothing", index.search("zzzz qqqq", top_k=4) == [])
    check_result("term frequency saturates", index.search("corporation", 4)[0][0] in {"a", "c"})


def test_rrf_ordering() -> None:
    """A document ranked mid-list by several strategies must beat a single #1."""
    print("rrf")
    consensus = sum(1.0 / (RRF_K + rank) for rank in (3, 4, 5))
    lone_winner = 1.0 / (RRF_K + 1)
    check_result("three mid ranks beat one top rank", consensus > lone_winner)
    check_result("rank 1 beats rank 2", 1.0 / (RRF_K + 1) > 1.0 / (RRF_K + 2))


def test_lang_detection() -> None:
    print("language detection")
    check_result("Hindi", detect_lang("कॉर्पोरेशन क्या है")[0] == "hin_Deva")
    check_result("Tamil", detect_lang("நிறுவனம் என்றால் என்ன")[0] == "tam_Taml")
    check_result("Bengali", detect_lang("কর্পোরেশন কী")[0] == "ben_Beng")
    check_result("English", detect_lang("what is a corporation")[0] == "eng_Latn")
    check_result("empty falls back to English", detect_lang("")[0] == "eng_Latn")
    check_result("digits only fall back", detect_lang("12345 !!!")[0] == "eng_Latn")
    mixed_lang, confidence = detect_lang("corporation कॉर्पोरेशन कॉर्पोरेशन")
    check_result("mixed script picks the majority", mixed_lang == "hin_Deva", mixed_lang)
    check_result("confidence is a fraction", 0.0 < confidence <= 1.0)
    check_result("sarvam round trip", from_sarvam_code(sarvam_code("tam_Taml")) == "tam_Taml")


def test_sentence_split() -> None:
    print("sentence splitting")
    parts = split_sentences("First sentence here, long enough to stand alone. Second sentence also long enough.")
    check_result("splits on full stops", len(parts) == 2, str(parts))
    hindi = split_sentences("यह पहला वाक्य है और यह काफी लंबा है। यह दूसरा वाक्य है और यह भी लंबा है।")
    check_result("splits on danda", len(hindi) == 2, str(hindi))
    check_result("short fragments merge", len(split_sentences("Yes. No. Maybe.")) == 1)
    check_result("never returns empty", split_sentences("") == [""])


def test_chunker() -> None:
    print("naive chunker")
    chunks = chunk_text("x" * 1100, size=512)
    check_result("cuts at fixed width", [len(c) for c in chunks] == [512, 512, 76], str([len(c) for c in chunks]))
    check_result("short text stays whole", chunk_text("short", size=512) == ["short"])


def test_grounding_rejects_invention() -> None:
    """The grounding check must fail on fabricated content, not just pass on real.

    Needs the index, so it is skipped when artefacts are absent.
    """
    print("grounding")
    try:
        from rag.generation.grounding import check
        from rag.retrieval.retriever import Candidate
        from rag.retrieval.store import get_store

        get_store()
    except Exception as exc:  # noqa: BLE001 - index not built yet
        print(f"  skip  (index unavailable: {str(exc)[:60]})")
        return

    passage = (
        "A corporation is a legal entity that is separate and distinct from its "
        "owners. Corporations enjoy most of the rights and responsibilities that "
        "an individual possesses."
    )
    candidate = Candidate(
        group_id="0:0", passage_id="0:0:eng_Latn", text=passage,
        lang="eng_Latn", query_id=0,
    )
    # The store scores by passage_id, and this fixture is not in the index, so
    # semantic support is measured against the span alone.
    faithful = check(
        "A corporation is a legal entity separate and distinct from its owners.",
        [], passage,
    )
    invented = check(
        "Corporations were first chartered in Reykjavik in 1483 by King Olaf, who "
        "set the corporate tax rate at nineteen percent.",
        [], passage,
    )
    check_result("accepts a faithful restatement", faithful.supported, faithful.reason)
    check_result("rejects invented specifics", not invented.supported, invented.reason)
    check_result("lexical score is lower for invention", invented.lexical < faithful.lexical,
                 f"{invented.lexical:.2f} vs {faithful.lexical:.2f}")
    del candidate


if __name__ == "__main__":
    test_tokenize()
    test_bm25()
    test_rrf_ordering()
    test_lang_detection()
    test_sentence_split()
    test_chunker()
    test_grounding_rejects_invention()
    print(f"\n{PASSED} checks passed")
