"""Spoken framing keyed to grounding confidence — the audible guardrail.

The system's certainty is carried by the voice, not only by a badge in the UI.
Three tiers, each with a prefix spoken before the content, in the asker's
language:

* ``high``    - stated directly, no hedge
* ``low``     - hedged aloud before the content is read
* ``refuse``  - a spoken refusal, never silence and never an error tone

A fourth tier was deliberately not added: more gradations are not audible to a
listener in one pass.
"""

from __future__ import annotations

Tier = str  # "high" | "low" | "refuse"

PREFIXES: dict[str, dict[Tier, str]] = {
    "eng_Latn": {
        "high": "",
        "low": "I found something related, but I'm not fully sure this answers it — ",
        "refuse": "I don't have a passage that answers that. My sources are a 1,200-question slice of MS MARCO, so I only know what those questions happened to cover. ",
    },
    "hin_Deva": {
        "high": "",
        "low": "मुझे कुछ मिला जो इससे जुड़ा है, पर मैं पूरी तरह निश्चित नहीं हूँ कि यह इसका उत्तर है — ",
        "refuse": "इसका उत्तर देने वाला कोई अंश मेरे पास नहीं है। मेरे स्रोत MS MARCO के 1,200 प्रश्नों का एक हिस्सा हैं, इसलिए मैं उतना ही जानता हूँ जितना वे प्रश्न कवर करते हैं। ",
    },
    "tam_Taml": {
        "high": "",
        "low": "தொடர்புடைய ஒன்றைக் கண்டேன், ஆனால் இது முழுமையான பதில் என்று உறுதியாகச் சொல்ல முடியவில்லை — ",
        "refuse": "இதற்கு பதிலளிக்கும் பத்தி என்னிடம் இல்லை. என் ஆதாரங்கள் MS MARCO-வின் 1,200 கேள்விகளின் ஒரு பகுதி மட்டுமே, அவை உள்ளடக்கியதை மட்டுமே நான் அறிவேன். ",
    },
    "ben_Beng": {
        "high": "",
        "low": "সম্পর্কিত কিছু পেয়েছি, তবে এটি সম্পূর্ণ উত্তর কিনা নিশ্চিত নই — ",
        "refuse": "এর উত্তর দেওয়ার মতো কোনো অনুচ্ছেদ আমার কাছে নেই। আমার উৎস MS MARCO-এর ১,২০০টি প্রশ্নের একটি অংশ, তাই সেগুলি যা কভার করে আমি কেবল ততটুকুই জানি। ",
    },
}

# Spoken when the query guardrail rejects the question before retrieval runs.
OFF_TOPIC: dict[str, str] = {
    "eng_Latn": "That falls outside what this system covers. My corpus is a 1,200-question slice of MS MARCO — try one of the example questions.",
    "hin_Deva": "यह इस प्रणाली के दायरे से बाहर है। कृपया MS MARCO अंशों से उत्तर देने योग्य कुछ पूछें।",
    "tam_Taml": "இது இந்த அமைப்பின் எல்லைக்கு வெளியே உள்ளது. MS MARCO பத்திகளிலிருந்து பதிலளிக்கக்கூடிய ஒன்றைக் கேளுங்கள்.",
    "ben_Beng": "এটি এই সিস্টেমের আওতার বাইরে। MS MARCO অনুচ্ছেদ থেকে উত্তরযোগ্য কিছু জিজ্ঞাসা করুন।",
}

# Spoken when a passage answering the question was found in another language.
CROSS_LINGUAL_NOTE: dict[str, str] = {
    "eng_Latn": " (answered from a source passage in {source})",
    "hin_Deva": " ({source} भाषा के स्रोत अंश से उत्तर दिया गया)",
    "tam_Taml": " ({source} மொழி மூலப் பத்தியிலிருந்து பதில்)",
    "ben_Beng": " ({source} ভাষার উৎস অনুচ্ছেদ থেকে উত্তর)",
}


def prefix_for(lang: str, tier: Tier) -> str:
    return PREFIXES.get(lang, PREFIXES["eng_Latn"]).get(tier, "")


def off_topic_for(lang: str) -> str:
    return OFF_TOPIC.get(lang, OFF_TOPIC["eng_Latn"])


def cross_lingual_note(lang: str, source_name: str) -> str:
    return CROSS_LINGUAL_NOTE.get(lang, CROSS_LINGUAL_NOTE["eng_Latn"]).format(source=source_name)
