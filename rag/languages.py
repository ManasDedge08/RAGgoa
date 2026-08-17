"""The language registry: one place describing every language the system knows.

MSMARCO-XI ships 13 Indian languages plus the English source columns. Sarvam's
coverage is not uniform across them, and the differences are load-bearing:

* 10 have speech-to-text, a Bulbul voice, and translation support
* 4 (Assamese, Nepali, Sanskrit, Urdu) have speech-to-text only — no voice to
  answer with, and no translator to localise the spoken framing

The four are still worth indexing: retrieval and cross-lingual answering work
because the embedding model covers them. They simply answer in text, with
English framing. Pretending otherwise would produce silent turns.

Scripts do not map one-to-one either. Hindi, Marathi, Nepali and Sanskrit all
use Devanagari; Bengali and Assamese share a block. Script detection can narrow
a typed query to a script but cannot resolve it further, so ``ambiguous_with``
records the collision and the UI offers a picker. Spoken queries are unaffected
— Sarvam returns the language code.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    code: str  # dataset code, e.g. "hin_Deva"
    name: str
    sarvam: str  # BCP-47 locale used by Sarvam
    file: str | None  # parquet slice prefix; None for English
    script: str
    stt: bool = True
    tts: bool = True
    translate: bool = True

    @property
    def voice(self) -> bool:
        """Can a spoken answer be produced in this language?"""
        return self.tts


REGISTRY: dict[str, Language] = {
    lang.code: lang
    for lang in (
        # English is not a slice of its own: it comes from the Eng_* columns
        # present inside every language file.
        Language("eng_Latn", "English", "en-IN", None, "Latin"),
        Language("hin_Deva", "Hindi", "hi-IN", "hin", "Devanagari"),
        Language("ben_Beng", "Bengali", "bn-IN", "ben", "Bengali"),
        Language("tam_Taml", "Tamil", "ta-IN", "tam", "Tamil"),
        Language("tel_Telu", "Telugu", "te-IN", "tel", "Telugu"),
        Language("mar_Deva", "Marathi", "mr-IN", "mar", "Devanagari"),
        Language("guj_Gujr", "Gujarati", "gu-IN", "guj", "Gujarati"),
        Language("kan_Knda", "Kannada", "kn-IN", "kan", "Kannada"),
        Language("mal_Mlym", "Malayalam", "ml-IN", "mal", "Malayalam"),
        Language("pan_Guru", "Punjabi", "pa-IN", "pan", "Gurmukhi"),
        Language("ori_Orya", "Odia", "od-IN", "ori", "Odia"),
        # Speech in, but no Bulbul voice and no translator.
        Language("asm_Beng", "Assamese", "as-IN", "asm", "Bengali", tts=False, translate=False),
        Language("nep_Deva", "Nepali", "ne-IN", "nep", "Devanagari", tts=False, translate=False),
        Language("san_Deva", "Sanskrit", "sa-IN", "san", "Devanagari", tts=False, translate=False),
        Language("urd_Arab", "Urdu", "ur-IN", "urd", "Arabic", tts=False, translate=False),
    )
}

# Unicode blocks per script. Devanagari and Bengali each cover several
# languages, which is exactly why detection alone is not enough.
SCRIPT_RANGES: dict[str, tuple[tuple[int, int], ...]] = {
    "Latin": ((0x0041, 0x005A), (0x0061, 0x007A)),
    "Devanagari": ((0x0900, 0x097F),),
    "Bengali": ((0x0980, 0x09FF),),
    "Gurmukhi": ((0x0A00, 0x0A7F),),
    "Gujarati": ((0x0A80, 0x0AFF),),
    "Odia": ((0x0B00, 0x0B7F),),
    "Tamil": ((0x0B80, 0x0BFF),),
    "Telugu": ((0x0C00, 0x0C7F),),
    "Kannada": ((0x0C80, 0x0CFF),),
    "Malayalam": ((0x0D00, 0x0D7F),),
    "Arabic": ((0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF)),
}

# The language assumed when a script is shared and nothing else disambiguates.
# Chosen by speaker population, and surfaced in the UI rather than hidden.
SCRIPT_DEFAULT: dict[str, str] = {
    "Latin": "eng_Latn",
    "Devanagari": "hin_Deva",
    "Bengali": "ben_Beng",
    "Gurmukhi": "pan_Guru",
    "Gujarati": "guj_Gujr",
    "Odia": "ori_Orya",
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu",
    "Kannada": "kan_Knda",
    "Malayalam": "mal_Mlym",
    "Arabic": "urd_Arab",
}


def languages_for_script(script: str) -> list[str]:
    return [code for code, lang in REGISTRY.items() if lang.script == script]


def ambiguous_with(code: str) -> list[str]:
    """Other languages a typed query in this script could equally be."""
    lang = REGISTRY[code]
    return [c for c in languages_for_script(lang.script) if c != code]


def by_sarvam(locale: str) -> str:
    for code, lang in REGISTRY.items():
        if lang.sarvam == locale:
            return code
    return "eng_Latn"
