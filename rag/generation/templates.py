"""Spoken framing keyed to grounding confidence — the audible guardrail.

The system's certainty is carried by the voice, not only by a badge in the UI.
Three tiers, each with a prefix spoken before the content, in the asker's
language:

* ``high``    - stated directly, no hedge
* ``low``     - hedged aloud before the content is read
* ``refuse``  - a spoken refusal, never silence and never an error tone

A fourth tier was deliberately not added: more gradations are not audible to a
listener in one pass.

Provenance of these strings: English, Hindi, Tamil and Bengali are hand-written
and reviewed. The other seven voice languages were produced with Sarvam's
translate endpoint and are committed as static text, so nothing is translated
at request time. Assamese, Nepali, Sanskrit and Urdu fall back to English
framing — Sarvam translates none of them, and none has a voice to speak with.

The cross-lingual note is assembled rather than translated: it is a short
parenthetical, and translating it dropped the placeholder in several scripts.
"""

from __future__ import annotations

from ..languages import REGISTRY

Tier = str  # "high" | "low" | "refuse"

PREFIXES: dict[str, dict[Tier, str]] = {
    "eng_Latn": {
        "high": "",
        "low": "I found something related, but I'm not fully sure this answers it — ",
        "refuse": "I don't have a passage that answers that. My sources cover only a small set of web-search questions. ",
    },
    "hin_Deva": {
        "high": "",
        "low": "मुझे कुछ मिला जो इससे जुड़ा है, पर मैं पूरी तरह निश्चित नहीं हूँ कि यह इसका उत्तर है — ",
        "refuse": "इसका उत्तर देने वाला कोई अंश मेरे पास नहीं है। मेरे स्रोतों में वेब-खोज के कुछ ही प्रश्न शामिल हैं। ",
    },
    "ben_Beng": {
        "high": "",
        "low": "সম্পর্কিত কিছু পেয়েছি, তবে এটি সম্পূর্ণ উত্তর কিনা নিশ্চিত নই — ",
        "refuse": "এর উত্তর দেওয়ার মতো কোনো অনুচ্ছেদ আমার কাছে নেই। আমার উৎসে কেবল কিছু ওয়েব-সার্চ প্রশ্ন রয়েছে। ",
    },
    "tam_Taml": {
        "high": "",
        "low": "தொடர்புடைய ஒன்றைக் கண்டேன், ஆனால் இது முழுமையான பதில் என்று உறுதியாகச் சொல்ல முடியவில்லை — ",
        "refuse": "இதற்குப் பதிலளிக்கும் பத்தி என்னிடம் இல்லை. என் ஆதாரங்களில் சில வலைத் தேடல் கேள்விகள் மட்டுமே உள்ளன. ",
    },
    "tel_Telu": {
        "high": "",
        "low": "నాకు సంబంధించినది ఏదో కనపడింది, కానీ అది మీ ప్రశ్నకు సమాధానమిస్తుందో లేదో నాకు పూర్తిగా తెలియదు. — ",
        "refuse": "దానికి సమాధానం ఉన్న పాఠ్యాంశం నా వద్ద లేదు. నా మూలాలు వెబ్-శోధన ప్రశ్నల యొక్క చిన్న సమితిని మాత్రమే కవర్ చేస్తాయి. ",
    },
    "mar_Deva": {
        "high": "",
        "low": "मला संबंधित काहीतरी आढळले, परंतु ते तुमच्या प्रश्नाचे उत्तर देते का हे मला पूर्णपणे निश्चित नाही. — ",
        "refuse": "त्या प्रश्नाचे उत्तर देणारा लेख माझ्याकडे नाही. माझ्या स्रोतांमध्ये वेब-शोध प्रश्नांच्या एका छोट्या संचांचाच समावेश आहे. ",
    },
    "guj_Gujr": {
        "high": "",
        "low": "મને સંબંધિત કંઈક મળ્યું છે, પરંતુ મને સંપૂર્ણપણે ખાતરી નથી કે તે તમારા પ્રશ્નનો જવાબ આપે છે. — ",
        "refuse": "મારી પાસે એનો જવાબ આપતો કોઈ ફકરો નથી. મારા સ્રોતો વેબ-શોધ પ્રશ્નોના માત્ર એક નાના સમૂહને આવરી લે છે. ",
    },
    "kan_Knda": {
        "high": "",
        "low": "ನನಗೆ ಸಂಬಂಧಪಟ್ಟದ್ದು ಕಂಡುಬಂದಿತು, ಆದರೆ ಅದು ನಿಮ್ಮ ಪ್ರಶ್ನೆಗೆ ಉತ್ತರಿಸುತ್ತದೆಯೇ ಎಂದು ನನಗೆ ಸಂಪೂರ್ಣವಾಗಿ ಖಚಿತವಿಲ್ಲ. — ",
        "refuse": "ಆ ಪ್ರಶ್ನೆಗೆ ಉತ್ತರಿಸುವಂತಹ ಯಾವುದೇ ಪಠ್ಯ ಲಭ್ಯವಿಲ್ಲ ನನಗೆ. ನನ್ನ ಮೂಲಗಳು ವೆಬ್-ಸರ್ಚ್ ಪ್ರಶ್ನೆಗಳ ಒಂದು ಸಣ್ಣ ಸಮೂಹವನ್ನು ಮಾತ್ರ ಒಳಗೊಂಡಿವೆ. ",
    },
    "mal_Mlym": {
        "high": "",
        "low": "എനിക്ക് ബന്ധപ്പെട്ട എന്തെങ്കിലും കണ്ടെത്തി, എന്നാൽ അത് നിങ്ങളുടെ ചോദ്യത്തിന് ഉത്തരം നൽകുന്നുണ്ടോ എന്ന് എനിക്ക് പൂർണ്ണമായും ഉറപ്പില്ല. — ",
        "refuse": "അതിനുള്ള ഉത്തരം നൽകുന്ന ഒരു ഭാഗം എന്റെ പക്കൽ ഇല്ല. എന്റെ സ്രോതസ്സുകൾ വെബ്-തിരയൽ ചോദ്യങ്ങളുടെ ഒരു ചെറിയ സെറ്റ് മാത്രമേ ഉൾക്കൊള്ളുന്നുള്ളൂ. ",
    },
    "pan_Guru": {
        "high": "",
        "low": "ਮੈਨੂੰ ਸੰਬੰਧਿਤ ਕੁਝ ਮਿਲਿਆ, ਪਰ ਮੈਨੂੰ ਪੂਰੀ ਤਰ੍ਹਾਂ ਯਕੀਨ ਨਹੀਂ ਹੈ ਕਿ ਇਹ ਤੁਹਾਡੇ ਪ੍ਰਸ਼ਨ ਦਾ ਉੱਤਰ ਦਿੰਦਾ ਹੈ। — ",
        "refuse": "ਮੇਰੇ ਕੋਲ ਉਸਦਾ ਉੱਤਰ ਦੇਣ ਵਾਲਾ ਪਾਠ ਨਹੀਂ ਹੈ। ਮੇਰੇ ਸਰੋਤ ਵੈੱਬ-ਖੋਜ ਦੇ ਕੁੱਝ ਹੀ ਸਵਾਲਾਂ ਨੂੰ ਕਵਰ ਕਰਦੇ ਹਨ। ",
    },
    "ori_Orya": {
        "high": "",
        "low": "ମୁଁ କିଛି ସମ୍ବନ୍ଧିତ ପାଇଲି, କିନ୍ତୁ ଏହା ଆପଣଙ୍କ ପ୍ରଶ୍ନର ଉତ୍ତର ଦେଉଛି କି ନାହିଁ ମୁଁ ସମ୍ପୂର୍ଣ୍ଣ ଭାବରେ ନିଶ୍ଚିତ ନୁହେଁ। — ",
        "refuse": "ମୋର ପାଖରେ ସେଭଳି କୌଣସି ଅନୁଚ୍ଛେଦ ନାହିଁ ଯେଉଁଥିରେ ଏହାର ଉତ୍ତର ରହିବ। ମୋର ସୂତ୍ରଗୁଡ଼ିକ କେବଳ ଏକ ଛୋଟ ସେଟ୍ ୱେବ୍-ସର୍ଚ୍ଚ ପ୍ରଶ୍ନଗୁଡ଼ିକୁ କଭର କରେ। ",
    },
    "asm_Beng": {
        "high": "",
        "low": "I found something related, but I'm not fully sure this answers it — ",
        "refuse": "I don't have a passage that answers that. My sources cover only a small set of web-search questions. ",
    },
    "nep_Deva": {
        "high": "",
        "low": "I found something related, but I'm not fully sure this answers it — ",
        "refuse": "I don't have a passage that answers that. My sources cover only a small set of web-search questions. ",
    },
    "san_Deva": {
        "high": "",
        "low": "I found something related, but I'm not fully sure this answers it — ",
        "refuse": "I don't have a passage that answers that. My sources cover only a small set of web-search questions. ",
    },
    "urd_Arab": {
        "high": "",
        "low": "I found something related, but I'm not fully sure this answers it — ",
        "refuse": "I don't have a passage that answers that. My sources cover only a small set of web-search questions. ",
    },
}

# Spoken when the query guardrail rejects a question before retrieval runs.
OFF_TOPIC: dict[str, str] = {
    "eng_Latn": "That falls outside what this system covers. Try one of the example questions.",
    "hin_Deva": "यह इस प्रणाली के दायरे से बाहर है। कृपया नीचे दिए गए उदाहरणों में से कोई प्रश्न आज़माएँ।",
    "ben_Beng": "এটি এই সিস্টেমের আওতার বাইরে। নিচের উদাহরণ প্রশ্নগুলির একটি চেষ্টা করুন।",
    "tam_Taml": "இது இந்த அமைப்பின் எல்லைக்கு வெளியே உள்ளது. கீழே உள்ள எடுத்துக்காட்டுக் கேள்விகளில் ஒன்றை முயற்சிக்கவும்.",
    "tel_Telu": "ఈ వ్యవస్థ కవర్ చేసే వాటికి ఇది వెలుపల ఉంది. ఉదాహరణ ప్రశ్నలను ఒకటి ప్రయత్నించి చూడండి.",
    "mar_Deva": "हे या प्रणालीच्या व्याप्तीबाहेर आहे. उदाहरणातील प्रश्नांपैकी एक वापरून पहा.",
    "guj_Gujr": "તે આ પ્રણાલીમાં આવરી લેવામાં આવતું નથી. ઉદાહરણ પ્રશ્નોમાંથી એક પ્રયાસ કરો.",
    "kan_Knda": "ಅದು ಈ ವ್ಯವಸ್ಥೆಯು ಒಳಗೊಳ್ಳುವ ವಿಷಯಗಳಿಗೆ ಹೊರಗಿದೆ. ಮಾದರಿ ಪ್ರಶ್ನೆಗಳಲ್ಲಿ ಒಂದನ್ನು ಪ್ರಯತ್ನಿಸಿ ನೋಡಿ.",
    "mal_Mlym": "ഈ സംവിധാനം ഉൾക്കൊള്ളുന്നതിനു പുറത്താണിത്. ഉദാഹരണ ചോദ്യങ്ങളിൽ ഏതെങ്കിലും ഒന്ന് പരീക്ഷിച്ചു നോക്കൂ.",
    "pan_Guru": "ਇਹ ਇਸ ਪ੍ਰਣਾਲੀ ਦੇ ਅਧੀਨ ਨਹੀਂ ਆਉਂਦਾ। ਉਦਾਹਰਨ ਪ੍ਰਸ਼ਨਾਂ ਵਿੱਚੋਂ ਇੱਕ ਦੀ ਕੋਸ਼ਿਸ਼ ਕਰੋ।",
    "ori_Orya": "ଏହି ସିଷ୍ଟମ୍‌ର ପରିସର ବାହାରେ ଏହା ଅଛି। ଉଦାହରଣ ପ୍ରଶ୍ନଗୁଡ଼ିକ ମଧ୍ୟରୁ ଗୋଟିଏ ଚେଷ୍ଟା କରନ୍ତୁ।",
    "asm_Beng": "That falls outside what this system covers. Try one of the example questions.",
    "nep_Deva": "That falls outside what this system covers. Try one of the example questions.",
    "san_Deva": "That falls outside what this system covers. Try one of the example questions.",
    "urd_Arab": "That falls outside what this system covers. Try one of the example questions.",
}

# Appended when the answering passage came from another language.
CROSS_LINGUAL_NOTE: dict[str, str] = {
    "eng_Latn": " (answered from a source passage in {source})",
    "hin_Deva": " ({source} भाषा के स्रोत अंश से उत्तर दिया गया)",
    "ben_Beng": " ({source} ভাষার উৎস অনুচ্ছেদ থেকে উত্তর)",
    "tam_Taml": " ({source} மொழி மூலப் பத்தியிலிருந்து பதில்)",
    "tel_Telu": " ({source})",
    "mar_Deva": " ({source})",
    "guj_Gujr": " ({source})",
    "kan_Knda": " ({source})",
    "mal_Mlym": " ({source})",
    "pan_Guru": " ({source})",
    "ori_Orya": " ({source})",
    "asm_Beng": " (answered from a source passage in {source})",
    "nep_Deva": " (answered from a source passage in {source})",
    "san_Deva": " (answered from a source passage in {source})",
    "urd_Arab": " (answered from a source passage in {source})",
}


def prefix_for(lang: str, tier: Tier) -> str:
    return PREFIXES.get(lang, PREFIXES["eng_Latn"]).get(tier, "")


def off_topic_for(lang: str) -> str:
    return OFF_TOPIC.get(lang, OFF_TOPIC["eng_Latn"])


def cross_lingual_note(lang: str, source_name: str) -> str:
    template = CROSS_LINGUAL_NOTE.get(lang, CROSS_LINGUAL_NOTE["eng_Latn"])
    return template.format(source=source_name)
