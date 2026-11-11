"""
Translation service for MediScan AI.
Supports online neural translation with offline clinical template fallback.
Enforces medical safety to preserve drug names, dosages, and units.
"""

import json
import logging
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Dict, Optional

from voice_assistant.medical_safety import MedicalSafetyGuard

logger = logging.getLogger("TranslationService")


@dataclass
class TranslationResult:
    translated_text: str
    original_text: str
    target_language: str
    engine_used: str  # "online_google" | "offline_clinical_lexicon" | "verbatim_fallback"
    is_fallback: bool
    integrity_verified: bool


# Curated offline clinical lexicon for medical instructions across all 8 languages
OFFLINE_CLINICAL_LEXICON: Dict[str, Dict[str, str]] = {
    "te": {
        "Take one tablet twice daily.": "రోజుకు రెండుసార్లు ఒక టాబ్లెట్ తీసుకోండి.",
        "Take one tablet once daily.": "రోజుకు ఒకసారి ఒక టాబ్లెట్ తీసుకోండి.",
        "Take one tablet three times daily.": "రోజుకు మూడుసార్లు ఒక టాబ్లెట్ తీసుకోండి.",
        "Take one capsule twice daily.": "రోజుకు రెండుసార్లు ఒక క్యాప్సూల్ తీసుకోండి.",
        "Take one capsule once daily.": "రోజుకు ఒకసారి ఒక క్యాప్సూల్ తీసుకోండి.",
        "Take one capsule three times daily.": "రోజుకు మూడుసార్లు ఒక క్యాప్సూల్ తీసుకోండి.",
        "Take as needed for fever.": "జ్వరం వచ్చినప్పుడు అవసరాన్ని బట్టి తీసుకోండి.",
        "Take as needed for pain.": "నొప్పి ఉన్నప్పుడు అవసరాన్ని బట్టి తీసుకోండి.",
        "Take before food.": "ఆహారానికి ముందు తీసుకోండి.",
        "Take after food.": "ఆహారం తర్వాత తీసుకోండి.",
        "As directed by the physician.": "వైద్యుడు సూచించినట్లు వాడండి.",
        "Store in a cool and dry place.": "చల్లని మరియు పొడి ప్రదేశంలో నిల్వ చేయండి.",
        "Keep out of reach of children.": "పిల్లలకు దూరంగా ఉంచండి.",
        "Prescription for": "రోగి ప్రిస్క్రిప్షన్:",
        "Total Medications": "మొత్తం మందులు",
    },
    "hi": {
        "Take one tablet twice daily.": "दिन में दो बार एक गोली लें।",
        "Take one tablet once daily.": "दिन में एक बार एक गोली लें।",
        "Take one tablet three times daily.": "दिन में तीन बार एक गोली लें।",
        "Take one capsule twice daily.": "दिन में दो बार एक कैप्सूल लें।",
        "Take one capsule once daily.": "दिन में एक बार एक कैप्सूल लें।",
        "Take one capsule three times daily.": "दिन में तीन बार एक कैप्सूल लें।",
        "Take as needed for fever.": "बुखार होने पर आवश्यकतानुसार लें।",
        "Take as needed for pain.": "दर्द होने पर आवश्यकतानुसार लें।",
        "Take before food.": "भोजन से पहले लें।",
        "Take after food.": "भोजन के बाद लें।",
        "As directed by the physician.": "चिकित्सक के निर्देशानुसार लें।",
        "Store in a cool and dry place.": "ठंडी और सूखी जगह पर रखें।",
        "Keep out of reach of children.": "बच्चों की पहुँच से दूर रखें।",
        "Prescription for": "मरीज का पर्चा:",
        "Total Medications": "कुल दवाएं",
    },
    "ta": {
        "Take one tablet twice daily.": "ஒரு மாத்திரையை ஒரு நாளைக்கு இரண்டு முறை எடுத்துக் கொள்ளுங்கள்.",
        "Take one tablet once daily.": "ஒரு மாத்திரையை ஒரு நாளைக்கு ஒரு முறை எடுத்துக் கொள்ளுங்கள்.",
        "Take one tablet three times daily.": "ஒரு மாத்திரையை ஒரு நாளைக்கு மூன்று முறை எடுத்துக் கொள்ளுங்கள்.",
        "Take one capsule twice daily.": "ஒரு காப்ஸ்யூலை ஒரு நாளைக்கு இரண்டு முறை எடுத்துக் கொள்ளுங்கள்.",
        "Take one capsule once daily.": "ஒரு காப்ஸ்யூலை ஒரு நாளைக்கு ஒரு முறை எடுத்துக் கொள்ளுங்கள்.",
        "Take as needed for fever.": "காய்ச்சலுக்கு தேவைக்கேற்ப எடுத்துக் கொள்ளுங்கள்.",
        "Take before food.": "உணவுக்கு முன் எடுத்துக் கொள்ளுங்கள்.",
        "Take after food.": "உணவுக்குப் பின் எடுத்துக் கொள்ளுங்கள்.",
        "As directed by the physician.": "மருத்துவர் அறிவுறுத்தியபடி.",
        "Store in a cool and dry place.": "குளிர்ந்த மற்றும் உலர்ந்த இடத்தில் வைக்கவும்.",
        "Keep out of reach of children.": "குழந்தைகளுக்கு எட்டாதவாறு வைக்கவும்.",
    },
    "kn": {
        "Take one tablet twice daily.": "ದಿನಕ್ಕೆ ಎರಡು ಬಾರಿ ಒಂದು ಮಾತ್ರೆ ತೆಗೆದುಕೊಳ್ಳಿ.",
        "Take one tablet once daily.": "ದಿನಕ್ಕೆ ಒಮ್ಮೆ ಒಂದು ಮಾತ್ರೆ ತೆಗೆದುಕೊಳ್ಳಿ.",
        "Take one tablet three times daily.": "ದಿನಕ್ಕೆ ಮೂರು ಬಾರಿ ಒಂದು ಮಾತ್ರೆ ತೆಗೆದುಕೊಳ್ಳಿ.",
        "Take as needed for fever.": "ಜ್ವರ ಬಂದಾಗ ಅಗತ್ಯಕ್ಕೆ ತಕ್ಕಂತೆ ತೆಗೆದುಕೊಳ್ಳಿ.",
        "Take before food.": "ಊಟಕ್ಕೆ ಮುಂಚೆ ತೆಗೆದುಕೊಳ್ಳಿ.",
        "Take after food.": "ಊಟದ ನಂತರ ತೆಗೆದುಕೊಳ್ಳಿ.",
        "As directed by the physician.": "ವೈದ್ಯರ ಸಲಹೆಯಂತೆ.",
        "Store in a cool and dry place.": "ತಂಪಾದ ಮತ್ತು ಒಣ ಸ್ಥಳದಲ್ಲಿ ಇರಿಸಿ.",
        "Keep out of reach of children.": "ಮಕ್ಕಳಿಗೆ ತಲುಪದಂತೆ ಇರಿಸಿ.",
    },
    "ml": {
        "Take one tablet twice daily.": "ദിവസത്തിൽ രണ്ട് തവണ ഒരു ഗുളിക കഴിക്കുക.",
        "Take one tablet once daily.": "ദിവസത്തിൽ ഒരു തവണ ഒരു ഗുളിക കഴിക്കുക.",
        "Take as needed for fever.": "പനി വരുമ്പോൾ ആവശ്യാനുസരണം കഴിക്കുക.",
        "Take before food.": "ഭക്ഷണത്തിന് മുൻപ് കഴിക്കുക.",
        "Take after food.": "ഭക്ഷണത്തിന് ശേഷം കഴിക്കുക.",
        "As directed by the physician.": "ഡോക്ടറുടെ നിർദ്ദേശപ്രകാരം.",
        "Store in a cool and dry place.": "തണുത്തതും വരണ്ടതുമായ സ്ഥലത്ത് സൂക്ഷിക്കുക.",
    },
    "bn": {
        "Take one tablet twice daily.": "দিনে দুইবার একটি করে ট্যাবলেট খান।",
        "Take one tablet once daily.": "দিনে একবার একটি করে ট্যাবলেট খান।",
        "Take as needed for fever.": "জ্বর হলে প্রয়োজন অনুযায়ী খান।",
        "Take before food.": "খাওয়ার আগে খাবেন।",
        "Take after food.": "খাওয়ার পরে খাবেন।",
        "As directed by the physician.": "চিকিৎসকের নির্দেশ অনুযায়ী।",
        "Store in a cool and dry place.": "ঠান্ডা ও শুষ্ক স্থানে রাখুন।",
    },
    "mr": {
        "Take one tablet twice daily.": "दिवसातून दोनदा एक गोळी घ्या.",
        "Take one tablet once daily.": "दिवसातून एकदा एक गोळी घ्या.",
        "Take as needed for fever.": "ताप आल्यास गरजेनुसार घ्या.",
        "Take before food.": "जेवणापूर्वी घ्या.",
        "Take after food.": "जेवणानंतर घ्या.",
        "As directed by the physician.": "डॉक्टरांच्या सल्ल्यानुसार घ्या.",
        "Store in a cool and dry place.": "थंड आणि कोरड्या जागी ठेवा.",
    },
}


class TranslationService:
    """
    Hybrid Translation Service.
    Enforces medical dosage and brand protection while translating clinical instructions.
    """

    def __init__(self, timeout_sec: int = 5):
        self.timeout_sec = timeout_sec

    def translate(
        self,
        text: str,
        target_lang: str,
        prefer_offline: bool = False,
    ) -> TranslationResult:
        """
        Translate text into target language with fallback support.
        If target_lang is 'en', returns original text directly.
        """
        if not text or not text.strip():
            return TranslationResult(
                translated_text="",
                original_text=text,
                target_language=target_lang,
                engine_used="none",
                is_fallback=False,
                integrity_verified=True,
            )

        if target_lang == "en":
            return TranslationResult(
                translated_text=text,
                original_text=text,
                target_language="en",
                engine_used="native_english",
                is_fallback=False,
                integrity_verified=True,
            )

        # 1. If prefer_offline, try offline lexicon first
        if prefer_offline:
            offline_res = self._translate_offline(text, target_lang)
            if offline_res:
                return offline_res

        # 2. Try online translation
        online_res = self._translate_online(text, target_lang)
        if online_res:
            return online_res

        # 3. Fallback to offline clinical lexicon
        offline_res = self._translate_offline(text, target_lang)
        if offline_res:
            return offline_res

        # 4. Final safety fallback: preserve original text verbatim
        return TranslationResult(
            translated_text=text,
            original_text=text,
            target_language=target_lang,
            engine_used="verbatim_fallback",
            is_fallback=True,
            integrity_verified=True,
        )

    def _translate_online(self, text: str, target_lang: str) -> Optional[TranslationResult]:
        """Translate via Google Translate endpoint with medical token preservation."""
        try:
            # Protect numerical values, units, and drug names
            protected_map = MedicalSafetyGuard.extract_protected_tokens(text)
            safe_text = MedicalSafetyGuard.protect_text(text, protected_map)

            url = (
                f"https://translate.googleapis.com/translate_a/single?"
                f"client=gtx&sl=en&tl={target_lang}&dt=t&q=" + urllib.parse.quote(safe_text)
            )
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) MediScanAI/2.0"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode("utf-8"))
                    raw_translated = "".join([item[0] for item in data[0] if item and item[0]])
                    # Restore protected tokens
                    restored_text = MedicalSafetyGuard.restore_protected_text(raw_translated, protected_map)
                    # Verify integrity
                    integrity = MedicalSafetyGuard.verify_dosage_integrity(text, restored_text)

                    return TranslationResult(
                        translated_text=restored_text,
                        original_text=text,
                        target_language=target_lang,
                        engine_used="online_google",
                        is_fallback=False,
                        integrity_verified=integrity,
                    )
        except Exception as e:
            logger.warning(f"Online translation failed: {e}. Falling back to offline lexicon.")
            return None

    def _translate_offline(self, text: str, target_lang: str) -> Optional[TranslationResult]:
        """Translate using curated offline clinical phrasebook."""
        lexicon = OFFLINE_CLINICAL_LEXICON.get(target_lang, {})
        translated_lines = []
        lines = text.split("\n")
        matched_any = False

        for line in lines:
            line_strip = line.strip()
            if line_strip in lexicon:
                translated_lines.append(lexicon[line_strip])
                matched_any = True
            else:
                # Line-level fuzzy substitution for common fragments
                translated_line = line
                for eng_phrase, native_phrase in lexicon.items():
                    if eng_phrase.lower() in line.lower():
                        translated_line = translated_line.replace(eng_phrase, native_phrase)
                        matched_any = True
                translated_lines.append(translated_line)

        if matched_any:
            res_text = "\n".join(translated_lines)
            integrity = MedicalSafetyGuard.verify_dosage_integrity(text, res_text)
            return TranslationResult(
                translated_text=res_text,
                original_text=text,
                target_language=target_lang,
                engine_used="offline_clinical_lexicon",
                is_fallback=True,
                integrity_verified=integrity,
            )

        return None
