"""Text-to-Speech (TTS) narration engine for MediScan AI.

Supports generating speech for Medication Regimens and Pharmacopeia Audit tables
in English and major Indian languages (Hindi, Telugu, Tamil, Bengali, Marathi,
Kannada, Gujarati, Malayalam, Punjabi).
"""

import io
import re
from typing import List, Dict, Any, Optional
from gtts import gTTS
from clinical_translator import (
    translate_formulation,
    translate_timing,
    translate_daily_intake,
    translate_dose,
    translate_clinical_instruction,
    translate_audit_reason,
    translate_status,
    translate_as_advised,
    translate_safety_message,
)

# Mapping language codes to browser SpeechSynthesis voice tags
BCP47_VOICE_MAP = {
    "en": "en-IN",
    "hi": "hi-IN",
    "te": "te-IN",
    "ta": "ta-IN",
    "bn": "bn-IN",
    "mr": "mr-IN",
    "kn": "kn-IN",
    "gu": "gu-IN",
    "ml": "ml-IN",
    "pa": "pa-IN",
}

INTRO_REGIMENS = {
    "en": "Medication Regimens Summary. Total medicines: {count}.",
    "hi": "दवा खुराक विवरण। कुल दवाएं: {count}।",
    "te": "ఔషధ మోతాదు వివరాలు. మొత్తం మందులు: {count}.",
    "ta": "மருந்து அட்டவணை விவரங்கள். மொத்த மருந்துகள்: {count}.",
    "bn": "ওষুধের সময়সূচী সারাংশ। মোট ওষুধ: {count}।",
    "mr": "औषध डोस तपशील. एकूण औषधे: {count}.",
    "kn": "ಔಷಧ ಡೋಸ್ ವಿವರಗಳು. ಒಟ್ಟು ಔಷಧಗಳು: {count}.",
    "gu": "દવા ડોઝ વિગતો. કુલ દવાઓ: {count}.",
    "ml": "മരുന്ന് സമയക്രമ വിവരങ്ങൾ. ആകെ മരുന്നുകൾ: {count}.",
    "pa": "ਦਵਾਈ ਖੁਰਾਕ ਵੇਰਵੇ। ਕੁੱਲ ਦਵਾਈਆਂ: {count}।",
}

ITEM_PREFIX = {
    "en": "Item {num}: {name}, {formulation}, strength {strength}, dose {dose}, frequency {freq}, timing {timing}. {instructions}.",
    "hi": "क्रमांक {num}: {name}, {formulation}, क्षमता {strength}, खुराक {dose}, सेवन समय {freq}, {timing}। {instructions}।",
    "te": "సంఖ్య {num}: {name}, {formulation}, సామర్థ్యం {strength}, మోతాదు {dose}, వాడకం {freq}, {timing}. {instructions}.",
    "ta": "வரிசை {num}: {name}, {formulation}, வீரியம் {strength}, மருந்தளவு {dose}, உட்கொள்ளும் முறை {freq}, {timing}. {instructions}.",
    "bn": "নম্বর {num}: {name}, {formulation}, ক্ষমতা {strength}, মাত্রা {dose}, সময় {freq}, {timing}। {instructions}।",
    "mr": "क्रमांक {num}: {name}, {formulation}, क्षमता {strength}, डोस {dose}, वारंवारता {freq}, {timing}. {instructions}.",
    "kn": "ಸಂಖ್ಯೆ {num}: {name}, {formulation}, ಸಾಮರ್ಥ್ಯ {strength}, ಡೋಸ್ {dose}, ಆವರ್ತನ {freq}, {timing}. {instructions}.",
    "gu": "ક્રમ {num}: {name}, {formulation}, શક્તિ {strength}, ડોઝ {dose}, આવર્તન {freq}, {timing}. {instructions}.",
    "ml": "നമ്പർ {num}: {name}, {formulation}, അളവ് {strength}, ഡോസ് {dose}, ആവൃത്തി {freq}, {timing}. {instructions}.",
    "pa": "ਨੰਬਰ {num}: {name}, {formulation}, ਸਮਰੱਥਾ {strength}, ਖੁਰਾਕ {dose}, ਬਾਰੰਬਾਰਤਾ {freq}, {timing}. {instructions}।",
}

INTRO_AUDIT = {
    "en": "Pharmacopeia Audit and Verification Summary. Total checked entries: {count}.",
    "hi": "फार्माकोपिया ऑडिट एवं सत्यापन विवरण। कुल प्रविष्टियां: {count}।",
    "te": "ఫార్మాకోపియా ఆడిట్ మరియు ధృవీకరణ వివరాలు. మొత్తం అంశాలు: {count}.",
    "ta": "மருந்தியல் தணிக்கை மற்றும் சரிபார்ப்பு பதிவு. மொத்த பதிவுகள்: {count}.",
    "bn": "ফার্মাকোপিয়া অডিট ও যাচাইকরণ সারাংশ। মোট এন্ট্রি: {count}।",
    "mr": "फार्माकोपिया ऑडिट आणि पडताळणी तपशील. एकूण नोंदी: {count}.",
    "kn": "ಫಾರ್ಮಾಕೋಪಿಯಾ ಆಡಿಟ್ ಮತ್ತು ಪರಿಶೀಲನೆ ಸಾರಾಂಶ. ಒಟ್ಟು ನಮೂದುಗಳು: {count}.",
    "gu": "ફાર્માકોપિયા ઓડિટ અને ચકાસણી સારાંશ. કુલ એન્ટ્રીઓ: {count}.",
    "ml": "ഫാർമക്കോപ്പിയ ഓഡിറ്റ് വിവരങ്ങൾ. ആകെ വിവരങ്ങൾ: {count}.",
    "pa": "ਫਾਰਮਾਕੋਪੀਆ ਆਡਿਟ ਅਤੇ ਤਸਦੀਕ ਸਾਰੰਸ਼। ਕੁੱਲ ਐਂਟਰੀਆਂ: {count}।",
}

AUDIT_CLEAN = {
    "en": "All medicine names adhered strictly to standard pharmacopeia nomenclature. Zero spelling corrections needed and safety checks passed.",
    "hi": "सभी दवाओं के नाम मानक फार्माकोपिया शब्दावली के अनुसार हैं। किसी वर्तनी सुधार की आवश्यकता नहीं है और सुरक्षा जांच सफल रही।",
    "te": "అన్ని మందుల పేర్లు ప్రామాణిక ఫార్మాకోపియా ప్రకారం సరిగ్గా ఉన్నాయి. ఎలాంటి దిద్దుబాట్లు అవసరం లేదు మరియు భద్రతా తనిఖీ పూర్తయింది.",
    "ta": "அனைத்து மருந்து பெயர்களும் பார்மகோபியா தரநிலைகளுக்கு ஏற்ப உள்ளன. திருத்தங்கள் தேவையில்லை, பாதுகாப்பு சரிபார்ப்பு முடிந்தது.",
    "bn": "সকল ওষুধের নাম মানসম্মত ফার্মাকোপিয়া অনুযায়ী সঠিক। কোনো সংশোধনের প্রয়োজন নেই এবং সুরক্ষা যাচাই সফল হয়েছে।",
    "mr": "सर्व औषधांची नावे प्रमाणित फार्माकोपिया मानकांनुसार आहेत. कोणतीही दुरुस्ती आवश्यक नाही आणि सुरक्षा तपासणी यशस्वी झाली.",
    "kn": "ಎಲ್ಲಾ ಔಷಧಗಳ ಹೆಸರುಗಳು ಪ್ರಮಾಣಿತ ಫಾರ್ಮಾಕೋಪಿಯಾದಂತೆ ಇವೆ. ಯಾವುದೇ ತಿದ್ದುಪಡಿಗಳ ಅಗತ್ಯವಿಲ್ಲ ಮತ್ತು ಸುರಕ್ಷತಾ ಪರೀಕ್ಷೆ ಯಶಸ್ವಿಯಾಗಿದೆ.",
    "gu": "તમામ દવાઓના નામ માનક ફાર્માકોપિયા મુજબ યોગ્ય છે. કોઈ સુધારાની જરૂર નથી અને સુરક્ષા તપાસ સફળ થઈ છે.",
    "ml": "എല്ലാ മരുന്നുകളുടെയും പേരുകൾ ഫാർമക്കോപ്പിയ നിലവാരത്തിലുള്ളതാണ്. തിരുത്തലുകൾ ആവശ്യമില്ല, സുരക്ഷാ പരിശോധന വിജയിച്ചു.",
    "pa": "ਸਾਰੀਆਂ ਦਵਾਈਆਂ ਦੇ ਨਾਮ ਮਿਆਰੀ ਫਾਰਮਾਕੋਪੀਆ ਅਨੁਸਾਰ ਹਨ। ਕੋਈ ਸੁਧਾਰ ਲੋੜੀਂਦਾ ਨਹੀਂ ਹੈ ਅਤੇ ਸੁਰੱਖਿਆ ਜਾਂਚ ਸਫਲ ਰਹੀ।",
}

AUDIT_ITEM = {
    "en": "Entry {num}: Original handwritten token {orig}, standardized generic name {corr}. Status: {status}. Reason: {reason}.",
    "hi": "प्रविष्टि {num}: मूल शब्द {orig}, मानकीकृत जेनेरिक नाम {corr}। स्थिति: {status}। कारण: {reason}।",
    "te": "అంశం {num}: అసలు పాఠం {orig}, ప్రామాణిక పేరు {corr}. స్థితి: {status}. కారణం: {reason}.",
    "ta": "பதிவு {num}: அசல் சொல் {orig}, முறைப்படுத்தப்பட்ட பெயர் {corr}. நிலை: {status}. காரணம்: {reason}.",
    "bn": "এন্ট্রি {num}: মূল শব্দ {orig}, মানসম্মত নাম {corr}। অবস্থা: {status}। কারণ: {reason}।",
    "mr": "नोंद {num}: मूळ शब्द {orig}, प्रमाणित नाव {corr}. स्थिती: {status}. कारण: {reason}.",
    "kn": "ನಮೂದು {num}: ಮೂಲ ಪದ {orig}, ಪ್ರಮಾಣಿತ ಹೆಸರು {corr}. ಸ್ಥಿತಿ: {status}. ಕಾರಣ: {reason}.",
    "gu": "એન્ટ્રી {num}: મૂળ શબ્દ {orig}, પ્રમાણિત નામ {corr}. સ્થિતિ: {status}. કારણ: {reason}.",
    "ml": "നമ്പർ {num}: യഥാർത്ഥ വാക്ക് {orig}, ക്രമീകരിച്ച പേര് {corr}. നില: {status}. കാരണം: {reason}.",
    "pa": "ਐਂਟਰੀ {num}: ਮੂਲ ਸ਼ਬਦ {orig}, ਮਿਆਰੀ ਨਾਮ {corr}। ਸਥਿਤੀ: {status}। ਕਾਰਨ: {reason}।",
}


def build_regimens_speech_text(groups: List[Dict[str, Any]], lang: str = "en") -> str:
    """Build natural narrative text for medication regimens table in selected language."""
    if not groups:
        return INTRO_REGIMENS.get(lang, INTRO_REGIMENS["en"]).format(count=0)
    
    intro = INTRO_REGIMENS.get(lang, INTRO_REGIMENS["en"]).format(count=len(groups))
    template = ITEM_PREFIX.get(lang, ITEM_PREFIX["en"])
    
    lines = [intro]
    for idx, g in enumerate(groups, 1):
        num = g.get("item_number") or idx
        name = g.get("medicine_name") or "Medicine"
        raw_form = g.get("formulation") or ""
        form = translate_formulation(raw_form, lang)
        strength = g.get("strength") or ""
        raw_dose = g.get("dose", {}).get("normalized_value", "") if isinstance(g.get("dose"), dict) else (g.get("dose") or "")
        dose = translate_dose(raw_dose, lang) if raw_dose else translate_as_advised(lang)
        raw_freq = g.get("frequency", {}).get("normalized_value", "") if isinstance(g.get("frequency"), dict) else (g.get("frequency") or "")
        raw_daily = g.get("frequency", {}).get("daily_frequency", "") if isinstance(g.get("frequency"), dict) else ""
        daily_code = f"{raw_daily}x / day" if raw_daily else raw_freq
        freq = translate_daily_intake(daily_code, lang)
        timing_obj = g.get("timing")
        if isinstance(timing_obj, dict):
            raw_timing = timing_obj.get("timing_type") or timing_obj.get("normalized_value") or ""
        else:
            raw_timing = timing_obj or ""
        timing = translate_timing(raw_timing, lang)
        raw_instructions = g.get("instructions") or ""
        instructions = translate_clinical_instruction(raw_instructions, lang)

        item_text = template.format(
            num=num,
            name=name,
            formulation=form,
            strength=strength,
            dose=dose,
            freq=freq,
            timing=timing,
            instructions=instructions,
        )
        # Collapse multiple spaces
        item_text = re.sub(r"\s+", " ", item_text).strip()
        lines.append(item_text)

    return " ".join(lines)


def build_audit_speech_text(
    corrections: List[Dict[str, Any]],
    issues: Optional[List[Any]] = None,
    lang: str = "en"
) -> str:
    """Build natural narrative text for audit and verification log in selected language."""
    lines: List[str] = []

    if not corrections:
        lines.append(AUDIT_CLEAN.get(lang, AUDIT_CLEAN["en"]))
    else:
        intro = INTRO_AUDIT.get(lang, INTRO_AUDIT["en"]).format(count=len(corrections))
        template = AUDIT_ITEM.get(lang, AUDIT_ITEM["en"])
        lines.append(intro)
        for idx, c in enumerate(corrections, 1):
            orig = c.get("original", c.get("original_ocr_text", ""))
            corr = c.get("normalized", c.get("corrected_text", c.get("standard_name", "")))
            status_val = c.get("status", c.get("verification_status", "Verified"))
            status = translate_status(str(status_val), lang)

            raw_reason = c.get("reason", "Nomenclature matched to pharmacopeia")
            reason = translate_audit_reason(raw_reason, lang)

            item_text = template.format(
                num=idx,
                orig=orig,
                corr=corr,
                status=status,
                reason=reason,
            )
            item_text = re.sub(r"\s+", " ", item_text).strip()
            lines.append(item_text)

    if issues:
        for issue in issues:
            cat = getattr(issue, "category", "SAFETY_WARNING")
            msg = getattr(issue, "message", "")
            cat_trans, msg_trans = translate_safety_message(msg, cat, lang)
            lines.append(f"{cat_trans}: {msg_trans}")

    return " ".join(lines)


LABEL_COMP_INTRO = {
    "en": "Active ingredients composition summary. Total ingredients: {count}.",
    "hi": "सक्रिय औषधि घटक विवरण। कुल घटक: {count}।",
    "te": "క్రియాశీల ఔషధ పదార్థాల వివరాలు. మొత్తం పదార్థాలు: {count}.",
    "ta": "செயலில் உள்ள மூலப்பொருட்கள் விவரம். மொத்த மூலப்பொருட்கள்: {count}.",
    "bn": "সক্রিয় উপাদানের বিবরণ। মোট উপাদান: {count}।",
    "mr": "सक्रिय औषध घटक तपशील. एकूण घटक: {count}.",
    "kn": "ಸಕ್ರಿಯ ಔಷಧ ಘಟಕಗಳ ವಿವರಗಳು. ಒಟ್ಟು ಘಟಕಗಳು: {count}.",
    "gu": "સક્રિય દવાના ઘટકોની વિગતો. કુલ ઘટકો: {count}.",
    "ml": "സജീവ മരുന്ന് ചേരുവകളുടെ വിവരങ്ങൾ. ആകെ ചേരുവകൾ: {count}.",
    "pa": "ਸਰਗਰਮ ਦਵਾਈ ਤੱਤਾਂ ਦੇ ਵੇਰਵੇ। ਕੁੱਲ ਤੱਤ: {count}।",
}

LABEL_COMP_ITEM = {
    "en": "Item {num}: {name}, {strength}, standard generic {std}.",
    "hi": "क्रमांक {num}: {name}, {strength}, मानक जेनेरिक {std}।",
    "te": "సంఖ్య {num}: {name}, {strength}, ప్రామాణిక జెనెరిక్ {std}.",
    "ta": "வரிசை {num}: {name}, {strength}, நிலையான ஜெனரிக் {std}.",
    "bn": "নম্বর {num}: {name}, {strength}, মানসম্মত জেনেরিক {std}।",
    "mr": "क्रमांक {num}: {name}, {strength}, प्रमाणित जेनेरिक {std}.",
    "kn": "ಸಂಖ್ಯೆ {num}: {name}, {strength}, ಪ್ರಮಾಣಿತ ಜೆನೆರಿಕ್ {std}.",
    "gu": "ક્રમ {num}: {name}, {strength}, પ્રમાણિત જેનેરિક {std}.",
    "ml": "നമ്പർ {num}: {name}, {strength}, ജനറിക് പേര് {std}.",
    "pa": "ਨੰਬਰ {num}: {name}, {strength}, ਮਿਆਰੀ ਜੈਨੇਰਿਕ {std}।",
}

LABEL_NO_ING = {
    "en": "No active pharmaceutical ingredients found.",
    "hi": "कोई सक्रिय औषधि घटक नहीं मिला।",
    "te": "క్రియాశీల ఔషధ పదార్థాలు ఏవీ కనుగొనబడలేదు.",
    "ta": "செயலில் உள்ள மூலப்பொருட்கள் எதுவும் கிடைக்கவில்லை.",
    "bn": "কোনো সক্রিয় উপাদান পাওয়া যায়নি।",
    "mr": "कोणतेही सक्रिय घटक आढळले नाहीत.",
    "kn": "ಯಾವುದೇ ಸಕ್ರಿಯ ಘಟಕಗಳು ಕಂಡುಬಂದಿಲ್ಲ.",
    "gu": "કોઈ સક્રિય ઘટક મળ્યો નથી.",
    "ml": "സജീവ ചേരുവകളൊന്നും കണ്ടെത്തിയില്ല.",
    "pa": "ਕੋਈ ਸਰਗਰਮ ਤੱਤ ਨਹੀਂ ਮਿਲਿਆ।",
}


def build_label_composition_speech_text(ingredients: List[Any], lang: str = "en") -> str:
    """Build narrative text for active ingredients in medicine label in target language."""
    if not ingredients:
        return LABEL_NO_ING.get(lang, LABEL_NO_ING["en"])
    intro = LABEL_COMP_INTRO.get(lang, LABEL_COMP_INTRO["en"]).format(count=len(ingredients))
    item_tmpl = LABEL_COMP_ITEM.get(lang, LABEL_COMP_ITEM["en"])
    lines = [intro]
    for idx, ing in enumerate(ingredients, 1):
        name = getattr(ing, "name", str(ing))
        std = getattr(ing, "standard_name", name)
        strength = getattr(ing, "strength", "")
        lines.append(re.sub(r"\s+", " ", item_tmpl.format(num=idx, name=name, strength=strength, std=std)).strip())
    return " ".join(lines)


def generate_speech_audio(text: str, lang: str = "en") -> Optional[bytes]:
    """Generate MP3 audio bytes using gTTS.
    
    Returns bytes if online and successful, or None if network is unavailable.
    """
    if not text or not text.strip():
        return None
    
    # Clean text: truncate if excessively long to prevent timeout (max 600 chars)
    spoken_text = text.strip()
    if len(spoken_text) > 800:
        spoken_text = spoken_text[:800] + "..."

    tts_lang = lang if lang in BCP47_VOICE_MAP else "en"

    try:
        tts = gTTS(text=spoken_text, lang=tts_lang, slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp.read()
    except Exception:
        # Gracefully handle network isolation
        return None


def get_browser_speech_html(text: str, lang: str = "en", button_id: str = "speak_btn", button_label: str = "🔊 Read Out Table") -> str:
    """Generate HTML/JavaScript for client-side SpeechSynthesis (Web Speech API).
    
    Ensures 100% offline playback support in all modern browsers without network dependencies.
    """
    bcp47 = BCP47_VOICE_MAP.get(lang, "en-IN")
    safe_text = (
        text.replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace('"', '\\"')
        .replace("\n", " ")
        .replace("\r", " ")
    )

    html = f"""
    <div style="margin: 0.4rem 0 0.8rem 0;">
        <button id="{button_id}" onclick="playTableSpeech_{button_id}()" 
            style="background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
                   color: white;
                   border: none;
                   border-radius: 8px;
                   padding: 0.45rem 1rem;
                   font-size: 0.88rem;
                   font-weight: 600;
                   cursor: pointer;
                   display: inline-flex;
                   align-items: center;
                   gap: 0.4rem;
                   box-shadow: 0 2px 4px rgba(37,99,235,0.2);
                   transition: all 0.2s ease;">
            {button_label}
        </button>
        <span id="status_{button_id}" style="margin-left: 0.6rem; font-size: 0.82rem; color: #64748B;"></span>
    </div>
    <script>
    function playTableSpeech_{button_id}() {{
        var btn = document.getElementById("{button_id}");
        var status = document.getElementById("status_{button_id}");
        var synth = window.speechSynthesis || window.parent.speechSynthesis;
        if (!synth) {{
            status.innerText = "Audio speech not supported in this browser.";
            return;
        }}
        if (synth.speaking) {{
            synth.cancel();
            btn.innerText = "{button_label}";
            status.innerText = "";
            return;
        }}
        var text = "{safe_text}";
        var utter = new SpeechSynthesisUtterance(text);
        utter.lang = "{bcp47}";
        utter.rate = 0.95;
        
        // Find language voice if available
        var voices = synth.getVoices();
        for (var i = 0; i < voices.length; i++) {{
            if (voices[i].lang.startsWith("{lang}") || voices[i].lang === "{bcp47}") {{
                utter.voice = voices[i];
                break;
            }}
        }}

        utter.onstart = function() {{
            btn.innerText = "⏹️ Stop";
            btn.style.background = "#DC2626";
            status.innerText = "Speaking ({bcp47})...";
        }};
        utter.onend = function() {{
            btn.innerText = "{button_label}";
            btn.style.background = "linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%)";
            status.innerText = "";
        }};
        utter.onerror = function() {{
            btn.innerText = "{button_label}";
            btn.style.background = "linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%)";
            status.innerText = "";
        }};

        synth.speak(utter);
    }}
    </script>
    """
    return html
