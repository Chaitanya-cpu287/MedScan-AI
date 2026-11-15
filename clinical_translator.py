"""Clinical Translation & Multilingual Normalization Engine for MediScan AI.

Translates clinical instructions, formulations, dosages, timings, frequencies,
audit reasons, label notices, and safety issues into English and major Indian languages
(Hindi, Telugu, Tamil, Bengali, Marathi, Kannada, Gujarati, Malayalam, Punjabi).
"""

import re
from typing import Dict, List, Any, Optional, Tuple

FORMULATIONS_MAP: Dict[str, Dict[str, str]] = {
    "tab": {
        "en": "Tablet", "hi": "टैबलेट", "te": "టాబ్లెట్", "ta": "மாத்திரை",
        "bn": "ট্যাবলেট", "mr": "टॅबलेट", "kn": "ಟ್ಯಾಬ್ಲೆಟ್", "gu": "ટેબ્લેટ",
        "ml": "ഗുളിക", "pa": "ਟੈਬਲੇਟ"
    },
    "cap": {
        "en": "Capsule", "hi": "कैप्सूल", "te": "క్యాప్సూల్", "ta": "கேப்சூல்",
        "bn": "ক্যাপসুল", "mr": "कॅप्सूल", "kn": "ಕ್ಯಾಪ್ಸುಲ್", "gu": "કેપ્સ્યુલ",
        "ml": "ക്യാപ്സൂൾ", "pa": "ਕੈਪਸੂਲ"
    },
    "syp": {
        "en": "Syrup", "hi": "सिरप", "te": "సిరప్", "ta": "சிரப்",
        "bn": "সিরাপ", "mr": "सिरप", "kn": "ಸಿರಪ್", "gu": "સિરપ",
        "ml": "സിറപ്പ്", "pa": "ਸ਼ਰਬਤ / ਸਿਰਪ"
    },
    "susp": {
        "en": "Suspension", "hi": "सस्पेंशन / सिरप", "te": "సస్పెన్షన్ / సిరప్", "ta": "திரவ மருந்து",
        "bn": "সাসপেনশন / সিরাপ", "mr": "सस्पेन्शन / सिरप", "kn": "ಸಸ್ಪೆನ್ಷನ್ / ಸಿರಪ್", "gu": "સસ્પેન્શન / સિરપ",
        "ml": "സിറപ്പ്", "pa": "ਸਿਰਪ"
    },
    "inj": {
        "en": "Injection", "hi": "इंजेक्शन", "te": "ఇంజెక్షన్", "ta": "ஊசி",
        "bn": "ইনজেকশন", "mr": "इंजेक्शन", "kn": "ಇಂಜೆಕ್ಷನ್", "gu": "ઇન્જેક્શન",
        "ml": "കുത്തിവയ്പ്പ്", "pa": "ਇੰਜੈਕਸ਼ਨ"
    },
    "oint": {
        "en": "Ointment", "hi": "मरहम", "te": "మందు లేపనం", "ta": "களிம்பு",
        "bn": "মলম", "mr": "मलम", "kn": "ಮುಲಾಮು", "gu": "મલમ",
        "ml": "തൈലം", "pa": "ਮਲ੍ਹਮ"
    },
    "gel": {
        "en": "Gel", "hi": "जेल", "te": "జెల్", "ta": "ஜெல்",
        "bn": "জেল", "mr": "जेल", "kn": "ಜೆಲ್", "gu": "જેલ",
        "ml": "ജൽ", "pa": "ਜੈੱਲ"
    },
    "drops": {
        "en": "Drops", "hi": "ड्रॉप्स", "te": "చుక్కల మందు", "ta": "சொட்டு மருந்து",
        "bn": "ড্রপস", "mr": "थेंब", "kn": "ಹನಿಗಳು", "gu": "ટીપાં",
        "ml": "ഡ്രോപ്സ്", "pa": "ਬੂੰਦਾਂ"
    },
    "inh": {
        "en": "Inhaler", "hi": "इनहेलर", "te": "ఇన్హేలర్", "ta": "இன்ஹேலர்",
        "bn": "ইনহেলার", "mr": "इनहेलर", "kn": "ಇನ್ಹೇಲರ್", "gu": "ઇન્હેલર",
        "ml": "ഇൻഹേലർ", "pa": "ਇਨਹੇਲਰ"
    },
}

TIMING_MAP: Dict[str, Dict[str, str]] = {
    "unspecified": {
        "en": "As advised", "hi": "सलाह अनुसार", "te": "పేర్కొనబడలేదు", "ta": "குறிப்பிடப்படவில்லை",
        "bn": "নির্দিষ্ট নয়", "mr": "अनिर्दिष्ट", "kn": "ನಿರ್ದಿಷ್ಟಪಡಿಸಿಲ್ಲ", "gu": "સ્પષ્ટ નથી",
        "ml": "വ്യക്തമാക്കിയിട്ടില്ല", "pa": "ਦੱਸਿਆ ਨਹੀਂ ਗਿਆ"
    },
    "before food": {
        "en": "Before food", "hi": "भोजन से पहले", "te": "భోజనానికి ముందు", "ta": "உணவுக்கு முன்",
        "bn": "খাবারের আগে", "mr": "जेवणापूर्वी", "kn": "ಊಟಕ್ಕೆ ಮುಂಚೆ", "gu": "જમ્યા પહેલાં",
        "ml": "ഭക്ഷണത്തിന് മുമ്പ്", "pa": "ਖਾਣੇ ਤੋਂ ਪਹਿਲਾਂ"
    },
    "after food": {
        "en": "After food", "hi": "भोजन के बाद", "te": "భోజనం తర్వాత", "ta": "உணவுக்குப் பின்",
        "bn": "খাবারের পরে", "mr": "जेवणानंतर", "kn": "ಊಟದ ನಂತರ", "gu": "જમ્યા પછી",
        "ml": "ഭക്ഷണത്തിന് ശേഷം", "pa": "ਖਾਣੇ ਤੋਂ ਬਾਅਦ"
    },
    "with food": {
        "en": "With food", "hi": "भोजन के साथ", "te": "భోజనంతో పాటు", "ta": "உணவுடன்",
        "bn": "খাবারের সাথে", "mr": "जेवणासोबत", "kn": "ಊಟದೊಂದಿಗೆ", "gu": "જમવા સાથે",
        "ml": "ഭക്ഷണത്തോടൊപ്പം", "pa": "ਖਾਣੇ ਦੇ ਨਾਲ"
    },
    "empty stomach": {
        "en": "On empty stomach", "hi": "खाली पेट", "te": "ఖాళీ కడుపుతో", "ta": "வெறும் வயிற்றில்",
        "bn": "খালি পেটে", "mr": "उपाशी पोटी", "kn": "ಖಾಲಿ ಹೊಟ್ಟೆಯಲ್ಲಿ", "gu": "ખાલી પેટે",
        "ml": "വെറും വയറ്റിൽ", "pa": "ਖਾਲੀ ਪੇਟ"
    },
    "at bedtime": {
        "en": "At bedtime", "hi": "सोते समय", "te": "పడుకునే ముందు", "ta": "படுக்கும் முன்",
        "bn": "ঘুমানোর সময়", "mr": "झोपताना", "kn": "ಮಲಗುವ ಮುನ್ನ", "gu": "સૂતી વખતે",
        "ml": "കിടക്കുന്നതിന് മുൻപ്", "pa": "ਸੌਣ ਵੇਲੇ"
    },
}

DAILY_INTAKE_MAP: Dict[str, Dict[str, str]] = {
    "1x / day": {
        "en": "1x / day", "hi": "दिन में 1 बार", "te": "రోజుకు 1 సారి", "ta": "தினமும் 1 முறை",
        "bn": "দিনে ১ বার", "mr": "दिवसातून १ वेळा", "kn": "ದಿನಕ್ಕೆ ೧ ಬಾರಿ", "gu": "દિવસમાં ૧ વાર",
        "ml": "ദിവസത്തിൽ 1 തവണ", "pa": "ਦਿਨ ਵਿੱਚ 1 ਵਾਰ"
    },
    "2x / day": {
        "en": "2x / day", "hi": "दिन में 2 बार", "te": "రోజుకు 2 సార్లు", "ta": "தினமும் 2 முறை",
        "bn": "দিনে ২ বার", "mr": "दिवसातून २ वेळा", "kn": "ದಿನಕ್ಕೆ ೨ ಬಾರಿ", "gu": "દિવસમાં ૨ વાર",
        "ml": "ദിവസത്തിൽ 2 തവണ", "pa": "ਦਿਨ ਵਿੱਚ 2 ਵਾਰ"
    },
    "3x / day": {
        "en": "3x / day", "hi": "दिन में 3 बार", "te": "రోజుకు 3 సార్లు", "ta": "தினமும் 3 முறை",
        "bn": "দিনে ৩ বার", "mr": "दिवसातून ३ वेळा", "kn": "ದಿನಕ್ಕೆ ೩ ಬಾರಿ", "gu": "દિવસમાં ૩ વાર",
        "ml": "ദിവസത്തിൽ 3 തവണ", "pa": "ਦਿਨ ਵਿੱਚ 3 ਵਾਰ"
    },
    "4x / day": {
        "en": "4x / day", "hi": "दिन में 4 बार", "te": "రోజుకు 4 సార్లు", "ta": "தினமும் 4 முறை",
        "bn": "দিনে ৪ বার", "mr": "दिवसातून ४ वेळा", "kn": "ದಿನಕ್ಕೆ ೪ ಬಾರಿ", "gu": "દિવસમાં ૪ વાર",
        "ml": "ദിവസത്തിൽ 4 തവണ", "pa": "ਦਿਨ ਵਿੱਚ 4 ਵਾਰ"
    },
}

DOSE_UNITS_MAP: Dict[str, Dict[str, str]] = {
    "tablet": {
        "en": "tablet", "hi": "टैबलेट", "te": "టాబ్లెట్", "ta": "மாத்திரை",
        "bn": "ট্যাবলেট", "mr": "टॅबलेट", "kn": "ಟ್ಯಾಬ್ಲೆಟ್", "gu": "ટેબ્લેટ",
        "ml": "ഗുളിക", "pa": "ਟੈਬਲੇਟ"
    },
    "tablets": {
        "en": "tablets", "hi": "टैबलेट", "te": "టాబ్లెట్లు", "ta": "மாத்திரைகள்",
        "bn": "ট্যাবলেট", "mr": "टॅबलेट्स", "kn": "ಟ್ಯಾಬ್ಲೆಟ್‌ಗಳು", "gu": "ટેબ્લેટ્સ",
        "ml": "ഗുളികകൾ", "pa": "ਟੈਬਲੇਟਾਂ"
    },
    "capsule": {
        "en": "capsule", "hi": "कैप्सूल", "te": "క్యాప్సూల్", "ta": "கேப்சூல்",
        "bn": "ক্যাপসুল", "mr": "कॅप्सूल", "kn": "ಕ್ಯಾಪ್ಸುಲ್", "gu": "કેપ્સ્યુલ",
        "ml": "ക്യാപ്സൂൾ", "pa": "ਕੈਪਸੂਲ"
    },
    "capsules": {
        "en": "capsules", "hi": "कैप्सूल", "te": "క్యాప్సూల్స్", "ta": "கேப்சூல்கள்",
        "bn": "ক্যাপসুল", "mr": "कॅप्सूल्स", "kn": "ಕ್ಯಾಪ್ಸುಲ್‌ಗಳು", "gu": "કેપ્સ્યુલ્સ",
        "ml": "ക്യാപ്സൂളുകൾ", "pa": "ਕੈਪਸੂਲ"
    },
    "puff": {
        "en": "puff", "hi": "पफ", "te": "పఫ్", "ta": "பஃப்",
        "bn": "পাফ", "mr": "पफ", "kn": "ಪಫ್", "gu": "પફ",
        "ml": "പഫ്", "pa": "ਪਫ਼"
    },
    "puffs": {
        "en": "puffs", "hi": "पफ", "te": "పఫ్‌లు", "ta": "பஃப்",
        "bn": "পাফ", "mr": "पफ", "kn": "ಪಫ್‌ಗಳು", "gu": "પફ",
        "ml": "പഫുകൾ", "pa": "ਪਫ਼"
    },
    "drop": {
        "en": "drop", "hi": "बूंद", "te": "చుక్క", "ta": "சொட்டு",
        "bn": "ফোঁটা", "mr": "थेंब", "kn": "ಹನಿ", "gu": "ટીપું",
        "ml": "തുള്ളി", "pa": "ਬੂੰਦ"
    },
    "drops": {
        "en": "drops", "hi": "बूंदें", "te": "చుక్కలు", "ta": "சொட்டுகள்",
        "bn": "ফোঁটা", "mr": "थेंब", "kn": "ಹನಿಗಳು", "gu": "ટીપાં",
        "ml": "തുള്ളികൾ", "pa": "ਬੂੰਦਾਂ"
    },
}

QTY_WORDS_MAP: Dict[str, Dict[str, str]] = {
    "one": {"hi": "एक", "te": "ఒక", "ta": "ஒரு", "bn": "একটি", "mr": "एक", "kn": "ಒಂದು", "gu": "એક", "ml": "ഒരു", "pa": "ਇੱਕ"},
    "two": {"hi": "दो", "te": "రెండు", "ta": "இரண்டு", "bn": "দুটি", "mr": "दोन", "kn": "ಎರಡು", "gu": "બે", "ml": "രണ്ട്", "pa": "ਦੋ"},
    "three": {"hi": "तीन", "te": "మూడు", "ta": "மூன்று", "bn": "তিনটি", "mr": "तीन", "kn": "ಮೂರು", "gu": "ત્રણ", "ml": "മൂന്ന്", "pa": "ਤਿੰਨ"},
    "four": {"hi": "चार", "te": "నాలుగు", "ta": "நான்கு", "bn": "চারটি", "mr": "चार", "kn": "ನಾಲ್ಕು", "gu": "ચાર", "ml": "നാല്", "pa": "ਚਾਰ"},
    "half": {"hi": "आधी", "te": "సగం", "ta": "அரை", "bn": "অর্ধেক", "mr": "अर्धी", "kn": "ಅರ್ಧ", "gu": "અડધી", "ml": "പകുതി", "pa": "ਅੱਧੀ"},
}

FREQ_PHRASES_MAP: Dict[str, Dict[str, str]] = {
    "once daily": {
        "hi": "दिन में एक बार", "te": "రోజుకు ఒకసారి", "ta": "தினமும் ஒரு முறை",
        "bn": "দিনে একবার", "mr": "दिवसातून एकदा", "kn": "ದಿನಕ್ಕೆ ಒಂದು ಬಾರಿ", "gu": "દિવસમાં એક વાર",
        "ml": "ദിവസത്തിൽ ഒരു തവണ", "pa": "ਦਿਨ ਵਿੱਚ ਇੱਕ ਵਾਰ",
    },
    "twice daily": {
        "hi": "दिन में दो बार", "te": "రోజుకు రెండుసార్లు", "ta": "தினமும் இரண்டு முறை",
        "bn": "দিনে দুবার", "mr": "दिवसातून दोनदा", "kn": "ದಿನಕ್ಕೆ ಎರಡು ಬಾರಿ", "gu": "દિવસમાં બે વાર",
        "ml": "ദിവസത്തിൽ രണ്ട് തവണ", "pa": "ਦਿਨ ਵਿੱਚ ਦੋ ਵਾਰ",
    },
    "three times daily": {
        "hi": "दिन में तीन बार", "te": "రోజుకు మూడుసార్లు", "ta": "தினமும் மூன்று முறை",
        "bn": "দিনে তিনবার", "mr": "दिवसातून तीन वेळा", "kn": "ದಿನಕ್ಕೆ ಮೂರು ಬಾರಿ", "gu": "દિવસમાં ત્રણ વાર",
        "ml": "ദിവസത്തിൽ മൂന്ന് തവണ", "pa": "ਦਿਨ ਵਿੱਚ ਤਿੰਨ ਵਾਰ",
    },
    "four times daily": {
        "hi": "दिन में चार बार", "te": "రోజుకు నాలుగుసార్లు", "ta": "தினமும் நான்கு முறை",
        "bn": "দিনে চারবার", "mr": "दिवसातून चार वेळा", "kn": "ದಿನಕ್ಕೆ ನಾಲ್ಕು ಬಾರಿ", "gu": "દિવસમાં ચાર વાર",
        "ml": "ദിവസത്തിൽ നാല് തവണ", "pa": "ਦਿਨ ਵਿੱਚ ਚਾਰ ਵਾਰ",
    },
    "every 8 hours": {
        "hi": "हर 8 घंटे में", "te": "ప్రతి 8 గంటలకు ఒకసారి", "ta": "ஒவ்வொரு 8 மணி நேரத்திற்கும்",
        "bn": "প্রতি ৮ ঘণ্টা পর", "mr": "दर ८ तासांनी", "kn": "ಪ್ರತಿ ೮ ಗಂಟೆಗೊಮ್ಮೆ", "gu": "દર ૮ કલાકે",
        "ml": "ഓരോ 8 മണിക്കൂറിലും", "pa": "ਹਰ 8 ਘੰਟੇ ਬਾਅਦ",
    },
    "every 12 hours": {
        "hi": "हर 12 घंटे में", "te": "ప్రతి 12 గంటలకు ఒకసారి", "ta": "ஒவ்வொரு 12 மணி நேரத்திற்கும்",
        "bn": "প্রতি ১২ ঘণ্টা পর", "mr": "दर १२ तासांनी", "kn": "ಪ್ರತಿ ೧೨ ಗಂಟೆಗೊಮ್ಮೆ", "gu": "દર ૧૨ કલાકે",
        "ml": "ഓരോ 12 മണിക്കൂറിലും", "pa": "ਹਰ 12 ਘੰਟੇ ਬਾਅਦ",
    },
    "every 6 hours": {
        "hi": "हर 6 घंटे में", "te": "ప్రతి 6 గంటలకు ఒకసారి", "ta": "ஒவ்வொரு 6 மணி நேரத்திற்கும்",
        "bn": "প্রতি ৬ ঘণ্টা পর", "mr": "दर ६ तासांनी", "kn": "ಪ್ರತಿ ೬ ಗಂಟೆಗೊಮ್ಮೆ", "gu": "દર ૬ કલાકે",
        "ml": "ഓരോ 6 മണിക്കൂറിലും", "pa": "ਹਰ 6 ਘੰਟੇ ਬਾਅਦ",
    },
    "at bedtime": {
        "hi": "सोते समय", "te": "పడుకునే ముందు", "ta": "படுக்கும் முன்",
        "bn": "ঘুমানোর সময়", "mr": "झोपताना", "kn": "ಮಲಗುವ ಮುನ್ನ", "gu": "સૂતી વખતે",
        "ml": "കിടക്കുന്നതിന് മുൻപ്", "pa": "ਸੌਣ ਵੇਲੇ",
    },
    "as needed": {
        "hi": "आवश्यकतानुसार", "te": "అవసరాన్ని బట్టి", "ta": "தேவைக்கேற்ப",
        "bn": "প্রয়োজনমতো", "mr": "गरजेनुसार", "kn": "ಅಗತ್ಯಕ್ಕೆ ತಕ್ಕಂತೆ", "gu": "જરૂર મુજબ",
        "ml": "ആവശ്യാനുസരണം", "pa": "ਲੋੜ ਅਨੁਸਾਰ",
    },
    "immediately": {
        "hi": "तुरंत", "te": "వెంటనే", "ta": "உடனடியாக",
        "bn": "তাৎক্ষণিক", "mr": "त्वरित", "kn": "ತಕ್ಷಣ", "gu": "તરત જ",
        "ml": "ഉടൻ തന്നെ", "pa": "ਤੁਰੰਤ",
    },
}

INDICATION_MAP: Dict[str, Dict[str, str]] = {
    "fever": {"hi": "बुखार होने पर", "te": "జ్వరం వచ్చినప్పుడు", "ta": "காய்ச்சலின் போது", "bn": "জ্বর হলে", "mr": "ताप आल्यास", "kn": "ಜ್ವರ ಬಂದಾಗ", "gu": "તાવ આવે ત્યારે", "ml": "പനി വരുമ്പോൾ", "pa": "ਬੁਖਾਰ ਹੋਣ 'ਤੇ"},
    "pain": {"hi": "दर्द होने पर", "te": "నొప్పి ఉన్నప్పుడు", "ta": "வலியின் போது", "bn": "ব্যথা হলে", "mr": "वेदना असल्यास", "kn": "ನೋವು ಇದ್ದಾಗ", "gu": "દુખાવો થાય ત્યારે", "ml": "വേദനയുള്ളപ്പോൾ", "pa": "ਦਰਦ ਹੋਣ 'ਤੇ"},
    "headache": {"hi": "सिरदर्द होने पर", "te": "తలనొప్పి ఉన్నప్పుడు", "ta": "தலைவலியின் போது", "bn": "মাথাব্যথা হলে", "mr": "डोकेदुखी असल्यास", "kn": "ತಲೆನೋವು ಇದ್ದಾಗ", "gu": "માથાના દુખાવામાં", "ml": "തലവേദനയുള്ളപ്പോൾ", "pa": "ਸਿਰ ਦਰਦ ਹੋਣ 'ਤੇ"},
    "cough": {"hi": "खांसी होने पर", "te": "దగ్గు ఉన్నప్పుడు", "ta": "இருமலின் போது", "bn": "কাশি হলে", "mr": "खोकला असल्यास", "kn": "ಕೆಮ್ಮು ಇದ್ದಾಗ", "gu": "ઉધરસ થાય ત્યારે", "ml": "ചുമയുള്ളപ്പോൾ", "pa": "ਖੰਘ ਹੋਣ 'ਤੇ"},
    "cold": {"hi": "जुकाम होने पर", "te": "జలుబు ఉన్నప్పుడు", "ta": "சளியின் போது", "bn": "সর্দি হলে", "mr": "सर्दी असल्यास", "kn": "ಶೀತ ಇದ್ದಾಗ", "gu": "શરદી થાય ત્યારે", "ml": "ജലദോഷമുള്ളപ്പോൾ", "pa": "ਜ਼ੁਕਾਮ ਹੋਣ 'ਤੇ"},
    "infection": {"hi": "संक्रमण के लिए", "te": "ఇన్ఫెక్షన్ కొరకు", "ta": "தொற்றுநோய்க்கு", "bn": "সংক্রমণের জন্য", "mr": "संसर्गासाठी", "kn": "ಸೋಂಕಿಗೆ", "gu": "ચેપ માટે", "ml": "അണുബാധയ്ക്ക്", "pa": "ਇਨਫੈਕਸ਼ਨ ਲਈ"},
    "relief": {"hi": "राहत के लिए", "te": "ఉపశమనం కోసం", "ta": "நிவாரணத்திற்கு", "bn": "উপশমের জন্য", "mr": "आरामासाठी", "kn": "ಉಪಶಮನಕ್ಕಾಗಿ", "gu": "રાહત માટે", "ml": "ആശ്വാസത്തിന്", "pa": "ਰਾਹਤ ਲਈ"},
    "sleep": {"hi": "नींद के लिए", "te": "నిద్ర కోసం", "ta": "தூக்கத்திற்கு", "bn": "ঘুমের জন্য", "mr": "झोपेसाठी", "kn": "ನಿದ್ರೆಗಾಗಿ", "gu": "ઊંઘ માટે", "ml": "ഉറക്കത്തിന്", "pa": "ਨੀਂਦ ਲਈ"},
}

VERB_TAKE_END: Dict[str, str] = {
    "hi": "लें।", "te": "తీసుకోండి.", "ta": "எடுத்துக் கொள்ளவும்.",
    "bn": "নিন।", "mr": "घ्या.", "kn": "ತೆಗೆದುಕೊಳ್ಳಿ.",
    "gu": "લો.", "ml": "കഴിക്കുക.", "pa": "ਲਓ।",
}

DURATION_UNIT_MAP: Dict[str, Dict[str, str]] = {
    "day": {"hi": "दिन के लिए", "te": "రోజు పాటు", "ta": "நாளுக்கு", "bn": "দিনের জন্য", "mr": "दिवसासाठी", "kn": "ದಿನದವರೆಗೆ", "gu": "દિવસ માટે", "ml": "ദിവസത്തേക്ക്", "pa": "ਦਿਨ ਲਈ"},
    "days": {"hi": "दिनों के लिए", "te": "రోజుల పాటు", "ta": "நாட்களுக்கு", "bn": "দিনের জন্য", "mr": "दिवसांसाठी", "kn": "ದಿನಗಳವರೆಗೆ", "gu": "દિવસ માટે", "ml": "ദിവസത്തേക്ക്", "pa": "ਦਿਨਾਂ ਲਈ"},
    "week": {"hi": "सप्ताह के लिए", "te": "వారం పాటు", "ta": "வாரத்திற்கு", "bn": "সপ্তাহের জন্য", "mr": "आठवड्यासाठी", "kn": "ವಾರದವರೆಗೆ", "gu": "અઠવાડિયા માટે", "ml": "ആഴ്ചത്തേക്ക്", "pa": "ਹਫ਼ਤੇ ਲਈ"},
    "weeks": {"hi": "सप्ताह के लिए", "te": "వారాల పాటు", "ta": "வாரங்களுக்கு", "bn": "সপ্তাহের জন্য", "mr": "आठवड्यांसाठी", "kn": "ವಾರಗಳವರೆಗೆ", "gu": "અઠવાડિયા માટે", "ml": "ആഴ്ചത്തേക്ക്", "pa": "ਹਫ਼ਤਿਆਂ ਲਈ"},
    "month": {"hi": "महीने के लिए", "te": "నెల పాటు", "ta": "மாதத்திற்கு", "bn": "মাসের জন্য", "mr": "महिन्यासाठी", "kn": "ತಿಂಗಳವರೆಗೆ", "gu": "મહિના માટે", "ml": "മാസത്തേക്ക്", "pa": "ਮਹੀਨੇ ਲਈ"},
    "months": {"hi": "महीनों के लिए", "te": "నెలల పాటు", "ta": "மாதங்களுக்கு", "bn": "মাসের জন্য", "mr": "महिन्यांसाठी", "kn": "ತಿಂಗಳುಗಳವರೆಗೆ", "gu": "મહિના માટે", "ml": "മാസത്തേക്ക്", "pa": "ਮਹੀਨਿਆਂ ਲਈ"},
}

COMMON_INSTRUCTIONS_MAP: Dict[str, Dict[str, str]] = {
    "take one tablet once daily": {
        "en": "Take one tablet once daily.",
        "hi": "दिन में एक बार एक टैबलेट लें।",
        "te": "రోజుకు ఒకసారి ఒక టాబ్లెట్ తీసుకోండి.",
        "ta": "தினமும் ஒரு முறை ஒரு மாத்திரை எடுத்துக் கொள்ளவும்.",
        "bn": "দিনে একবার একটি ট্যাবলেট নিন।",
        "mr": "दिवसातून एकदा एक टॅबलेट घ्या.",
        "kn": "ದಿನಕ್ಕೆ ಒಂದು ಬಾರಿ ಒಂದು ಟ್ಯಾಬ್ಲೆಟ್ ತೆಗೆದುಕೊಳ್ಳಿ.",
        "gu": "દિવસમાં એક વાર એક ટેબ્લેટ લો.",
        "ml": "ദിവസത്തിൽ ഒരു തവണ ഒരു ഗുളിക കഴിക്കുക.",
        "pa": "ਦਿਨ ਵਿੱਚ ਇੱਕ ਵਾਰ ਇੱਕ ਟੈਬਲੇਟ ਲਓ।",
    },
    "take one tablet twice daily": {
        "en": "Take one tablet twice daily.",
        "hi": "दिन में दो बार एक टैबलेट लें।",
        "te": "రోజుకు రెండుసార్లు ఒక టాబ్లెట్ తీసుకోండి.",
        "ta": "தினமும் இரண்டு முறை ஒரு மாத்திரை எடுத்துக் கொள்ளவும்.",
        "bn": "দিনে দুবার একটি ট্যাবলেট নিন।",
        "mr": "दिवसातून दोनदा एक टॅबलेट घ्या.",
        "kn": "ದಿನಕ್ಕೆ ಎರಡು ಬಾರಿ ಒಂದು ಟ್ಯಾಬ್ಲೆಟ್ ತೆಗೆದುಕೊಳ್ಳಿ.",
        "gu": "દિવસમાં બે વાર એક ટેબ્લેટ લો.",
        "ml": "ദിവസത്തിൽ രണ്ട് തവണ ഒരു ഗുളിക കഴിക്കുക.",
        "pa": "ਦਿਨ ਵਿੱਚ ਦੋ ਵਾਰ ਇੱਕ ਟੈਬਲੇਟ ਲਓ।",
    },
    "take one tablet three times daily": {
        "en": "Take one tablet three times daily.",
        "hi": "दिन में तीन बार एक टैबलेट लें।",
        "te": "రోజుకు మూడుసార్లు ఒక టాబ్లెట్ తీసుకోండి.",
        "ta": "தினமும் மூன்று முறை ஒரு மாத்திரை எடுத்துக் கொள்ளவும்.",
        "bn": "দিনে তিনবার একটি ট্যাবলেট নিন।",
        "mr": "दिवसातून तीन वेळा एक टॅबलेट घ्या.",
        "kn": "ದಿನಕ್ಕೆ ಮೂರು ಬಾರಿ ಒಂದು ಟ್ಯಾಬ್ಲೆಟ್ ತೆಗೆದುಕೊಳ್ಳಿ.",
        "gu": "દિવસમાં ત્રણ વાર એક ટેબ્લેટ લો.",
        "ml": "ദിവസത്തിൽ മൂന്ന് തവണ ഒരു ഗുളിക കഴിക്കുക.",
        "pa": "ਦਿਨ ਵਿੱਚ ਤਿੰਨ ਵਾਰ ਇੱਕ ਟੈਬਲੇਟ ਲਓ।",
    },
    "take one capsule once daily": {
        "en": "Take one capsule once daily.",
        "hi": "दिन में एक बार एक कैप्सूल लें।",
        "te": "రోజుకు ఒకసారి ఒక క్యాప్సూల్ తీసుకోండి.",
        "ta": "தினமும் ஒரு முறை ஒரு கேப்சூல் எடுத்துக் கொள்ளவும்.",
        "bn": "দিনে একবার একটি ক্যাপসুল নিন।",
        "mr": "दिवसातून एकदा एक कॅप्सूल घ्या.",
        "kn": "ದಿನಕ್ಕೆ ಒಂದು ಬಾರಿ ಒಂದು ಕ್ಯಾಪ್ಸುಲ್ ತೆಗೆದುಕೊಳ್ಳಿ.",
        "gu": "દિવસમાં એક વાર એક કેપ્સ્યુલ લો.",
        "ml": "ദിവസത്തിൽ ഒരു തവണ ഒരു ക്യാപ്സൂൾ കഴിക്കുക.",
        "pa": "ਦਿਨ ਵਿੱਚ ਇੱਕ ਵਾਰ ਇੱਕ ਕੈਪਸੂਲ ਲਓ।",
    },
    "take one capsule twice daily": {
        "en": "Take one capsule twice daily.",
        "hi": "दिन में दो बार एक कैप्सूल लें।",
        "te": "రోజుకు రెండుసార్లు ఒక క్యాప్సూల్ తీసుకోండి.",
        "ta": "தினமும் இரண்டு முறை ஒரு கேப்சூல் எடுத்துக் கொள்ளவும்.",
        "bn": "দিনে দুবার একটি ক্যাপসুল নিন।",
        "mr": "दिवसातून दोनदा एक कॅप्सूल घ्या.",
        "kn": "ದಿನಕ್ಕೆ ಎರಡು ಬಾರಿ ಒಂದು ಕ್ಯಾಪ್ಸುಲ್ ತೆಗೆದುಕೊಳ್ಳಿ.",
        "gu": "દિવસમાં બે વાર એક કેપ્સ્યુલ લો.",
        "ml": "ദിവസത്തിൽ രണ്ട് തവണ ഒരു ക്യാപ്സൂൾ കഴിക്കുക.",
        "pa": "ਦਿਨ ਵਿੱਚ ਦੋ ਵਾਰ ਇੱਕ ਕੈਪਸੂਲ ਲਓ।",
    },
    "take one capsule three times daily for 5 days": {
        "en": "Take one capsule three times daily for 5 days.",
        "hi": "5 दिनों के लिए दिन में तीन बार एक कैप्सूल लें।",
        "te": "5 రోజుల పాటు రోజుకు మూడుసార్లు ఒక క్యాప్సూల్ తీసుకోండి.",
        "ta": "5 நாட்களுக்கு தினமும் மூன்று முறை ஒரு கேப்சூல் எடுத்துக் கொள்ளவும்.",
        "bn": "৫ দিনের জন্য দিনে তিনবার একটি ক্যাপসুল নিন।",
        "mr": "५ दिवसांसाठी दिवसातून तीन वेळा एक कॅप्सूल घ्या.",
        "kn": "೫ ದಿನಗಳವರೆಗೆ ದಿನಕ್ಕೆ ಮೂರು ಬಾರಿ ಒಂದು ಕ್ಯಾಪ್ಸುಲ್ ತೆಗೆದುಕೊಳ್ಳಿ.",
        "gu": "૫ દિવસ માટે દિવસમાં ત્રણ વાર એક કેપ્સ્યુલ લો.",
        "ml": "5 ദിവസത്തേക്ക് ദിവസത്തിൽ മൂന്ന് തവണ ഒരു ക്യാപ്സൂൾ കഴിക്കുക.",
        "pa": "5 ਦਿਨਾਂ ਲਈ ਦਿਨ ਵਿੱਚ ਤਿੰਨ ਵਾਰ ਇੱਕ ਕੈਪਸੂਲ ਲਓ।",
    },
    "take one tablet as needed for fever": {
        "en": "Take one tablet as needed for fever.",
        "hi": "बुखार होने पर आवश्यकतानुसार एक टैबलेट लें।",
        "te": "జ్వరం వచ్చినప్పుడు అవసరాన్ని బట్టి ఒక టాబ్లెట్ తీసుకోండి.",
        "ta": "காய்ச்சல் ஏற்படும் போது தேவைக்கேற்ப ஒரு மாத்திரை உட்கொள்ளவும்.",
        "bn": "জ্বর হলে প্রয়োজনমতো একটি ট্যাবলেট নিন।",
        "mr": "ताप आल्यास गरजेनुसार एक टॅबलेट घ्या.",
        "kn": "ಜ್ವರ ಬಂದಾಗ ಅಗತ್ಯಕ್ಕೆ ತಕ್ಕಂತೆ ಒಂದು ಟ್ಯಾಬ್ಲೆಟ್ ತೆಗೆದುಕೊಳ್ಳಿ.",
        "gu": "તાવ આવે ત્યારે જરૂર મુજબ એક ટેબ્લેટ લો.",
        "ml": "പനി വരുമ്പോൾ ആവശ്യാനുസരണം ഒരു ഗുളിക കഴിക്കുക.",
        "pa": "ਬੁਖਾਰ ਹੋਣ 'ਤੇ ਲੋੜ ਅਨੁਸਾਰ ਇੱਕ ਟੈਬਲੇਟ ਲਓ।",
    },
    "take one tablet before food": {
        "en": "Take one tablet before food.",
        "hi": "भोजन से पहले एक टैबलेट लें।",
        "te": "భోజనానికి ముందు ఒక టాబ్లెట్ తీసుకోండి.",
        "ta": "உணவுக்கு முன் ஒரு மாத்திரை உட்கொள்ளவும்.",
        "bn": "খাবারের আগে একটি ট্যাবলেট নিন।",
        "mr": "जेवणापूर्वी एक टॅबलेट घ्या.",
        "kn": "ಊಟಕ್ಕೆ ಮುಂಚೆ ಒಂದು ಟ್ಯಾಬ್ಲೆಟ್ ತೆಗೆದುಕೊಳ್ಳಿ.",
        "gu": "જમ્યા પહેલાં એક ટેબ્લેટ લો.",
        "ml": "ഭക്ഷണത്തിന് മുമ്പ് ഒരു ഗുളിക കഴിക്കുക.",
        "pa": "ਖਾਣੇ ਤੋਂ ਪਹਿਲਾਂ ਇੱਕ ਟੈਬਲੇਟ ਲਓ।",
    },
}

REASONS_MAP: Dict[str, Dict[str, str]] = {
    "exact match in pharmacopeia database": {
        "en": "Exact match in pharmacopeia database",
        "hi": "फार्माकोपिया डेटाबेस में पूर्ण मिलान",
        "te": "ఫార్మాకోపియా డేటాబేస్‌లో సరిపోలింది",
        "ta": "பார்மகோபியா தரவுத்தளத்தில் துல்லியமாக பொருந்தியது",
        "bn": "ফার্মাকোপিয়া ডাটাবেসে সঠিক মিল পাওয়া গেছে",
        "mr": "फार्माकोपिया डेटाबेसमध्ये अचूक जुळणी",
        "kn": "ಫಾರ್ಮಾಕೋಪಿಯಾ ಡೇಟಾಬೇಸ್‌ನಲ್ಲಿ ನಿಖರ ಹೊಂದಾಣಿಕೆ",
        "gu": "ફાર્માકોપિયા ડેટાબેઝમાં સંપૂર્ણ મેળ",
        "ml": "ഫാർമക്കോപ്പിയ ഡാറ്റാബേസിൽ പൂർണ്ണ പൊരുത്തം",
        "pa": "ਫਾਰਮਾਕੋਪੀਆ ਡੇਟਾਬੇਸ ਵਿੱਚ ਪੂਰਾ ਮੇਲ",
    },
    "pharmacopeial generic verification": {
        "en": "Pharmacopeial generic verification",
        "hi": "फार्माकोपिया जेनेरिक सत्यापन",
        "te": "ఫార్మాకోపియా జెనెరిక్ ధృవీకరణ",
        "ta": "பார்மகோபியா ஜெனரிக் சரிபார்ப்பு",
        "bn": "ফার্মাকোপিয়া জেনেরিক যাচাইকরণ",
        "mr": "फार्माकोपिया जेनेरिक पडताळणी",
        "kn": "ಫಾರ್ಮಾಕೋಪಿಯಾ ಜೆನೆರಿಕ್ ಪರಿಶೀಲನೆ",
        "gu": "ફાર્માકોપિયા જેનેરિક ચકાસણી",
        "ml": "ഫാർമക്കോപ്പിയ ജനറിക് പരിശോധന",
        "pa": "ਫਾਰਮਾਕੋਪੀਆ ਜੈਨੇਰਿਕ ਤਸਦੀਕ",
    },
    "nomenclature matched to pharmacopeia": {
        "en": "Nomenclature matched to pharmacopeia",
        "hi": "फार्माकोपिया मानक से मिलान हुआ",
        "te": "ఫార్మాకోపియా నామకరణంతో సరిపోలింది",
        "ta": "பார்மகோபியா தரத்துடன் பொருந்தியது",
        "bn": "ফার্মাকোপিয়ার সাথে মিলেছে",
        "mr": "फार्माकोपिया मानकांशी जुळले",
        "kn": "ಫಾರ್ಮಾಕೋಪಿಯಾದೊಂದಿಗೆ ಹೊಂದಾಣಿಕೆಯಾಗಿದೆ",
        "gu": "ફાર્માકોપિયા સાથે મેળ ખાતો",
        "ml": "ഫാർമക്കോപ്പിയയുമായി പൊരുത്തപ്പെട്ടു",
        "pa": "ਫਾਰਮਾਕੋਪੀਆ ਨਾਲ ਮੇਲ ਖਾਧਾ",
    },
    "spelling normalized to standard nomenclature": {
        "en": "Spelling normalized to standard nomenclature",
        "hi": "वर्तनी मानक शब्दावली में सुधारी गई",
        "te": "ప్రామాణిక నామకరణానికి సరిచేయబడింది",
        "ta": "எழுத்துப்பிழை நிலையான பெயருக்கு மாற்றப்பட்டது",
        "bn": "বানান মানসম্মত নামে সংশোধন করা হয়েছে",
        "mr": "अचूक मानक नावामध्ये दुरुस्ती केली",
        "kn": "ಕಾಗುಣಿತವನ್ನು ಪ್ರಮಾಣಿತ ಹೆಸರಿಗೆ ಸರಿಪಡಿಸಲಾಗಿದೆ",
        "gu": "જોડણી પ્રમાણિત નામમાં સુધારી",
        "ml": "അക്ഷരവിന്യാസം നിലവാരമുള്ള പേരിലേക്ക് മാറ്റി",
        "pa": "ਸ਼ਬਦ-ਜੋੜ ਮਿਆਰੀ ਨਾਮ ਵਿੱਚ ਸੁਧਾਰਿਆ ਗਿਆ",
    },
    "brand name resolved to standard generic active ingredient": {
        "en": "Brand name resolved to standard generic active ingredient",
        "hi": "ब्रांड नाम को मानक जेनेरिक घटक में बदला गया",
        "te": "బ్రాండ్ పేరు ప్రామాణిక జెనెరిక్‌గా మార్చబడింది",
        "ta": "வணிகப் பெயர் நிலையான ஜெனரிக் மருந்தாக மாற்றப்பட்டது",
        "bn": "ব্র্যান্ড নাম মানসম্মত জেনেরিকে রূপান্তরিত হয়েছে",
        "mr": "ब्रँड नाव प्रमाणित जेनेरिक घटकामध्ये रूपांतरित केले",
        "kn": "ಬ್ರಾಂಡ್ ಹೆಸರನ್ನು ಪ್ರಮಾಣಿತ ಜೆನೆರಿಕ್ ಹೆಸರಿಗೆ ಪರಿವರ್ತಿಸಲಾಗಿದೆ",
        "gu": "બ્રાન્ડ નામ પ્રમાણિત જેનેરિકમાં રૂપાંતરિત કર્યું",
        "ml": "ബ്രാൻഡ് പേര് ജനറിക് പേരിലേക്ക് മാറ്റി",
        "pa": "ਬ੍ਰਾਂਡ ਨਾਮ ਮਿਆਰੀ ਜੈਨੇਰਿਕ ਵਿੱਚ ਬਦਲਿਆ ਗਿਆ",
    },
    "fuzzy matched to pharmacopeia": {
        "en": "Fuzzy matched to pharmacopeia",
        "hi": "अनुमानित मिलान द्वारा सुधारा गया",
        "te": "అక్షర సారూప్యత ఆధారంగా సరిచేయబడింది",
        "ta": "எழுத்து ஒற்றுமையின் அடிப்படையில் திருத்தப்பட்டது",
        "bn": "অক্ষরের মিল অনুযায়ী সংশোধিত",
        "mr": "समान अक्षरांनुसार दुरुस्त केले",
        "kn": "ಹೋಲಿಕೆ ಆಧಾರದಲ್ಲಿ ಸರಿಪಡಿಸಲಾಗಿದೆ",
        "gu": "સમાનતાના આધારે સુધાર્યું",
        "ml": "സാമ്യതയുടെ അടിസ്ഥാനത്തിൽ തിരുത്തി",
        "pa": "ਮਿਲਦੇ-ਜੁਲਦੇ ਅੱਖਰਾਂ ਅਨੁਸਾਰ ਸੁਧਾਰਿਆ ਗਿਆ",
    },
}

SAFETY_CATEGORY_MAP: Dict[str, Dict[str, str]] = {
    "DOSAGE_MISMATCH": {
        "en": "DOSAGE MISMATCH", "hi": "खुराक में विसंगति", "te": "మోతాదు వ్యత్యాసం", "ta": "மருந்தளவு முரண்பாடு",
        "bn": "মাত্রার অসঙ্গতি", "mr": "डोस विसंगती", "kn": "ಡೋಸ್ ವ್ಯತ್ಯಾಸ", "gu": "ડોઝ વિસંગતતા",
        "ml": "ഡോസ് വ്യത്യാസം", "pa": "ਖੁਰਾਕ ਦਾ ਅੰਤਰ"
    },
    "UNKNOWN_MEDICINE": {
        "en": "UNKNOWN MEDICINE", "hi": "अज्ञात दवा", "te": "తెలియని ఔషధం", "ta": "தெரியாத மருந்து",
        "bn": "অজানা ওষুধ", "mr": "अज्ञात औषध", "kn": "ಅಪರಿಚಿತ ಔಷಧ", "gu": "અજાણી દવા",
        "ml": "അപരിചിതമായ മരുന്ന്", "pa": "ਅਣਜਾਣ ਦਵਾਈ"
    },
    "FREQUENCY_MISMATCH": {
        "en": "FREQUENCY MISMATCH", "hi": "आवृत्ति में विसंगति", "te": "ఫ్రీక్వెన్సీ వ్యత్యాసం", "ta": "இடைவெளி முரண்பாடு",
        "bn": "সেবনের অসঙ্গতি", "mr": "वारंवारता विसंगती", "kn": "ಆವರ್ತನ ವ್ಯತ್ಯಾಸ", "gu": "આવર્તન વિસંગતતા",
        "ml": "ആവൃത്തി വ്യത്യാസം", "pa": "ਸਮੇਂ ਦਾ ਅੰਤਰ"
    },
    "SAFETY_WARNING": {
        "en": "SAFETY WARNING", "hi": "सुरक्षा चेतावनी", "te": "భద్రతా హెచ్చరిక", "ta": "பாதுகாப்பு எச்சரிக்கை",
        "bn": "সুরক্ষা সতর্কতা", "mr": "सुरक्षा इशारा", "kn": "ಸುರಕ್ಷತಾ ಎಚ್ಚರಿಕೆ", "gu": "સુરક્ષા ચેતવણી",
        "ml": "സുരക്ഷാ മുന്നറിയിപ്പ്", "pa": "ਸੁਰੱਖਿਆ ਚੇਤਾਵਨੀ"
    },
    "HALLUCINATION": {
        "en": "UNVERIFIED TOKEN", "hi": "असत्यापित दवा प्रविष्टि", "te": "ధృవీకరించబడని అంశం", "ta": "சரிபார்க்கப்படாத பதிவு",
        "bn": "অযাচাইকৃত তথ্য", "mr": "अप्रमाणित नोंद", "kn": "ಪರಿಶೀಲಿಸದ ನಮೂದು", "gu": "અપ્રમાણિત એન્ટ્રી",
        "ml": "സ്ഥിരീകരിക്കാത്ത വിവരം", "pa": "ਅਪ੍ਰਮਾਣਿਤ ਇੰਦਰਾਜ਼"
    },
    "UNCERTAINTY": {
        "en": "CLINICIAN REVIEW", "hi": "डॉक्टर की समीक्षा आवश्यक", "te": "వైద్యుల సమీక్ష అవసరం", "ta": "மருத்துவர் சரிபார்ப்பு",
        "bn": "ডাক্তারের পরামর্শ প্রয়োজন", "mr": "डॉक्टरांची तपासणी आवश्यक", "kn": "ವೈದ್ಯರ ಪರಿಶೀಲನೆ ಅಗತ್ಯ", "gu": "ડૉક્ટરની ચકાસણી જરૂરી",
        "ml": "ഡോക്ടറുടെ പരിശോധന ആവശ്യമാണ്", "pa": "ਡਾਕਟਰ ਦੀ ਜਾਂਚ ਲੋੜੀਂਦੀ"
    },
    "SOUNDALIKE": {
        "en": "SOUND-ALIKE ALERT", "hi": "मिलते-जुलते नाम की चेतावनी", "te": "సమాన ధ్వని గల మందుల హెచ్చరిక", "ta": "ஒத்த பெயர் எச்சரிக்கை",
        "bn": "সদৃশ নামের সতর্কতা", "mr": "सारख्या नावाचा इशारा", "kn": "ಒಂದೇ ರೀತಿಯ ಹೆಸರಿನ ಎಚ್ಚರಿಕೆ", "gu": "સમાન નામની ચેતવણી",
        "ml": "സമാന പേര് മുന്നറിയിപ്പ്", "pa": "ਮਿਲਦੇ-ਜੁਲਦੇ ਨਾਮ ਦੀ ਚੇਤਾਵਨੀ"
    },
    "OMISSION": {
        "en": "POSSIBLE OMISSION", "hi": "संभावित छूटी हुई दवा", "te": "విడిపోయిన అంశం కావచ్చు", "ta": "விடுபட்டிருக்கக்கூடிய விவரம்",
        "bn": "সম্ভাব্য বাদ পড়া অংশ", "mr": "गहाळ असलेला घटक", "kn": "ಬಿಟ್ಟುಹೋದ ಮಾಹಿತಿ", "gu": "બાકી રહી ગયેલ વિગત",
        "ml": "വിട്ടുപോയ വിവരം", "pa": "ਛੁੱਟਿਆ ਹੋਇਆ ਵੇਰਵਾ"
    },
    "REGULATORY": {
        "en": "REGULATORY NOTICE", "hi": "नियामक सूचना", "te": "నియంత్రణ సూచన", "ta": "ஒழுங்குமுறை அறிவிப்பு",
        "bn": "নিয়ন্ত্রক বিজ্ঞপ্তি", "mr": "नियामक सूचना", "kn": "ನಿಯಂತ್ರಣ ಸೂಚನೆ", "gu": "નિયમનકારી નોંધ",
        "ml": "നിയമപരമായ അറിയിപ്പ്", "pa": "ਨਿਯਮਤ ਸੂਚਨਾ"
    },
    "STORAGE": {
        "en": "STORAGE ADVICE", "hi": "भंडारण निर्देश", "te": "నిల్వ సూచన", "ta": "சேமிப்பு ஆலோசனை",
        "bn": "সংরক্ষণ নির্দেশ", "mr": "साठवणूक सूचना", "kn": "ಸಂಗ್ರಹಣೆ ಸಲಹೆ", "gu": "સંગ્રહ સલાહ",
        "ml": "സൂക്ഷിക്കേണ്ട വിവരം", "pa": "ਸੰਭਾਲ ਸਲਾਹ"
    },
}

STATUS_MAP: Dict[str, Dict[str, str]] = {
    "verified": {
        "en": "Verified", "hi": "सत्यापित", "te": "ధృవీకరించబడింది", "ta": "சரிபார்க்கப்பட்டது",
        "bn": "যাচাইকৃত", "mr": "पडताळणी झालेली", "kn": "ಪರಿಶೀಲಿಸಲಾಗಿದೆ", "gu": "ચકાસાયેલ",
        "ml": "പരിശോധിച്ചു", "pa": "ਤਸਦੀਕ ਕੀਤਾ",
    },
    "review": {
        "en": "Requires Review", "hi": "समीक्षा आवश्यक", "te": "సమీక్ష అవసరం", "ta": "மறுஆய்வு தேவை",
        "bn": "পর্যালোচনা প্রয়োজন", "mr": "तपासणी आवश्यक", "kn": "ಪರಿಶೀಲನೆ ಅಗತ್ಯ", "gu": "ચકાસણી જરૂરી",
        "ml": "പരിശോധന ആവശ്യമാണ്", "pa": "ਜਾਂਚ ਲੋੜੀਂਦੀ",
    },
    "unchanged": {
        "en": "Verified Exact", "hi": "पूर्णतः सटीक", "te": "ఖచ్చితంగా సరిపోలింది", "ta": "துல்லியமானது",
        "bn": "সঠিক", "mr": "अचूक", "kn": "ನಿಖರವಾಗಿದೆ", "gu": "સચોટ",
        "ml": "കൃത്യമാണ്", "pa": "ਬਿਲਕੁਲ ਸਹੀ",
    },
}


def translate_as_advised(lang: str = "en") -> str:
    """Return localized 'As advised' string."""
    return TIMING_MAP["unspecified"].get(lang, "As advised")


def translate_status(status_val: str, lang: str = "en") -> str:
    """Translate verification status string into target language."""
    if not status_val or lang == "en":
        return str(status_val)
    sl = str(status_val).lower()
    if "review" in sl or "yes" in sl or "ambig" in sl:
        return STATUS_MAP["review"].get(lang, str(status_val))
    if "unchanged" in sl:
        return STATUS_MAP["unchanged"].get(lang, str(status_val))
    return STATUS_MAP["verified"].get(lang, str(status_val))


def translate_formulation(form: str, lang: str = "en") -> str:
    """Translate formulation token (TAB., CAP., etc.) into target language."""
    if not form or lang == "en":
        return form
    cleaned = form.lower().replace(".", "").strip()
    if cleaned in FORMULATIONS_MAP:
        return FORMULATIONS_MAP[cleaned].get(lang, form)
    for k, v in FORMULATIONS_MAP.items():
        if k in cleaned:
            return v.get(lang, form)
    return form


def translate_timing(timing_str: str, lang: str = "en") -> str:
    """Translate timing string (including normalized AC/PC/HS labels) into target language."""
    if not timing_str:
        return TIMING_MAP["unspecified"].get(lang, "As advised") if lang != "en" else "Unspecified"

    key = timing_str.lower().strip()
    # Filter pure digits or explicit unspecified values
    if key in ("unspecified", "not specified", "none", "") or key.isdigit():
        return TIMING_MAP["unspecified"].get(lang, "As advised") if lang != "en" else "Unspecified"

    if lang == "en":
        return timing_str

    if key in TIMING_MAP:
        return TIMING_MAP[key].get(lang, timing_str)

    if "empty stomach" in key:
        return TIMING_MAP["empty stomach"].get(lang, timing_str)
    if "before" in key or "pre-prandial" in key or re.search(r"\bac\b", key):
        return TIMING_MAP["before food"].get(lang, timing_str)
    if "after" in key or "post-prandial" in key or re.search(r"\bpc\b", key):
        return TIMING_MAP["after food"].get(lang, timing_str)
    if "with meal" in key or "with food" in key:
        return TIMING_MAP["with food"].get(lang, timing_str)
    if "bedtime" in key or "night" in key or re.search(r"\bhs\b", key):
        return TIMING_MAP["at bedtime"].get(lang, timing_str)

    return timing_str


def translate_daily_intake(daily_str: str, lang: str = "en") -> str:
    """Translate daily intake string (e.g. '3x / day') into target language."""
    if not daily_str or lang == "en":
        return daily_str
    key = daily_str.lower().strip()
    if key in DAILY_INTAKE_MAP:
        return DAILY_INTAKE_MAP[key].get(lang, daily_str)
    m = re.match(r"(\d+)x\s*/\s*day", key)
    if m:
        n = m.group(1)
        sub_key = f"{n}x / day"
        if sub_key in DAILY_INTAKE_MAP:
            return DAILY_INTAKE_MAP[sub_key].get(lang, daily_str)
    return daily_str


def translate_dose(dose_str: str, lang: str = "en") -> str:
    """Translate dose quantity (e.g. '1 tablet', '3 tablets') into target language."""
    if not dose_str or lang == "en":
        return dose_str
    words = dose_str.split()
    translated_words = []
    for w in words:
        wl = w.lower().strip()
        if wl in DOSE_UNITS_MAP:
            translated_words.append(DOSE_UNITS_MAP[wl].get(lang, w))
        elif wl in QTY_WORDS_MAP:
            translated_words.append(QTY_WORDS_MAP[wl].get(lang, w))
        else:
            translated_words.append(w)
    return " ".join(translated_words)


def translate_clinical_instruction(inst: str, lang: str = "en") -> str:
    """Translate clinical instruction sentences dynamically into target language."""
    if not inst or lang == "en":
        return inst
    cleaned = inst.lower().strip().rstrip(".")

    # 1. Exact phrase lookup
    if cleaned in COMMON_INSTRUCTIONS_MAP:
        return COMMON_INSTRUCTIONS_MAP[cleaned].get(lang, inst)

    # 2. Composable sentence translation matching _build_readable_sentence output
    # Pattern: Take/Administer/Instill/Inhale/Apply <qty> <noun> [freq] [indication] [food] [duration]
    qty_trans = ""
    for q_en, q_map in QTY_WORDS_MAP.items():
        if re.search(rf"\b{q_en}\b", cleaned):
            qty_trans = q_map.get(lang, "")
            break

    noun_trans = ""
    for u_en, u_map in DOSE_UNITS_MAP.items():
        if re.search(rf"\b{u_en}\b", cleaned):
            noun_trans = u_map.get(lang, "")
            break
    if not noun_trans and "syrup" in cleaned:
        m_ml = re.search(r"\b(\d+\s*ml)\s*syrup\b", cleaned)
        ml_part = m_ml.group(1) if m_ml else ""
        noun_trans = f"{ml_part} {FORMULATIONS_MAP['syp'].get(lang, '')}".strip()
    elif not noun_trans and "injection" in cleaned:
        noun_trans = FORMULATIONS_MAP["inj"].get(lang, "")
    elif not noun_trans and "topically" in cleaned:
        noun_trans = FORMULATIONS_MAP["oint"].get(lang, "")

    freq_trans = ""
    for f_en, f_map in FREQ_PHRASES_MAP.items():
        if f_en in cleaned:
            freq_trans = f_map.get(lang, "")
            break

    ind_trans = ""
    for i_en, i_map in INDICATION_MAP.items():
        if re.search(rf"\bfor\s+{i_en}\b", cleaned):
            ind_trans = i_map.get(lang, "")
            break

    food_trans = ""
    if "empty stomach" in cleaned:
        food_trans = TIMING_MAP["empty stomach"].get(lang, "")
    elif "before food" in cleaned or "before meal" in cleaned:
        food_trans = TIMING_MAP["before food"].get(lang, "")
    elif "after food" in cleaned or "after meal" in cleaned:
        food_trans = TIMING_MAP["after food"].get(lang, "")
    elif "with food" in cleaned or "with meal" in cleaned:
        food_trans = TIMING_MAP["with food"].get(lang, "")

    dur_trans = ""
    m_dur = re.search(r"\bfor\s+(\d+)\s+(days?|weeks?|months?)\b", cleaned)
    if m_dur:
        d_num, d_unit = m_dur.groups()
        u_trans = DURATION_UNIT_MAP.get(d_unit, {}).get(lang, "")
        if u_trans:
            dur_trans = f"{d_num} {u_trans}"

    if noun_trans and (freq_trans or food_trans or dur_trans or ind_trans):
        parts = []
        if dur_trans:
            parts.append(dur_trans)
        if ind_trans:
            parts.append(ind_trans)
        if food_trans:
            parts.append(food_trans)
        if freq_trans:
            parts.append(freq_trans)
        dose_phrase = f"{qty_trans} {noun_trans}".strip()
        parts.append(dose_phrase)
        verb_end = VERB_TAKE_END.get(lang, "")
        if verb_end:
            parts.append(verb_end)
        return " ".join(p for p in parts if p).strip()

    # Fallback partial match
    for k, v in COMMON_INSTRUCTIONS_MAP.items():
        if k in cleaned:
            return v.get(lang, inst)
    return inst


def translate_audit_reason(reason: str, lang: str = "en") -> str:
    """Translate audit reason text into target language."""
    if not reason or lang == "en":
        return reason
    cleaned = reason.lower().strip()
    if cleaned in REASONS_MAP:
        return REASONS_MAP[cleaned].get(lang, reason)
    for k, v in REASONS_MAP.items():
        if k in cleaned:
            return v.get(lang, reason)
    return reason


def translate_safety_message(message: str, category: str, lang: str = "en") -> Tuple[str, str]:
    """Translate safety category and all issue message templates into target language."""
    cat_trans = SAFETY_CATEGORY_MAP.get(category, {}).get(lang, category)
    if not message or lang == "en":
        return cat_trans, message

    # 1. Dose number 'X' in 'Y' for Med does not appear in raw OCR text
    m = re.search(r"Dose number '([^']+)' in '([^']+)' for ([^ ]+) does not appear in raw OCR text", message, re.IGNORECASE)
    if m:
        num_val, full_val, med_name = m.groups()
        templates = {
            "te": f"{med_name} కొరకు '{full_val}' లోని '{num_val}' సంఖ్య అసలు OCR పాఠంలో కనిపించలేదు.",
            "hi": f"{med_name} के लिए '{full_val}' में '{num_val}' संख्या मूल OCR टेक्स्ट में नहीं मिली।",
            "ta": f"{med_name} க்கான '{full_val}' இல் உள்ள '{num_val}' அளவு அசல் OCR உரையில் காணப்படவில்லை.",
            "bn": f"{med_name} এর জন্য '{full_val}' এর '{num_val}' সংখ্যা মূল OCR টেক্সটে পাওয়া যায়নি।",
            "mr": f"{med_name} साठी '{full_val}' मधील '{num_val}' प्रमाण मूळ OCR मजकुरात आढळले नाही.",
            "kn": f"{med_name} ಗಾಗಿ '{full_val}' ನಲ್ಲಿರುವ '{num_val}' ಸಂಖ್ಯೆ ಮೂಲ OCR ಪಠ್ಯದಲ್ಲಿ ಕಂಡುಬಂದಿಲ್ಲ.",
            "gu": f"{med_name} માટે '{full_val}' માંથી '{num_val}' સંખ્યા મૂળ OCR લખાણમાં મળી નથી.",
            "ml": f"{med_name} നായുള്ള '{full_val}' ലെ '{num_val}' അളവ് യഥാർത്ഥ OCR ടെക്സ്റ്റിൽ കണ്ടെത്തിയില്ല.",
            "pa": f"{med_name} ਲਈ '{full_val}' ਵਿੱਚ '{num_val}' ਸੰਖਿਆ ਮੂਲ OCR ਲਿਖਤ ਵਿੱਚ ਨਹੀਂ ਮਿਲੀ।",
        }
        return cat_trans, templates.get(lang, message)

    # 2. Medicine 'X' requires clinician verification (...)
    m_unc = re.search(r"Medicine '([^']+)' requires clinician verification", message, re.IGNORECASE)
    if m_unc:
        med_name = m_unc.group(1)
        templates = {
            "te": f"ఔషధం '{med_name}' కొరకు వైద్యుల ధృవీకరణ అవసరం.",
            "hi": f"दवा '{med_name}' के लिए चिकित्सक की पुष्टि आवश्यक है।",
            "ta": f"மருந்து '{med_name}' க்கு மருத்துவரின் சரிபார்ப்பு தேவை.",
            "bn": f"ওষুধ '{med_name}' এর জন্য চিকিৎসকের যাচাইকরণ প্রয়োজন।",
            "mr": f"औषध '{med_name}' साठी डॉक्टरांची तपासणी आवश्यक आहे.",
            "kn": f"ಔಷಧಿ '{med_name}' ಗಾಗಿ ವೈದ್ಯರ ಪರಿಶೀಲನೆ ಅಗತ್ಯವಿದೆ.",
            "gu": f"દવા '{med_name}' માટે ડૉક્ટરની ચકાસણી જરૂરી છે.",
            "ml": f"മരുന്ന് '{med_name}' ഡോക്ടറുടെ പരിശോധന ആവശ്യമാണ്.",
            "pa": f"ਦਵਾਈ '{med_name}' ਲਈ ਡਾਕਟਰ ਦੀ ਪੁਸ਼ਟੀ ਲੋੜੀਂਦੀ ਹੈ।",
        }
        return cat_trans, templates.get(lang, message)

    # 3. Possible hallucinated medicine: 'X' has no trace in raw OCR text.
    m_hal = re.search(r"Possible hallucinated medicine: '([^']+)'", message, re.IGNORECASE)
    if m_hal:
        med_name = m_hal.group(1)
        templates = {
            "te": f"అసలు OCR పాఠంలో '{med_name}' ఆనవాలు కనిపించలేదు (ధృవీకరించండి).",
            "hi": f"संभावित त्रुटि: मूल OCR टेक्स्ट में '{med_name}' का कोई संकेत नहीं मिला।",
            "ta": f"அசல் OCR உரையில் '{med_name}' காணப்படவில்லை (சரிபார்க்கவும்).",
            "bn": f"মূল OCR টেক্সটে '{med_name}' এর কোনো চিহ্ন পাওয়া যায়নি।",
            "mr": f"मूळ OCR मजकुरात '{med_name}' आढळले नाही.",
            "kn": f"ಮೂಲ OCR ಪಠ್ಯದಲ್ಲಿ '{med_name}' ಕಂಡುಬಂದಿಲ್ಲ.",
            "gu": f"મૂળ OCR લખાણમાં '{med_name}' મળ્યું નથી.",
            "ml": f"യഥാർത്ഥ OCR ടെക്സ്റ്റിൽ '{med_name}' കണ്ടെത്തിയില്ല.",
            "pa": f"ਮੂਲ OCR ਲਿਖਤ ਵਿੱਚ '{med_name}' ਨਹੀਂ ਮਿਲਿਆ।",
        }
        return cat_trans, templates.get(lang, message)

    # 4. Possible omitted medicine with dosage 'X' present in OCR text.
    m_omi = re.search(r"Possible omitted medicine with dosage '([^']+)'", message, re.IGNORECASE)
    if m_omi:
        raw_dose = m_omi.group(1)
        templates = {
            "te": f"OCR పాఠంలో '{raw_dose}' మోతాదు ఉన్న మందు విడిపోయి ఉండవచ్చు.",
            "hi": f"मूल टेक्स्ट में '{raw_dose}' खुराक वाली कोई दवा छूट गई हो सकती है।",
            "ta": f"OCR உரையில் '{raw_dose}' அளவுள்ள மருந்து விடுபட்டிருக்கலாம்.",
            "bn": f"OCR টেক্সটে '{raw_dose}' মাত্রার ওষুধ বাদ পড়ে থাকতে পারে।",
            "mr": f"OCR मजकुरात '{raw_dose}' डोस असलेले औषध गहाळ असू शकते.",
            "kn": f"OCR ಪಠ್ಯದಲ್ಲಿ '{raw_dose}' ಡೋಸ್ ಇರುವ ಔಷಧ ಬಿಟ್ಟುಹೋಗಿರಬಹುದು.",
            "gu": f"OCR લખાણમાં '{raw_dose}' ડોઝ વાળી દવા બાકી રહી ગઈ હોઈ શકે છે.",
            "ml": f"OCR ടെക്സ്റ്റിൽ '{raw_dose}' അളവുള്ള മരുന്ന് വിട്ടുപോയിരിക്കാം.",
            "pa": f"OCR ਲਿਖਤ ਵਿੱਚ '{raw_dose}' ਖੁਰਾਕ ਵਾਲੀ ਦਵਾਈ ਛੁੱਟ ਗਈ ਹੋ ਸਕਦੀ ਹੈ।",
        }
        return cat_trans, templates.get(lang, message)

    # 5. Possible dosage specification 'X' in raw OCR not attached to an active ingredient.
    m_lbl_omi = re.search(r"Possible dosage specification '([^']+)'", message, re.IGNORECASE)
    if m_lbl_omi:
        raw_dose = m_lbl_omi.group(1)
        templates = {
            "te": f"OCR పాఠంలో ఉన్న '{raw_dose}' మోతాదు ఏ క్రియాశీల పదార్థానికి జతచేయబడలేదు.",
            "hi": f"OCR टेक्स्ट में '{raw_dose}' की मात्रा किसी सक्रिय घटक से नहीं जुड़ी है।",
            "ta": f"OCR உரையில் உள்ள '{raw_dose}' அளவு எந்த மூலப்பொருளுடனும் இணைக்கப்படவில்லை.",
            "bn": f"OCR টেক্সটের '{raw_dose}' মাত্রা কোনো সক্রিয় উপাদানের সাথে যুক্ত হয়নি।",
            "mr": f"OCR मजकुरातील '{raw_dose}' प्रमाण कोणत्याही घटकाशी जोडलेले नाही.",
            "kn": f"OCR ಪಠ್ಯದಲ್ಲಿರುವ '{raw_dose}' ಪ್ರಮಾಣವು ಯಾವುದೇ ಘಟಕಕ್ಕೆ ಜೋಡಿಸಲ್ಪಟ್ಟಿಲ್ಲ.",
            "gu": f"OCR લખાણમાં રહેલ '{raw_dose}' કોઈ સક્રિય ઘटक સાથે જોડાયેલ નથી.",
            "ml": f"OCR ടെക്സ്റ്റിലെ '{raw_dose}' അളവ് ഒരു ചേരുവയുമായും ബന്ധിപ്പിച്ചിട്ടില്ല.",
            "pa": f"OCR ਲਿਖਤ ਵਿੱਚ '{raw_dose}' ਦੀ ਮਾਤਰਾ ਕਿਸੇ ਸਰਗਰਮ ਤੱਤ ਨਾਲ ਜੁੜੀ ਨਹੀਂ ਹੈ।",
        }
        return cat_trans, templates.get(lang, message)

    # 6. No active pharmaceutical ingredient or strength identified
    if "no active pharmaceutical ingredient" in message.lower():
        templates = {
            "te": "మెడిసిన్ లేబుల్ నుండి ఎటువంటి క్రియాశీల ఔషధ పదార్థం గుర్తించబడలేదు.",
            "hi": "दवा के लेबल से कोई सक्रिय दवा घटक या क्षमता नहीं मिली।",
            "ta": "மருந்து லேபிளில் செயலில் உள்ள மூலப்பொருள் எதுவும் கண்டறியப்படவில்லை.",
            "bn": "ওষুধের লেবেল থেকে কোনো সক্রিয় উপাদান শনাক্ত হয়নি।",
            "mr": "औषधाच्या लेबलवरून कोणताही सक्रिय घटक आढळला नाही.",
            "kn": "ಔಷಧದ ಲೇಬಲ್‌ನಿಂದ ಯಾವುದೇ ಸಕ್ರಿಯ ಘಟಕ ಕಂಡುಬಂದಿಲ್ಲ.",
            "gu": "દવાના લેબલમાંથી કોઈ સક્રિય ઘટક મળ્યો નથી.",
            "ml": "മരുന്ന് ലേബലിൽ നിന്ന് സജീവ ഘടകങ്ങളൊന്നും കണ്ടെത്തിയില്ല.",
            "pa": "ਦਵਾਈ ਦੇ ਲੇਬਲ ਤੋਂ ਕੋਈ ਸਰਗਰਮ ਸਮੱਗਰੀ ਨਹੀਂ ਮਿਲੀ।",
        }
        return cat_trans, templates.get(lang, message)

    # 7. Schedule H medication label detected
    if "schedule h" in message.lower():
        templates = {
            "te": "షెడ్యూల్ H మందు లేబుల్ గుర్తించబడింది; డాక్టర్ చీటీతోనే అమ్మాలి అనే నిబంధనను సరిచూడండి.",
            "hi": "शेड्यूल H दवा लेबल मिला; डॉक्टर के पर्चे की वैधानिक चेतावनी की पुष्टि करें।",
            "ta": "அட்டவணை H மருந்து கண்டறியப்பட்டது; மருத்துவர் பரிந்துரை எச்சரிக்கையை உறுதிப்படுத்தவும்.",
            "bn": "শিডিউল এইচ ওষুধের লেবেল শনাক্ত হয়েছে; প্রেসক্রিপশন সতর্কতা যাচাই করুন।",
            "mr": "शेड्यूल H औषध लेबल आढळले; डॉक्टरांच्या चिठ्ठीची सूचना तपासा.",
            "kn": "ಶೆಡ್ಯೂಲ್ H ಔಷಧ ಲೇಬಲ್ ಪತ್ತೆಯಾಗಿದೆ; ವೈದ್ಯರ ಚೀಟಿ ಎಚ್ಚರಿಕೆಯನ್ನು ಖಚಿತಪಡಿಸಿ.",
            "gu": "શેડ્યૂલ H દવાનું લેબલ મળ્યું છે; ડૉક્ટરની સલાહ મુજબની ચેતવણી ચકાસો.",
            "ml": "ഷെഡ്യൂൾ എച്ച് മരുന്ന് ലേബൽ കണ്ടെത്തി; കുറിപ്പടി മുന്നറിയിപ്പ് ഉറപ്പാക്കുക.",
            "pa": "ਸ਼ਡਿਊਲ H ਦਵਾਈ ਦਾ ਲੇਬਲ ਮਿਲਿਆ; ਡਾਕਟਰ ਦੀ ਪਰਚੀ ਵਾਲੀ ਚੇਤਾਵਨੀ ਦੀ ਪੁਸ਼ਟੀ ਕਰੋ।",
        }
        return cat_trans, templates.get(lang, message)

    # 8. No explicit storage conditions extracted
    if "storage conditions" in message.lower():
        templates = {
            "te": "ప్రత్యేక నిల్వ సూచనలు కనుగొనబడలేదు; సాధారణ ఉష్ణోగ్రత జాగ్రత్తలు పాటించండి.",
            "hi": "भंडारण की कोई विशेष शर्त नहीं मिली; सामान्य तापमान संबंधी सावधानी बरतें।",
            "ta": "குறிப்பிட்ட சேமிப்பு விதிகள் இல்லை; சாதாரண வெப்பநிலை முன்னெச்சரிக்கைகளைப் பின்பற்றவும்.",
            "bn": "সুনির্দিষ্ট সংরক্ষণের নিয়ম পাওয়া যায়নি; সাধারণ তাপমাত্রা সতর্কতা অনুসরণ করুন।",
            "mr": "साठवणुकीची विशिष्ट सूचना आढळली नाही; सामान्य तापमान काळजी घ्या.",
            "kn": "ನಿರ್ದಿಷ್ಟ ಸಂಗ್ರಹಣೆ ಸೂಚನೆ ಕಂಡುಬಂದಿಲ್ಲ; ಸಾಮಾನ್ಯ ತಾಪಮಾನದ ಎಚ್ಚರಿಕೆ ಪಾಲಿಸಿ.",
            "gu": "સંગ્રહની કોઈ ખાસ સૂચના મળી નથી; સામાન્ય તાપમાનની સાવચેતી રાખો.",
            "ml": "പ്രത്യേക സംഭരണ നിർദ്ദേശങ്ങളൊന്നും കണ്ടില്ല; സാധാരണ താപനിലയിൽ സൂക്ഷിക്കുക.",
            "pa": "ਸੰਭਾਲ ਦੀਆਂ ਕੋਈ ਖਾਸ ਹਦਾਇਤਾਂ ਨਹੀਂ ਮਿਲੀਆਂ; ਆਮ ਤਾਪਮਾਨ ਦੀਆਂ ਸਾਵਧਾਨੀਆਂ ਵਰਤੋ।",
        }
        return cat_trans, templates.get(lang, message)

    # 9. Medicine 'X' not found in pharmacopeia
    m2 = re.search(r"Medicine '([^']+)' not found in pharmacopeia", message, re.IGNORECASE)
    if m2:
        med_name = m2.group(1)
        templates = {
            "te": f"ఔషధం '{med_name}' ఫార్మాకోపియా డేటాబేస్‌లో కనుగొనబడలేదు.",
            "hi": f"दवा '{med_name}' फार्माकोपिया डेटाबेस में नहीं मिली।",
            "ta": f"மருந்து '{med_name}' பார்மகோபியாவில் காணப்படவில்லை.",
            "bn": f"ওষুধ '{med_name}' ফার্মাকোপিয়ায় পাওয়া যায়নি।",
            "mr": f"औषध '{med_name}' फार्माकोपियामध्ये आढळले नाही.",
            "kn": f"ಔಷಧಿ '{med_name}' ಫಾರ್ಮಾಕೋಪಿಯಾದಲ್ಲಿ ಕಂಡುಬಂದಿಲ್ಲ.",
            "gu": f"દવા '{med_name}' ફાર્માકોપિયામાં મળી નથી.",
            "ml": f"മരുന്ന് '{med_name}' ഫാർമക്കോപ്പിയയിൽ കണ്ടെത്തിയില്ല.",
            "pa": f"ਦਵਾਈ '{med_name}' ਫਾਰਮਾਕੋਪੀਆ ਵਿੱਚ ਨਹੀਂ ਮਿਲੀ।",
        }
        return cat_trans, templates.get(lang, message)

    return cat_trans, message


FIELD_NAMES_MAP: Dict[str, Dict[str, str]] = {
    "medicine_name": {
        "en": "Medicine Name", "hi": "दवा का नाम", "te": "ఔషధం పేరు", "ta": "மருந்து பெயர்",
        "bn": "ওষুধের নাম", "mr": "औषधाचे नाव", "kn": "ಔಷಧದ ಹೆಸರು", "gu": "દવાનું નામ",
        "ml": "മരുന്നിന്റെ പേര്", "pa": "ਦਵਾਈ ਦਾ ਨਾਮ"
    },
    "field": {
        "en": "Field", "hi": "फ़ील्ड", "te": "విభాగం", "ta": "புலம்",
        "bn": "ক্ষেত্র", "mr": "घटक", "kn": "ವಿಭಾಗ", "gu": "વિભાગ",
        "ml": "വിഭാഗം", "pa": "ਖੇਤਰ"
    },
    "formulation": {
        "en": "Formulation", "hi": "दवा का रूप", "te": "ఔషధ రూపం", "ta": "மருந்து வடிவம்",
        "bn": "ওষুধের রূপ", "mr": "औषध स्वरूप", "kn": "ಔಷಧ ರೂಪ", "gu": "દવાનું સ્વરૂપ",
        "ml": "രൂപം", "pa": "ਦਵਾਈ ਦਾ ਰੂਪ"
    },
    "strength": {
        "en": "Strength", "hi": "क्षमता", "te": "సామర్థ్యం", "ta": "அளவு வீரியம்",
        "bn": "ক্ষমতা", "mr": "क्षमता", "kn": "ಸಾಮರ್ಥ್ಯ", "gu": "ક્ષમતા",
        "ml": "ശേഷി", "pa": "ਸਮਰੱਥਾ"
    },
    "dose": {
        "en": "Dosage", "hi": "खुराक", "te": "మోతాదు", "ta": "மருந்து அளவு",
        "bn": "ডোজ", "mr": "डोस", "kn": "ಡೋಸ್", "gu": "ડોઝ",
        "ml": "ഡോസ്", "pa": "ਖੁਰਾਕ"
    },
    "frequency": {
        "en": "Frequency", "hi": "बारंबारता", "te": "ఫ్రీక్వెన్సీ", "ta": "அதிர்வெண்",
        "bn": "ফ্রিকোয়েন্সি", "mr": "वारंवारता", "kn": "ಆವರ್ತನ", "gu": "વારંવારતા",
        "ml": "ആവൃത്തി", "pa": "ਵਾਰਵਾਰਤਾ"
    },
    "food_timing": {
        "en": "Timing", "hi": "भोजन समय", "te": "తీసుకోవలసిన సమయం", "ta": "உணவு நேரம்",
        "bn": "খাবারের সময়", "mr": "जेवणाची वेळ", "kn": "ಆಹಾರದ ಸಮಯ", "gu": "ખોરાકનો સમય",
        "ml": "ഭക്ഷണ സമയം", "pa": "ਖਾਣੇ ਦਾ ਸਮਾਂ"
    },
    "active_ingredient": {
        "en": "Active Ingredient", "hi": "सक्रिय घटक", "te": "క్రియాశీల పదార్థం", "ta": "செயலில் உள்ள மூலப்பொருள்",
        "bn": "সক্রিয় উপাদান", "mr": "सक्रिय घटक", "kn": "ಸಕ್ರಿಯ ಘಟಕಾಂಶ", "gu": "સક્રિય ઘટક",
        "ml": "സജീവ ഘടകം", "pa": "ਸਰਗਰਮ ਸਮੱਗਰੀ"
    },
}


def translate_field_name(field: str, lang: str = "en") -> str:
    """Translate internal field name into localized readable name."""
    if not field or lang == "en":
        return field
    fl = field.lower().strip()
    if fl in FIELD_NAMES_MAP:
        return FIELD_NAMES_MAP[fl].get(lang, field)
    for k, v in FIELD_NAMES_MAP.items():
        if k in fl:
            return v.get(lang, field)
    return field


AUDIT_COLS_MAP = {
    "en": {"field": "Field", "original": "Original OCR Text", "normalized": "Standardized Generic", "confidence": "Confidence", "reason": "Pharmacopeia Validation"},
    "hi": {"field": "फ़ील्ड", "original": "मूल ओसीआर टेक्स्ट", "normalized": "मानकीकृत जेनेरिक", "confidence": "विश्वसनीयता", "reason": "सत्यापन कारण"},
    "te": {"field": "విభాగం", "original": "అసలు OCR పాఠం", "normalized": "ప్రామాణిక జెనెరిక్", "confidence": "విశ్వసనీయత", "reason": "ధృవీకరణ కారణం"},
    "ta": {"field": "புலம்", "original": "அசல் OCR உரை", "normalized": "நிலையான ஜெனரிக்", "confidence": "நம்பகத்தன்மை", "reason": "சரிபார்ப்பு காரணம்"},
    "bn": {"field": "ক্ষেত্র", "original": "মূল ওসিআর লেখা", "normalized": "মানসম্মত জেনেরিক", "confidence": "নির্ভুলতা", "reason": "যাচাইকরণের কারণ"},
    "mr": {"field": "घटक", "original": "मूळ ओसीआर मजकूर", "normalized": "प्रमाणित जेनेरिक", "confidence": "विश्वासार्हता", "reason": "तपासणी कारण"},
    "kn": {"field": "ವಿಭಾಗ", "original": "ಮೂಲ ಒಸಿಆರ್ ಪಠ್ಯ", "normalized": "ಪ್ರಮಾಣಿತ ಜೆನೆರಿಕ್", "confidence": "ವಿಶ್ವಾಸಾರ್ಹತೆ", "reason": "ಪರಿಶೀಲನೆ ಕಾರಣ"},
    "gu": {"field": "વિભાગ", "original": "મૂળ ઓસીઆર લખાણ", "normalized": "પ્રમાણિત જેનેરિક", "confidence": "વિશ્વસનીયતા", "reason": "ચકાસણી કારણ"},
    "ml": {"field": "വിഭാഗം", "original": "യഥാർത്ഥ ടെക്സ്റ്റ്", "normalized": "ജനറിക് പേര്", "confidence": "വിശ്വാസ്യത", "reason": "പരിശോധന കാരണം"},
    "pa": {"field": "ਖੇਤਰ", "original": "ਮੂਲ ਓਸੀਆਰ ਲਿਖਤ", "normalized": "ਮਿਆਰੀ ਜੈਨੇਰਿਕ", "confidence": "ਭਰੋਸੇਯੋਗਤਾ", "reason": "ਤਸਦੀਕ ਕਾਰਨ"},
}


def format_audit_dataframe(records: List[Dict[str, Any]], lang: str = "en") -> List[Dict[str, Any]]:
    """Format audit dictionary records with translated headers, field names, and cell reasons."""
    cols = AUDIT_COLS_MAP.get(lang, AUDIT_COLS_MAP["en"])
    out = []
    for r in records:
        f_raw = r.get("field", r.get("Field", ""))
        f_val = translate_field_name(f_raw, lang)
        orig_val = r.get("original", r.get("Original", ""))
        norm_val = r.get("normalized", r.get("Normalized", ""))
        conf_raw = r.get("confidence", 100)
        conf_val = f"{conf_raw}%" if isinstance(conf_raw, (int, float)) else str(conf_raw)
        reason_raw = r.get("reason", r.get("Reason", ""))
        reason_val = translate_audit_reason(reason_raw, lang)

        out.append({
            cols["field"]: f_val,
            cols["original"]: orig_val,
            cols["normalized"]: norm_val,
            cols["confidence"]: conf_val,
            cols["reason"]: reason_val,
        })
    return out


def _clean_html(html_str: str) -> str:
    """Strip all leading and trailing whitespace from each line so Markdown parsers never treat it as an indented code block."""
    return "\n".join(line.strip() for line in html_str.splitlines() if line.strip())


def build_localized_prescription_card(norm: Any, lang: str = "en") -> str:
    """Build a styled clinical card displaying prescription details in target language without any indentation."""
    if lang == "en":
        return ""

    patient_label = {"hi": "मरीज़", "te": "రోగి", "ta": "நோயாளி", "bn": "রোগী", "mr": "रुग्ण", "kn": "ರೋಗಿ", "gu": "દર્દી", "ml": "രോഗി", "pa": "ਮਰੀਜ਼"}.get(lang, "Patient")
    date_label = {"hi": "तारीख", "te": "తేదీ", "ta": "தேதி", "bn": "তারিখ", "mr": "तारीख", "kn": "ದಿನಾಂಕ", "gu": "તારીખ", "ml": "തീയതി", "pa": "ਤਾਰੀਖ"}.get(lang, "Date")
    meds_title = {"hi": "दवा विवरण एवं सेवन निर्देश", "te": "ఔషధ వివరాలు మరియు వినియోగ సూచనలు", "ta": "மருந்து விவரங்கள் & உட்கொள்ளும் வழிமுறைகள்", "bn": "ওষুধের বিবরণ ও সেবনবিধি", "mr": "औषध तपशील आणि सूचना", "kn": "ಔಷಧ ವಿವರಗಳು ಮತ್ತು ಸೂಚನೆಗಳು", "gu": "દવા વિગતો અને સૂચનાઓ", "ml": "മരുന്ന് വിവരങ്ങളും നിർദ്ദേശങ്ങളും", "pa": "ਦਵਾਈ ਵੇਰਵੇ ਅਤੇ ਨਿਰਦੇਸ਼"}.get(lang, "Medication Details")
    unspecified = {"hi": "अनिर्दिष्ट", "te": "పేర్కొనబడలేదు", "ta": "குறிப்பிடப்படவில்லை", "bn": "নির্দিষ্ট নয়", "mr": "अनिर्दिष्ट", "kn": "ನಿರ್ದಿಷ್ಟಪಡಿಸಿಲ್ಲ", "gu": "સ્પષ્ટ નથી", "ml": "വ്യക്തമാക്കിയിട്ടില്ല", "pa": "ਦੱਸਿਆ ਨਹੀਂ ਗਿਆ"}.get(lang, "Not specified")

    p_name = getattr(norm, "patient_name", "") or unspecified
    d_val = getattr(norm, "date", "") or unspecified

    items_html_list = []
    meds = getattr(norm, "medications", []) or []
    for idx, m in enumerate(meds, 1):
        m_name = getattr(m, "medicine_name", "")
        m_form = translate_formulation(getattr(m, "formulation", ""), lang)
        m_dosage = getattr(m, "dosage", "") or getattr(m, "strength", "") or ""
        if m_dosage == "Not specified":
            m_dosage = ""
        raw_inst = getattr(m, "readable_sentence", "") or getattr(m, "dosage_instruction", "") or getattr(m, "instructions", "") or ""
        m_inst = translate_clinical_instruction(raw_inst, lang)

        inst_html = f'<div style="color: #334155; font-size: 0.88rem; margin-top: 0.35rem;">👉 {m_inst}</div>' if m_inst else ""
        dose_part = f" {m_dosage}" if m_dosage else ""

        item_block = f'''<div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 0.85rem 1rem; margin-bottom: 0.65rem;">
<div style="font-weight: 700; color: #1E3A8A; font-size: 0.96rem;">{idx}. {m_form} {m_name}{dose_part}</div>
{inst_html}
</div>'''
        items_html_list.append(_clean_html(item_block))

    items_html = "\n".join(items_html_list)

    card_html = f'''<div style="background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 14px; padding: 1.2rem; margin-bottom: 1.2rem; box-shadow: 0 2px 4px rgba(0,0,0,0.04);">
<div style="display: flex; gap: 1.8rem; border-bottom: 1px solid #E2E8F0; padding-bottom: 0.65rem; margin-bottom: 0.85rem; font-size: 0.92rem;">
<div><b>👤 {patient_label}:</b> <span style="color: #0284C7; font-weight: 600;">{p_name}</span></div>
<div><b>📅 {date_label}:</b> <span style="color: #0284C7; font-weight: 600;">{d_val}</span></div>
</div>
<div style="font-weight: 700; color: #0F172A; font-size: 1.02rem; margin-bottom: 0.75rem;">📋 {meds_title}</div>
{items_html}
</div>'''

    return _clean_html(card_html)


def build_localized_prescription_text(norm: Any, lang: str = "en") -> str:
    """Generates clean standardized plain text in target language."""
    if lang == "en":
        return norm.to_formatted_text() if hasattr(norm, "to_formatted_text") else ""

    patient_label = {"hi": "मरीज़", "te": "రోగి", "ta": "நோயாளி", "bn": "রোগী", "mr": "रुग्ण", "kn": "ರೋಗಿ", "gu": "દર્દી", "ml": "രോഗി", "pa": "ਮਰੀਜ਼"}.get(lang, "PATIENT")
    date_label = {"hi": "तारीख", "te": "తేదీ", "ta": "தேதி", "bn": "তারিখ", "mr": "तारीख", "kn": "ದಿನಾಂಕ", "gu": "તારીખ", "ml": "തീയതി", "pa": "ਤਾਰੀਖ"}.get(lang, "DATE")
    meds_title = {"hi": "दवा विवरण एवं सेवन निर्देश", "te": "ఔషధ వివరాలు మరియు వినియోగ సూచనలు", "ta": "மருந்து விவரங்கள் & வழிமுறைகள்", "bn": "ওষুধের বিবরণ ও সেবনবিধি", "mr": "औषध तपशील आणि सूचना", "kn": "ಔಷಧ ವಿವರಗಳು ಮತ್ತು ಸೂಚನೆಗಳು", "gu": "દવા વિગતો અને સૂચનાઓ", "ml": "മരുന്ന് വിവരങ്ങളും നിർദ്ദേശങ്ങളും", "pa": "ਦਵਾਈ ਵੇਰਵੇ ਅਤੇ ਨਿਰਦੇਸ਼"}.get(lang, "MEDICATION DETAILS")
    unspecified = {"hi": "अनिर्दिष्ट", "te": "పేర్కొనబడలేదు", "ta": "குறிப்பிடப்படவில்லை", "bn": "নির্দিষ্ট নয়", "mr": "अनिर्दिष्ट", "kn": "ನಿರ್ದಿಷ್ಟಪಡಿಸಿಲ್ಲ", "gu": "સ્પષ્ટ નથી", "ml": "വ്യക്തമാക്കിയിട്ടില്ല", "pa": "ਦੱਸਿਆ ਨਹੀਂ ਗਿਆ"}.get(lang, "Not specified")

    lines = []
    p_name = getattr(norm, "patient_name", "") or unspecified
    d_val = getattr(norm, "date", "") or unspecified
    lines.append(f"{patient_label}: {p_name}")
    lines.append(f"{date_label}: {d_val}")
    lines.append("")
    lines.append(meds_title)
    lines.append("")

    meds = getattr(norm, "medications", []) or []
    if not meds:
        no_meds = {"hi": "कोई दवा नहीं मिली।", "te": "మందులు కనుగొనబడలేదు.", "ta": "மருந்துகள் எதுவும் கண்டறியப்படவில்லை.", "bn": "কোনো ওষুধ পাওয়া যায়নি।", "mr": "कोणतेही औषध आढळले नाही.", "kn": "ಯಾವುದೇ ಔಷಧಗಳು ಕಂಡುಬಂದಿಲ್ಲ.", "gu": "કોઈ દવા મળી નથી.", "ml": "മരുന്നുകളൊന്നും കണ്ടെത്തിയില്ല.", "pa": "ਕੋਈ ਦਵਾਈ ਨਹੀਂ ਮਿਲੀ।"}.get(lang, "No medications identified.")
        lines.append(no_meds)
    else:
        for idx, m in enumerate(meds, 1):
            m_name = getattr(m, "medicine_name", "")
            m_form = translate_formulation(getattr(m, "formulation", ""), lang)
            m_dosage = getattr(m, "dosage", "") or getattr(m, "strength", "") or ""
            if m_dosage == "Not specified":
                m_dosage = ""
            raw_inst = getattr(m, "readable_sentence", "") or getattr(m, "dosage_instruction", "") or getattr(m, "instructions", "") or ""
            m_inst = translate_clinical_instruction(raw_inst, lang)

            header = f"{idx}. {m_form} {m_name}"
            if m_dosage:
                header += f" {m_dosage}"
            lines.append(header)
            if m_inst:
                lines.append(f"   {m_inst}")
            lines.append("")

    return "\n".join(lines).strip()
