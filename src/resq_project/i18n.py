"""
Minimal in-app i18n for the HP Disaster Relief Agent — English and Hindi only.

`t()` looks up a Hindi translation for a given English template string
(optionally with `{name}`-style placeholders filled via keyword args) when
the user's language preference is Hindi; otherwise it returns the English
text unchanged. A key missing from HI falls back to the English text
itself, so an untranslated string degrades gracefully instead of crashing.

Widget *values* that flow into backend logic (district codes, disaster
types, need categories) are never translated — only their on-screen
labels are, via the `district_label` / `disaster_type_label` / `need_label`
helpers used with Streamlit's `format_func=`.

Scope note: this covers on-screen UI chrome (labels, buttons, headers,
captions, badges/enum codes) plus the emergency-contacts list. A few deep,
dynamically-constructed sentences (GLOF/wildfire alert messages built in
workflow.py/tools.py from live data) are left in English — translating
those would mean threading language through several more data-generation
functions for comparatively little payoff. The AI-generated final report
and chatbot answers are already translated via the LLM (see workflow.py /
chatbot.py), independent of this module.
"""

import streamlit as st

LANGUAGES = ["English", "Hindi"]


def current_language() -> str:
    return st.session_state.get("language_pref", "English")


def t(text: str, **kwargs) -> str:
    """Translate `text` to Hindi if that's the active language; else return
    it unchanged. `{placeholder}` substitutions happen after lookup."""
    if current_language() == "Hindi":
        text = HI.get(text, text)
    return text.format(**kwargs) if kwargs else text


# ── Display-only label maps (underlying widget VALUE stays in English) ───
DISTRICT_LABELS_HI = {
    "KANGRA": "कांगड़ा", "MANDI": "मंडी", "SHIMLA": "शिमला", "KULLU": "कुल्लू",
    "SOLAN": "सोलन", "SIRMOUR": "सिरमौर", "BILASPUR": "बिलासपुर",
    "HAMIRPUR": "हमीरपुर", "CHAMBA": "चंबा", "UNA": "ऊना",
    "KINNAUR": "किन्नौर", "LAHUL AND SPITI": "लाहौल और स्पीति",
}

DISASTER_TYPE_LABELS_HI = {
    "Flash Flood": "अचानक बाढ़", "Landslide": "भूस्खलन", "Cloudburst": "बादल फटना",
    "GLOF": "हिमानी झील प्रकोप बाढ़ (GLOF)", "Wildfire": "जंगल की आग",
    "Avalanche": "हिमस्खलन", "Drought": "सूखा", "Road Blockage": "सड़क अवरोध",
}

NEED_LABELS_HI = {
    "Medical": "चिकित्सा", "Shelter": "आश्रय", "Food": "भोजन",
    "Rescue": "बचाव", "Evacuation": "निकासी", "Water": "पानी",
    "Transport": "परिवहन",
}

CODE_LABELS_HI = {
    "CRITICAL": "अति गंभीर", "HIGH": "उच्च", "MEDIUM": "मध्यम", "MODERATE": "मध्यम",
    "LOW": "निम्न", "MINIMAL": "न्यूनतम", "UNKNOWN": "अज्ञात", "N/A": "उपलब्ध नहीं",
    "RED": "लाल", "ORANGE": "नारंगी", "YELLOW": "पीला", "GREEN": "हरा",
    "WATCH": "निगरानी", "ADVISORY": "परामर्श",
    "SENT": "भेजा गया", "APPROVED": "स्वीकृत", "REJECTED": "अस्वीकृत",
    "APPROVED_SENT": "स्वीकृत और भेजा गया", "EDITED": "संपादित",
    "MATCHED": "मिलान हुआ", "PARTIAL": "आंशिक", "UNMATCHED": "कोई मिलान नहीं",
    "INCREASE": "वृद्धि", "DECREASE": "कमी",
}


def district_label(code: str) -> str:
    if current_language() == "Hindi":
        return DISTRICT_LABELS_HI.get(code, code.title())
    return code.title()


def disaster_type_label(value: str) -> str:
    if current_language() == "Hindi":
        return DISASTER_TYPE_LABELS_HI.get(value, value)
    return value


def need_label(value: str) -> str:
    if current_language() == "Hindi":
        return NEED_LABELS_HI.get(value, value)
    return value


def code_label(value: str) -> str:
    """Short enum-code labels (risk tier, alert level, wildfire level, etc.)."""
    if current_language() == "Hindi":
        return CODE_LABELS_HI.get(str(value).upper(), value)
    return value


# ── English → Hindi dictionary ────────────────────────────────────────────
HI: dict[str, str] = {
    # Landing page
    "🏔️ HP Disaster Relief Agent": "🏔️ HP आपदा राहत एजेंट",
    "This language will be used across the whole app — reports, chat "
    "answers, and on-screen labels/buttons.":
        "यह भाषा पूरे ऐप में उपयोग होगी — रिपोर्ट, चैट उत्तर और स्क्रीन पर लेबल/बटन सभी इसी भाषा में दिखेंगे।",
    "Tell us who you are before we start — this personalizes AI-generated "
    "reports and chat answers, and lets you send coordination emails "
    "directly from the app.":
        "शुरू करने से पहले हमें अपने बारे में बताएं — इससे AI-जनित रिपोर्ट और चैट उत्तर आपके "
        "अनुसार तैयार होंगे, और आप ऐप से सीधे समन्वय ईमेल भेज सकेंगे।",
    "Your name *": "आपका नाम *",
    "Language preference": "भाषा प्राथमिकता",
    "📧 **Email & SMTP password are optional** — only required if you want "
    "the app to send coordination emails **directly to relief authorities** "
    "on your behalf. You can leave these blank and still use the rest of "
    "the app.":
        "📧 **ईमेल और SMTP पासवर्ड वैकल्पिक हैं** — केवल तभी आवश्यक हैं जब आप चाहते हैं कि ऐप "
        "आपकी ओर से **सीधे राहत प्राधिकारियों को** समन्वय ईमेल भेजे। आप इन्हें खाली छोड़कर भी "
        "बाकी ऐप का उपयोग कर सकते हैं।",
    "Your email": "आपका ईमेल",
    "SMTP password": "SMTP पासवर्ड",
    "Gmail app password for the email above — only used if you actually "
    "send a coordination email.":
        "ऊपर दिए ईमेल के लिए Gmail ऐप पासवर्ड — केवल तभी उपयोग होता है जब आप वास्तव में "
        "कोई समन्वय ईमेल भेजते हैं।",
    "🚀 Continue to app": "🚀 ऐप में जारी रखें",
    "Please enter your name to continue.": "जारी रखने के लिए कृपया अपना नाम दर्ज करें।",

    # Header
    "🏔️ HP Disaster Relief Resource Matching Agent": "🏔️ HP आपदा राहत संसाधन मिलान एजेंट",
    "Himachal Pradesh • Multi-hazard: Flood | Landslide | GLOF | Wildfire | "
    "Avalanche | Cloudburst":
        "हिमाचल प्रदेश • बहु-खतरा: बाढ़ | भूस्खलन | GLOF | जंगल की आग | हिमस्खलन | बादल फटना",
    "Powered by: LangGraph • Ollama llama3.2:1b • ChromaDB • Open-Meteo • "
    "Sources: HIMCOSTE 2023 | CWC | NHP Hospitals | DAY-NULM | HP Education Dept":
        "संचालित: LangGraph • Ollama llama3.2:1b • ChromaDB • Open-Meteo • "
        "स्रोत: HIMCOSTE 2023 | CWC | NHP अस्पताल | DAY-NULM | HP शिक्षा विभाग",

    # HP Disaster Assistant expander
    "💬 HP Disaster Assistant": "💬 HP आपदा सहायक",
    "One chatbot for two jobs: grounded Q&A over ingested HP disaster data "
    "(hospitals, shelters, CWC stations, GLOF monitoring, disaster "
    "guidance), and guided, **informational-only** help with property "
    "damage — it never files, submits, or stores anything.":
        "एक चैटबॉट, दो काम: HP आपदा डेटा (अस्पताल, आश्रय, CWC स्टेशन, GLOF निगरानी, आपदा "
        "मार्गदर्शन) पर आधारित प्रश्नोत्तर, और संपत्ति क्षति के लिए मार्गदर्शित, **केवल जानकारी "
        "हेतु** सहायता — यह कभी कुछ दर्ज, सबमिट या संग्रहीत नहीं करता।",

    # Volunteer matching expander
    "🤝 Volunteer Need–Resource Matching & Coordination (human-in-the-loop)":
        "🤝 स्वयंसेवक आवश्यकता–संसाधन मिलान और समन्वय (मानव-पर्यवेक्षित)",
    "Matches reported **needs** against available **volunteer, NGO, and "
    "relief resources** (deterministic scoring by category, location, "
    "quantity, availability). Every coordination message is emailed to the "
    "**agent coordinator email** after approval — nothing is dispatched "
    "automatically.":
        "रिपोर्ट की गई **आवश्यकताओं** का उपलब्ध **स्वयंसेवक, NGO और राहत संसाधनों** से मिलान "
        "करता है (श्रेणी, स्थान, मात्रा, उपलब्धता के आधार पर निश्चित स्कोरिंग)। हर समन्वय संदेश "
        "स्वीकृति के बाद **एजेंट समन्वयक ईमेल** पर भेजा जाता है — कुछ भी स्वतः प्रेषित नहीं होता।",
    "Coordinator email for demo approvals: `{email}`":
        "डेमो स्वीकृतियों हेतु समन्वयक ईमेल: `{email}`",
    "Add a need from a field message / tweet (optional, rule-based extraction)":
        "फील्ड संदेश/ट्वीट से आवश्यकता जोड़ें (वैकल्पिक, नियम-आधारित निष्कर्षण)",
    "e.g. URGENT: 30 people trapped, need rescue and medical at Manali ward 3":
        "उदा.: अत्यावश्यक: 30 लोग फंसे हैं, मनाली वार्ड 3 में बचाव और चिकित्सा की आवश्यकता है",
    "➕ Extract & add need": "➕ निष्कर्षित कर आवश्यकता जोड़ें",
    "**Worklist — {n} needs** ({matched} matched, {unmatched} need "
    "attention), sorted by urgency.":
        "**कार्य सूची — {n} आवश्यकताएं** ({matched} मिलान हुईं, {unmatched} पर ध्यान चाहिए), "
        "तात्कालिकता अनुसार क्रमबद्ध।",
    "Match score": "मिलान स्कोर",
    "Coverage": "कवरेज",
    "Gap": "अंतर",
    "Provider verified": "प्रदाता सत्यापित",
    "Yes": "हाँ",
    "No": "नहीं",
    "Coordination message (editable)": "समन्वय संदेश (संपादन योग्य)",
    "✅ Approve & mark sent": "✅ स्वीकृत करें और भेजा गया चिह्नित करें",
    "📝 Log edit (no send)": "📝 संपादन दर्ज करें (न भेजें)",
    "🚫 Reject / escalate": "🚫 अस्वीकार करें / आगे बढ़ाएं",
    "Email send failed: {error}": "ईमेल भेजने में विफल: {error}",
    "Coordinator decision logged: **{decision}**": "समन्वयक निर्णय दर्ज: **{decision}**",
    "**🧾 Human-in-the-loop audit log** (last {n})":
        "**🧾 मानव-पर्यवेक्षित लेखा परीक्षा लॉग** (अंतिम {n})",
    "no match": "कोई मिलान नहीं",
    "· ✏️ edited": "· ✏️ संपादित",

    # Sidebar — login details
    "Your details (set at login)": "आपका विवरण (लॉगिन पर सेट)",
    "Your Name": "आपका नाम",
    "✏️ Edit login details": "✏️ लॉगिन विवरण संपादित करें",

    # Sidebar — situation details
    "📋 Situation Details": "📋 स्थिति विवरण",
    "Agent coordinator email": "एजेंट समन्वयक ईमेल",
    "Volunteer/NGO coordination drafts are addressed to this email for "
    "demo approval.":
        "स्वयंसेवक/NGO समन्वय ड्राफ्ट डेमो स्वीकृति हेतु इस ईमेल को संबोधित होते हैं।",
    "District *": "जिला *",
    "Select the affected district in Himachal Pradesh": "हिमाचल प्रदेश में प्रभावित जिला चुनें",
    "Location Description": "स्थान विवरण",
    "e.g. Near Kullu bus stand, Beas riverbank": "उदा.: कुल्लू बस स्टैंड के पास, ब्यास नदी किनारे",
    "**Approximate Coordinates** (auto-derived)": "**अनुमानित निर्देशांक** (स्वतः प्राप्त)",
    "geocoded (OSM)": "जियोकोडेड (OSM)",
    "district center": "जिला केंद्र",
    "**🔥 Wildfire Proneness**": "**🔥 जंगल की आग की संभावना**",
    " · PRONE": " · प्रवण",
    "{n} past fires ≤10km": "{n} पिछली आग ≤10 किमी में",
    "Disaster Type *": "आपदा प्रकार *",
    "Immediate Needs *": "तत्काल आवश्यकताएं *",
    "🚨 Find Relief Resources": "🚨 राहत संसाधन खोजें",
    "{d} Risk Profile": "{d} जोखिम प्रोफ़ाइल",
    "Landslides (2023): **{n}**": "भूस्खलन (2023): **{n}**",
    "Key Rivers: {rivers}": "प्रमुख नदियाँ: {rivers}",
    "**Data Sources**": "**डेटा स्रोत**",
    """
    <small>
    🏥 NHP Hospitals (289 HP facilities)<br>
    🏫 HP Edu Dept Schools (shelter proxy)<br>
    🏠 DAY-NULM Shelters (54 cities)<br>
    🤝 Volunteer & NGO resource pool (demo coordination registry)<br>
    🌊 CWC Stations (52 HP stations)<br>
    ⛰️ HIMCOSTE Landslide Inventory 2023<br>
    🌧️ Open-Meteo (no API key)<br>
    🗺️ OpenRouteService (routing)
    </small>
    """:
        """
    <small>
    🏥 NHP अस्पताल (289 HP सुविधाएं)<br>
    🏫 HP शिक्षा विभाग स्कूल (आश्रय प्रॉक्सी)<br>
    🏠 DAY-NULM आश्रय (54 शहर)<br>
    🤝 स्वयंसेवक और NGO संसाधन पूल (डेमो समन्वय रजिस्ट्री)<br>
    🌊 CWC स्टेशन (52 HP स्टेशन)<br>
    ⛰️ HIMCOSTE भूस्खलन सूची 2023<br>
    🌧️ Open-Meteo (API कुंजी नहीं चाहिए)<br>
    🗺️ OpenRouteService (मार्ग निर्धारण)
    </small>
    """,

    # Main panel — default map view
    "### 🗺️ Himachal Pradesh — Disaster Risk Map": "### 🗺️ हिमाचल प्रदेश — आपदा जोखिम मानचित्र",
    "ℹ️ Fill in the sidebar and click **Find Relief Resources** to activate "
    "the agent.":
        "ℹ️ साइडबार भरें और एजेंट सक्रिय करने के लिए **राहत संसाधन खोजें** पर क्लिक करें।",
    "📍 Loading hospitals and shelters for {d}...":
        "📍 {d} के लिए अस्पताल और आश्रय लोड हो रहे हैं...",
    "🏥 {nh} hospitals · 🏠 {ns} shelters shown for {d}":
        "🏥 {nh} अस्पताल · 🏠 {ns} आश्रय {d} के लिए दिखाए गए",
    "⚠️ Please fill in District, Disaster Type, and at least one Need.":
        "⚠️ कृपया जिला, आपदा प्रकार, और कम से कम एक आवश्यकता भरें।",
    "Risk: {v}": "जोखिम: {v}",
    "Landslides 2023: {v}": "भूस्खलन 2023: {v}",
    "🔄 Running disaster response pipeline... (6 LangGraph nodes)":
        "🔄 आपदा प्रतिक्रिया पाइपलाइन चल रही है... (6 LangGraph नोड्स)",
    "Agent error: {error}": "एजेंट त्रुटि: {error}",

    # Results — top metrics
    "🚨 Urgency": "🚨 तात्कालिकता",
    "🌧️ IMD Alert": "🌧️ IMD चेतावनी",
    "⛰️ Risk Tier": "⛰️ जोखिम स्तर",
    "🌡️ Temp": "🌡️ तापमान",
    "💧 Rain (24h)": "💧 वर्षा (24घं)",
    "🏥 Resources": "🏥 संसाधन",
    "🧊 GLOF": "🧊 GLOF",
    "🚨 **Urgency {score}/100 ({level})** — {breakdown}":
        "🚨 **तात्कालिकता {score}/100 ({level})** — {breakdown}",
    "🧊 **GLOF {level}** — {message}": "🧊 **GLOF {level}** — {message}",
    "🔥 **Wildfire proneness: {level}** — {message} ({n} past fire detections within 10 km)":
        "🔥 **जंगल की आग की संभावना: {level}** — {message} ({n} पिछली आग की घटनाएं 10 किमी के भीतर)",
    "🟢 **Wildfire proneness: {level}** — {message}":
        "🟢 **जंगल की आग की संभावना: {level}** — {message}",

    # Tabs
    "📋 Response Report": "📋 प्रतिक्रिया रिपोर्ट",
    "🏥 Resources": "🏥 संसाधन",
    "🗺️ Route & Map": "🗺️ मार्ग और मानचित्र",
    "🌊 CWC Stations": "🌊 CWC स्टेशन",
    "🧊 GLOF Watch": "🧊 GLOF निगरानी",
    "⚙️ Agent Log": "⚙️ एजेंट लॉग",

    # Tab 1 — Response report
    "🚨 ESCALATION REQUIRED — Insufficient local resources found":
        "🚨 आगे बढ़ाना आवश्यक — पर्याप्त स्थानीय संसाधन नहीं मिले",
    "No report generated.": "कोई रिपोर्ट तैयार नहीं हुई।",
    "**📞 Emergency Contacts**": "**📞 आपातकालीन संपर्क**",
    "**🤝 Coordination Message — human approval required**":
        "**🤝 समन्वय संदेश — मानव स्वीकृति आवश्यक**",
    "Draft message for the coordinator email. Review/edit, then approve to "
    "send it by email. Nothing is sent automatically before approval.":
        "समन्वयक ईमेल हेतु ड्राफ्ट संदेश। समीक्षा/संपादन करें, फिर ईमेल भेजने के लिए स्वीकृत "
        "करें। स्वीकृति से पहले कुछ भी स्वतः नहीं भेजा जाता।",

    # Tab 2 — Resources
    "✅ **Priority Resource: {name}**": "✅ **प्राथमिकता संसाधन: {name}**",
    "**Type:** {v}": "**प्रकार:** {v}",
    "**Contact:** {v}": "**संपर्क:** {v}",
    "**District:** {v}": "**जिला:** {v}",
    "**🏥 Hospitals ({n} found)**": "**🏥 अस्पताल ({n} मिले)**",
    "**Specialities:** {v}": "**विशेषताएं:** {v}",
    "**🏠 Shelters ({n} found)**": "**🏠 आश्रय ({n} मिले)**",
    "**Capacity:** {v}": "**क्षमता:** {v}",
    "School shelter details are activated through district administration.":
        "स्कूल आश्रय का विवरण जिला प्रशासन के माध्यम से सक्रिय किया जाता है।",
    "**Matching Reasoning:** {v}": "**मिलान तर्क:** {v}",

    # Tab 3 — Route & Map
    "**🚑 Distance & Estimated Time to Each Resource**":
        "**🚑 प्रत्येक संसाधन की दूरी और अनुमानित समय**",
    "Hospital": "अस्पताल",
    "Shelter": "आश्रय",
    "Resource": "संसाधन",
    "at {v}": "{v} पर",
    "Season: {v}": "मौसम: {v}",
    "📍 Distance": "📍 दूरी",
    "⏱️ Time": "⏱️ समय",
    "≈ straight-line estimate (add ORS_API_KEY for road distance)":
        "≈ सीधी-रेखा अनुमान (सड़क दूरी हेतु ORS_API_KEY जोड़ें)",
    "approx location · {v}": "अनुमानित स्थान · {v}",
    "No hospital/shelter routes available.": "कोई अस्पताल/आश्रय मार्ग उपलब्ध नहीं है।",
    "🛣️ Routing unavailable — {v}": "🛣️ मार्ग निर्धारण अनुपलब्ध — {v}",
    "**Route Steps:**": "**मार्ग चरण:**",
    "**⛰️ Road Risk Assessment — {d}**": "**⛰️ सड़क जोखिम आकलन — {d}**",
    "📍 Approximate location: {lat}, {lon} ({src})":
        "📍 अनुमानित स्थान: {lat}, {lon} ({src})",
    "Your Location": "आपका स्थान",

    # Tab 4 — CWC Stations
    "### 🌊 CWC River Monitoring Stations": "### 🌊 CWC नदी निगरानी स्टेशन",
    "These are official Central Water Commission stations. Check "
    "**[ffs.india-water.gov.in](https://ffs.india-water.gov.in)** for live "
    "water level data.":
        "ये केंद्रीय जल आयोग के आधिकारिक स्टेशन हैं। लाइव जल स्तर डेटा हेतु "
        "**[ffs.india-water.gov.in](https://ffs.india-water.gov.in)** देखें।",
    "📡 **Nearest CWC Station:** {name} | River: {river} | Distance: {dist} km":
        "📡 **निकटतम CWC स्टेशन:** {name} | नदी: {river} | दूरी: {dist} किमी",
    "**District:** {v}\n": "**जिला:** {v}",
    "**Site Type:** {v}": "**स्थल प्रकार:** {v}",
    "**Coordinates:** {v}": "**निर्देशांक:** {v}",
    "**Live Data:** {v}": "**लाइव डेटा:** {v}",

    # Tab 5 — GLOF Watch
    "### 🧊 Glacial Lake Outburst Flood (GLOF) Watch":
        "### 🧊 हिमानी झील प्रकोप बाढ़ (GLOF) निगरानी",
    "⏳ **Note:** This data is based on **previous-year monthly satellite "
    "monitoring** by the Central Water Commission (September 2025) — it "
    "reflects **water-spread-area trends, not real-time water levels**. "
    "Expanding lakes indicate *elevated* GLOF risk; always verify with "
    "live CWC advisories.":
        "⏳ **नोट:** यह डेटा केंद्रीय जल आयोग की **पिछले वर्ष की मासिक उपग्रह निगरानी** "
        "(सितंबर 2025) पर आधारित है — यह **जल-प्रसार क्षेत्र के रुझान दर्शाता है, वास्तविक-समय "
        "जल स्तर नहीं**। बढ़ती झीलें *बढ़े हुए* GLOF जोखिम को दर्शाती हैं; हमेशा लाइव CWC "
        "परामर्श से सत्यापित करें।",
    "✅ No expanding glacial lakes flagged near this location in the latest "
    "monitoring.":
        "✅ नवीनतम निगरानी में इस स्थान के पास कोई बढ़ती हिमानी झील चिह्नित नहीं हुई।",
    "Nearest expanding lake": "निकटतम बढ़ती झील",
    "Distance": "दूरी",
    "Area change": "क्षेत्र परिवर्तन",
    "**Monitored glacial lakes (nearest first):**":
        "**निगरानी में हिमानी झीलें (निकटतम पहले):**",
    "**Basin / River:** {v}": "**बेसिन / नदी:** {v}",
    "**Distance from you:** {v} km": "**आपसे दूरी:** {v} किमी",
    "**Monitored period:** {v}": "**निगरानी अवधि:** {v}",
    "**Source:** {v}": "**स्रोत:** {v}",
    "No glacial-lake monitoring records available for this area.":
        "इस क्षेत्र हेतु कोई हिमानी-झील निगरानी रिकॉर्ड उपलब्ध नहीं है।",

    # Tab 6 — Agent Log
    "### ⚙️ LangGraph Agent Execution Log": "### ⚙️ LangGraph एजेंट निष्पादन लॉग",
    "**Errors:**": "**त्रुटियां:**",
    "**LangGraph Pipeline:**": "**LangGraph पाइपलाइन:**",

    # Footer
    """
IIT Mandi AAI Himshikhar 2026 Capstone Project |
HP Disaster Relief Resource Matching Agent |
Data: HIMCOSTE 2023 • CWC (incl. GLOF Glacial Lake Monitoring Sep 2025) • VIIRS Wildfire History • NHP • DAY-NULM • HP Education Dept • Open-Meteo
""":
        """
IIT मंडी AAI हिमशिखर 2026 कैप्स्टोन परियोजना |
HP आपदा राहत संसाधन मिलान एजेंट |
डेटा: HIMCOSTE 2023 • CWC (GLOF हिमानी झील निगरानी सितंबर 2025 सहित) • VIIRS वाइल्डफ़ायर इतिहास • NHP • DAY-NULM • HP शिक्षा विभाग • Open-Meteo
""",

    # ── chatbot.py — HP Assistant chat ────────────────────────────────
    "👋 Ask me anything about HP hospitals, shelters, river/GLOF monitoring, "
    "or disaster guidance. Need help with **property damage**? Tap the "
    "button below.":
        "👋 HP अस्पतालों, आश्रयों, नदी/GLOF निगरानी, या आपदा मार्गदर्शन के बारे में कुछ भी "
        "पूछें। **संपत्ति क्षति** में मदद चाहिए? नीचे बटन दबाएं।",
    "🚨 **Disaster alert acknowledged.** Let's start with immediate safety, "
    "then work through the property damage.\n\n"
    "**Is anyone injured or trapped right now?**":
        "🚨 **आपदा चेतावनी दर्ज।** पहले तत्काल सुरक्षा पर ध्यान देते हैं, फिर संपत्ति क्षति "
        "पर चर्चा करेंगे।\n\n**क्या अभी कोई घायल या फंसा हुआ है?**",
    "🚑 Yes — injured / trapped": "🚑 हाँ — घायल / फंसा हुआ",
    "Yes — injured / trapped": "हाँ — घायल / फंसा हुआ",
    "No injuries reported": "कोई चोट दर्ज नहीं",
    "**Call the numbers above now** and head to the nearest hospital, or "
    "wait for help if it's not safe to move. Let's also check the "
    "property damage.\n\n**How badly is the property damaged?**":
        "**अभी ऊपर दिए नंबरों पर कॉल करें** और निकटतम अस्पताल जाएं, या यदि हिलना सुरक्षित न "
        "हो तो सहायता की प्रतीक्षा करें। अब संपत्ति क्षति भी जांचते हैं।\n\n**संपत्ति कितनी "
        "क्षतिग्रस्त हुई है?**",
    "Good to hear. Let's check the property damage.\n\n**How badly is the "
    "property damaged?**":
        "यह सुनकर अच्छा लगा। अब संपत्ति क्षति जांचते हैं।\n\n**संपत्ति कितनी क्षतिग्रस्त हुई है?**",
    "🏚️ Fully damaged": "🏚️ पूर्ण क्षतिग्रस्त",
    "Fully damaged": "पूर्ण क्षतिग्रस्त",
    "🧱 Partially damaged": "🧱 आंशिक क्षतिग्रस्त",
    "Partially damaged": "आंशिक क्षतिग्रस्त",
    "**Is it safe for you to stay in the house tonight?**":
        "**क्या आज रात घर में रहना आपके लिए सुरक्षित है?**",
    "🔧 Minor damage": "🔧 मामूली क्षति",
    "Minor damage": "मामूली क्षति",
    "✅ Not damaged": "✅ क्षतिग्रस्त नहीं",
    "Not damaged": "क्षतिग्रस्त नहीं",
    "That's a relief — no compensation or claim process is needed. Stay "
    "alert to further disaster advisories and re-check the property once "
    "the situation clears.":
        "यह राहत की बात है — किसी मुआवज़े या दावा प्रक्रिया की आवश्यकता नहीं है। आगे की "
        "आपदा सलाह हेतु सतर्क रहें और स्थिति सामान्य होने पर संपत्ति की दोबारा जांच करें।",
    "Yes — safe to stay": "हाँ — रहना सुरक्षित है",
    "Yes — safe to stay tonight": "हाँ — आज रात रहना सुरक्षित है",
    "No — not safe": "नहीं — सुरक्षित नहीं",
    "No — not safe to stay tonight": "नहीं — आज रात रहना सुरक्षित नहीं",
    "Your question": "आपका प्रश्न",
    "e.g. Which hospitals are in Kullu? Or ask about property damage help.":
        "उदा.: कुल्लू में कौन-से अस्पताल हैं? या संपत्ति क्षति सहायता के बारे में पूछें।",
    "Ask ➤": "पूछें ➤",
    "🏚️ Property damage help": "🏚️ संपत्ति क्षति सहायता",
    "🗑️ Clear": "🗑️ साफ़ करें",
    "I need help with property damage": "मुझे संपत्ति क्षति में मदद चाहिए",
    "Sure — **is anyone injured or trapped right now?**":
        "ज़रूर — **क्या अभी कोई घायल या फंसा हुआ है?**",
    "📚 Sources: ": "📚 स्रोत: ",
    "📍 Finding nearest hospitals...": "📍 निकटतम अस्पताल खोजे जा रहे हैं...",
    "📍 Finding nearest shelters...": "📍 निकटतम आश्रय खोजे जा रहे हैं...",
    "🔎 Searching HP disaster data...": "🔎 HP आपदा डेटा खोजा जा रहा है...",
    "**📞 Emergency numbers — tap to call:**": "**📞 आपातकालीन नंबर — कॉल हेतु टैप करें:**",
    "No hospitals found in the local directory for this district — call "
    "1077 (district control room) for assistance.":
        "इस जिले हेतु स्थानीय निर्देशिका में कोई अस्पताल नहीं मिला — सहायता हेतु 1077 (जिला "
        "नियंत्रण कक्ष) पर कॉल करें।",
    "**🏥 Nearest hospitals ({n}, sorted by distance):**":
        "**🏥 निकटतम अस्पताल ({n}, दूरी अनुसार क्रमबद्ध):**",
    " (approx. location)": " (अनुमानित स्थान)",
    "No shelters found in the local directory for this district — call "
    "1077 (district control room) for the nearest relief camp.":
        "इस जिले हेतु स्थानीय निर्देशिका में कोई आश्रय नहीं मिला — निकटतम राहत शिविर हेतु "
        "1077 (जिला नियंत्रण कक्ष) पर कॉल करें।",
    "**🏠 Nearest shelters ({n}, sorted by distance):**":
        "**🏠 निकटतम आश्रय ({n}, दूरी अनुसार क्रमबद्ध):**",
    " · Capacity: {v}": " · क्षमता: {v}",
    "Source: {v}": "स्रोत: {v}",
    "⚠️ This amount is **approximate** and is finalized only after "
    "**official verification** by the Patwari / Revenue department. Do "
    "not treat this as a guaranteed payout.":
        "⚠️ यह राशि **अनुमानित** है और पटवारी/राजस्व विभाग द्वारा **आधिकारिक सत्यापन** के बाद "
        "ही अंतिम होगी। इसे निश्चित भुगतान न मानें।",
    "**📋 Claim procedure (informational):**": "**📋 दावा प्रक्रिया (केवल जानकारी हेतु):**",
    "Report the damage to your area **Patwari / Revenue department "
    "(DDMA)** — Revenue staff carry out the official damage assessment "
    "in HP.":
        "अपने क्षेत्र के **पटवारी / राजस्व विभाग (DDMA)** को क्षति की सूचना दें — HP में "
        "राजस्व कर्मचारी आधिकारिक क्षति आकलन करते हैं।",
    "Take **dated photos and video** of the damage from multiple angles "
    "as evidence.":
        "साक्ष्य हेतु क्षति की **तारीख सहित फोटो और वीडियो** कई कोणों से लें।",
    "Keep ready: **Aadhaar, bank account details, ration card, and "
    "ownership/tenancy proof.**":
        "तैयार रखें: **आधार, बैंक खाता विवरण, राशन कार्ड, और स्वामित्व/किरायेदारी प्रमाण।**",
    "Submit the relief application at the **Tehsil / SDM (Revenue) office.**":
        "राहत आवेदन **तहसील / SDM (राजस्व) कार्यालय** में जमा करें।",
    "A **Patwari verifies the damage** on-site and files a report.":
        "एक **पटवारी मौके पर क्षति सत्यापित** करता है और रिपोर्ट दाखिल करता है।",
    "After verification, relief is **credited to your bank account** "
    "under SDRF norms.":
        "सत्यापन के बाद, SDRF मानदंडों के तहत राहत राशि **आपके बैंक खाते में जमा** होती है।",
    "⚠️ Informational only — nothing here is filed, submitted, or stored. "
    "Always confirm with the Patwari / Tehsil office and HP SDMA.":
        "⚠️ केवल जानकारी हेतु — यहां कुछ भी दर्ज, सबमिट या संग्रहीत नहीं होता। हमेशा पटवारी/"
        "तहसील कार्यालय और HP SDMA से पुष्टि करें।",
    "🔄 Start over": "🔄 फिर से शुरू करें",
    "Fully damaged house": "पूर्ण क्षतिग्रस्त घर",
    "Partially damaged house": "आंशिक क्षतिग्रस्त घर",
    "HP Special Relief Package, July 2025 — revised after major disasters; "
    "amounts are approximate until officially verified.":
        "HP विशेष राहत पैकेज, जुलाई 2025 — बड़ी आपदाओं के बाद संशोधित; राशि आधिकारिक "
        "सत्यापन तक अनुमानित है।",
    "**Where to go:** Patwari / Tehsil (Revenue) office  \n"
    "**Helplines:** 1077 & 1070  \n"
    "**HP SDMA:** hpsdma.nic.in":
        "**कहां जाएं:** पटवारी / तहसील (राजस्व) कार्यालय  \n"
        "**हेल्पलाइन:** 1077 और 1070  \n"
        "**HP SDMA:** hpsdma.nic.in",

    # ── ndma_alerts.py ─────────────────────────────────────────────────
    "📰 No NDMA alerts for Himachal Pradesh in the last 7 days.":
        "📰 पिछले 7 दिनों में हिमाचल प्रदेश हेतु कोई NDMA चेतावनी नहीं है।",
    "📰 NDMA Alerts — Himachal Pradesh (last 7 days, {n})":
        "📰 NDMA चेतावनियां — हिमाचल प्रदेश (पिछले 7 दिन, {n})",
    "🌐 Translated from Hindi · Original: {v}": "🌐 हिंदी से अनुवादित · मूल: {v}",
    "⚠️ English version unavailable from source — showing original text.":
        "⚠️ स्रोत से अंग्रेज़ी संस्करण उपलब्ध नहीं — मूल पाठ दिखाया जा रहा है।",
    "Area: {v}": "क्षेत्र: {v}",
    "Source: NDMA SACHET · {author} · [View CAP alert]({link})":
        "स्रोत: NDMA SACHET · {author} · [CAP चेतावनी देखें]({link})",
}
