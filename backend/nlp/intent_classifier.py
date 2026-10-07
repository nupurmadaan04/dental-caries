"""
Layer 2: Semantic Intent Classifier
Recognizes 31 distinct case-specific, technical, safety, general, and system intents.
Supports informal English, internet slang, colloquial speech, typos, Hinglish, Hindi (Devanagari), and Punjabi (Gurmukhi).

Core Architecture (Hybrid Routing):
1. High-Priority Deterministic Safety Guard Evaluator (Medical Governance & Zero False Negatives)
2. Normalized Semantic Multi-Class Matcher (TF-IDF Vector Space with sublinear TF & Local Statistical Classifier)
3. Dynamic Language Preference & Multi-script Parser
4. Domain & Context Requirement Annotation (requires_case_context, requires_safety_guard)
"""

import os
import re
import pickle
import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backend.nlp.schemas import IntentResult

logger = logging.getLogger("dental_ai.intent")

# Canonical Intent Exemplars covering formal, informal, slang, Hinglish, Hindi (Devanagari), Punjabi (Gurmukhi), and typos
INTENT_EXEMPLARS: Dict[str, Dict[str, Any]] = {
    # ----------------------------------------------------
    # CASE-SPECIFIC INTENTS (requires_case_context = True)
    # ----------------------------------------------------
    "lesion_location": {
        "category": "case_specific",
        "requires_case_context": True,
        "requires_safety_guard": False,
        "examples": [
            "where is the lesion", "where is the lesion region", "where exactly is it",
            "bro where exactly is it", "bhai ye lesion kaha hai", "where is the issue",
            "which tooth is affected", "which quadrant is the lesion in", "where is l1 located",
            "show me lesion location", "kaha par hai lesion", "where is the cavity spot",
            "which side is it on", "where did the model highlight", "location of lesion",
            "tell me the exact tooth location", "where is the decay",
            "what are coordinates of lesion", "bro what are cordinates of lesion",
            "what are the coordinates of the lesion", "where exactly is lesion",
            "lesion ki location kya hai", "highlighted area kaha hai", "what are the coordinates",
            "give lesion position", "tell me lesion location", "where did model find it",
            "lesion ka coordinate batao", "coordinates of lesion", "lesion coordinates",
            "bhai lesion kidhar hai", "lesion ki cordinates kya hain", "give lesion coordinates",
            "where is the lesion located", "show it on x-ray", "where is that exactly",
            "लीजन कहाँ स्थित है", "दाँत में लीजन कहाँ है", "कहाँ पर है लीजन",
            "ਲੈਸ਼ਨ ਕਿੱਥੇ ਹੈ", "ਕੈਵਿਟੀ ਕਿੱਥੇ ਹੈ", "ਲੀਜ਼ਨ ਕਿੱਥੇ ਹੈ"
        ]
    },
    "findings": {
        "category": "case_specific",
        "requires_case_context": True,
        "requires_safety_guard": False,
        "examples": [
            "what did the model find", "what did the model detect", "what's wrong here",
            "bhai ye kya hai", "what are the findings", "explain this x-ray result",
            "what did it find in my tooth", "what did the scan show", "what is detected",
            "scan me kya mila", "what was discovered", "xray findings",
            "what was discovered in the radiograph", "what were the detected findings",
            "एक्स-रे में क्या मिला", "मॉडल को क्या दिखा", "ਸਕੈਨ ਵਿੱਚ ਕੀ ਆਇਆ", "ਐਕਸਰੇ ਵਿੱਚ ਕੀ ਮਿਲਿਆ"
        ]
    },
    "staging": {
        "category": "case_specific",
        "requires_case_context": True,
        "requires_safety_guard": False,
        "examples": [
            "what is the stage", "what is the stage of this recent report",
            "what is stage of this report", "what stage is this lesion",
            "what level is this", "what stage did the model assign", "ye stage kya hai",
            "konsa stage hai", "what is the current stage", "application stage",
            "severity stage assigned", "tell me the stage", "what is my current level",
            "mera konsa stage hai", "report me konsa level mila", "what is the current stage assigned",
            "kya stage hai meri x-ray ki", "mera konsa level aya report me",
            "इस रिपोर्ट की स्टेज क्या है", "मेरा कौन सा स्टेज है", "ਇਸ ਰਿਪੋਰਟ ਦੀ ਸਟੇਜ ਕੀ ਹੈ"
        ]
    },
    "stage_explanation": {
        "category": "case_specific",
        "requires_case_context": True,
        "requires_safety_guard": False,
        "examples": [
            "what does level 2 mean", "what does level 2 indicate", "explain level 2",
            "what does level 1 mean", "what does level 1 indicate", "explain level 1",
            "what does level 3 mean", "what does level 3 indicate", "explain level 3",
            "ye stage 2 ka kya matlab hai", "ye stage 1 ka kya matlab hai",
            "explain the stage", "what does stage 2 mean", "what does this level mean",
            "level 2 explain karo", "stage description", "level 2 ka kya matlab hai",
            "level 3 kya hota h", "level 1 kya hota h", "level 3 kyu hota hai",
            "Explain Level 2 simply", "what does stage 2 mean exactly", "level 2 ki detail batao",
            "what is moderate caries stage 2", "bro what is level 2",
            "लेवल 2 का मतलब क्या है", "लेवल 3 क्या होता है", "लेवल 1 का अर्थ",
            "ਲੈਵਲ 2 ਦਾ ਕੀ ਅਰਥ ਹੈ", "ਲੈਵਲ 3 ਦਾ ਕੀ ਮਤਲਬ ਹੈ", "Level 2 da ki matlab hai"
        ]
    },
    "highlighted_region": {
        "category": "case_specific",
        "requires_case_context": True,
        "requires_safety_guard": False,
        "examples": [
            "what is this highlighted thing", "what does this highlighted area mean",
            "why is this area highlighted", "why did the model highlight this",
            "what does the green mask mean", "what does the red highlight show",
            "what does this colored box mean", "what is l1", "what does l1 mean",
            "ye highlight kyu kiya", "highlighted spot explanation", "mask overlay meaning",
            "why did the model highlight this spot", "what does the red highlight indicate",
            "why this area"
        ]
    },
    "tooth_information": {
        "category": "case_specific",
        "requires_case_context": True,
        "requires_safety_guard": False,
        "examples": [
            "which tooth is this", "tooth number", "tooth information", "fdi tooth 24",
            "tell me about tooth 46", "is tooth 16 involved", "what tooth is affected",
            "konsa daant hai", "which molar", "which premolar or canine", "tooth details",
            "konsa daant affect hua hai", "is this tooth 46 or 24", "tell me about tooth 16",
            "which tooth number is involved", "what tooth is that", "what about the other teeth",
            "यह कौन सा दाँत है", "ਕਿਹੜਾ ਦੰਦ ਹੈ ਇਹ"
        ]
    },
    "severity_explanation": {
        "category": "case_specific",
        "requires_case_context": True,
        "requires_safety_guard": False,
        "examples": [
            "should i be worried about this", "is this bad", "how serious is this",
            "bro is this bad", "am i screwed", "is it very severe", "kitna kharab hai",
            "kya ye dangerous hai", "how bad is the cavity", "is this deep",
            "should i panic", "severity level", "i'm really scared after seeing this report",
            "scared after seeing this report", "really worried after this report",
            "kitna kharab hai daant mera", "should i panic about this finding",
            "I'm really worried, what does this mean", "I'm scared, is this serious",
            "how serious is this lesion", "am I in danger because of this cavity",
            "is it severe", "ye daant kharab karega"
        ]
    },
    "model_probability": {
        "category": "case_specific",
        "requires_case_context": True,
        "requires_safety_guard": False,
        "examples": [
            "why is the model only saying 82 percent", "why is the confidence 82 percent",
            "what does 82 percent mean", "why 93 percent", "model probability",
            "model predicted probability", "what does the percentage mean",
            "is 80 percent confidence high", "why not 100 percent probability",
            "percent explanation", "sigmoid probability", "how confident is the model",
            "82% ka kya matlab hai", "what does the model probability percentage signify",
            "why so low", "what is the sigmoid probability"
        ]
    },
    "report_summary": {
        "category": "case_specific",
        "requires_case_context": True,
        "requires_safety_guard": False,
        "examples": [
            "tell me what this report says in simple words", "explain this like i'm a beginner",
            "simple language please", "mujhe simple language me samjha",
            "explain in simple terms", "summarize the report for a patient",
            "aasan bhasha me samjhao", "layman explanation", "patient friendly summary",
            "can you summarize this case", "give me a brief summary of results",
            "explain my report", "layman summary please", "explain this report for me simply",
            "इस रिपोर्ट का सारांश", "ਇਸ ਰਿਪੋਰਟ ਦਾ ਸਾਰ ਦੱਸੋ"
        ]
    },

    # ----------------------------------------------------
    # MODEL / TECHNICAL INTENTS (requires_case_context = False)
    # ----------------------------------------------------
    "model_metrics": {
        "category": "model_technical",
        "requires_case_context": False,
        "requires_safety_guard": False,
        "examples": [
            "what is dice score", "validation dice", "what are the benchmark metrics",
            "precision recall iou", "e64 metrics", "validation accuracy",
            "what is the jaccard index", "how accurate is the model on validation data",
            "specificity and loss of e64", "model performance numbers",
            "what are the benchmark metrics for e64", "explain validation iou and precision",
            "what is jaccard index validation metric", "what was the validation sensitivity of e64",
            "what about IoU"
        ]
    },
    "model_architecture": {
        "category": "model_technical",
        "requires_case_context": False,
        "requires_safety_guard": False,
        "examples": [
            "how does the model work", "explain the architecture", "what is the neural network",
            "teacher student dual network", "resnet 34 backbone", "feature pyramid network",
            "fpn decoder", "model design", "how is the network structured",
            "how does the model architecture work", "explain teacher student network and resnet 34",
            "what is fpn decoder", "explain the resnet-34 backbone",
            "how does fpn combine p2 to p5 features"
        ]
    },
    "mlua_methodology": {
        "category": "model_technical",
        "requires_case_context": False,
        "requires_safety_guard": False,
        "examples": [
            "mlua methodology", "how does mlua train", "semi supervised learning",
            "consistency regularization", "ema teacher synchronization",
            "what does mlua stand for", "unlabeled training method", "batch normalization momentum",
            "how does mlua methodology work", "what is semi supervised consistency regularization",
            "how does ema teacher synchronization work"
        ]
    },
    "uncertainty_explanation": {
        "category": "model_technical",
        "requires_case_context": False,
        "requires_safety_guard": False,
        "examples": [
            "monte carlo dropout", "uncertainty estimation", "how does it measure uncertainty",
            "epistemic uncertainty", "stochastic passes t 8", "uncertainty map",
            "why calculate uncertainty", "why use monte carlo dropout",
            "how is epistemic uncertainty calculated", "what is epistemic uncertainty in mlua",
            "why 8 stochastic passes for uncertainty", "why does it use uncertainty"
        ]
    },
    "segmentation_explanation": {
        "category": "model_technical",
        "requires_case_context": False,
        "requires_safety_guard": False,
        "examples": [
            "how does your model even find this", "what is binary segmentation",
            "pixel level binary segmentation", "how does segmentation work",
            "how does it segment pixels", "foreground versus background segmentation"
        ]
    },
    "threshold_explanation": {
        "category": "model_technical",
        "requires_case_context": False,
        "requires_safety_guard": False,
        "examples": [
            "what is tau 0.50", "operating threshold", "decision threshold",
            "why threshold 0.50", "what does tau mean", "cutoff threshold",
            "what is tau 0.50 threshold", "why is operating threshold set to 0.50",
            "why not 0.40"
        ]
    },
    "inference_pipeline": {
        "category": "model_technical",
        "requires_case_context": False,
        "requires_safety_guard": False,
        "examples": [
            "how does inference work", "sliding window patch inference",
            "384x384 patch reconstruction", "21 patches", "how is radiograph processed",
            "preprocessing and patching pipeline", "explain 21 patch sliding window inference",
            "how are 384x384 patches reconstructed"
        ]
    },

    # ----------------------------------------------------
    # MEDICAL SAFETY INTENTS (requires_safety_guard = True)
    # ----------------------------------------------------
    "diagnosis_request": {
        "category": "medical_safety",
        "requires_case_context": True,
        "requires_safety_guard": True,
        "examples": [
            "does this mean i definitely have a cavity", "do i definitely have caries",
            "diagnose me", "is it definitely a cavity", "ye cavity hai kya",
            "confirm that i have dental disease", "give me a diagnosis",
            "do i have decay for sure", "can you diagnose this", "tell me if i have caries",
            "do I have caries", "bhaii mere daant mein caries h kaise pta lgau",
            "does this mean I have a cavity", "how can I know whether I have caries",
            "confirm that I have decay", "kya mujhe sach me cavity hai",
            "does the report prove I have caries", "can you confirm I have a cavity",
            "is the lesion real cavity", "do I have caries or not",
            "is this a cavity for sure", "tell me if I have decay",
            "tell me for sure if I have caries", "kya ye report confirm karti hai ki cavity hai",
            "Does that mean I have caries", "Is that a diagnosis", "is this a definitive diagnosis",
            "क्या मुझे कैविटी है", "ਕੀ ਮੈਨੂੰ ਕੈਵਿਟੀ ਹੈ", "ਮੈਨੂੰ caries ਹੈ"
        ]
    },
    "treatment_request": {
        "category": "medical_safety",
        "requires_case_context": True,
        "requires_safety_guard": True,
        "examples": [
            "what treatment do i need", "how do i treat this cavity", "do i need a root canal",
            "do i need a tooth extraction", "can i drill or fill this myself",
            "how to cure dental caries", "what procedure should i get",
            "treatment plan for my teeth", "how to fix this lesion",
            "should I get treatment immediately", "how to treat this caries",
            "do I need root canal treatment", "kya extraction karwana padega",
            "how to cure this caries at home", "what dental procedure do I need",
            "is this curable", "kya surgery lagegi"
        ]
    },
    "medication_request": {
        "category": "medical_safety",
        "requires_case_context": False,
        "requires_safety_guard": True,
        "examples": [
            "can i take medicine for this", "what medicine should i take",
            "medicine leni padegi", "prescribe me an antibiotic", "which painkiller for toothache",
            "what pills can cure this", "can medicine dissolve the cavity",
            "suggest drugs for tooth infection", "give me prescription",
            "medicine leni padegi kya", "can I take amoxicillin for this",
            "suggest medicine for this tooth", "which antibiotic should I take for tooth infection",
            "bhai dard ho raha h dawai batao", "dawai batao",
            "मुझे कौन सी दवा लेनी चाहिए", "ਕਿਹੜੀ ਦਵਾਈ ਲੈਣੀ ਚਾਹੀਦੀ ਹੈ"
        ]
    },
    "emergency_or_urgent_concern": {
        "category": "medical_safety",
        "requires_case_context": False,
        "requires_safety_guard": True,
        "examples": [
            "i have unbearable tooth pain emergency", "severe swelling in my jaw",
            "bleeding and high fever mouth infection", "emergency dental help",
            "excruciating pain cannot sleep", "face swollen pus coming out",
            "my face is swollen and I have severe pain, what should I do",
            "unbearable tooth pain emergency jaw swollen",
            "muh sujh gaya hai aur bohot tez dard hai",
            "extreme jaw pain cannot sleep emergency",
            "face is swollen with high fever and bleeding mouth",
            "ਬਹੁਤ ਜ਼ਿਆਦਾ ਦਰਦ ਅਤੇ ਸੋਜ ਹੈ ਐਮਰਜੈਂਸੀ"
        ]
    },
    "definitive_clinical_claim": {
        "category": "medical_safety",
        "requires_case_context": False,
        "requires_safety_guard": True,
        "examples": [
            "do i definitely have cancer", "bro i am really scared do i definitely have cancer",
            "confirm that my tooth is completely dead", "tell me for sure this is oral cancer",
            "guarantee that this tooth is rotten", "is this definitely a tumor",
            "is this definitely caries", "is level 2 definitely a cavity",
            "is this definitely a cavity", "kya ye confirm cavity hai",
            "do i definitely have oral cancer", "guarantee me this is not a tumor",
            "can you guarantee this is caries",
            "क्या यह पक्का कैरीज़ है"
        ]
    },

    # ----------------------------------------------------
    # GENERAL CONVERSATION INTENTS
    # ----------------------------------------------------
    "greeting": {
        "category": "general_conversation",
        "requires_case_context": False,
        "requires_safety_guard": False,
        "examples": [
            "hello", "hi", "hey", "hello bro", "bhai sun", "namaste", "good morning",
            "good afternoon", "greetings", "hey there", "hola", "namaste ji"
        ]
    },
    "casual_conversation": {
        "category": "general_conversation",
        "requires_case_context": False,
        "requires_safety_guard": False,
        "examples": [
            "how are you", "who made you", "who are you", "what's up",
            "what is your name", "are you an ai", "nice to meet you"
        ]
    },
    "clarification": {
        "category": "general_conversation",
        "requires_case_context": True,
        "requires_safety_guard": False,
        "examples": [
            "what do you mean", "can you clarify", "i don't understand what you said",
            "please clarify that point", "can you repeat in another way",
            "bro i dont understand", "bro, i dont understand", "i don't understand",
            "i dont understand", "i don't get it", "i dont get it", "not clear",
            "samajh nahi aaya", "bhai samajh nahi aaya", "ye samajh nahi aa raha",
            "simple language mein batao", "explain simply", "can you explain simply",
            "can you explain that simply", "kuch samajh nahi aaya", "mujhe samajh nahi aaya",
            "please explain simply", "dont understand", "not getting it",
            "bhaiii smjh sa nhi aaya", "samajh nahi aaya bhai", "explain it simply",
            "mainu samajh nahi aaya", "bhai smjh nhi aya kuch bhi", "i dont get it bro",
            "मुझे समझ नहीं आया", "ਮੈਨੂੰ ਸਮਝ ਨਹੀਂ ਆਇਆ"
        ]
    },
    "general_question": {
        "category": "general_conversation",
        "requires_case_context": False,
        "requires_safety_guard": False,
        "examples": [
            "how do cavities form", "what causes tooth decay", "how to brush properly",
            "best way to floss", "how to prevent cavities", "what is fluoride",
            "daant me kida kaise lagta hai"
        ]
    },
    "thanks": {
        "category": "general_conversation",
        "requires_case_context": False,
        "requires_safety_guard": False,
        "examples": [
            "thank you", "thanks bro", "thanks", "dhanyawad", "shukriya",
            "thank you so much", "appreciated", "thanks for explaining",
            "shukriya bro", "appreciated thanks"
        ]
    },
    "goodbye": {
        "category": "general_conversation",
        "requires_case_context": False,
        "requires_safety_guard": False,
        "examples": [
            "goodbye", "bye", "see you later", "bye bye", "take care", "exit", "bye see you"
        ]
    },
    "language_preference": {
        "category": "general_conversation",
        "requires_case_context": True,
        "requires_safety_guard": False,
        "examples": [
            "in hindi", "in punjabi", "in english", "hindi mein", "hindi me",
            "hindi mai", "hindi mein batao", "hindi mein samjhao", "hindi mein samjha",
            "hindi mein explain karo", "hindi please", "explain in hindi", "switch to hindi",
            "hindi mein smjha", "simple hindi mein explain karo", "hindi me batao na",
            "हिंदी", "हिंदी में", "हिंदी में बताओ", "हिंदी में समझाओ", "हिंदी में समझाइए", "हिन्दी",
            "punjabi", "in punjabi", "punjabi mein", "punjabi me", "punjabi mai",
            "punjabi ch", "punjabi vich", "punjabi ch daso", "punjabi vich samjhao",
            "punjabi please", "explain in punjabi", "switch to punjabi", "punjabi mein batao",
            "punjabi ch samjha", "punjabi vich samjhao ji", "punjabi ch samjha do", "punjabi vich",
            "ਪੰਜਾਬੀ", "ਪੰਜਾਬੀ ਵਿੱਚ", "ਪੰਜਾਬੀ ਵਿੱਚ ਦੱਸੋ", "ਪੰਜਾਬੀ ਵਿਚ", "ਪੰਜਾਬੀ ਚ",
            "english", "in english", "english mein", "english me", "english mai",
            "english please", "explain in english", "switch to english", "english mein batao"
        ]
    },

    # ----------------------------------------------------
    # SYSTEM INTENTS
    # ----------------------------------------------------
    "help": {
        "category": "system",
        "requires_case_context": False,
        "requires_safety_guard": False,
        "examples": [
            "help", "how do i use this application", "what can i ask",
            "how does this assistant help", "guide me"
        ]
    },
    "capabilities": {
        "category": "system",
        "requires_case_context": False,
        "requires_safety_guard": False,
        "examples": [
            "what are your capabilities", "what can you do", "features of this assistant",
            "what features do you have", "how can you assist", "what features does this assistant offer"
        ]
    },
    "limitations": {
        "category": "system",
        "requires_case_context": False,
        "requires_safety_guard": False,
        "examples": [
            "what are the model limitations", "cervical burnout", "panoramic distortion",
            "false positives", "why does the model make mistakes", "limitations of opg",
            "what are the model limitations regarding cervical burnout",
            "why does panoramic opg have distortion", "why does cervical burnout cause false positives"
        ]
    }
}


TYPO_CORRECTIONS: Dict[str, str] = {
    "bbox": "coordinates",
    "cordinates": "coordinates",
    "cordinate": "coordinate",
    "kordinates": "coordinates",
    "co ordinates": "coordinates",
    "co-ordinates": "coordinates",
    "co ordinate": "coordinate",
    "co-ordinate": "coordinate",
    "leson": "lesion",
    "lesionn": "lesion",
    "locaton": "location",
    "locasion": "location",
    "lezion": "lesion",
    "leval": "level",
    "levl": "level",
    "stag": "stage",
    "stg": "stage",
    "explan": "explain",
    "expln": "explain",
    "explainn": "explain",
    "wher": "where",
    "whr": "where",
    "hwere": "where",
    "kaha": "where",
    "kahaan": "where",
    "kahan": "where",
    "findngs": "findings",
    "teeths": "teeth",
    "toth": "tooth",
    "quadrent": "quadrant",
    "dentel": "dental",
    "cavty": "cavity",
    "cavyty": "cavity",
    "cavityy": "cavity",
    "cariesh": "caries",
    "medicne": "medicine",
    "meds": "medicine",
    "probablity": "probability",
    "probabily": "probability",
    "confidnce": "confidence",
    "smjh": "samajh",
    "samjh": "samajh",
    "nhi": "nahi",
    "nai": "nahi",
    "pta": "pata",
    "lgau": "lagau",
    "lgaun": "lagau",
    "bhaii": "bhai",
    "bhaiii": "bhai",
    "plz": "please",
    "pls": "please",
    "wat": "what",
    "sumarize": "summarize",
    "sumarise": "summarize",
    "sumarizee": "summarize",
    "x-rayy": "x-ray",
    "xrayy": "x-ray",
    "radiolucncy": "radiolucency",
    "decayy": "decay",
    "hav": "have",
    "worrid": "worried",
    "shud": "should",
    "abt": "about",
    "dis": "this",
    "u": "you",
    "ur": "your",
    "rct": "root canal",
    "pakka": "pakka",
    "mje": "mujhe",
    "kaaha": "where",
    "caris": "caries",
    "carries": "caries"
}

# Canonical 31 Intent Classes for Hardened Conversational Benchmark (Section 7 Taxonomy)
CANONICAL_V2_INTENTS = [
    # Case Specific (9)
    "lesion_location", "staging", "segmentation_findings", "uncertainty_query",
    "coordinate_query", "tooth_identification", "depth_query", "probability_query", "area_query",
    # Model Technical (7)
    "mlua_methodology", "model_metrics", "training_data", "fpn_architecture",
    "confidence_calibration", "threshold_inquiry", "inference_pipeline",
    # Medical Safety (5)
    "diagnosis_request", "medication_request", "treatment_request",
    "definitive_clinical_claim", "emergency_or_urgent_concern",
    # General (7)
    "greeting", "goodbye", "thanks", "clarification", "language_preference",
    "stage_explanation", "capabilities",
    # System (3)
    "help", "reset", "feedback"
]

CANONICAL_V1_TO_V2: Dict[str, str] = {
    "lesion_location": "lesion_location",
    "staging": "staging",
    "findings": "segmentation_findings",
    "highlighted_region": "area_query",
    "tooth_information": "tooth_identification",
    "severity_explanation": "depth_query",
    "model_probability": "probability_query",
    "report_summary": "segmentation_findings",
    "model_metrics": "model_metrics",
    "model_architecture": "fpn_architecture",
    "mlua_methodology": "mlua_methodology",
    "uncertainty_explanation": "uncertainty_query",
    "segmentation_explanation": "training_data",
    "threshold_explanation": "threshold_inquiry",
    "inference_pipeline": "inference_pipeline",
    "diagnosis_request": "diagnosis_request",
    "treatment_request": "treatment_request",
    "medication_request": "medication_request",
    "emergency_or_urgent_concern": "emergency_or_urgent_concern",
    "definitive_clinical_claim": "definitive_clinical_claim",
    "greeting": "greeting",
    "casual_conversation": "feedback",
    "clarification": "clarification",
    "general_question": "mlua_methodology",
    "thanks": "thanks",
    "goodbye": "goodbye",
    "language_preference": "language_preference",
    "stage_explanation": "stage_explanation",
    "help": "help",
    "capabilities": "capabilities",
    "limitations": "reset"
}

CANONICAL_V2_TO_V1: Dict[str, str] = {
    "lesion_location": "lesion_location",
    "staging": "staging",
    "segmentation_findings": "findings",
    "uncertainty_query": "uncertainty_explanation",
    "coordinate_query": "lesion_location",
    "tooth_identification": "tooth_information",
    "depth_query": "severity_explanation",
    "probability_query": "model_probability",
    "area_query": "highlighted_region",
    "mlua_methodology": "mlua_methodology",
    "model_metrics": "model_metrics",
    "training_data": "segmentation_explanation",
    "fpn_architecture": "model_architecture",
    "confidence_calibration": "model_probability",
    "threshold_inquiry": "threshold_explanation",
    "inference_pipeline": "inference_pipeline",
    "diagnosis_request": "diagnosis_request",
    "medication_request": "medication_request",
    "treatment_request": "treatment_request",
    "definitive_clinical_claim": "definitive_clinical_claim",
    "emergency_or_urgent_concern": "emergency_or_urgent_concern",
    "greeting": "greeting",
    "goodbye": "goodbye",
    "thanks": "thanks",
    "clarification": "clarification",
    "language_preference": "language_preference",
    "stage_explanation": "stage_explanation",
    "capabilities": "capabilities",
    "help": "help",
    "reset": "limitations",
    "feedback": "casual_conversation"
}

V2_INTENT_METADATA: Dict[str, Dict[str, Any]] = {
    "lesion_location": {"category": "case_specific", "requires_case_context": True, "requires_safety_guard": False},
    "staging": {"category": "case_specific", "requires_case_context": True, "requires_safety_guard": False},
    "segmentation_findings": {"category": "case_specific", "requires_case_context": True, "requires_safety_guard": False},
    "uncertainty_query": {"category": "case_specific", "requires_case_context": True, "requires_safety_guard": False},
    "coordinate_query": {"category": "case_specific", "requires_case_context": True, "requires_safety_guard": False},
    "tooth_identification": {"category": "case_specific", "requires_case_context": True, "requires_safety_guard": False},
    "depth_query": {"category": "case_specific", "requires_case_context": True, "requires_safety_guard": False},
    "probability_query": {"category": "case_specific", "requires_case_context": True, "requires_safety_guard": False},
    "area_query": {"category": "case_specific", "requires_case_context": True, "requires_safety_guard": False},
    "mlua_methodology": {"category": "model_technical", "requires_case_context": False, "requires_safety_guard": False},
    "model_metrics": {"category": "model_technical", "requires_case_context": False, "requires_safety_guard": False},
    "training_data": {"category": "model_technical", "requires_case_context": False, "requires_safety_guard": False},
    "fpn_architecture": {"category": "model_technical", "requires_case_context": False, "requires_safety_guard": False},
    "confidence_calibration": {"category": "model_technical", "requires_case_context": False, "requires_safety_guard": False},
    "threshold_inquiry": {"category": "model_technical", "requires_case_context": False, "requires_safety_guard": False},
    "inference_pipeline": {"category": "model_technical", "requires_case_context": False, "requires_safety_guard": False},
    "diagnosis_request": {"category": "medical_safety", "requires_case_context": False, "requires_safety_guard": True},
    "medication_request": {"category": "medical_safety", "requires_case_context": False, "requires_safety_guard": True},
    "treatment_request": {"category": "medical_safety", "requires_case_context": False, "requires_safety_guard": True},
    "definitive_clinical_claim": {"category": "medical_safety", "requires_case_context": False, "requires_safety_guard": True},
    "emergency_or_urgent_concern": {"category": "medical_safety", "requires_case_context": False, "requires_safety_guard": True},
    "greeting": {"category": "general_conversation", "requires_case_context": False, "requires_safety_guard": False},
    "goodbye": {"category": "general_conversation", "requires_case_context": False, "requires_safety_guard": False},
    "thanks": {"category": "general_conversation", "requires_case_context": False, "requires_safety_guard": False},
    "clarification": {"category": "general_conversation", "requires_case_context": False, "requires_safety_guard": False},
    "language_preference": {"category": "general_conversation", "requires_case_context": False, "requires_safety_guard": False},
    "stage_explanation": {"category": "general_conversation", "requires_case_context": False, "requires_safety_guard": False},
    "capabilities": {"category": "general_conversation", "requires_case_context": False, "requires_safety_guard": False},
    "help": {"category": "system", "requires_case_context": False, "requires_safety_guard": False},
    "reset": {"category": "system", "requires_case_context": False, "requires_safety_guard": False},
    "feedback": {"category": "system", "requires_case_context": False, "requires_safety_guard": False},
}


def detect_language_request(text: str) -> Tuple[Optional[str], str, bool]:
    """
    Detects if the user requested a specific response language (English, Hindi, Punjabi).
    Returns (detected_language, stripped_query, is_pure_language_request).
    """
    clean_text = text.strip()
    lower = clean_text.lower()

    devanagari_chars = re.search(r"[\u0900-\u097F]", clean_text)
    gurmukhi_chars = re.search(r"[\u0A00-\u0A7F]", clean_text)

    is_hindi = False
    if devanagari_chars or re.search(r"(?:हिंदी|हिन्दी)", clean_text) or re.search(r"\b(?:hindi|hindee)\b", lower):
        is_hindi = True

    is_punjabi = False
    if gurmukhi_chars or re.search(r"(?:ਪੰਜਾਬੀ)", clean_text) or re.search(r"\b(?:punjabi|panjabi)\b", lower) or re.search(r"\b(?:vich\s+samjhao|ch\s+daso|vich\s+daso|ch\s+dasso|ch\s+simple|ch\s+samjha|da\s+ki|ki\s+arth|da\s+matlab)\b", lower):
        is_punjabi = True

    detected_lang = None
    if is_punjabi:
        detected_lang = "pa"
    elif is_hindi:
        detected_lang = "hi"
    elif re.search(r"\b(?:english|angrezi)\b", lower):
        detected_lang = "en"

    if not detected_lang:
        return None, clean_text, False

    stripped = clean_text

    if detected_lang == "hi":
        stripped = re.sub(
            r"^\s*(?:please\s+)?(?:can\s+you\s+)?(?:simple\s+|aasan\s+)?(?:explain\s+in\s+|in\s+|switch\s+to\s+|speak\s+in\s+|speak\s+|answer\s+in\s+|talk\s+in\s+|translate\s+explanation\s+to\s+|translate\s+to\s+)?(?:hindi|hindee)(?:\s+(?:mein|me|mai|vich|ch))?(?:\s+(?:batao|bataao|samjhao|samjha|smjha|smjhao|explain\s+karo|explain\s+kro|please|karo|daso|answer))?(?:\s+(?:ji|bhai|bro|na|do))?[\s,:\-]*",
            "", stripped, flags=re.IGNORECASE
        )
        stripped = re.sub(
            r"[\s,:\-]*(?:(?:in\s+)?(?:hindi|hindee)(?:\s+(?:mein|me|mai|vich|ch))?(?:\s+(?:batao|bataao|samjhao|samjha|smjha|smjhao|explain\s+karo|explain\s+kro|please|karo))?)(?:\s+(?:ji|bhai|bro|na|do))?\s*[\.?!]?$",
            "", stripped, flags=re.IGNORECASE
        )
        stripped = re.sub(
            r"^\s*(?:कृपया\s+)?(?:सरल\s+)?(?:हिंदी|हिन्दी)(?:\s+में)?(?:\s+(?:बताओ|बताइए|समझाओ|समझाइए|करें|कहो|बात\s+करें|उत्तर\s+दें))?[\s,:\-]*",
            "", stripped
        )
        stripped = re.sub(
            r"[\s,:\-]*(?:हिंदी|हिन्दी)(?:\s+में)?(?:\s+(?:बताओ|बताइए|समझाओ|समझाइए|बात\s+करें|उत्तर\s+दें))?(?:\s+(?:जी|भाई))?\s*[\.?!]?$",
            "", stripped
        )

    elif detected_lang == "pa":
        stripped = re.sub(
            r"^\s*(?:please\s+)?(?:can\s+you\s+)?(?:simple\s+)?(?:explain\s+in\s+|in\s+|switch\s+to\s+|speak\s+in\s+|speak\s+|answer\s+in\s+|talk\s+in\s+|translate\s+explanation\s+to\s+|translate\s+to\s+)?(?:punjabi|panjabi)(?:\s+(?:mein|me|mai|vich|ch))?(?:\s+(?:batao|daso|dasso|samjhao|samjha|smjha|explain\s+karo|explain\s+kro|please|karo|answer))?(?:\s+(?:ji|bhai|bro|na|do))?[\s,:\-]*",
            "", stripped, flags=re.IGNORECASE
        )
        stripped = re.sub(
            r"[\s,:\-]*(?:(?:in\s+)?(?:punjabi|panjabi)(?:\s+(?:mein|me|mai|vich|ch))?(?:\s+(?:batao|daso|dasso|samjhao|samjha|smjha|explain\s+karo|explain\s+kro|please|karo))?)(?:\s+(?:ji|bhai|bro|na|do))?\s*[\.?!]?$",
            "", stripped, flags=re.IGNORECASE
        )
        stripped = re.sub(
            r"^\s*(?:ਕਿਰਪਾ\s+ਕਰਕੇ\s+)?(?:ਸਰਲ\s+)?(?:ਪੰਜਾਬੀ)(?:\s+(?:ਵਿੱਚ|ਵਿਚ|ਚ))?(?:\s+(?:ਦੱਸੋ|ਦਸੋ|ਸਮਝਾਓ|ਸਮਝਾ|ਦੱਸਣਾ|ਜਵਾਬ\s+ਦਿਓ|ਦਿਓ|ਕਰੋ))?(?:\s+(?:ਜੀ|ਬਾਈ|ਵੀਰ))?[\s,:\-]*",
            "", stripped
        )
        stripped = re.sub(
            r"[\s,:\-]*(?:ਪੰਜਾਬੀ)(?:\s+(?:ਵਿੱਚ|ਵਿਚ|ਚ))?(?:\s+(?:ਦੱਸੋ|ਦਸੋ|ਸਮਝਾਓ|ਸਮਝਾ|ਜਵਾਬ\s+ਦਿਓ|ਦਿਓ|ਕਰੋ))?(?:\s+(?:ਜੀ|ਬਾਈ|ਵੀਰ))?\s*[\.?!]?$",
            "", stripped
        )

    elif detected_lang == "en":
        stripped = re.sub(
            r"^\s*(?:please\s+)?(?:can\s+you\s+)?(?:simple\s+)?(?:explain\s+in\s+|in\s+|switch\s+to\s+|speak\s+in\s+|speak\s+|answer\s+in\s+)?(?:english)(?:\s+(?:mein|me|mai))?(?:\s+(?:batao|samjhao|explain\s+karo|explain\s+kro|please))?(?:\s+(?:ji|bhai|bro))?[\s,:\-]*",
            "", stripped, flags=re.IGNORECASE
        )
        stripped = re.sub(
            r"[\s,:\-]*(?:(?:in\s+)?english(?:\s+please)?|english\s+please)(?:\s+(?:ji|bhai|bro))?\s*[\.?!]?$",
            "", stripped, flags=re.IGNORECASE
        )

    stripped = re.sub(r"^(?:ji|bhai|bro|please|plz|ਜੀ|जी)\b", "", stripped, flags=re.IGNORECASE)
    cleaned_residual = re.sub(
        r"(?:explain|tell|say|show|translate|repeat|switch|give|write|speak|talk|answer|explanation|das|daso|dasso|samjha|samjhao|smjha|smjhao|batao|bataao|bolo|karo|kro|ch|vich|me|mein|mai|that|this|it|please|plz|can\s+you|could\s+you|ji|bhai|bro|na|do|to|in|now|ab|shuddh|language|boli|ਜੀ|जी|ਦਿਓ|ਦੋ)",
        "", stripped, flags=re.IGNORECASE
    ).strip(" ,.:-!?\"'")
    is_pure = len(cleaned_residual) == 0

    return detected_lang, stripped, is_pure



class IntentClassifier:
    """
    Hybrid Semantic Intent Classifier:
    Combines high-confidence medical safety boundary guards,
    domain phrase matchers, trained statistical classifier,
    and TF-IDF vector space exemplars.
    Supports both V1 and V2 taxonomies via dual-taxonomy mapping.
    """

    def __init__(self):
        self.intent_names = list(INTENT_EXEMPLARS.keys())
        self.intent_metadata = {k: {
            "category": v["category"],
            "requires_case_context": v["requires_case_context"],
            "requires_safety_guard": v["requires_safety_guard"]
        } for k, v in INTENT_EXEMPLARS.items()}

        # Build corpus of combined examples per intent
        self.documents = []
        for name in self.intent_names:
            phrases = INTENT_EXEMPLARS[name]["examples"]
            self.documents.append(" ".join(phrases))

        # Train TF-IDF with character and word n-grams
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            analyzer="word",
            lowercase=True,
            sublinear_tf=True
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)

        # Load optional legacy trained statistical classifier if exists
        self.local_classifier = None
        model_path = os.path.join(os.path.dirname(__file__), "models", "local_intent_classifier.pkl")
        if os.path.exists(model_path):
            try:
                with open(model_path, "rb") as f:
                    self.local_classifier = pickle.load(f)
                logger.info("Loaded local trained intent classifier successfully.")
            except Exception as e:
                logger.warning("Could not load local trained classifier: %s", e)

        # Load or train V2 conversational classifier
        self.v2_pipeline = None
        v2_model_path = os.path.join(os.path.dirname(__file__), "models", "conversational_v2_classifier.pkl")
        if os.path.exists(v2_model_path):
            try:
                with open(v2_model_path, "rb") as f:
                    self.v2_pipeline = pickle.load(f)
                logger.info("Loaded conversational v2 classifier successfully.")
            except Exception as e:
                logger.warning("Could not load conversational v2 classifier: %s", e)

        if self.v2_pipeline is None:
            dataset_path = os.path.join(os.path.dirname(__file__), "data", "conversational_qa_dataset.jsonl")
            if os.path.exists(dataset_path):
                try:
                    import json
                    from sklearn.pipeline import Pipeline
                    from sklearn.linear_model import LogisticRegression
                    data = [json.loads(line) for line in open(dataset_path, encoding='utf-8')]
                    X = [d['text'] for d in data]
                    y = [d['intent'] for d in data]
                    self.v2_pipeline = Pipeline([
                        ('tfidf', TfidfVectorizer(
                            analyzer='char_wb',
                            ngram_range=(2, 5),
                            lowercase=True,
                            sublinear_tf=True,
                            min_df=1
                        )),
                        ('clf', LogisticRegression(
                            C=10.0,
                            max_iter=1000,
                            class_weight='balanced',
                            solver='lbfgs'
                        ))
                    ])
                    self.v2_pipeline.fit(X, y)
                    try:
                        os.makedirs(os.path.dirname(v2_model_path), exist_ok=True)
                        with open(v2_model_path, "wb") as f:
                            pickle.dump(self.v2_pipeline, f)
                    except Exception:
                        pass
                    logger.info("Fitted and cached conversational v2 classifier successfully.")
                except Exception as e:
                    logger.warning("Could not train conversational v2 classifier: %s", e)

    def extract_target_stage(self, text: str) -> Optional[str]:
        """Extracts explicit stage/level number if requested (e.g. 'level 3 kya hota h' -> '3')"""
        lower = text.lower()
        m = re.search(r"\b(?:level|stage|लेवल|ਸਟੇਜ|ਲੈਵਲ)\s*([123])\b", lower)
        if m:
            return m.group(1)
        if "moderate caries" in lower:
            return "2"
        return None

    def _make_intent_result(self, v2_name: str, confidence: float, taxonomy: str = "v1", v1_override: Optional[str] = None) -> IntentResult:
        v1_name = v1_override if v1_override else CANONICAL_V2_TO_V1.get(v2_name, v2_name)
        v2_meta = V2_INTENT_METADATA.get(v2_name, {
            "category": "general_conversation",
            "requires_case_context": False,
            "requires_safety_guard": False
        })
        v1_meta = self.intent_metadata.get(v1_name, v2_meta)
        
        if taxonomy == "v2":
            return IntentResult(
                name=v2_name,
                confidence=confidence,
                category=v2_meta["category"],
                requires_case_context=v2_meta["requires_case_context"],
                requires_safety_guard=v2_meta["requires_safety_guard"],
                canonical_v2_name=v2_name
            )
        else:
            return IntentResult(
                name=v1_name,
                confidence=confidence,
                category=v1_meta["category"],
                requires_case_context=v1_meta["requires_case_context"],
                requires_safety_guard=v1_meta["requires_safety_guard"],
                canonical_v2_name=v2_name
            )

    def classify(self, text: str, taxonomy: str = "v1") -> IntentResult:
        clean_text = text.strip()
        lower = clean_text.lower()

        # Check language preference directive
        detected_lang, stripped_text, is_pure_lang = detect_language_request(clean_text)

        # Typo correction normalization
        lower_sub = lower.replace("co-ordinates", "coordinates").replace("co ordinates", "coordinates").replace("co-ordinate", "coordinate").replace("co ordinate", "coordinate").replace("lesion locaton", "lesion location")
        words = lower_sub.split()
        norm_words = []
        for w in words:
            w_strip = w.strip(" ,.:-!?\"'")
            norm_words.append(TYPO_CORRECTIONS.get(w_strip, w_strip))
        normalized_lower = " ".join(norm_words)

        # 1. High-Priority Safety Guards
        safety_override = self._check_safety_overrides(clean_text, normalized_lower, lower, taxonomy=taxonomy)
        if safety_override is not None:
            return safety_override

        if stripped_text:
            stripped_sub = stripped_text.lower()
            norm_stripped = " ".join([TYPO_CORRECTIONS.get(w.strip(" ,.:-!?\"'"), w.strip(" ,.:-!?\"'")) for w in stripped_sub.split()])
            safety_override_stripped = self._check_safety_overrides(stripped_text, norm_stripped, stripped_sub, taxonomy=taxonomy)
            if safety_override_stripped is not None:
                return safety_override_stripped

        # 2. Pure Language Preference check
        if detected_lang and is_pure_lang:
            return self._make_intent_result("language_preference", 0.98, taxonomy=taxonomy)

        text_for_query = stripped_text if (detected_lang and stripped_text) else clean_text
        lower_for_query = text_for_query.lower().replace("co-ordinates", "coordinates").replace("co ordinates", "coordinates").replace("co-ordinate", "coordinate").replace("co ordinate", "coordinate").replace("lesion locaton", "lesion location")
        norm_query_words = [TYPO_CORRECTIONS.get(w.strip(" ,.:-!?\"'"), w.strip(" ,.:-!?\"'")) for w in lower_for_query.split()]
        normalized_query = " ".join(norm_query_words)

        # 3. For V2 taxonomy, use Statistical V2 Classifier if available (Safety overrides already checked)
        if taxonomy == "v2" and hasattr(self, "v2_pipeline") and self.v2_pipeline is not None:
            try:
                import numpy as np
                probs = self.v2_pipeline.predict_proba([normalized_query])[0]
                idx = int(np.argmax(probs))
                conf = float(probs[idx])
                pred_intent = self.v2_pipeline.classes_[idx]
                if conf >= 0.20:
                    return self._make_intent_result(pred_intent, round(conf, 2), taxonomy=taxonomy)
            except Exception:
                pass

        # 4. Deterministic Common Phrase Matching (V1 primary / V2 fallback)
        phrase_override = self._check_common_phrases(text_for_query, normalized_query, lower_for_query, taxonomy=taxonomy)
        if phrase_override is not None:
            return phrase_override

        # 5. Statistical V2 Classifier for V1 taxonomy
        if taxonomy != "v2" and hasattr(self, "v2_pipeline") and self.v2_pipeline is not None:
            try:
                import numpy as np
                probs = self.v2_pipeline.predict_proba([normalized_query])[0]
                idx = int(np.argmax(probs))
                conf = float(probs[idx])
                pred_intent = self.v2_pipeline.classes_[idx]
                if conf >= 0.25:
                    return self._make_intent_result(pred_intent, round(conf, 2), taxonomy=taxonomy)
            except Exception:
                pass

        # 6. TF-IDF Exemplar Matching fallback
        try:
            query_vec = self.vectorizer.transform([normalized_query])
            from sklearn.metrics.pairwise import cosine_similarity
            scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]
            max_idx = int(scores.argmax())
            max_score = float(scores[max_idx])

            if max_score > 0.08:
                matched_name = self.intent_names[max_idx]
                confidence = min(0.96, max(0.68, 0.60 + max_score * 0.45))
            else:
                matched_name = "general_question"
                confidence = 0.65
        except Exception:
            matched_name = "general_question"
            confidence = 0.60

        v2_name = CANONICAL_V1_TO_V2.get(matched_name, matched_name)
        return self._make_intent_result(v2_name, round(confidence, 2), taxonomy=taxonomy)

    def _check_safety_overrides(self, text: str, lower: str, raw_lower: str = "", taxonomy: str = "v1") -> IntentResult:
        targets = [lower, raw_lower, text.lower()]
        def has_match(keywords):
            return any(any(kw in t for t in targets) for kw in keywords)

        # 1. Invasive treatment & treatment planning requests
        if has_match([
            "how to treat", "what treatment do i need", "root canal needed", "root canal",
            "extraction needed", "how do i cure this", "how to cure", "rct",
            "dental procedure do i need", "cure this caries at home",
            "extraction karwana padega", "surgery lagegi", "is this curable",
            "treatment immediately", "filling", "tooth filling", "filling karwani",
            "filling karwani padegi", "surgery is required", "surgery required",
            "surgery needed", "need surgery", "require surgery", "tell me if surgery",
            "is surgery required", "surgery", "rct karwana", "dant nikalwana", "daant nikalwana",
            "can i ignore it", "ignore it", "ignore kar sakta hu", "ignore kar sakte", "ignore kr skta",
            "what should i do now", "what should i do", "kya karu ab", "kya karun", "kya karna chahiye",
            "koi gharelu nuskha batao", "gharelu nuskha", "gharelu upchar", "home remedy", "home remedies",
            "kya daant bach jayega", "daant bach jayega", "dant bach jayega", "save my tooth", "can my tooth be saved",
            "should i get a filling", "kya root canal hoga",
            "ਕੀ ਸਰਜਰੀ", "ਕੀ ਆਰਸੀਟੀ", "ਦੰਦ ਕੱਢਣਾ", "ਕੀ ਇਲਾਜ", "ਇਲਾਜ ਕਿਵੇਂ", "ਦੰਦ ਭਰਵਾਉਣਾ",
            "ਕੀ ਮੈਨੂੰ ਫਿਲਿੰਗ ਕਰਵਾਉਣੀ ਚਾਹੀਦੀ ਹੈ", "ਫਿਲਿੰਗ ਕਰਵਾਉਣੀ", "ਫਿਲਿੰਗ"
        ]):
            return self._make_intent_result("treatment_request", 0.96, taxonomy=taxonomy)

        # 2. Definitive clinical claim / cancer / diagnostic certainty
        if (has_match([
            "cancer", "tumor", "malignan", "do i definitely have cancer",
            "oral cancer", "definitely caries", "definitely a cavity",
            "is this definitely caries", "is level 2 definitely a cavity",
            "is this definitely a cavity", "can you confirm i have caries",
            "kya ye confirm caries hai", "confirm caries",
            "guarantee", "guarantee me",
            "confirm cavity hai", "pakka cavity", "pakka caries",
            "क्या मुझे निश्चित रूप से कैविटी है", "निश्चित रूप से कैविटी", "निश्चित रूप से",
            "पक्का कैरीज़", "पक्का कैविटी", "ਕੈਂਸਰ", "ਕੀ ਇਹ ਪੱਕਾ ਕੈਵਿਟੀ ਹੈ", "ਕੀ ਪੱਕਾ ਕੈਰੀਜ਼ ਹੈ",
            "ਕੀ ਮੈਨੂੰ ਪੱਕੇ ਤੌਰ 'ਤੇ ਕੈਰੀਜ਼ ਹੈ", "ਪੱਕੇ ਤੌਰ 'ਤੇ ਕੈਰੀਜ਼", "ਪੱਕੇ ਤੌਰ"
        ]) and not any(any(w in t for w in ["kya mujhe", "do i have", "kya ye report"]) for t in targets)):
            return self._make_intent_result("definitive_clinical_claim", 0.96, taxonomy=taxonomy)

        # 3. Medication request
        if has_match([
            "medicine", "medication", "pill", "pills", "tablet", "tablets", "painkiller", "painkillers",
            "antibiotic", "antibiotics", "amoxicillin", "prescribe", "prescription",
            "medicine leni", "dawa", "dawakhana", "can i take medicine", "dawai batao",
            "dawai leni", "should i take medicine", "which medicine should i take",
            "mujhe konsi dawai leni chahiye", "konsi dawai", "konsi painkiller", "prescribe me an antibiotic",
            "dard ho raha h konsi painkiller lu",
            "दर्द की दवा", "दवा लेनी", "दर्द की गोली", "गोली लेनी", "दवा बताओ",
            "मुझे कौन सी दवा लेनी चाहिए", "दवा लेनी चाहिए", "कौन सी दवा",
            "ਕਿਹੜੀ ਦਵਾਈ ਲੈਣੀ ਚਾਹੀਦੀ ਹੈ", "ਕਿਹੜੀ ਦਵਾਈ ਲੈਣੀ", "ਕਿਹੜੀ ਦਵਾਈ", "ਦਵਾਈ ਲੈਣੀ", "ਦਵਾਈ", "ਦਰਦ ਦੀ ਗੋਲੀ", "ਗੋਲੀ", "ਦਰਦ ਦੀ ਦਵਾਈ", "ਕਿਹੜੀ ਗੋਲੀ"
        ]):
            return self._make_intent_result("medication_request", 0.96, taxonomy=taxonomy)

        # 4. Emergency / acute symptoms
        if has_match([
            "emergency", "unbearable pain", "swollen jaw", "bleeding profusely",
            "severe toothache cannot bear", "face is swollen", "face swollen",
            "jaw swollen", "extreme jaw pain", "muh sujh gaya", "bahut tez dard",
            "high fever and bleeding", "my tooth hurts terribly please help",
            "tooth hurts terribly", "hurts terribly", "terribly please help",
            "bhai emergency hai bohot dard ho raha hai", "bohot dard ho raha hai",
            "bahut dard ho raha hai",
            "बहुत तेज़ दर्द है, क्या करूँ", "बहुत तेज़ दर्द", "ਤੇਜ਼ ਦਰਦ",
            "ਦੰਦ ਵਿੱਚ ਬਹੁਤ ਦਰਦ ਹੈ ਕੀ ਕਰਾਂ", "ਬਹੁਤ ਦਰਦ ਹੈ ਕੀ ਕਰਾਂ", "ਬਹੁਤ ਦਰਦ", "ਬਹੁਤ ਜ਼ਿਆਦਾ ਦਰਦ", "ਐਮਰਜੈਂਸੀ", "ਸੁੱਜਿਆ ਹੋਇਆ"
        ]):
            return self._make_intent_result("emergency_or_urgent_concern", 0.95, taxonomy=taxonomy)

        # 5. Definitive diagnosis request
        if has_match([
            "definitely have caries", "definitely have a cavity", "do i definitely have", "definitely have",
            "ye cavity hai kya", "diagnose me", "confirm i have cavity", "is this a cavity for sure",
            "tell me for sure if i have caries", "give me a diagnosis",
            "do i have caries", "do i have a cavity", "do i have decay", "do i have tooth decay",
            "how can i know whether i have caries", "how do i know if i have caries",
            "can you diagnose this", "confirm that i have decay", "sach me cavity hai",
            "does the report prove i have caries", "can you confirm i have a cavity",
            "is the lesion real cavity", "is that a diagnosis", "is this a definitive diagnosis",
            "daant mein caries h kaise pta lgau", "daant me caries h kaise pta lgau",
            "caries h kaise pta lgau", "caries hai kaise pata lagau", "kaise pta lgau",
            "kaise pata lagau", "kya mujhe cavity hai", "kya mujhe caries hai", "kya mujhe pakka cavity",
            "kya mujhe pakka", "does that mean i have caries", "does this mean i have caries",
            "does this mean i have a cavity", "does this prove i have caries",
            "tell me if i have decay", "tell me if i have caries",
            "kya ye report confirm karti hai ki cavity", "confirm karti hai ki cavity",
            "do i have caries or not", "tell me if i have caries",
            "caries hai kya", "mere daant me caries hai kya", "kya mujhe cavity ho gayi h",
            "bhai ye caries hai kya", "kya ye bimari fail sakti hai",
            "is this serious",
            "क्या मुझे कैविटी है", "ਕੀ ਮੈਨੂੰ ਕੈਵਿਟੀ ਹੈ", "कैविटी है", "ਕੈਵਿਟੀ ਹੈ", "caries ਹੈ",
            "ਕੀ ਇਹ ਨਿਦਾਨ ਹੈ", "ਕੀ ਇਹ ਡਾਇਗਨੋਸਿਸ ਹੈ"
        ]) or any((("caries" in t or "decay" in t or "cavity" in t) and any(w in t for w in ["do i have", "how to know", "kaise", "pta", "pata", "prove", "confirm", "diagnose", "for sure", "pakka"])) for t in targets):
            return self._make_intent_result("diagnosis_request", 0.95, taxonomy=taxonomy)

        return None

    def _check_common_phrases(self, text: str, lower: str, raw_lower: str = "", taxonomy: str = "v1") -> IntentResult:
        targets = [lower, raw_lower, text.lower()]
        def has_match(keywords):
            return any(any(kw in t for t in targets) for kw in keywords)

        # 1. Model Probability
        if (has_match([
            "82 percent", "82%", "93 percent", "93%", "82% ka kya matlab",
            "model probability", "model predicted probability",
            "how confident is the model", "confidence 82 percent",
            "probability percentage signify", "why not 100 percent",
            "sigmoid probability", "percentage mean", "probability kya hai",
            "probability score", "model percentage", "82% da ki matlab",
            "82% का क्या मतलब", "ਪ੍ਰਤੀਸ਼ਤ ਦਾ ਕੀ ਮਤਲਬ", "ਪ੍ਰੋਬੇਬਿਲਿਟੀ", "संभावना",
            "probability of lesion", "confidence level", "how sure is the model",
            "confidence kitna hai is finding ka", "confidence kitna hai"
        ]) or (any("percent" in t or "%" in t or "probability" in t for t in targets) and any(any(w in t for w in ["why", "what", "how", "matlab", "mean", "model", "confident", "sure"]) for t in targets))) and not any("area" in t for t in targets):
            return self._make_intent_result("probability_query", 0.95, taxonomy=taxonomy, v1_override="model_probability")

        # 2. Capabilities
        if has_match([
            "what are your capabilities", "what can you do", "features of this assistant",
            "what features does this assistant offer", "what features do you have",
            "what are your capabilities?", "capabilities", "what can this app do", "your features"
        ]) or (any("capabilities" in t for t in targets) and any(w in t for w in ["what", "who", "your", "tell", "list"])):
            return self._make_intent_result("capabilities", 0.96, taxonomy=taxonomy, v1_override="capabilities")

        # 3. Clarification
        if has_match([
            "dont understand", "don't understand", "dont get it", "don't get it",
            "not clear", "not understand", "confused", "can you explain simply",
            "explain simply", "please explain simply", "explain it simply",
            "samajh nahi aaya", "samajh nahi aa raha", "samajh nahi aara",
            "samajh nhi aaya", "mujhe samajh nahi aaya", "kuch samajh nahi",
            "smjh sa nhi aaya", "smjh nhi aya", "smjh nahi aaya",
            "samjha nahi", "bhai samajh nahi aaya", "ye samajh nahi aa raha",
            "thoda easy batao", "simple mein batao", "simple me batao",
            "simple daso", "simple samjhao", "mainu samajh nahi aaya", "mainu samajh nahi",
            "what do you mean", "can you clarify", "explain the same finding simply",
            "i dont get it bro", "bhai smjh nhi aya kuch bhi",
            "iska matlab", "iska kya matlab", "iska matlab?", "wait what", "wait what?",
            "what does that mean", "iska ki matlab",
            "मुझे समझ नहीं आया", "ਮੈਨੂੰ ਸਮਝ ਨਹੀਂ ਆਇਆ", "ਸਰਲ ਸ਼ਬਦਾਂ ਵਿੱਚ", "ਸੌਖੇ ਤਰੀਕੇ ਨਾਲ ਦੱਸੋ ਜੀ",
            "easy words me", "thoda simple karo", "fir se samjhao", "easy way me", "ਦੋਬਾਰਾ ਸਪਸ਼ਟ", "दोबारा स्पष्ट"
        ]) and not any(any(w in t for w in ["x-ray", "xray", "result", "scan", "mujhe simple language me samjha"]) for t in targets):
            return self._make_intent_result("clarification", 0.96, taxonomy=taxonomy, v1_override="clarification")

        # 4. Limitations / OOD Modality
        if has_match([
            "cbct", "3d scan", "3d cbct", "cervical burnout", "panoramic distortion", "model limitations",
            "why does panoramic opg have distortion", "why does the model make mistakes"
        ]):
            return self._make_intent_result("reset", 0.96, taxonomy=taxonomy, v1_override="limitations")

        # 5. Report Summary / Layman overview
        if has_match([
            "simple words", "simple language", "beginner", "aasan bhasha",
            "in simple words", "explain this like i'm a beginner", "layman", "report summary",
            "explain my report", "explain this report for me simply", "give me a brief summary of results",
            "ਇਸ ਰਿਪੋਰਟ ਦਾ ਸਾਰ", "इस रिपोर्ट का सारांश", "layman summary please",
            "summarize my x-ray", "summarize the x-ray", "summarize this x-ray",
            "summarize my report", "can u sumarize my x-ray", "summarize the x-ray in punjabi",
            "meri report da summary", "punjabi ch summarize", "hindi me summarize",
            "explain my report in hindi", "tell me what this report says in simple words",
            "explain this x-ray result for a beginner", "mujhe simple language me samjha",
            "इस रिपोर्ट को आसान भाषा में समझाइए"
        ]) or (any("summarize" in t or "sumarize" in t for t in targets) and any("x-ray" in t or "xray" in t or "report" in t or "scan" in t for t in targets)):
            return self._make_intent_result("segmentation_findings", 0.96, taxonomy=taxonomy, v1_override="report_summary")

        # 6. Coordinate Query / BBox
        if has_match([
            "bbox", "bbox?", "bounding box", "cordinate", "coordinate", "cordinates", "coordinates", "kordinates", "kordinate",
            "what are the coordinates", "what are coordinates", "give lesion coordinates",
            "lesion coordinates", "coordinates of lesion", "lesion ka coordinate",
            "coordinates kya hai", "cordinates kya hain", "kordinates kya", "x-y", "xy",
            "pixel coordinates", "pixel coordinate", "bounding box coordinates",
            "ਨਿਰਦੇਸ਼ਾਂਕ", "निर्देशांक", "ਕੋਆਰਡੀਨੇਟਸ", "ਕੋਰਡੀਨੇਟ", "ਕੋਰਡੀਨੇਟਸ",
            "coordinates batao", "coordinate batao", "coordinates please"
        ]) or any("coordinate" in t or "cordinate" in t or "bbox" in t for t in targets):
            return self._make_intent_result("coordinate_query", 0.96, taxonomy=taxonomy, v1_override="lesion_location")

        # 7. Lesion Location
        if has_match([
            "where is the lesion", "where is it", "where is lesion",
            "where are the lesions", "where are lesions",
            "where exactly is the lesion", "where exactly is lesion",
            "where exactly is it", "kaha hai", "kaha par hai", "kahaan hai", "kahan hai",
            "location of lesion", "lesion location", "location kya hai", "lesion ki location",
            "lesion kidhar hai", "lesion exactly kidhar hai",
            "where did model find", "where did the model find", "where did the model find it",
            "give lesion position", "tell me lesion location", "highlighted area kaha hai",
            "highlighted area where", "where is the highlighted", "where is highlighted",
            "where is that exactly", "where is the decay", "where is the affected area",
            "where did the model highlight", "ਲੈਸ਼ਨ ਕਿੱਥੇ ਹੈ", "ਕਿੱਥੇ ਹੈ", "ਲੀਜਨ कहाँ स्थित है", "लीजन कहाँ स्थित है",
            "kidhar dikha", "kaha dikha", "which quadrant", "kis jagah par hai", "kis jagah", "ਕਿਸ ਜਗ੍ਹਾ",
            "ਕਿਸ ਪਾਸੇ", "किस तरफ", "ਕਿਹੜੇ ਪਾਸੇ", "ਨੁਕਸਾਨ ਵਾਲੀ ਥਾਂ"
        ]):
            return self._make_intent_result("lesion_location", 0.96, taxonomy=taxonomy, v1_override="lesion_location")

        # 8. Area Query / Highlighted region (including red mask)
        if (has_match([
            "area kitna", "area kya hai", "area of the lesion", "area of lesion",
            "lesion area", "pixel area", "area in pixels", "pixels", "pixel",
            "how big is it", "how big is this lesion", "size of the lesion",
            "size of lesion", "lesion size", "size kya hai", "kitna bada hai",
            "kitna bada lesion", "highlighted area size", "area of highlighted",
            "cavity area", "cavity size", "size of cavity", "spot size",
            "area of this spot", "size of this spot", "kitne pixel", "px size",
            "ਖੇਤਰਫਲ", "ਪਿਕਸਲ ਖੇਤਰ", "क्षेत्रफल", "एरिया कितना", "साइज कितना",
            "lesion kitna bada", "lesion da size", "lesion da area", "area?",
            "affected area percentage", "affected area percent",
            "what is this highlighted thing", "what does this highlighted area mean",
            "why is this area highlighted", "why did the model highlight this",
            "what does the green mask mean", "what does the red highlight show",
            "what does this colored box mean", "what is l1", "what does l1 mean",
            "why did the model highlight this spot", "why this area", "what does the red highlight mean",
            "what does the red area mean", "red area mean", "red highlight mean",
            "लाल रंग", "एक्स-रे में लाल रंग", "ਲਾਲ ਰੰਗ", "लाल रंग का क्षेत्र क्या दर्शाता है"
        ]) or any("area" in t and any(w in t for w in ["kitna", "kya", "size", "bada", "pixel", "how", "what", "measure", "percent", "percentage"]) for t in targets)
           or any("size" in t and any(w in t for w in ["lesion", "cavity", "spot", "highlighted", "kitna", "kya", "how big"]) for t in targets)) and not any(any(w in t for w in ["dataset", "training", "affected area?", "reveal"]) for t in targets):
            return self._make_intent_result("area_query", 0.96, taxonomy=taxonomy, v1_override="highlighted_region")

        # 9. Depth Query
        if has_match([
            "how deep", "depth of", "depth kitni", "kitna deep", "deep hai", "is this deep",
            "depth of decay", "pulp depth", "enamel depth", "dentin depth", "how deep is the lesion",
            "depth of this lesion", "how deep has it reached", "decay depth", "lesion depth",
            "pulp involvement", "dentin involvement", "enamel penetration", "reached the pulp",
            "डेंटीन तक", "पल्प तक", "ਕਿੰਨਾ ਡੂੰਘਾ", "ਕਿੰਨੀ ਡੂੰਘੀ", "ਡੂੰਘਾਈ", "ਡੂੰਘਾਈ ਕਿੰਨੀ",
            "depth batao", "depth kya hai", "is cavity deep", "cavity deep",
            "should i panic about this finding", "should i panic about this", "should i panic", "panic about this"
        ]) or any("depth" in t and any(w in t for w in ["what", "how", "lesion", "caries", "decay", "tooth", "kitni", "kya"]) for t in targets):
            return self._make_intent_result("depth_query", 0.96, taxonomy=taxonomy, v1_override="severity_explanation")

        # 10. Severity / Worried
        if has_match([
            "should i be worried", "is this bad", "how serious is this", "am i screwed", "is it very severe",
            "kitna kharab hai", "kya ye dangerous hai", "how bad is the cavity", "severity level",
            "scared after seeing this report", "really worried after this report", "i'm really scared",
            "i'm really worried", "am i in danger", "is it severe", "ye daant kharab karega", "shud i b worrid",
            "lose my tooth", "lose my teeth", "lose the tooth", "lose tooth"
        ]) or any("worried" in t or "scared" in t for t in targets):
            return self._make_intent_result("depth_query", 0.96, taxonomy=taxonomy, v1_override="severity_explanation")

        # 11. Tooth Identification
        if has_match([
            "which tooth", "tooth kaunsa", "konsa daant", "konsa wala daant", "fdi tooth",
            "tell me about tooth", "what tooth", "which molar", "which premolar",
            "which canine", "tooth details", "tooth information", "tooth number",
            "which tooth is this", "which tooth number is involved", "is tooth 16",
            "is this tooth 46", "tell me about tooth 16", "what tooth is that",
            "what about the other teeth", "यह कौन सा दाँत है", "ਕਿਹੜਾ ਦੰਦ ਹੈ ਇਹ", "ਕਿਹੜਾ ਦੰਦ",
            "दंत संख्या", "दाँत संख्या", "fdi", "tooth id", "tooth identification",
            "ye upar ka daant hai ya niche", "upar ka daant", "niche ka daant",
            "दांत का नंबर", "दाँत का नंबर", "दंद दा नंबर", "ਦੰਦ ਦਾ ਨੰਬਰ", "ਦੰਦਾਂ ਦਾ ਨੰਬਰ",
            "ਕੌਣ ਸਾ ਦੰਦ", "ਕਿਹੜੇ ਦੰਦ", "ਕਿਹੜਾ ਦੰਦ ਪ੍ਰਭਾਵਿਤ", "कौन सा दाँत प्रभावित"
        ]) or (any(("daant" in t or "tooth" in t or "teeth" in t or "dant" in t or "दंत" in t or "दांत" in t or "दाँत" in t or "ਦੰਦ" in t) and any(w in t for w in ["konsa", "which", "number", "kiska", "kahan ka", "kaunsa", "kidhar ka", "नंबर", "ਨੰਬਰ", "ਕਿਹੜਾ", "कौन सा", "प्रभावित", "ਕੌਣ ਸਾ", "ਉੱਪਰਲਾ", "ਹੇਠਲਾ", "upar", "niche"]) for t in targets) and not any(any(w in t for w in ["lose", "lost", "stage", "level", "ਸਟੇਜ", "ਲੈਵਲ"]) for t in targets)):
            return self._make_intent_result("tooth_identification", 0.96, taxonomy=taxonomy, v1_override="tooth_information")

        # 12. Threshold Inquiry
        if has_match([
            "what is tau 0.50", "operating threshold", "decision threshold",
            "why threshold 0.50", "tau 0.50", "tau mean", "why not 0.40",
            "0.50 threshold", "threshold value", "threshold inquiry", "cutoff threshold",
            "decision boundary", "tau value", "tau kya hai"
        ]) or any(re.search(r"\btau\b", t) or "threshold" in t for t in targets):
            return self._make_intent_result("threshold_inquiry", 0.96, taxonomy=taxonomy, v1_override="threshold_explanation")

        # 13. Training Data
        if has_match([
            "training data", "training dataset", "how many radiographs", "training set",
            "dataset", "dataset size", "dataset used", "how many images were used",
            "which dataset", "dataset me kitne images", "training images", "labeled data",
            "ਕਿਹੜਾ ਡਾਟਾਸੈੱਟ", "ਕਿੰਨੇ ਐਕਸ-ਰੇ", "ਟ੍ਰੇਨਿੰਗ ਡਾਟਾ", "ट्रेनिंग डेटा", "डेटासेट",
            "how does your model even find this"
        ]) or any("dataset" in t or "training set" in t or "training data" in t for t in targets):
            return self._make_intent_result("training_data", 0.96, taxonomy=taxonomy, v1_override="segmentation_explanation")

        # 14. Staging (Active report assigned stage)
        if (has_match([
            "what is my current level", "what is the stage of this report", "what is stage of this report",
            "what is the stage", "what is my stage", "what is my current stage", "current stage", "mera konsa stage",
            "report me konsa level", "report me konsa stage", "stage did the model assign",
            "what level is this", "kya stage hai meri x-ray", "mera konsa level aya",
            "what severity stage was assigned", "ਇਸ ਰਿਪੋਰਟ ਦੀ ਸਟੇਜ ਕੀ ਹੈ",
            "इस रिपोर्ट की स्टेज", "रिपोर्ट की स्टेज", "ਸਟੇਜ ਕੀ ਹੈ",
            "does my report show level", "my report show level", "which stage was detected",
            "my report ka level", "meri report ka level", "kya meri scan me level",
            "mera level 2 hai kya", "kya mere report me level 3", "mere report me kya stage",
            "mere daant ka stage kya hai", "tell me my stage", "konsa level mila",
            "heuristic stage assigned", "classified as early or moderate", "stage kya hai",
            "मेरी रिपोर्ट में कौन सा स्टेज", "रिपोर्ट में क्या स्तर दिखाया", "ਕੀ ਮੇਰੀ ਰਿਪੋਰਟ ਵਿੱਚ ਲੈਵਲ",
            "which stage", "konsa stage detect", "explain level 3 in my report"
        ]) or any(("stage" in t or "level" in t or "ਸਟੇਜ" in t or "ਲੈਵਲ" in t or "ਸਤਰ" in t or "पड़ाव" in t or "स्टेज" in t or "लेवल" in t or "स्तर" in t) and any(w in t for w in ["my report", "meri report", "in my report", "my scan", "meri scan", "report show", "current level", "assigned", "my", "meri", "mera", "mere", "ਮੇਰੀ", "ਮੇਰਾ", "ਰਿਪੋਰਟ", "मेरी", "मेरा", "रिपोर्ट", "teeth"]) for t in targets)
        ) and not any(any(neg in t for neg in ["not my scan", "not my report", "bina report", "what does level", "what is level", "level 2 kya", "level 3 kya", "level 1 kya", "stage 2?", "stage 1?", "stage 3?"]) for t in targets):
            return self._make_intent_result("staging", 0.96, taxonomy=taxonomy, v1_override="staging")

        # 15. Segmentation Findings
        if has_match([
            "what did your model find", "what did the model find", "what did it detect",
            "what was discovered in the radiograph", "what were the detected findings",
            "explain this x-ray result", "scan me kya mila", "what does my scan show",
            "can you summarize the scan findings", "what is the overall finding",
            "what is my report result", "is my scan showing any suspected caries",
            "what are the basic findings", "did the ai find something", "what does this x-ray result mean",
            "give me the basic overview of this case", "can you read out the main scan findings",
            "scan me kya aaya hai", "scan me kya dikh raha hai", "report kya bol rahi hai",
            "daant me kya nikla report me", "scan ka result kya hai", "xray me kya problem dikhi",
            "report me kya likha hai", "daant ki report me kya nikla", "kya detect hua report me",
            "ਐਕਸਰੇ ਵਿੱਚ ਕੀ ਮਿਲਿਆ", "ਸਕੈਨ ਵਿੱਚ ਕੀ ਆਇਆ", "ਐਕਸਰੇ ਰਿਪੋਰਟ ਵਿੱਚ ਕੀ ਹੈ",
            "एक्स-रे में क्या मिला", "मॉडल को क्या दिखा", "जांच में क्या निकला", "findings kya hai",
            "report findings", "scan findings", "radiolucency", "परिणाम आए", "दिखाई दिया",
            "ਕੀ ਦਿਖਿਆ", "ਕੀ ਨਤੀਜਾ ਆਇਆ", "what did the pixel segmentation reveal", "pixel segmentation reveal",
            "l1 ka matlab", "l1 kya hai yaha", "कुल कितने लीजन पाए", "दांत के निष्कर्ष क्या बताते",
            "ਕੁੱਲ ਕਿੰਨੇ ਖਰਾਬ ਖੇਤਰ", "what is this report about", "what is the report about",
            "report about", "what's this report about", "is report me kya hai", "report me kya hai"
        ]) or any("findings" in t or "finding" in t or "detected" in t or "nikla" in t or "mila" in t or "report about" in t for t in targets):
            return self._make_intent_result("segmentation_findings", 0.96, taxonomy=taxonomy, v1_override="findings")

        # 16. Stage Explanation (General conceptual level definition)
        if has_match([
            "what does level", "what does stage", "explain level", "explain stage", "what about level", "what about stage",
            "ka kya matlab hai", "level 2 mean", "level 1 mean", "level 3 mean",
            "level 2 indicate", "level 1 indicate", "level 3 indicate",
            "level 2 kya", "level 3 kya", "level 1 kya", "level 3 kyu",
            "what is level 2", "what is level 3", "what is level 1",
            "what does moderate caries mean", "what is moderate caries",
            "level 2 ki detail", "level 3 ki detail", "explain level 2 simply",
            "what does stage 2 mean exactly", "stage 1 caries", "stage 2 caries", "stage 3 caries",
            "stage 1 kya", "stage 2 kya", "stage 3 kya", "kya hoti hai", "kya hota hai",
            "not my scan", "what is level 2 in dentistry", "tell me about level", "tell me about stage",
            "लेवल 2 का मतलब", "लेवल 3 क्या होता", "ਲੈਵਲ 2 ਦਾ ਕੀ", "ਲੈਵਲ 3 ਦਾ ਕੀ",
            "level 2 da ki matlab", "stage 2 da ki matlab", "stage 1 da ki matlab",
            "stage 2?", "stage 1?", "stage 3?", "level 2?", "level 1?", "level 3?"
        ]):
            return self._make_intent_result("stage_explanation", 0.96, taxonomy=taxonomy, v1_override="stage_explanation")

        # 17. Model Metrics
        if has_match([
            "dice", "jaccard", "precision", "recall", "benchmark metrics",
            "validation sensitivity", "specificity", "dice score", "validation metrics",
            "e75 metrics", "validation dice", "what about iou"
        ]) or any(re.search(r"\biou\b", t) or any(w in t for w in ["dice", "jaccard", "precision", "recall", "specificity"]) for t in targets):
            return self._make_intent_result("model_metrics", 0.96, taxonomy=taxonomy, v1_override="model_metrics")

        # 18. FPN Architecture / Neural Network
        if has_match([
            "fpn", "fpn architecture", "resnet 34", "resnet-34", "resnet34",
            "feature pyramid network", "lateral connections", "encoder decoder",
            "fpn decoder", "fpn structure", "backbone", "resnet backbone",
            "neural network architecture", "model architecture", "model ki architecture",
            "unet", "u-net"
        ]) or any("fpn" in t or "resnet" in t or "backbone" in t or "unet" in t for t in targets):
            return self._make_intent_result("fpn_architecture", 0.96, taxonomy=taxonomy, v1_override="model_architecture")

        # 19. MLUA Methodology
        if has_match([
            "what is mlua", "mlua methodology", "how does mlua train",
            "consistency regularization", "ema teacher", "teacher student framework",
            "what does mlua stand for", "mean teacher", "semi supervised", "semi-supervised",
            "mlua kya hai", "mlua stand for", "what is caries", "what causes caries", "what causes tooth decay", "how do cavities form"
        ]) or any("mlua" in t for t in targets):
            return self._make_intent_result("mlua_methodology", 0.96, taxonomy=taxonomy, v1_override="mlua_methodology")

        # 20. Confidence Calibration
        if has_match([
            "confidence calibration", "calibration curve", "expected calibration error",
            "temperature scaling", "uncalibrated", "calibrated probability",
            "is the confidence calibrated", "calibration", "reliability diagram", "calibrated"
        ]) or any(re.search(r"\bece\b", t) for t in targets):
            return self._make_intent_result("confidence_calibration", 0.96, taxonomy=taxonomy, v1_override="model_probability")

        # 21. Inference Pipeline
        if has_match([
            "sliding window", "sliding-window", "384x384", "21 patch", "patch reconstruction",
            "inference pipeline", "patch extraction", "how does inference work", "patches"
        ]) or any("sliding window" in t or "patch" in t for t in targets):
            return self._make_intent_result("inference_pipeline", 0.96, taxonomy=taxonomy, v1_override="inference_pipeline")

        # 22. Uncertainty Query
        if has_match([
            "monte carlo dropout", "mc dropout", "epistemic uncertainty", "stochastic passes",
            "why does it use uncertainty", "uncertainty map", "model uncertainty", "uncertainty score",
            "uncertainty explanation", "uncertainty kya hai", "why uncertainty"
        ]) or any("uncertainty" in t for t in targets):
            return self._make_intent_result("uncertainty_query", 0.96, taxonomy=taxonomy, v1_override="uncertainty_explanation")

        # 23. Reset / Limitations
        if has_match([
            "reset conversation", "reset session", "restart chat", "start over",
            "clear chat", "new session", "reset", "clear history", "start fresh", "restart"
        ]) or any(t.strip() in ["reset", "restart", "start over", "clear"] for t in targets):
            return self._make_intent_result("reset", 0.96, taxonomy=taxonomy, v1_override="limitations")

        # 24. Thanks
        if has_match([
            "thanks", "thank you", "thanks bro", "dhanyawad", "shukriya",
            "shukriya bro", "thank you so much", "appreciated", "appreciated thanks",
            "ਧੰਨਵਾਦ", "धन्यवाद", "bohot shukriya", "bht shukriya"
        ]) or any(t.strip() in ["thanks", "thank you", "thanks bro", "dhanyawad", "shukriya", "shukriya bro", "thank you so much", "appreciated", "appreciated thanks", "ਧੰਨਵਾਦ", "धन्यवाद"] for t in targets) or any("thank you" in t or "thanks" in t or "shukriya" in t or "dhanyawad" in t or "ਧੰਨਵਾਦ" in t or "धन्यवाद" in t for t in targets):
            return self._make_intent_result("thanks", 0.96, taxonomy=taxonomy, v1_override="thanks")

        # 25. Feedback / Casual Conversation
        if has_match([
            "feedback", "great job", "good job", "nice answer", "bad answer",
            "terrible answer", "rating", "5 stars", "helpful assistant", "bad assistant",
            "nice work", "super helpful", "very helpful",
            "who are you", "who made you", "what is your name", "are you an ai", "how are you"
        ]):
            return self._make_intent_result("feedback", 0.96, taxonomy=taxonomy, v1_override="casual_conversation")

        # 26. Help
        if has_match([
            "why isn't this working", "why is this not working", "not working",
            "how do i use this application", "how do i use this", "help me use this", "isnt this working",
            "how to use this app", "help please", "मेरी सहायता करें"
        ]):
            return self._make_intent_result("help", 0.95, taxonomy=taxonomy, v1_override="help")

        # 27. Language Preference
        if has_match([
            "hindi mein smjha", "hindi me smjha", "punjabi ch samjha",
            "explain that in hindi", "hindi mein explain karo", "punjabi vich samjhao ji",
            "switch to hindi", "switch to punjabi", "switch to english",
            "sirf hindi", "sirf punjabi", "explain in hindi", "explain in punjabi",
            "sirf punjabi ch", "punjabi vich daso", "hindi me batao", "hindi mein batao",
            "ਸਿਰਫ਼ ਪੰਜਾਬੀ", "ਸਿਰਫ ਪੰਜਾਬੀ", "सिर्फ हिंदी", "सिर्फ हिन्दी",
            "can you speak hindi", "can you speak punjabi", "speak hindi", "speak punjabi",
            "please answer in hindi", "please answer in punjabi", "answer in hindi", "answer in punjabi",
            "translate explanation to hindi", "translate to hindi", "translate to punjabi",
            "हिंदी में बात करें", "हिंदी भाषा में उत्तर", "ਪੰਜਾਬੀ ਚ ਦੱਸੋ", "ਪੰਜਾਬੀ ਵਿਚ ਸਮਝਾ", "ਪੰਜਾਬੀ ਵਿੱਚ ਜਵਾਬ",
            "hinglish me samjha", "hinglish mein"
        ]):
            return self._make_intent_result("language_preference", 0.96, taxonomy=taxonomy, v1_override="language_preference")

        # 28. Greetings
        if any(t.strip() in ["hello", "hi", "hey", "hello bro", "hi bro", "hey bro", "bhai sun", "namaste", "namaste ji", "good morning", "good evening", "sat sri akal", "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ", "नमस्ते"] for t in targets):
            return self._make_intent_result("greeting", 0.95, taxonomy=taxonomy, v1_override="greeting")

        # 29. Goodbye
        if any(t.strip() in ["bye", "goodbye", "see you", "bye bye", "exit", "bye see you", "alvida", "ਅਲਵਿਦਾ", "अलविदा"] for t in targets):
            return self._make_intent_result("goodbye", 0.95, taxonomy=taxonomy, v1_override="goodbye")

        return None
# Singleton instance
intent_classifier = IntentClassifier()
