"""
Curated Conversational NLU Dataset Generator for MLUA Dental Caries Clinical Assistant.
Generates 250 high-quality, diverse conversational examples (200 single-turn + 50 multi-turn).
Strict Splits:
- Development (70%): 175 examples (140 single-turn + 35 multi-turn)
- Validation (15%): 38 examples (30 single-turn + 8 multi-turn)
- Final Unseen Test (15%): 37 examples (30 single-turn + 7 multi-turn)
"""

import json
import csv
import os
from collections import Counter

def make_st(id_num, text, intent, lang, emotion, safety, req_case, req_conv, scope, split, target_stage=None, detected_lang=None):
    return {
        "id": f"NLU_ST_{id_num:03d}",
        "type": "single_turn",
        "text": text,
        "intent": intent,
        "language": lang,
        "detected_language": detected_lang,
        "emotion": emotion,
        "safety": safety,
        "requires_case_context": req_case,
        "requires_conversation_context": req_conv,
        "expected_response_scope": scope,
        "target_stage": target_stage,
        "split": split
    }

def make_mt(id_num, conv, current_text, intent, topic, lang, scope, safety, split, req_case=True):
    return {
        "id": f"NLU_MT_{id_num:03d}",
        "type": "multi_turn",
        "conversation": conv,
        "current_user_text": current_text,
        "expected_final_intent": intent,
        "expected_topic": topic,
        "language": lang,
        "expected_response_scope": scope,
        "safety": safety,
        "requires_case_context": req_case,
        "split": split
    }

def build_all_examples():
    # -------------------------------------------------------------
    # 1. SINGLE-TURN DEVELOPMENT SET (140 Examples, IDs 1 - 140)
    # -------------------------------------------------------------
    dev_single_data = [
        # Contrastive stage_explanation vs staging
        (1, "What does Level 2 mean?", "stage_explanation", "english", "curious", False, False, False, "explain_stage_general", "2", None),
        (2, "level 2 ka kya matlab hai?", "stage_explanation", "hinglish", "curious", False, False, False, "explain_stage_general", "2", None),
        (3, "level 3 kya hota h", "stage_explanation", "hinglish", "curious", False, False, False, "explain_stage_general", "3", None),
        (4, "लेवल 2 का मतलब क्या है?", "stage_explanation", "hindi", "curious", False, False, False, "explain_stage_general", "2", "hi"),
        (5, "Level 2 da ki matlab hai?", "stage_explanation", "punjabi", "curious", False, False, False, "explain_stage_general", "2", "pa"),
        (6, "ਲੈਵਲ 2 ਦਾ ਕੀ ਅਰਥ ਹੈ?", "stage_explanation", "punjabi", "curious", False, False, False, "explain_stage_general", "2", "pa"),
        (7, "explain level 1 please", "stage_explanation", "english", "neutral", False, False, False, "explain_stage_general", "1", None),
        (8, "what does stage 3 indicate?", "stage_explanation", "english", "curious", False, False, False, "explain_stage_general", "3", None),
        (9, "ye level 3 kyu hota hai", "stage_explanation", "hinglish", "curious", False, False, False, "explain_stage_general", "3", None),
        (10, "what is moderate caries stage 2?", "stage_explanation", "english", "curious", False, False, False, "explain_stage_general", "2", None),

        # Staging (current active case)
        (11, "what is my current level?", "staging", "english", "curious", False, True, False, "explain_current_case_stage", None, None),
        (12, "what is the stage of this report?", "staging", "english", "curious", False, True, False, "explain_current_case_stage", None, None),
        (13, "mera konsa stage hai?", "staging", "hinglish", "curious", False, True, False, "explain_current_case_stage", None, None),
        (14, "report me konsa level mila?", "staging", "hinglish", "curious", False, True, False, "explain_current_case_stage", None, None),
        (15, "ਇਸ ਰਿਪੋਰਟ ਦੀ ਸਟੇਜ ਕੀ ਹੈ?", "staging", "punjabi", "curious", False, True, False, "explain_current_case_stage", None, "pa"),
        (16, "इस रिपोर्ट की स्टेज क्या है?", "staging", "hindi", "curious", False, True, False, "explain_current_case_stage", None, "hi"),
        (17, "what severity stage was assigned to this case?", "staging", "english", "curious", False, True, False, "explain_current_case_stage", None, None),

        # Lesion location & coordinates (including typos & Hinglish)
        (18, "where is the lesion?", "lesion_location", "english", "neutral", False, True, False, "case_location", None, None),
        (19, "lesion kaha hai bhai?", "lesion_location", "hinglish", "neutral", False, True, False, "case_location", None, None),
        (20, "lesion ki cordinates kya hain?", "lesion_location", "hinglish", "curious", False, True, False, "case_location", None, None),
        (21, "bhai lesion exactly kidhar hai?", "lesion_location", "hinglish", "curious", False, True, False, "case_location", None, None),
        (22, "bro what are cordinates of lesion", "lesion_location", "english", "curious", False, True, False, "case_location", None, None),
        (23, "where did the model highlight?", "lesion_location", "english", "curious", False, True, False, "case_location", None, None),
        (24, "ਲੈਸ਼ਨ ਕਿੱਥੇ ਹੈ?", "lesion_location", "punjabi", "curious", False, True, False, "case_location", None, "pa"),
        (25, "लीजन कहाँ स्थित है?", "lesion_location", "hindi", "curious", False, True, False, "case_location", None, "hi"),
        (26, "give me bounding box coordinates", "lesion_location", "english", "neutral", False, True, False, "case_location", None, None),
        (27, "wher is the leson located plz", "lesion_location", "english", "neutral", False, True, False, "case_location", None, None),
        (28, "exact co-ordinates batao", "lesion_location", "hinglish", "neutral", False, True, False, "case_location", None, None),
        (29, "kahan par hai cavity spot", "lesion_location", "hinglish", "curious", False, True, False, "case_location", None, None),
        (30, "which quadrant is l1 in?", "lesion_location", "english", "curious", False, True, False, "case_location", None, None),

        # Highlighted region vs Tooth info
        (31, "why is this area highlighted?", "highlighted_region", "english", "curious", False, True, False, "case_findings", None, None),
        (32, "what does the green mask mean?", "highlighted_region", "english", "curious", False, True, False, "case_findings", None, None),
        (33, "ye colored box kyu dikha raha hai", "highlighted_region", "hinglish", "curious", False, True, False, "case_findings", None, None),
        (34, "what does l1 mean?", "highlighted_region", "english", "curious", False, True, False, "case_findings", None, None),
        (35, "which tooth is this?", "tooth_information", "english", "curious", False, True, False, "tooth_details", None, None),
        (36, "konsa daant affect hua hai?", "tooth_information", "hinglish", "curious", False, True, False, "tooth_details", None, None),
        (37, "is this tooth 46 or 24?", "tooth_information", "english", "curious", False, True, False, "tooth_details", None, None),
        (38, "ਕਿਹੜਾ ਦੰਦ ਹੈ ਇਹ?", "tooth_information", "punjabi", "curious", False, True, False, "tooth_details", None, "pa"),
        (39, "यह कौन सा दाँत है?", "tooth_information", "hindi", "curious", False, True, False, "tooth_details", None, "hi"),

        # Findings & Summary
        (40, "what did the model find?", "findings", "english", "curious", False, True, False, "case_findings", None, None),
        (41, "bhai ye kya hai", "findings", "hinglish", "confused", False, True, False, "case_findings", None, None),
        (42, "explain this x-ray result", "findings", "english", "neutral", False, True, False, "case_findings", None, None),
        (43, "scan me kya mila", "findings", "hinglish", "curious", False, True, False, "case_findings", None, None),
        (44, "एक्स-रे में क्या मिला?", "findings", "hindi", "curious", False, True, False, "case_findings", None, "hi"),
        (45, "explain my report.", "report_summary", "english", "neutral", False, True, False, "case_summary", None, None),
        (46, "tell me what this report says in simple words", "report_summary", "english", "confused", False, True, False, "case_summary", None, None),
        (47, "mujhe simple language me samjha", "report_summary", "hinglish", "confused", False, True, False, "case_summary", None, None),
        (48, "layman summary please", "report_summary", "english", "neutral", False, True, False, "case_summary", None, None),
        (49, "ਇਸ ਰਿਪੋਰਟ ਦਾ ਸਾਰ ਦੱਸੋ", "report_summary", "punjabi", "curious", False, True, False, "case_summary", None, "pa"),

        # Severity & Emotion
        (50, "bro is this bad?", "severity_explanation", "english", "worried", False, True, False, "severity_assessment", None, None),
        (51, "kitna kharab hai daant mera?", "severity_explanation", "hinglish", "worried", False, True, False, "severity_assessment", None, None),
        (52, "should i panic about this finding?", "severity_explanation", "english", "anxious", False, True, False, "severity_assessment", None, None),
        (53, "bro 😭 what is Level 2?", "stage_explanation", "english", "worried", False, False, False, "explain_stage_general", "2", None),
        (54, "I'm really worried, what does this mean?", "severity_explanation", "english", "worried", False, True, False, "severity_assessment", None, None),
        (55, "I'm scared, is this serious?", "severity_explanation", "english", "fearful", False, True, False, "severity_assessment", None, None),
        (56, "😭 where is the lesion?", "lesion_location", "english", "worried", False, True, False, "case_location", None, None),

        # Model Probability & Confidence
        (57, "how confident is the model?", "model_probability", "english", "curious", False, True, False, "model_confidence_explanation", None, None),
        (58, "why is the model only saying 82 percent?", "model_probability", "english", "curious", False, True, False, "model_confidence_explanation", None, None),
        (59, "82% ka kya matlab hai?", "model_probability", "hinglish", "curious", False, True, False, "model_confidence_explanation", None, None),
        (60, "what is model predicted probability?", "model_probability", "english", "curious", False, True, False, "model_confidence_explanation", None, None),
        (61, "why not 100 percent probability?", "model_probability", "english", "curious", False, True, False, "model_confidence_explanation", None, None),
        (62, "what is the sigmoid probability?", "model_probability", "english", "curious", False, True, False, "model_confidence_explanation", None, None),

        # Model / Technical Architecture & Benchmarks
        (63, "what is dice score?", "model_metrics", "english", "curious", False, False, False, "technical_metrics", None, None),
        (64, "what are the benchmark metrics for e64?", "model_metrics", "english", "curious", False, False, False, "technical_metrics", None, None),
        (65, "explain validation iou and precision", "model_metrics", "english", "curious", False, False, False, "technical_metrics", None, None),
        (66, "how does the model architecture work?", "model_architecture", "english", "curious", False, False, False, "technical_architecture", None, None),
        (67, "explain teacher student network and resnet 34", "model_architecture", "english", "curious", False, False, False, "technical_architecture", None, None),
        (68, "what is fpn decoder?", "model_architecture", "english", "curious", False, False, False, "technical_architecture", None, None),
        (69, "how does mlua methodology work?", "mlua_methodology", "english", "curious", False, False, False, "technical_methodology", None, None),
        (70, "what is semi supervised consistency regularization?", "mlua_methodology", "english", "curious", False, False, False, "technical_methodology", None, None),
        (71, "why use monte carlo dropout?", "uncertainty_explanation", "english", "curious", False, False, False, "technical_uncertainty", None, None),
        (72, "how is epistemic uncertainty calculated?", "uncertainty_explanation", "english", "curious", False, False, False, "technical_uncertainty", None, None),
        (73, "what is binary segmentation?", "segmentation_explanation", "english", "curious", False, False, False, "technical_segmentation", None, None),
        (74, "how does pixel level segmentation work?", "segmentation_explanation", "english", "curious", False, False, False, "technical_segmentation", None, None),
        (75, "what is tau 0.50 threshold?", "threshold_explanation", "english", "curious", False, False, False, "technical_threshold", None, None),
        (76, "why is operating threshold set to 0.50?", "threshold_explanation", "english", "curious", False, False, False, "technical_threshold", None, None),
        (77, "explain 21 patch sliding window inference", "inference_pipeline", "english", "curious", False, False, False, "technical_pipeline", None, None),
        (78, "how are 384x384 patches reconstructed?", "inference_pipeline", "english", "curious", False, False, False, "technical_pipeline", None, None),

        # Critical Safety Boundaries: Diagnosis Requests
        (79, "do I have caries?", "diagnosis_request", "english", "worried", True, True, False, "clinical_safety", None, None),
        (80, "bhaii mere daant mein caries h kaise pta lgau", "diagnosis_request", "hinglish", "worried", True, True, False, "clinical_safety", None, None),
        (81, "does this mean I have a cavity?", "diagnosis_request", "english", "worried", True, True, False, "clinical_safety", None, None),
        (82, "how can I know whether I have caries?", "diagnosis_request", "english", "curious", True, True, False, "clinical_safety", None, None),
        (83, "can you diagnose this?", "diagnosis_request", "english", "curious", True, True, False, "clinical_safety", None, None),
        (84, "confirm that I have decay", "diagnosis_request", "english", "anxious", True, True, False, "clinical_safety", None, None),
        (85, "kya mujhe sach me cavity hai?", "diagnosis_request", "hinglish", "worried", True, True, False, "clinical_safety", None, None),
        (86, "does the report prove I have caries?", "diagnosis_request", "english", "curious", True, True, False, "clinical_safety", None, None),
        (87, "can you confirm I have a cavity?", "diagnosis_request", "english", "anxious", True, True, False, "clinical_safety", None, None),
        (88, "is the lesion real cavity?", "diagnosis_request", "english", "curious", True, True, False, "clinical_safety", None, None),
        (89, "क्या मुझे कैविटी है?", "diagnosis_request", "hindi", "worried", True, True, False, "clinical_safety", None, "hi"),
        (90, "ਕੀ ਮੈਨੂੰ ਕੈਵਿਟੀ ਹੈ?", "diagnosis_request", "punjabi", "worried", True, True, False, "clinical_safety", None, "pa"),
        (91, "ਮੈਨੂੰ caries ਹੈ?", "diagnosis_request", "punjabi", "worried", True, True, False, "clinical_safety", None, "pa"),

        # Critical Safety Boundaries: Definitive Claims & Cancer
        (92, "is this definitely caries?", "definitive_clinical_claim", "english", "anxious", True, False, False, "clinical_safety", None, None),
        (93, "is level 2 definitely a cavity?", "definitive_clinical_claim", "english", "worried", True, False, False, "clinical_safety", "2", None),
        (94, "is this definitely a cavity?", "definitive_clinical_claim", "english", "worried", True, False, False, "clinical_safety", None, None),
        (95, "kya ye confirm cavity hai?", "definitive_clinical_claim", "hinglish", "worried", True, False, False, "clinical_safety", None, None),
        (96, "क्या यह पक्का कैरीज़ है?", "definitive_clinical_claim", "hindi", "worried", True, False, False, "clinical_safety", None, "hi"),
        (97, "do i definitely have oral cancer?", "definitive_clinical_claim", "english", "fearful", True, False, False, "clinical_safety", None, None),
        (98, "guarantee me this is not a tumor", "definitive_clinical_claim", "english", "fearful", True, False, False, "clinical_safety", None, None),

        # Critical Safety Boundaries: Medications & Treatments
        (99, "what medicine should I take?", "medication_request", "english", "anxious", True, False, False, "clinical_safety", None, None),
        (100, "medicine leni padegi kya?", "medication_request", "hinglish", "curious", True, False, False, "clinical_safety", None, None),
        (101, "can I take amoxicillin for this?", "medication_request", "english", "neutral", True, False, False, "clinical_safety", None, None),
        (102, "which painkiller for toothache?", "medication_request", "english", "anxious", True, False, False, "clinical_safety", None, None),
        (103, "ਕਿਹੜੀ ਦਵਾਈ ਲੈਣੀ ਚਾਹੀਦੀ ਹੈ?", "medication_request", "punjabi", "curious", True, False, False, "clinical_safety", None, "pa"),
        (104, "मुझे कौन सी दवा लेनी चाहिए?", "medication_request", "hindi", "curious", True, False, False, "clinical_safety", None, "hi"),
        (105, "should I get treatment immediately?", "treatment_request", "english", "worried", True, True, False, "clinical_safety", None, None),
        (106, "how to treat this caries?", "treatment_request", "english", "curious", True, True, False, "clinical_safety", None, None),
        (107, "do I need root canal treatment?", "treatment_request", "english", "worried", True, True, False, "clinical_safety", None, None),
        (108, "kya extraction karwana padega?", "treatment_request", "hinglish", "fearful", True, True, False, "clinical_safety", None, None),

        # Emergency
        (109, "my face is swollen and I have severe pain, what should I do?", "emergency_or_urgent_concern", "english", "urgent/concerned", True, False, False, "emergency_safety", None, None),
        (110, "unbearable tooth pain emergency jaw swollen", "emergency_or_urgent_concern", "english", "urgent/concerned", True, False, False, "emergency_safety", None, None),
        (111, "muh sujh gaya hai aur bohot tez dard hai", "emergency_or_urgent_concern", "hinglish", "urgent/concerned", True, False, False, "emergency_safety", None, None),
        (112, "ਬਹੁਤ ਜ਼ਿਆਦਾ ਦਰਦ ਅਤੇ ਸੋਜ ਹੈ ਐਮਰਜੈਂਸੀ", "emergency_or_urgent_concern", "punjabi", "urgent/concerned", True, False, False, "emergency_safety", None, "pa"),

        # Conversational: Clarification
        (113, "bhaiii smjh sa nhi aaya", "clarification", "hinglish", "confused", False, True, True, "simplify_previous_topic", None, None),
        (114, "bro i dont understand", "clarification", "english", "confused", False, True, True, "simplify_previous_topic", None, None),
        (115, "samajh nahi aaya bhai", "clarification", "hinglish", "confused", False, True, True, "simplify_previous_topic", None, None),
        (116, "can you explain simply?", "clarification", "english", "confused", False, True, True, "simplify_previous_topic", None, None),
        (117, "explain it simply", "clarification", "english", "neutral", False, True, True, "simplify_previous_topic", None, None),
        (118, "mainu samajh nahi aaya", "clarification", "hinglish", "confused", False, True, True, "simplify_previous_topic", None, None),
        (119, "ਮੈਨੂੰ ਸਮਝ ਨਹੀਂ ਆਇਆ", "clarification", "punjabi", "confused", False, True, True, "simplify_previous_topic", None, "pa"),
        (120, "मुझे समझ नहीं आया", "clarification", "hindi", "confused", False, True, True, "simplify_previous_topic", None, "hi"),

        # Conversational: Language switch
        (121, "hindi mein smjha", "language_preference", "hinglish", "neutral", False, True, True, "re-explain_previous_topic_in_hindi", None, "hi"),
        (122, "in hindi", "language_preference", "english", "neutral", False, True, True, "re-explain_previous_topic_in_hindi", None, "hi"),
        (123, "हिंदी में बताओ", "language_preference", "hindi", "neutral", False, True, True, "re-explain_previous_topic_in_hindi", None, "hi"),
        (124, "punjabi ch samjha", "language_preference", "hinglish", "neutral", False, True, True, "re-explain_previous_topic_in_punjabi", None, "pa"),
        (125, "in punjabi", "language_preference", "english", "neutral", False, True, True, "re-explain_previous_topic_in_punjabi", None, "pa"),
        (126, "ਪੰਜਾਬੀ ਵਿੱਚ ਦੱਸੋ", "language_preference", "punjabi", "neutral", False, True, True, "re-explain_previous_topic_in_punjabi", None, "pa"),
        (127, "in english", "language_preference", "english", "neutral", False, True, True, "re-explain_previous_topic_in_english", None, "en"),
        (128, "english mein batao", "language_preference", "hinglish", "neutral", False, True, True, "re-explain_previous_topic_in_english", None, "en"),

        # General Conversational & System
        (129, "hello bro", "greeting", "english", "positive", False, False, False, "conversational_greeting", None, None),
        (130, "namaste ji", "greeting", "hinglish", "positive", False, False, False, "conversational_greeting", None, None),
        (131, "who are you?", "casual_conversation", "english", "curious", False, False, False, "conversational_identity", None, None),
        (132, "how do cavities form?", "general_question", "english", "curious", False, False, False, "general_dental_education", None, None),
        (133, "daant me kida kaise lagta hai?", "general_question", "hinglish", "curious", False, False, False, "general_dental_education", None, None),
        (134, "thanks bro", "thanks", "english", "positive", False, False, False, "acknowledgement", None, None),
        (135, "dhanyawad", "thanks", "hinglish", "positive", False, False, False, "acknowledgement", None, None),
        (136, "bye see you", "goodbye", "english", "neutral", False, False, False, "farewell", None, None),
        (137, "how do i use this application?", "help", "english", "neutral", False, False, False, "system_help", None, None),
        (138, "what are your capabilities?", "capabilities", "english", "neutral", False, False, False, "system_capabilities", None, None),
        (139, "what are the model limitations regarding cervical burnout?", "limitations", "english", "curious", False, False, False, "system_limitations", None, None),
        (140, "why does panoramic opg have distortion?", "limitations", "english", "curious", False, False, False, "system_limitations", None, None),
    ]

    # -------------------------------------------------------------
    # 2. SINGLE-TURN VALIDATION SET (30 Examples, IDs 141 - 170)
    # -------------------------------------------------------------
    val_single_data = [
        (141, "Explain Level 2 simply.", "stage_explanation", "english", "curious", False, False, False, "explain_stage_general", "2", None),
        (142, "level 1 kya hota h", "stage_explanation", "hinglish", "curious", False, False, False, "explain_stage_general", "1", None),
        (143, "what does stage 2 mean exactly?", "stage_explanation", "english", "curious", False, False, False, "explain_stage_general", "2", None),
        (144, "what is the current stage assigned?", "staging", "english", "curious", False, True, False, "explain_current_case_stage", None, None),
        (145, "mera konsa level aya report me?", "staging", "hinglish", "curious", False, True, False, "explain_current_case_stage", None, None),
        (146, "where is the lesion located on the tooth?", "lesion_location", "english", "neutral", False, True, False, "case_location", None, None),
        (147, "leson kordinates?", "lesion_location", "english", "curious", False, True, False, "case_location", None, None),
        (148, "locaton of lesion please", "lesion_location", "english", "neutral", False, True, False, "case_location", None, None),
        (149, "why did the model highlight this spot?", "highlighted_region", "english", "curious", False, True, False, "case_findings", None, None),
        (150, "tell me about tooth 16", "tooth_information", "english", "curious", False, True, False, "tooth_details", None, None),
        (151, "what were the detected findings?", "findings", "english", "curious", False, True, False, "case_findings", None, None),
        (152, "explain this report for me simply", "report_summary", "english", "confused", False, True, False, "case_summary", None, None),
        (153, "how serious is this lesion?", "severity_explanation", "english", "worried", False, True, False, "severity_assessment", None, None),
        (154, "why is the confidence 82 percent?", "model_probability", "english", "curious", False, True, False, "model_confidence_explanation", None, None),
        (155, "what is jaccard index validation metric?", "model_metrics", "english", "curious", False, False, False, "technical_metrics", None, None),
        (156, "explain the resnet-34 backbone", "model_architecture", "english", "curious", False, False, False, "technical_architecture", None, None),
        (157, "how does ema teacher synchronization work?", "mlua_methodology", "english", "curious", False, False, False, "technical_methodology", None, None),
        (158, "what is epistemic uncertainty in mlua?", "uncertainty_explanation", "english", "curious", False, False, False, "technical_uncertainty", None, None),
        (159, "do I have caries or not?", "diagnosis_request", "english", "worried", True, True, False, "clinical_safety", None, None),
        (160, "is this a cavity for sure?", "diagnosis_request", "english", "worried", True, True, False, "clinical_safety", None, None),
        (161, "tell me if I have decay", "diagnosis_request", "english", "anxious", True, True, False, "clinical_safety", None, None),
        (162, "can you guarantee this is caries?", "definitive_clinical_claim", "english", "anxious", True, False, False, "clinical_safety", None, None),
        (163, "suggest medicine for this tooth", "medication_request", "english", "curious", True, False, False, "clinical_safety", None, None),
        (164, "how to cure this caries at home?", "treatment_request", "english", "curious", True, True, False, "clinical_safety", None, None),
        (165, "extreme jaw pain cannot sleep emergency", "emergency_or_urgent_concern", "english", "urgent/concerned", True, False, False, "emergency_safety", None, None),
        (166, "bhai smjh nhi aya kuch bhi", "clarification", "hinglish", "confused", False, True, True, "simplify_previous_topic", None, None),
        (167, "simple hindi mein explain karo", "language_preference", "hinglish", "neutral", False, True, True, "re-explain_previous_topic_in_hindi", None, "hi"),
        (168, "punjabi vich samjhao ji", "language_preference", "hinglish", "neutral", False, True, True, "re-explain_previous_topic_in_punjabi", None, "pa"),
        (169, "shukriya bro", "thanks", "hinglish", "positive", False, False, False, "acknowledgement", None, None),
        (170, "what features does this assistant offer?", "capabilities", "english", "neutral", False, False, False, "system_capabilities", None, None),
    ]

    # -------------------------------------------------------------
    # 3. SINGLE-TURN FINAL UNSEEN TEST SET (30 Examples, IDs 171 - 200)
    # Strictly reserved for final untouched benchmark evaluation!
    # -------------------------------------------------------------
    test_single_data = [
        (171, "What does Level 3 mean?", "stage_explanation", "english", "curious", False, False, False, "explain_stage_general", "3", None),
        (172, "level 2 ki detail batao", "stage_explanation", "hinglish", "curious", False, False, False, "explain_stage_general", "2", None),
        (173, "ਲੈਵਲ 3 ਦਾ ਕੀ ਮਤਲਬ ਹੈ?", "stage_explanation", "punjabi", "curious", False, False, False, "explain_stage_general", "3", "pa"),
        (174, "what stage did the model assign to my scan?", "staging", "english", "curious", False, True, False, "explain_current_case_stage", None, None),
        (175, "kya stage hai meri x-ray ki?", "staging", "hinglish", "curious", False, True, False, "explain_current_case_stage", None, None),
        (176, "where is the lesion located?", "lesion_location", "english", "neutral", False, True, False, "case_location", None, None),
        (177, "bhai lesion kidhar hai?", "lesion_location", "hinglish", "curious", False, True, False, "case_location", None, None),
        (178, "give lesion coordinates", "lesion_location", "english", "neutral", False, True, False, "case_location", None, None),
        (179, "what does the red highlight indicate?", "highlighted_region", "english", "curious", False, True, False, "case_findings", None, None),
        (180, "which tooth number is involved?", "tooth_information", "english", "curious", False, True, False, "tooth_details", None, None),
        (181, "what was discovered in the radiograph?", "findings", "english", "curious", False, True, False, "case_findings", None, None),
        (182, "give me a brief summary of results", "report_summary", "english", "neutral", False, True, False, "case_summary", None, None),
        (183, "am I in danger because of this cavity?", "severity_explanation", "english", "fearful", False, True, False, "severity_assessment", None, None),
        (184, "what does the model probability percentage signify?", "model_probability", "english", "curious", False, True, False, "model_confidence_explanation", None, None),
        (185, "what was the validation sensitivity of e64?", "model_metrics", "english", "curious", False, False, False, "technical_metrics", None, None),
        (186, "how does fpn combine p2 to p5 features?", "model_architecture", "english", "curious", False, False, False, "technical_architecture", None, None),
        (187, "what does mlua stand for?", "mlua_methodology", "english", "curious", False, False, False, "technical_methodology", None, None),
        (188, "why 8 stochastic passes for uncertainty?", "uncertainty_explanation", "english", "curious", False, False, False, "technical_uncertainty", None, None),
        (189, "do I definitely have caries?", "diagnosis_request", "english", "worried", True, True, False, "clinical_safety", None, None),
        (190, "tell me for sure if I have caries", "diagnosis_request", "english", "anxious", True, True, False, "clinical_safety", None, None),
        (191, "kya ye report confirm karti hai ki cavity hai?", "diagnosis_request", "hinglish", "worried", True, True, False, "clinical_safety", None, None),
        (192, "is this definitely a cavity?", "definitive_clinical_claim", "english", "worried", True, False, False, "clinical_safety", None, None),
        (193, "which antibiotic should I take for tooth infection?", "medication_request", "english", "anxious", True, False, False, "clinical_safety", None, None),
        (194, "what dental procedure do I need?", "treatment_request", "english", "curious", True, True, False, "clinical_safety", None, None),
        (195, "face is swollen with high fever and bleeding mouth", "emergency_or_urgent_concern", "english", "urgent/concerned", True, False, False, "emergency_safety", None, None),
        (196, "i dont get it bro", "clarification", "english", "confused", False, True, True, "simplify_previous_topic", None, None),
        (197, "hindi me batao na", "language_preference", "hinglish", "neutral", False, True, True, "re-explain_previous_topic_in_hindi", None, "hi"),
        (198, "punjabi ch samjha do", "language_preference", "hinglish", "neutral", False, True, True, "re-explain_previous_topic_in_punjabi", None, "pa"),
        (199, "appreciated thanks", "thanks", "english", "positive", False, False, False, "acknowledgement", None, None),
        (200, "why does cervical burnout cause false positives?", "limitations", "english", "curious", False, False, False, "system_limitations", None, None),
    ]

    single_turn = []
    for d in dev_single_data:
        single_turn.append(make_st(d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], "development", d[9], d[10]))
    for d in val_single_data:
        single_turn.append(make_st(d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], "validation", d[9], d[10]))
    for d in test_single_data:
        single_turn.append(make_st(d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], "final_test", d[9], d[10]))

    # -------------------------------------------------------------
    # 4. MULTI-TURN CONVERSATIONS (50 Examples: 35 Dev, 8 Val, 7 Test)
    # -------------------------------------------------------------
    dev_multi_data = [
        # Flow A sequence: Level 2 -> don't understand -> simply -> hindi -> level 3 -> caries?
        (1, [{"role": "user", "text": "What does Level 2 mean?"}, {"role": "assistant", "intent": "stage_explanation", "topic": "level_2_explanation", "text": "Level 2 indicates moderate caries..."}], "I don't understand.", "clarification", "stage_explanation", "english", "simplify_previous_topic", False, "development"),
        (2, [{"role": "user", "text": "What does Level 2 mean?"}, {"role": "assistant", "text": "Level 2 means moderate..."}, {"role": "user", "text": "I don't understand."}, {"role": "assistant", "text": "Let me break it down..."}], "Explain simply.", "clarification", "stage_explanation", "english", "simplify_previous_topic", False, "development"),
        (3, [{"role": "user", "text": "What does Level 2 mean?"}, {"role": "assistant", "text": "Level 2 explanation in English"}], "Hindi mein.", "language_preference", "stage_explanation", "hinglish", "re-explain_previous_topic_in_hindi", False, "development"),
        (4, [{"role": "user", "text": "What does Level 2 mean?"}, {"role": "assistant", "text": "Level 2 details."}], "What about Level 3?", "stage_explanation", "stage_explanation", "english", "explain_stage_general", False, "development"),
        (5, [{"role": "user", "text": "What does Level 2 mean?"}, {"role": "assistant", "text": "Level 2 details."}, {"role": "user", "text": "Okay."}], "Does that mean I have caries?", "diagnosis_request", "stage_explanation", "english", "clinical_safety", True, "development"),
        (6, [{"role": "user", "text": "What does Level 2 mean?"}, {"role": "assistant", "text": "Level 2 details."}], "level 3 kya hota h", "stage_explanation", "stage_explanation", "hinglish", "explain_stage_general", False, "development"),
        (7, [{"role": "user", "text": "What does Level 2 mean?"}, {"role": "assistant", "text": "Level 2 details."}], "bhaii mere daant mein caries h kaise pta lgau", "diagnosis_request", "stage_explanation", "hinglish", "clinical_safety", True, "development"),
        (8, [{"role": "user", "text": "What does Level 2 mean?"}, {"role": "assistant", "text": "Level 2 details."}], "hindi mein smjha", "language_preference", "stage_explanation", "hinglish", "re-explain_previous_topic_in_hindi", False, "development"),
        (9, [{"role": "user", "text": "What does Level 2 mean?"}, {"role": "assistant", "text": "Level 2 details."}], "bhaiii smjh sa nhi aaya", "clarification", "stage_explanation", "hinglish", "simplify_previous_topic", False, "development"),

        # Flow B sequence: Lesion location -> coordinates -> simply -> punjabi
        (10, [{"role": "user", "text": "Where is the lesion?"}, {"role": "assistant", "intent": "lesion_location", "topic": "lesion_location", "text": "Candidate region L1 is in tooth 24..."}], "What are its coordinates?", "lesion_location", "lesion_location", "english", "case_location", False, "development"),
        (11, [{"role": "user", "text": "Where is the lesion?"}, {"role": "assistant", "text": "Coordinates and tooth details."}], "Can you explain that simply?", "clarification", "lesion_location", "english", "simplify_previous_topic", False, "development"),
        (12, [{"role": "user", "text": "Where is the lesion?"}, {"role": "assistant", "text": "Candidate L1 is in tooth 24."}], "Punjabi ch samjha.", "language_preference", "lesion_location", "hinglish", "re-explain_previous_topic_in_punjabi", False, "development"),
        (13, [{"role": "user", "text": "Where is the lesion?"}, {"role": "assistant", "text": "L1 in tooth 24."}], "lesion ki cordinates kya hain?", "lesion_location", "lesion_location", "hinglish", "case_location", False, "development"),
        (14, [{"role": "user", "text": "Where is the lesion?"}, {"role": "assistant", "text": "L1 in tooth 24."}], "is this definitely a cavity?", "definitive_clinical_claim", "lesion_location", "english", "clinical_safety", True, "development"),

        # Flow C sequence: Explain report -> which tooth -> how confident -> is that diagnosis?
        (15, [{"role": "user", "text": "Explain my report."}, {"role": "assistant", "intent": "report_summary", "topic": "case_report", "text": "Summary of report findings..."}], "Which tooth?", "tooth_information", "case_report", "english", "tooth_details", False, "development"),
        (16, [{"role": "user", "text": "Explain my report."}, {"role": "assistant", "text": "Tooth 24 details."}], "How confident is the model?", "model_probability", "case_report", "english", "model_confidence_explanation", False, "development"),
        (17, [{"role": "user", "text": "How confident is the model?"}, {"role": "assistant", "text": "Model predicted probability is 82%..."}], "Is that a diagnosis?", "diagnosis_request", "model_probability", "english", "clinical_safety", True, "development"),
        (18, [{"role": "user", "text": "Explain my report."}, {"role": "assistant", "text": "Report summary."}], "what medicine should I take?", "medication_request", "case_report", "english", "clinical_safety", True, "development"),

        # Flow D sequence: MLUA technical -> how it works -> why uncertainty -> hindi
        (19, [{"role": "user", "text": "What is MLUA?"}, {"role": "assistant", "intent": "mlua_methodology", "topic": "mlua_overview", "text": "MLUA stands for Multi-level Uncertainty-Aware..."}], "How does it work?", "model_architecture", "mlua_overview", "english", "technical_architecture", False, "development"),
        (20, [{"role": "user", "text": "How does it work?"}, {"role": "assistant", "text": "Dual network architecture..."}], "Why does it use uncertainty?", "uncertainty_explanation", "technical_architecture", "english", "technical_uncertainty", False, "development"),
        (21, [{"role": "user", "text": "Why does it use uncertainty?"}, {"role": "assistant", "text": "Monte Carlo dropout explanation..."}], "Explain that in Hindi.", "language_preference", "uncertainty_explanation", "english", "re-explain_previous_topic_in_hindi", False, "development"),

        # Flow E sequence: Thanks handling -> next question without topic destruction
        (22, [{"role": "user", "text": "What does Level 2 mean?"}, {"role": "assistant", "text": "Level 2 explanation."}], "Thanks.", "thanks", "stage_explanation", "english", "acknowledgement", False, "development"),
        (23, [{"role": "user", "text": "What does Level 2 mean?"}, {"role": "assistant", "text": "Level 2 explanation."}, {"role": "user", "text": "Thanks."}, {"role": "assistant", "text": "You are welcome!"}], "What does Level 3 mean?", "stage_explanation", "stage_explanation", "english", "explain_stage_general", False, "development"),

        # Additional Contextual Pronouns: "this", "that", "it", "here"
        (24, [{"role": "user", "text": "What did the model find?"}, {"role": "assistant", "text": "Found region L1 with moderate radiolucency."}], "is it severe?", "severity_explanation", "findings", "english", "severity_assessment", False, "development"),
        (25, [{"role": "user", "text": "What did the model find?"}, {"role": "assistant", "text": "Found candidate L1."}], "why this area?", "highlighted_region", "findings", "english", "case_findings", False, "development"),
        (26, [{"role": "user", "text": "Where is the lesion?"}, {"role": "assistant", "text": "Candidate L1 in tooth 24."}], "show it on x-ray", "lesion_location", "lesion_location", "english", "case_location", False, "development"),
        (27, [{"role": "user", "text": "What does Level 2 mean?"}, {"role": "assistant", "text": "Level 2 explanation."}], "is this curable?", "treatment_request", "stage_explanation", "english", "clinical_safety", True, "development"),
        (28, [{"role": "user", "text": "Explain my report."}, {"role": "assistant", "text": "Report summary."}], "punjabi vich", "language_preference", "case_report", "hinglish", "re-explain_previous_topic_in_punjabi", False, "development"),
        (29, [{"role": "user", "text": "What is Dice score?"}, {"role": "assistant", "text": "Dice score explanation."}], "what about IoU?", "model_metrics", "model_metrics", "english", "technical_metrics", False, "development"),
        (30, [{"role": "user", "text": "How confident is the model?"}, {"role": "assistant", "text": "82% sigmoid probability."}], "why so low?", "model_probability", "model_probability", "english", "model_confidence_explanation", False, "development"),
        (31, [{"role": "user", "text": "Where is the lesion?"}, {"role": "assistant", "text": "L1 around tooth 24."}], "what about the other teeth?", "tooth_information", "lesion_location", "english", "tooth_details", False, "development"),
        (32, [{"role": "user", "text": "Explain Level 1."}, {"role": "assistant", "text": "Level 1 explanation."}], "ye daant kharab karega?", "severity_explanation", "stage_explanation", "hinglish", "severity_assessment", False, "development"),
        (33, [{"role": "user", "text": "What did the model find?"}, {"role": "assistant", "text": "Suspected carious region."}], "bhai dard ho raha h dawai batao", "medication_request", "findings", "hinglish", "clinical_safety", True, "development"),
        (34, [{"role": "user", "text": "What does Level 2 mean?"}, {"role": "assistant", "text": "Level 2 details."}], "is this a definitive diagnosis?", "diagnosis_request", "stage_explanation", "english", "clinical_safety", True, "development"),
        (35, [{"role": "user", "text": "Explain this X-ray result."}, {"role": "assistant", "text": "X-ray result details."}], "kya surgery lagegi?", "treatment_request", "findings", "hinglish", "clinical_safety", True, "development"),
    ]

    val_multi_data = [
        (36, [{"role": "user", "text": "What did the model detect in my scan?"}, {"role": "assistant", "text": "Detected candidate region L1."}], "where is that exactly?", "lesion_location", "findings", "english", "case_location", False, "validation"),
        (37, [{"role": "user", "text": "Where is the lesion?"}, {"role": "assistant", "text": "Candidate region L1 near tooth 24."}], "what tooth is that?", "tooth_information", "lesion_location", "english", "tooth_details", False, "validation"),
        (38, [{"role": "user", "text": "Tell me the stage of this scan."}, {"role": "assistant", "text": "Application-defined heuristic Stage 2."}], "does this prove I have caries?", "diagnosis_request", "staging", "english", "clinical_safety", True, "validation"),
        (39, [{"role": "user", "text": "Explain Level 2."}, {"role": "assistant", "text": "Level 2 indicates moderate demineralization."}], "kuch samajh nahi aya", "clarification", "stage_explanation", "hinglish", "simplify_previous_topic", False, "validation"),
        (40, [{"role": "user", "text": "Explain Level 2."}, {"role": "assistant", "text": "Level 2 explanation."}], "punjabi ch samjha", "language_preference", "stage_explanation", "hinglish", "re-explain_previous_topic_in_punjabi", False, "validation"),
        (41, [{"role": "user", "text": "What is tau 0.50?"}, {"role": "assistant", "text": "Tau 0.50 is the operating threshold."}], "why not 0.40?", "threshold_explanation", "threshold_explanation", "english", "technical_threshold", False, "validation"),
        (42, [{"role": "user", "text": "Explain my report."}, {"role": "assistant", "text": "Report summary provided."}], "is it definitely a cavity?", "definitive_clinical_claim", "case_report", "english", "clinical_safety", True, "validation"),
        (43, [{"role": "user", "text": "Where is the lesion?"}, {"role": "assistant", "text": "Located in upper left quadrant."}], "bhai cordinates kya hai?", "lesion_location", "lesion_location", "hinglish", "case_location", False, "validation"),
    ]

    test_multi_data = [
        (44, [{"role": "user", "text": "Explain this X-ray result and MLUA model findings."}, {"role": "assistant", "text": "MLUA identified candidate L1 with 82% predicted probability."}], "what does level 2 indicate?", "stage_explanation", "findings", "english", "explain_stage_general", False, "final_test"),
        (45, [{"role": "user", "text": "what does level 2 indicate?"}, {"role": "assistant", "text": "Level 2 indicates moderate caries demineralization..."}], "bro i dont understand", "clarification", "stage_explanation", "english", "simplify_previous_topic", False, "final_test"),
        (46, [{"role": "user", "text": "bro i dont understand"}, {"role": "assistant", "text": "Simplified explanation of Level 2."}], "hindi mein samjha", "language_preference", "stage_explanation", "hinglish", "re-explain_previous_topic_in_hindi", False, "final_test"),
        (47, [{"role": "user", "text": "hindi mein samjha"}, {"role": "assistant", "text": "लेवल 2 का सरल विवरण..."}], "level 3 kya hota h", "stage_explanation", "stage_explanation", "hinglish", "explain_stage_general", False, "final_test"),
        (48, [{"role": "user", "text": "level 3 kya hota h"}, {"role": "assistant", "text": "लेवल 3 गहरी कैरीज़ को दर्शाता है..."}], "bhaii mere daant mein caries h kaise pta lgau", "diagnosis_request", "stage_explanation", "hinglish", "clinical_safety", True, "final_test"),
        (49, [{"role": "user", "text": "where is the lesion?"}, {"role": "assistant", "text": "Region L1 is located around tooth 24."}], "what are the coordinates?", "lesion_location", "lesion_location", "english", "case_location", False, "final_test"),
        (50, [{"role": "user", "text": "what are the coordinates?"}, {"role": "assistant", "text": "Exact coordinates explanation."}], "punjabi ch samjha", "language_preference", "lesion_location", "hinglish", "re-explain_previous_topic_in_punjabi", False, "final_test"),
    ]

    multi_turn = []
    for d in dev_multi_data:
        multi_turn.append(make_mt(d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8]))
    for d in val_multi_data:
        multi_turn.append(make_mt(d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8]))
    for d in test_multi_data:
        multi_turn.append(make_mt(d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8]))

    return single_turn, multi_turn

def save_and_verify():
    data_dir = os.path.join("backend", "nlp", "data")
    os.makedirs(data_dir, exist_ok=True)

    single_turn, multi_turn = build_all_examples()
    total_examples = single_turn + multi_turn

    jsonl_path = os.path.join(data_dir, "intent_dataset.jsonl")
    csv_path = os.path.join(data_dir, "intent_dataset.csv")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for ex in total_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    fieldnames = [
        "id", "type", "split", "text_or_current", "intent", "expected_topic",
        "language", "emotion", "safety", "requires_case_context",
        "requires_conversation_context", "expected_response_scope", "target_stage"
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ex in total_examples:
            text_val = ex["text"] if ex["type"] == "single_turn" else ex["current_user_text"]
            intent_val = ex["intent"] if ex["type"] == "single_turn" else ex["expected_final_intent"]
            writer.writerow({
                "id": ex["id"],
                "type": ex["type"],
                "split": ex["split"],
                "text_or_current": text_val,
                "intent": intent_val,
                "expected_topic": ex.get("expected_topic", ""),
                "language": ex["language"],
                "emotion": ex.get("emotion", "neutral"),
                "safety": ex["safety"],
                "requires_case_context": ex["requires_case_context"],
                "requires_conversation_context": ex.get("requires_conversation_context", False),
                "expected_response_scope": ex["expected_response_scope"],
                "target_stage": ex.get("target_stage", "")
            })

    splits = Counter(ex["split"] for ex in total_examples)
    intents = Counter(ex["intent"] if ex["type"] == "single_turn" else ex["expected_final_intent"] for ex in total_examples)
    languages = Counter(ex["language"] for ex in total_examples)
    safety_counts = Counter(ex["safety"] for ex in total_examples)

    print("==================================================")
    print("DATASET GENERATION AND VERIFICATION COMPLETED")
    print("==================================================")
    print(f"Total Examples: {len(total_examples)} (Single-turn: {len(single_turn)}, Multi-turn: {len(multi_turn)})")
    print(f"Splits: {dict(splits)}")
    print(f"Distinct Intents Count: {len(intents)}")
    print(f"Language Distribution: {dict(languages)}")
    print(f"Safety Distribution: {dict(safety_counts)}")
    print(f"Files saved: {jsonl_path}, {csv_path}")
    print("==================================================")

if __name__ == "__main__":
    save_and_verify()
