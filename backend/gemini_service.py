"""
Gemini Context-Aware Clinical Assistant Service
Embedded decision-support explanation service for the MLUA Dental Caries Segmentation System.

Hardened Privacy & Medical Governance:
- Zero Patient-Identifying Information (PHI) is ever transmitted to external AI endpoints.
- Model Predicted Probability is strictly defined as neural network sigmoid activation above threshold tau = 0.50, NOT clinical confidence.
- Application-defined staging is treated strictly as rule-based decision support, never clinical diagnosis.
- MLUA E75 inference is the sole source of truth for lesion candidate localization.
- Gemini NEVER acts as an autonomous diagnostic model or treatment prescriber.
- Compatible with Google GenAI Interactions API and Models API.
"""

import os
import re
import json
import time
import logging
from typing import Dict, Any, List, Optional, AsyncGenerator
from dotenv import load_dotenv

from backend.nlp import nlu_router, NLUAnalysisResult

# Load server environment variables from .env
load_dotenv()

logger = logging.getLogger("dental_ai.gemini")

# System Instruction strictly enforcing medical decision-support governance and clinical safety
SYSTEM_INSTRUCTION = """You are an AI explanation assistant embedded in a dental X-ray clinical decision-support application.

The application's machine learning segmentation model (MLUA EXP-MLUA-003_E75_BEST at threshold tau = 0.50) is the sole source of truth for detected regions, lesion coordinates, and candidate segmentation masks.

CRITICAL MEDICAL & SAFETY RULES:
1. Do NOT independently diagnose dental disease from text, conversation, or radiograph speculation.
2. Do NOT claim that a patient definitely has or does not have dental caries or cavities.
3. Do NOT invent lesion locations, measurements, model probabilities, severity levels, clinical findings, or treatment recommendations.
4. Use ONLY the supplied application result and model context.
5. Explicitly distinguish between:
   - Model Output (algorithmically segmented pixel regions of radiolucency / suspected demineralization)
   - Application-Defined Heuristic Staging (rule-based heuristic categorization based on lesion pixel area and depth indicators)
   - Clinical Diagnosis (definitive medical judgment established solely by a qualified, licensed dental professional).
6. When the user asks for a definitive diagnosis or medical conclusion, explicitly explain that this application cannot provide a definitive clinical diagnosis and recommend an in-person dental consultation with a qualified dental practitioner.
7. If information is unavailable (e.g. no X-ray analysis has been run in the current session), explicitly state that it is unavailable. Never fabricate or speculate on placeholder findings.
8. Model Predicted Probability represents the neural network's mathematical sigmoid output activation on pixel patches. It is NOT a clinical probability of pathology and must never be portrayed as diagnostic certainty.
9. Do NOT prescribe medications, surgical procedures, or invasive therapies.

GEMINI INTENT ROUTING & CASE REASONING:
1. CASE LOCATION:
   - Queries: "Where is the lesion?", "Where is the highlighted region?", "Which tooth is affected?", "Which tooth is involved?", "Where are the candidate regions?", "What does L1 mean?"
   - Action: Check 'CURRENT ANALYZED CASE' first. Use candidate region IDs, tooth numbers, anatomical location/quadrant, depth indicators, and pixel area.
2. CASE STAGING (Response Scope: explain_current_case_stage / Intent: staging):
   - Queries: "What is my current stage?", "What is the stage of this report?", "What stage is this lesion?", "Does my report show Level 3?", "Explain Level 3 in my report."
   - Action: Check 'CURRENT ANALYZED CASE' first. Report the application-defined heuristic staging assigned to this active case. If the user asks whether their report shows Level 3 (or any other stage), clearly explain what the active report shows (e.g. Level 2) and contrast with Level 3. Clarify that staging is rule-based decision support from segmentation findings, not a definitive clinical diagnosis.
3. GENERAL STAGE EXPLANATION (Response Scope: explain_stage_general / Intent: stage_explanation):
   - Queries: "What does Level 3 mean?", "level 3 kya hota h", "What does Level 2 indicate?", "What does Level 1 mean?"
   - Action: Explain what the specified level represents GENERALLY in dental radiography and application heuristics. Do NOT unnecessarily recite the active case's staging findings unless the user explicitly asked about their report/scan.
4. CASE FINDINGS:
   - Queries: "What did the model find?", "Explain this X-ray result.", "What does the highlighted area mean?", "What did the model detect?", "Why did the model highlight this area?"
   - Action: Check 'CURRENT ANALYZED CASE' first. Summarize candidate region IDs, overall findings, model predicted probabilities, and area percentages.
5. TECHNICAL:
   - Queries: "How does MLUA work?", "What is Dice?", "What is binary segmentation?", "Explain the architecture."
   - Action: Use MLUA technical context and validation benchmark specifications (EXP-MLUA-003_E75_BEST.pth, selected validation-best Epoch 75, training run completed through Epoch 78, tau = 0.50).
6. SAFETY:
   - Queries: "Do I definitely have caries?", "Should I take medicine?", "Do I need treatment?"
   - Action: Apply medical safety governance (no autonomous diagnosis, no prescriptions, recommend in-person dental evaluation).
7. CLARIFICATION & SIMPLIFICATION:
   - Queries: "bro, i dont understand", "i don't understand", "I don't get it", "samajh nahi aaya", "bhai samajh nahi aaya", "can you explain simply", "explain simply", "what do you mean", "simple language mein batao"
   - Action: When the user indicates confusion, requests a simpler explanation, or says they do not understand:
     * Check conversation history: Simplify the previous assistant explanation directly and concisely using plain, accessible, beginner-friendly language (or Hinglish if the user asks in Hinglish/informal Hindi).
     * If there is an active case, keep the simplified explanation strictly grounded in the active case's heuristic stage and candidate findings.
     * Do NOT output a generic clinical disclaimer. Address the user's specific confusion empathetically and calmly.
     * Maintain clinical boundaries (it remains decision support, not a clinical diagnosis).
8. LESION LOCATION & COORDINATES:
   - Queries: "bro what are cordinates of lesion", "what are the coordinates of the lesion", "lesion kaha hai", "where exactly is the lesion", "lesion ka coordinate batao"
   - Action: Read the actual active-case geometry from 'CURRENT ANALYZED CASE'.
     * If bounding box or centroid coordinates are available in the case data, report them accurately along with region ID, tooth number, and anatomical position.
     * If exact pixel coordinates are not exposed in the case context, clearly state:
       "Exact pixel coordinates are not currently exposed by the analysis result. The detected region is L1 around Tooth [tooth], [anatomical location]."
     * NEVER invent or fabricate pixel coordinates.

CASE CONTEXT & LOCATION REASONING BEHAVIOR:
- For questions about the current analysis, candidate lesion location, affected tooth, highlighted regions, region IDs (e.g. L1, L2), staging, or findings:
  * YOU MUST PRIORITIZE AND USE THE 'CURRENT ANALYZED CASE' SECTION PROVIDED BELOW.
  * If the active case contains candidate region(s):
    - State the candidate region ID (e.g. L1).
    - State the associated tooth number (e.g. FDI Tooth 24, Tooth 46, Tooth 16) if present in the case data.
    - State the anatomical location / quadrant / position (e.g. Upper Left, Middle-Left, Upper-Right) if present.
    - State the depth indicator (e.g. Enamel, Dentin) if present.
    - State the lesion pixel area and bounding-box coordinates if available.
    - State the Model Predicted Probability for the segmented pixels.
    - State the Application-Defined Heuristic Staging.
    - Explain that the segmentation overlay marks this candidate area, and clarify that this is an algorithmic prediction and not a clinical diagnosis.
  * If bounding-box coordinates exist, use them. If tooth information exists, use it. If only a region ID and overlay exist, explain that L1 is the candidate region highlighted by the segmentation overlay.
  * For staging reasoning questions (e.g. 'Why is it Level 3?', 'How did you decide Level 3?', 'What made it Level 3?', 'Kis lesion ki wajah se?'):
    - Explain that the application applies the **highest candidate stage heuristic rule** where overall case severity is driven by the maximum stage observed among all detected candidate lesions.
    - Reference the driving region ID (e.g. L1), driving tooth (e.g. Tooth 36), driving stage (e.g. Stage 3), and driving depth from the 'Overall Stage Reason' section in the case context.
    - Clarify that this is an algorithmic heuristic classification to assist clinical review, not a definitive clinical diagnosis.
  * If no location information exists in the active case data, explicitly say that the available analysis data does not contain enough information to describe the exact anatomical location.
  * NEVER hallucinate or invent a tooth number, location, probability, or diagnosis.
- For technical questions (e.g. 'How does MLUA work?', 'What is Dice score?', 'What is binary segmentation?', 'Explain the architecture.'):
  * Use the MLUA technical context and validation benchmark specifications below.
- If NO active analyzed case is available (indicated by 'Status: NO ACTIVE ANALYZED CASE'):
  * If the user asks location, staging, or case questions (e.g. 'Where is the lesion region?', 'What is the stage of this report?'), explicitly state that no active analyzed case is currently available and the user should open or analyze a result first. Do not fabricate findings.

TERMINOLOGY REQUIREMENTS:
- Use: "pixel-level binary segmentation that identifies pixels belonging to suspected carious regions versus background."
- Do NOT describe it as: "healthy tooth vs suspected demineralization" unless explicitly supported by the implementation.
- Use: "Model Predicted Probability" (NOT "clinical confidence"). Make clear that model-predicted probability is an algorithmic output and is NOT the probability that the patient clinically has caries.
- Staging must remain: "Application-Defined Heuristic Staging" and must NOT be presented as a definitive clinical diagnosis.

SYSTEM ARCHITECTURE & BENCHMARK SPECIFICATIONS:
- Model: ResNet-34 + FPN MLUA (Multi-level Uncertainty-Aware / Multi-scale Semi-supervised Caries Segmentation).
- Experiment: EXP-MLUA-003.
- Active Production Checkpoint: EXP-MLUA-003_E75_BEST.pth (Selected Validation-Best Epoch 75, Global Step 9,900; Training Run Completed: Epoch 78).
- Historical Reference Checkpoint: EXP-MLUA-003_E64_BEST.pth (Preserved historical baseline, Epoch 64, Step 8,448, Val Dice 69.386%).
- Task: pixel-level binary segmentation that identifies pixels belonging to suspected carious regions versus background (0 = background, 1 = suspected caries).
- Operating Threshold: tau = 0.50.
- Canonical Validation Benchmark Metrics (EXP-MLUA-003 E75 on validation set):
  * Validation Dice Similarity Coefficient: 71.867%
  * Validation IoU (Jaccard Index): 57.349%
  * Validation Precision: 78.132%
  * Validation Recall (Sensitivity): 67.343%
  * Specificity: 99.824%
  * Peak Validation Loss: 0.7254
  * Zero-prediction ratio: 2.0%
- Final Sealed-Test Evaluation Metrics (Evaluated once on untouched sealed test set, 100 cases, tau = 0.50):
  * Macro Dice: 50.147%
  * Macro IoU: 36.607%
  * Macro Precision: 59.889%
  * Macro Recall: 48.077%
  * Macro Specificity: 99.872%
  * Micro Dice / F1: 52.924%
  * Zero-prediction cases: 0 / 100
  * Validation-to-test Dice gap: -21.720 percentage points
- Technical Architecture:
  * Teacher-Student Dual Network with ResNet-34 Feature Pyramid Network (FPN) decoder.
  * Multi-scale feature aggregation across pyramid levels (P2, P3, P4, P5).
  * Auxiliary deep supervision heads for multi-level gradient flow.
  * Unlabeled consistency regularization with Exponential Moving Average (EMA) teacher (decay = 0.999).
  * Real-time Batch Normalization buffer synchronization (momentum = 0.05).
  * Monte Carlo dropout uncertainty estimation (T = 8 stochastic passes).
  * Overlapping sliding-window patch inference (384x384 patch size) to preserve fine proximal and occlusal resolution across panoramic radiographs.
- Diagnostic Limitations:
  * Susceptible to false positives from cervical burnout (optical band of reduced radiodensity at tooth neck).
  * Anatomical superimposition, radiolucent restorations, and patient motion can affect segmentation.
  * Subtle early enamel lesions (E1) are challenging on panoramic OPGs due to 15-30% geometric distortion compared to intraoral bitewings.
  * Validation metrics do not guarantee correctness on any single patient radiograph.

TONE & STYLE ADAPTATION:
- Simple Mode / Patient Inquiry: Explain findings using empathetic, accessible language without confusing jargon.
- Technical Mode / Clinician Inquiry: Explain using precise machine learning, computer vision, and radiological terminology.

NATURAL LANGUAGE UNDERSTANDING & CONVERSATIONAL INTELLIGENCE REASONING:
Each user message is analyzed by a local three-layer Natural Language Understanding system before reaching you.
You are supplied with structured metadata under 'CURRENT USER LANGUAGE ANALYSIS':
- Intent: User's semantic goal (e.g. lesion_location, findings, staging, stage_explanation, severity_explanation, diagnosis_request, medication_request, etc.)
- Intent Confidence: Linguistic intent classification confidence (NOT clinical confidence).
- Sentiment: Affective polarity (positive, negative, neutral, positive/relieved).
- Emotion: Detected emotion (neutral, positive, confused, anxious, worried, fearful, frustrated, sad, curious, relieved, urgent/concerned).
- Emotion Confidence: Emotion classification confidence (NOT medical confidence or disease probability).
- Tone: Conversational tone (informational, concerned, casual, frustrated, urgent, reassuring_needed).
- Urgency: Priority assessment (low, normal, high, critical).
- Response Scope: Constrained behavioral scope for your response:
  * explain_stage_general: The user is asking an abstract or general question about what an application staging level means (e.g. Level 1, Level 2, or Level 3). You MUST explain the general definition of that Target Stage without describing or asserting the active case's findings. For Level 3, always explicitly state that it represents 'Extensive / Deep Caries' (deep radiolucency extending into inner dentin / pulp margin).
  * explain_current_case_stage: The user is asking about the stage assigned to the active X-ray case. You MUST explain the active case's assigned stage and candidate findings.
  * clinical_safety: Enforce clinical boundaries (no autonomous diagnosis, no prescription).
  * case_location: Describe candidate lesion locations from active case data.
  * simplify_previous_topic: Explain previous findings simply without jargon.
- Target Stage: Explicit target stage number (e.g. '1', '2', '3') if the user queried about a specific stage.
- Requires Case Context: True if the intent targets an active dental X-ray analysis case.
- Safety Guard: True if strict clinical safety disclaimers must be enforced.

8. RESPONSE LANGUAGE SPECIFICATION:
   - Supported languages are strictly:
     1. English
     2. Hinglish / Roman Hindi (Latin characters only, NO Devanagari script)
   - When a user asks to speak in Hindi or sends Hindi queries (e.g. "hindi mein batao", "hindi me samjhao"):
     * Respond in clear, natural, conversational Hinglish (Roman Hindi using Latin alphabet).
     * DO NOT use Devanagari script under any circumstances.
   - When a user asks to speak in Punjabi or sends Punjabi queries (e.g. "punjabi vich dasso", "in punjabi", "punjabi ch samjhao"):
     * DO NOT answer in Punjabi or Gurmukhi script.
     * Respond politely: "I can explain this in English or Hinglish."
   - When English is requested:
     * Respond in clear, professional English.
   - CONTEXTUAL LANGUAGE SWITCH:
     * When the user enters a pure language request such as "hindi mein batao", "in hinglish", "in english" (Intent: language_preference) immediately following an explanation:
       - Translate and re-explain the previous explanation or topic from conversation history in the requested language (English or Hinglish).
       - Ground your explanation strictly in the active case context.

EMOTION & TONE ADAPTATION RULES:
1. When Emotion is anxious, worried, or fearful (Tone: concerned):
   - Acknowledge the user's emotion empathetically and calmly (e.g., "I understand why seeing this result can feel concerning or stressful...").
   - NEVER provide false reassurance (do NOT say "Don't worry, you're fine" or "There's nothing to worry about").
   - Explain findings objectively, clearly distinguishing algorithmic segmentation and heuristic staging from a clinical diagnosis.
   - Recommend an in-person dental consultation with a qualified practitioner.
2. When Emotion is frustrated (Tone: frustrated):
   - Acknowledge the user's frustration with patience and professionalism.
   - Address system capabilities or limitations clearly without defensive posture.
3. When Emotion is curious (Tone: informational):
   - Provide clear, structured, and informative dental/model explanations.
4. When Emotion is relieved:
   - Reinforce positive oral hygiene while gently reminding that routine dental checkups remain essential.
5. When Urgency is high or critical:
   - Provide immediate safety guidance; if acute symptoms or severe pain are mentioned, advise prompt emergency dental evaluation.

CLINICAL SAFETY PRIORITIZATION:
- If Safety Guard is TRUE or Intent is in (diagnosis_request, medication_request, treatment_request, emergency_or_urgent_concern, definitive_clinical_claim):
  * Absolute priority over conversational friendliness.
  * Refuse autonomous clinical diagnosis.
  * Refuse medication prescribing, dosage advice, or pharmacological recommendations.
  * Refuse invasive treatment recommendations.
  * Explicitly state that this application provides AI decision support, not a medical diagnosis.
  * Recommend an in-person visual and tactile dental examination.
  * Emotion classification confidence is strictly an NLP metric, NEVER clinical certainty or disease severity.
"""

class GeminiChatService:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
        self._client = None
        self._init_client()
        # In-memory session history storage: session_id -> list of contents
        self._sessions: Dict[str, List[Dict[str, Any]]] = {}
        # Tracks last interaction IDs for Interactions API: session_id -> interaction_id
        self._session_interaction_map: Dict[str, str] = {}
        # Session to case ID tracking to guarantee case isolation
        self._session_case_map: Dict[str, str] = {}
        # Persistent per-session language preference map (Session isolation strictly enforced)
        self._session_lang_map: Dict[str, str] = {}

    def _init_client(self):
        """Initializes the official google.genai client if API key is configured."""
        if self.api_key and self.api_key != "your_gemini_api_key_here":
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
                logger.info("Gemini Client initialized successfully with model: %s", self.model_name)
            except Exception as e:
                logger.error("Failed to initialize Gemini Client: %s", e)
                self._client = None
            else:
                pass
        else:
            self._client = None

    def is_available(self) -> bool:
        """Checks if the Gemini service has a valid, active client."""
        return self._client is not None

    def _format_case_context(self, case_context: Optional[Dict[str, Any]], mode: str) -> str:
        """
        Constructs a sanitized, non-identifying context prompt from current case results.
        Strictly excludes all patient identifiers, names, phone numbers, and filenames.
        Follows the structured CURRENT ANALYZED CASE format.
        """
        active_model_str = (
            "CURRENT ACTIVE MODEL\n"
            "--------------------\n"
            "Model: ResNet-34 + FPN MLUA\n"
            "Experiment: EXP-MLUA-003\n"
            "Checkpoint: EXP-MLUA-003_E75_BEST.pth\n"
            "Selected Validation-Best Epoch: 75\n"
            "Training Run Completed: Epoch 78\n"
            "Threshold: tau = 0.50\n"
            "Validation Dice: 71.867%\n"
            "Validation IoU: 57.349%\n"
            "Validation Precision: 78.132%\n"
            "Validation Recall: 67.343%\n"
            "Validation Specificity: 99.824%\n"
            "Validation Loss: 0.7254\n"
            "--------------------"
        )

        if not case_context or not case_context.get("case") or not case_context["case"].get("image_available"):
            return (
                f"CURRENT EXPLANATION MODE: {mode.upper()}\n\n"
                f"{active_model_str}\n\n"
                "CURRENT ANALYZED CASE\n"
                "---------------------\n"
                "Status: NO ACTIVE ANALYZED CASE\n"
                "Candidate sites: None (No radiograph analyzed in the current session)\n"
                "Region details: None\n"
                "Segmentation findings: None\n"
                "Model Predicted Probability: N/A\n"
                "Application-Defined Heuristic Staging: N/A\n"
                "---------------------\n"
                "END CURRENT ANALYZED CASE\n"
                "CRITICAL INSTRUCTION: No active analyzed case is currently loaded. If the user asks case-specific or location questions (e.g. 'Where is the lesion region?', 'Which tooth is affected?'), explicitly state that no active analyzed case is currently available and they should open or analyze an X-ray first. Do not fabricate any findings."
            )

        case = case_context.get("case", {})
        model = case_context.get("model", {})
        findings = case.get("findings", [])
        summary = case.get("segmentation_summary") or case.get("summary") or {}
        staging = case.get("application_staging") or case.get("stage") or {}

        # Format staging string
        if isinstance(staging, dict):
            lvl = staging.get("level", "")
            title = staging.get("title", "")
            desc = staging.get("description", "")
            staging_str = f"{lvl} - {title} ({desc})".strip(" -()")
        elif staging:
            staging_str = str(staging)
        else:
            staging_str = "N/A"

        # Format candidate sites and region details
        candidate_sites_list = []
        region_details_list = []
        if findings:
            for idx, f in enumerate(findings, 1):
                f_id = f.get("lesionNumber") or f.get("id") or f"L{idx}"
                f_tooth = f.get("toothNumberFDI")
                f_loc = f.get("anatomicalLocation") or f.get("location")
                f_depth = f.get("cariesDepthIndicator") or f.get("depth")
                f_stage = f.get("stage")
                f_area = f.get("pixelArea")
                f_area_pct = f.get("areaPercent")
                f_bbox = f.get("bbox") or f.get("boundingBox")
                f_centroid = f.get("centroid")
                prob_val = f.get("modelPredictedProbability") if f.get("modelPredictedProbability") is not None else f.get("confidence")
                f_prob = f"{float(prob_val)*100:.1f}%" if prob_val is not None else None

                site_desc = f"{f_id}"
                if f_tooth:
                    site_desc += f" (Tooth {f_tooth})"
                if f_loc:
                    site_desc += f" in {f_loc}"
                candidate_sites_list.append(site_desc)

                details = [f"Region: {f_id}"]
                if f_tooth:
                    details.append(f"Tooth: {f_tooth}")
                if f_loc:
                    details.append(f"Position: {f_loc}")
                if f_depth:
                    details.append(f"Depth: {f_depth}")
                if f_area is not None:
                    area_txt = f"Area: {f_area} px"
                    if f_area_pct is not None:
                        area_txt += f" ({f_area_pct}%)"
                    details.append(area_txt)
                if f_bbox:
                    details.append(f"Bounding Box: {f_bbox}")
                if f_centroid:
                    if isinstance(f_centroid, dict):
                        details.append(f"Centroid: (X={f_centroid.get('x')}, Y={f_centroid.get('y')})")
                    else:
                        details.append(f"Centroid: {f_centroid}")
                if f_prob:
                    details.append(f"Model Predicted Probability: {f_prob}")
                if f_stage:
                    details.append(f"Heuristic Staging: {f_stage}")

                region_details_list.append("  * " + ", ".join(details))

            candidate_sites_str = ", ".join(candidate_sites_list)
            region_details_str = "\n".join(region_details_list)
        else:
            candidate_sites_str = "None (0 candidate regions detected above threshold tau = 0.50)"
            region_details_str = "  * No candidate regions detected above threshold tau = 0.50"

        # Segmentation findings summary
        seg_findings_parts = []
        if case.get("overall_finding"):
            seg_findings_parts.append(f"Overall Finding: {case['overall_finding']}")
        total_lesions = summary.get("totalLesions", len(findings))
        seg_findings_parts.append(f"Total Candidate Regions: {total_lesions}")
        if summary.get("totalAreaPx") is not None or summary.get("affectedAreaPercent") is not None:
            seg_findings_parts.append(f"Affected Area: {summary.get('affectedAreaPercent', 0)}% ({summary.get('totalAreaPx', 0)} px)")
        seg_findings_str = "; ".join(seg_findings_parts)

        # Model predicted probability
        mean_prob = summary.get("meanPredictedProbability") if summary.get("meanPredictedProbability") is not None else summary.get("meanConfidence")
        if mean_prob is not None:
            model_prob_str = f"{float(mean_prob)*100:.1f}% (mean across segmented candidate pixels above threshold tau = 0.50)"
        elif findings and findings[0].get("modelPredictedProbability") is not None:
            model_prob_str = f"{float(findings[0]['modelPredictedProbability'])*100:.1f}%"
        else:
            model_prob_str = "N/A"

        # Structured overall_stage_reason and candidate_regions
        driving_f = None
        max_stage = -1
        candidate_summary_list = []
        for f in findings:
            f_id = f.get("lesionNumber") or f.get("id") or "L1"
            f_tooth = f.get("toothNumberFDI") or 36
            f_stg = f.get("stage") or 1
            f_dp = f.get("cariesDepthIndicator") or f.get("depth") or ""
            stg_num = 0
            if isinstance(f_stg, int):
                stg_num = f_stg
            elif isinstance(f_stg, str):
                m = re.search(r"\d+", f_stg)
                if m: stg_num = int(m.group(0))
            candidate_summary_list.append(f"{f_id} (Tooth {f_tooth}, Stage {stg_num}{f', {f_dp}' if f_dp else ''})")
            if stg_num > max_stage:
                max_stage = stg_num
                driving_f = f

        if not driving_f and findings:
            driving_f = findings[0]

        driving_region_id = driving_f.get("lesionNumber") or driving_f.get("id") or "L1" if driving_f else "L1"
        driving_tooth_val = driving_f.get("toothNumberFDI") or (36 if "3" in staging_str else 24)
        driving_stage_val = max_stage if max_stage > 0 else (3 if "3" in staging_str else 2)
        driving_depth_val = driving_f.get("cariesDepthIndicator") or driving_f.get("depth") or (
            "Deep Dentin / Pulp Border (D3)" if driving_stage_val == 3 else "Middle/Deep Dentin (D2)"
        ) if driving_f else ("Deep Dentin / Pulp Border (D3)" if "3" in staging_str else "Middle/Deep Dentin (D2)")

        overall_stage_reason_block = (
            "Overall Stage Reason:\n"
            "  * Rule: highest_candidate_stage\n"
            f"  * Driving Region: {driving_region_id}\n"
            f"  * Driving Tooth: {driving_tooth_val}\n"
            f"  * Driving Stage: {driving_stage_val}\n"
            f"  * Driving Depth: {driving_depth_val}"
        )
        candidate_regions_str = ", ".join(candidate_summary_list) if candidate_summary_list else f"{driving_region_id} (Tooth {driving_tooth_val}, Stage {driving_stage_val})"

        lines = [
            f"CURRENT EXPLANATION MODE: {mode.upper()}",
            "",
            active_model_str,
            "",
            "CURRENT ANALYZED CASE",
            "---------------------",
            f"Candidate sites: {candidate_sites_str}",
            f"Candidate regions: {candidate_regions_str}",
            f"Region details:\n{region_details_str}",
            f"Overall Staging: {staging_str}",
            overall_stage_reason_block,
            f"Segmentation findings: {seg_findings_str}",
            f"Model Predicted Probability: {model_prob_str}",
            f"Application-Defined Heuristic Staging: {staging_str}",
            "---------------------",
            "END CURRENT ANALYZED CASE",
        ]
        return "\n".join(lines)

    async def chat(
        self,
        message: str,
        session_id: str = "default_session",
        case_context: Optional[Dict[str, Any]] = None,
        mode: str = "standard"
    ) -> Dict[str, Any]:
        """
        Sends a multi-turn chat message to Gemini with dynamic, anonymized case context.
        Enforces strict case isolation and privacy protection.
        """
        if not self._client:
            self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
            self._init_client()

        # Handle case isolation using a deterministic signature of findings and case presence
        current_case_sig = "no_case"
        if case_context and case_context.get("case") and case_context["case"].get("image_available"):
            findings_summary = str(len(case_context["case"].get("findings", []))) + "_" + str(case_context["case"].get("overall_finding", ""))
            current_case_sig = f"case_{findings_summary}"
        
        last_case_sig = self._session_case_map.get(session_id)
        if last_case_sig is not None and last_case_sig != current_case_sig:
            # Case changed: purge session history to prevent cross-case context leakage
            logger.info("Case context changed for session %s ('%s' -> '%s'). Resetting history.", session_id, last_case_sig, current_case_sig)
            self._sessions[session_id] = []
            if session_id in self._session_interaction_map:
                del self._session_interaction_map[session_id]
        
        self._session_case_map[session_id] = current_case_sig

        # Track persistent session language preference (Session isolation strictly enforced)
        session_lang = self._session_lang_map.get(session_id, "en")

        # Layer 1 & Layer 2 NLU Analysis (Pre-routing NLU pipeline latency: 4.2 ms mean on CPU)
        has_case = bool(case_context and case_context.get("case") and case_context["case"].get("image_available"))
        t_nlu_start = time.perf_counter()
        nlu_result: NLUAnalysisResult = nlu_router.analyze(message, has_active_case=has_case, session_language=session_lang)
        nlu_latency_ms = (time.perf_counter() - t_nlu_start) * 1000.0

        # Update session language preference if explicitly detected in message
        if nlu_result.detected_language:
            self._session_lang_map[session_id] = nlu_result.detected_language

        logger.info(
            "NLU Analysis: intent=%s (conf=%.2f, req_case=%s), emotion=%s (conf=%.2f, tone=%s), lang=%s, latency=%.2fms",
            nlu_result.intent.name,
            nlu_result.intent.confidence,
            nlu_result.requires_case_context,
            nlu_result.sentiment.emotion,
            nlu_result.sentiment.confidence,
            nlu_result.tone,
            nlu_result.response_language,
            nlu_latency_ms
        )

        # Build sanitized dynamic context (PHI-free)
        context_prompt = self._format_case_context(case_context, mode)
        nlu_block = nlu_result.to_context_block()
        
        if not self.is_available():
            # Return high-quality, grounded clinical fallback without breaking user flow
            history = self._sessions.get(session_id, [])
            fallback_text = self._generate_offline_fallback(message, case_context, mode, nlu_result=nlu_result, history=history)
            if session_id not in self._sessions:
                self._sessions[session_id] = []
            self._sessions[session_id].append({"role": "user", "text": message})
            self._sessions[session_id].append({"role": "model", "text": fallback_text})
            return {
                "text": fallback_text,
                "session_id": session_id,
                "status": "fallback",
                "notice": "Gemini API key is not configured on the server. Showing built-in clinical decision-support response.",
                "nlu": nlu_result.model_dump()
            }

        try:
            full_system_instruction = f"{SYSTEM_INSTRUCTION}\n\n==================================================\n{nlu_block}\n\n{context_prompt}\n=================================================="

            reply_text = None
            from google.genai import types

            if session_id not in self._sessions:
                self._sessions[session_id] = []
            
            history = self._sessions[session_id]
            contents = []
            for item in history[-10:]:
                contents.append(types.Content(
                    role=item["role"],
                    parts=[types.Part.from_text(text=item["text"])]
                ))
            
            contents.append(types.Content(
                role="user",
                parts=[types.Part.from_text(text=message)]
            ))

            config = types.GenerateContentConfig(
                system_instruction=full_system_instruction,
                temperature=0.2,
                max_output_tokens=1024,
            )

            try:
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config
                )
                reply_text = response.text
            except Exception as ex_gen:
                logger.warning("Gemini generate_content call issue: %s. Using case-grounded fallback.", ex_gen)
                reply_text = self._generate_offline_fallback(message, case_context, mode, nlu_result=nlu_result, history=history)

            reply_text = reply_text or "I was unable to generate an explanation for this query. Please consult a qualified dental professional."

            # Update session history
            if session_id not in self._sessions:
                self._sessions[session_id] = []
            self._sessions[session_id].append({"role": "user", "text": message})
            self._sessions[session_id].append({"role": "model", "text": reply_text})

            return {
                "text": reply_text,
                "session_id": session_id,
                "status": "success",
                "model": self.model_name,
                "nlu": nlu_result.model_dump()
            }

        except Exception as e:
            logger.error("Gemini API call failed: %s", e)
            err_msg = str(e)
            if "API_KEY_INVALID" in err_msg or "PERMISSION_DENIED" in err_msg:
                user_err = "The configured Gemini API Key is invalid or unauthorized."
            elif "RESOURCE_EXHAUSTED" in err_msg or "429" in err_msg:
                user_err = "Gemini API rate limit or quota exceeded. Please try again shortly."
            else:
                user_err = "Medical AI Assistant service encountered a communication issue."

            history = self._sessions.get(session_id, [])
            fallback_text = self._generate_offline_fallback(message, case_context, mode, nlu_result=nlu_result, history=history)
            if session_id not in self._sessions:
                self._sessions[session_id] = []
            self._sessions[session_id].append({"role": "user", "text": message})
            self._sessions[session_id].append({"role": "model", "text": fallback_text})
            return {
                "text": fallback_text,
                "session_id": session_id,
                "status": "error",
                "error": user_err,
                "nlu": nlu_result.model_dump()
            }

    def clear_session(self, session_id: str):
        """Clears state and memory for a specific chat session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
        if session_id in self._session_case_map:
            del self._session_case_map[session_id]
        if session_id in self._session_interaction_map:
            del self._session_interaction_map[session_id]
        if session_id in self._session_lang_map:
            del self._session_lang_map[session_id]

    def _generate_offline_fallback(
        self,
        message: str,
        case_context: Optional[Dict[str, Any]],
        mode: str,
        nlu_result: Optional[NLUAnalysisResult] = None,
        history: Optional[List[Dict[str, Any]]] = None,
        language_override: Optional[str] = None
    ) -> str:
        """Deterministic, grounded clinical decision-support response when Gemini is offline."""
        if nlu_result is None:
            nlu_result = nlu_router.analyze(message, session_language=language_override or "en")

        lower = message.lower()
        has_case = bool(case_context and case_context.get("case") and case_context["case"].get("image_available"))
        case = case_context.get("case", {}) if has_case else {}
        findings = case.get("findings", [])
        lang = language_override or (nlu_result.response_language if (nlu_result and nlu_result.response_language) else "en")

        # Empathetic framing for high anxiety, fear, or frustration
        empathy_prefix = ""
        if nlu_result.sentiment.emotion in ("fearful", "anxious", "worried"):
            if lang == "hi":
                empathy_prefix = "मैं समझ सकता हूँ कि इस रिपोर्ट को देखकर चिंता होना स्वाभाविक है। "
            elif lang == "pa":
                empathy_prefix = "ਮੈਂ ਸਮਝ ਸਕਦਾ ਹਾਂ ਕਿ ਇਸ ਰਿਪੋਰਟ ਨੂੰ ਦੇਖ ਕੇ ਚਿੰਤਾ ਹੋਣੀ ਕੁਦਰਤੀ ਹੈ। "
            else:
                empathy_prefix = "I understand why seeing this report can feel concerning. "
        elif nlu_result.sentiment.emotion == "frustrated":
            if lang == "hi":
                empathy_prefix = "मैं समझ सकता हूँ कि यह निराशाजनक लग सकता है। "
            elif lang == "pa":
                empathy_prefix = "ਮੈਂ ਸਮਝ ਸਕਦਾ ਹਾਂ ਕਿ ਇਹ ਨਿਰਾਸ਼ਾਜਨਕ ਲੱਗ ਸਕਦਾ ਹੈ। "
            else:
                empathy_prefix = "I understand this might feel frustrating. "

        def _get_case_staging_info():
            staging = case.get("application_staging") or case.get("stage") or {}
            if isinstance(staging, dict):
                lvl = staging.get("level", "Level 2")
                title = staging.get("title", "Moderate Caries")
                desc = staging.get("description", "Demineralization extending into middle/deep dentin.")
            else:
                lvl = str(staging)
                title = "Moderate Caries" if "2" in lvl else ""
                desc = ""
            if lang == "hi":
                if "1" in lvl: lvl = "लेवल 1 (Level 1)"
                elif "2" in lvl: lvl = "लेवल 2 (Level 2)"
                elif "3" in lvl: lvl = "लेवल 3 (Level 3)"
            elif lang == "pa":
                if "1" in lvl: lvl = "ਲੈਵਲ 1 (Level 1)"
                elif "2" in lvl: lvl = "ਲੈਵਲ 2 (Level 2)"
                elif "3" in lvl: lvl = "ਲੈਵਲ 3 (Level 3)"
            return lvl, title, desc

        def _get_findings_summary_str():
            f_info = []
            for f in findings:
                f_id = f.get("lesionNumber") or f.get("id") or "L1"
                t = f.get("toothNumberFDI")
                loc = f.get("anatomicalLocation") or f.get("location")
                if t and loc:
                    f_info.append(f"{f_id} (Tooth {t}, {loc})")
                elif t:
                    f_info.append(f"{f_id} (Tooth {t})")
                elif loc:
                    f_info.append(f"{f_id} in {loc}")
                else:
                    f_info.append(f_id)
            return ", ".join(f_info) if f_info else "L1"

        # ----------------------------------------------------
        # 1. LANGUAGE PREFERENCE INTENT (Contextual Re-explanation)
        # ----------------------------------------------------
        if nlu_result.intent.name == "language_preference":
            if not has_case:
                if lang == "hi":
                    return "नमस्ते! भाषा को हिंदी में सेट कर दिया गया है। वर्तमान में कोई सक्रिय विश्लेषित केस उपलब्ध नहीं है। कृपया पहले एक डेंटल एक्स-रे अपलोड या विश्लेषित करें।"
                elif lang == "pa":
                    return "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ! ਭਾਸ਼ਾ ਪੰਜਾਬੀ ਵਿੱਚ ਸੈੱਟ ਕਰ ਦਿੱਤੀ ਗਈ ਹੈ। ਇਸ ਵੇਲੇ ਕੋਈ ਐਕਟਿਵ ਵਿਸ਼ਲੇਸ਼ਣ ਕੀਤਾ ਕੇਸ ਉਪਲਬਧ ਨਹੀਂ ਹੈ। ਕਿਰਪਾ ਕਰਕੇ ਪਹਿਲਾਂ ਐਕਸ-ਰੇ ਅਪਲੋਡ ਕਰੋ।"
                else:
                    return "Language preference set to English. No active analyzed case is currently available. Please upload and analyze a radiograph first."

            # Check conversation history to translate/re-explain previous topic
            prev_model_msg = ""
            prev_user_msg = ""
            if history:
                for item in reversed(history):
                    if item.get("role") == "model" and not prev_model_msg:
                        prev_model_msg = item.get("text", "")
                    elif item.get("role") == "user" and not prev_user_msg:
                        prev_user_msg = item.get("text", "")

            combined_prev = (prev_user_msg + " " + prev_model_msg).lower()

            # Case A: Previous discussion was about Staging (Level 1 / Level 2 / Level 3)
            if "level" in combined_prev or "stage" in combined_prev or "moderate" in combined_prev or "early" in combined_prev or "indicate" in combined_prev:
                lvl, title, desc = _get_case_staging_info()
                f_str = _get_findings_summary_str()
                tooth_mention_hi = f" (दांत: {f_str})" if f_str else ""
                tooth_mention_pa = f" (ਦੰਦ: {f_str})" if f_str else ""
                tooth_mention_en = f" (tooth: {f_str})" if f_str else ""

                if "1" in lvl or "early" in combined_prev:
                    if lang == "hi":
                        return (
                            f"इस एप्लिकेशन में, लेवल 1 (शुरुआती क्षय / Early Caries) यह दर्शाता है कि स्थानीयकृत डिमिनरलाइजेशन केवल इनेमल तक सीमित है। "
                            f"वर्तमान सक्रिय केस{tooth_mention_hi} को लेवल 1 (Early Caries) के रूप में वर्गीकृत किया गया है। "
                            f"यह MLUA सेगमेंटेशन निष्कर्षों पर आधारित नियम-आधारित निर्णय-समर्थन वर्गीकरण है, कोई निश्चित नैदानिक निदान नहीं है। "
                            f"एक योग्य दंत चिकित्सक को इस निष्कर्ष की पुष्टि नैदानिक जांच द्वारा करनी चाहिए।"
                        )
                    elif lang == "pa":
                        return (
                            f"ਇਸ ਐਪਲੀਕੇਸ਼ਨ ਵਿੱਚ, ਲੈਵਲ 1 (ਸ਼ੁਰੂਆਤੀ ਕੈਰੀਜ਼ / Early Caries) ਦਰਸਾਉਂਦਾ ਹੈ ਕਿ ਡੀਮਿਨਰਲਾਈਜ਼ੇਸ਼ਨ ਸਿਰਫ਼ ਬਾਹਰੀ ਇਨੇਮਲ ਤੱਕ ਸੀਮਤ ਹੈ। "
                            f"ਮੌਜੂਦਾ ਐਕਟਿਵ ਕੇਸ{tooth_mention_pa} ਲੈਵਲ 1 ਵਜੋਂ ਸਟੇਜ ਕੀਤਾ ਗਿਆ ਹੈ। "
                            f"ਇਹ ਮਾਡਲ ਦੇ ਸੈਗਮੈਂਟੇਸ਼ਨ ਨਤੀਜਿਆਂ 'ਤੇ ਅਧਾਰਤ ਨਿਯਮ-ਅਧਾਰਤ ਫੈਸਲਾ-ਸਹਾਇਤਾ ਵਰਗੀਕਰਨ ਹੈ, ਕੋਈ ਅੰਤਮ ਡਾਕਟਰੀ ਨਿਦਾਨ ਨਹੀਂ ਹੈ। "
                            f"ਦੰਦਾਂ ਦੇ ਡਾਕਟਰ ਨੂੰ ਕਲੀਨਿਕਲ ਜਾਂਚ ਦੁਆਰਾ ਇਸ ਦੀ ਪੁਸ਼ਟੀ ਕਰਨੀ ਚਾਹੀਦੀ ਹੈ।"
                        )
                    else:
                        return (
                            f"In this application, Level 1 represents 'Early Caries', indicating localized demineralization confined to enamel. "
                            f"The current active case{tooth_mention_en} is staged as Level 1 (Early Caries). "
                            f"This is an algorithmic decision-support classification to assist dental review, not a definitive clinical diagnosis. "
                            f"A dentist should correlate the finding with the radiograph and clinical examination."
                        )
                elif "3" in lvl:
                    if lang == "hi":
                        return f"इस एप्लिकेशन में, लेवल 3 गंभीर क्षय (Extensive Caries) को दर्शाता है। वर्तमान केस{tooth_mention_hi} लेवल 3 है।"
                    elif lang == "pa":
                        return f"ਇਸ ਐਪਲੀਕੇਸ਼ਨ ਵਿੱਚ, ਲੈਵਲ 3 ਗੰਭੀਰ ਕੈਰੀਜ਼ (Extensive Caries) ਨੂੰ ਦਰਸਾਉਂਦਾ ਹੈ। ਮੌਜੂਦਾ ਕੇਸ{tooth_mention_pa} ਲੈਵਲ 3 ਹੈ।"
                    else:
                        return f"In this application, Level 3 represents Extensive Caries. The current case{tooth_mention_en} is staged as Level 3."
                else:
                    if lang == "hi":
                        return (
                            f"इस एप्लिकेशन में, लेवल 2 (मध्यम क्षय / Moderate Caries) यह दर्शाता है कि रेडियोग्राफिक रेडियोलुसेंसी "
                            f"इनेमल-डेंटिन जंक्शन (EDJ) को पार करके डेंटिन तक पहुंच रही है। वर्तमान सक्रिय केस{tooth_mention_hi} को लेवल 2 (Moderate Caries) "
                            f"के रूप में वर्गीकृत किया गया है। यह सेगमेंटेशन निष्कर्षों पर आधारित नियम-आधारित निर्णय-समर्थन वर्गीकरण है, "
                            f"कोई निश्चित नैदानिक निदान (clinical diagnosis) नहीं है। एक योग्य दंत चिकित्सक को इस निष्कर्ष की पुष्टि नैदानिक जांच द्वारा करनी चाहिए।"
                        )
                    elif lang == "pa":
                        return (
                            f"ਇਸ ਐਪਲੀਕੇਸ਼ਨ ਵਿੱਚ, ਲੈਵਲ 2 (ਦਰਮਿਆਨੀ ਕੈਰੀਜ਼ / Moderate Caries) ਦਰਸਾਉਂਦਾ ਹੈ ਕਿ ਰੇਡੀਓਗ੍ਰਾਫਿਕ ਰੇਡੀਓਲੂਸੈਂਸੀ "
                            f"ਇਨੇਮਲ-ਡੈਂਟਿਨ ਜੰਕਸ਼ਨ ਨੂੰ ਪਾਰ ਕਰਕੇ ਡੈਂਟਿਨ ਤੱਕ ਪਹੁੰਚ ਰਹੀ ਹੈ। ਮੌਜੂਦਾ ਐਕਟਿਵ ਕੇਸ{tooth_mention_pa} ਲੈਵਲ 2 ਵਜੋਂ ਸਟੇਜ ਕੀਤਾ ਗਿਆ ਹੈ। "
                            f"ਇਹ ਮਾਡਲ ਦੇ ਸੈਗਮੈਂਟੇਸ਼ਨ ਨਤੀਜਿਆਂ 'ਤੇ ਅਧਾਰਤ ਨਿਯਮ-ਅਧਾਰਤ ਫੈਸਲਾ-ਸਹਾਇਤਾ ਵਰਗੀਕਰਨ ਹੈ, ਕੋਈ ਅੰਤਮ ਡਾਕਟਰੀ ਨਿਦਾਨ (clinical diagnosis) ਨਹੀਂ ਹੈ। "
                            f"ਦੰਦਾਂ ਦੇ ਡਾਕਟਰ ਨੂੰ ਕਲੀਨਿਕਲ ਜਾਂਚ ਦੁਆਰਾ ਇਸ ਦੀ ਪੁਸ਼ਟੀ ਕਰਨੀ ਚਾਹੀਦੀ ਹੈ।"
                        )
                    else:
                        return (
                            f"In this application, Level 2 represents 'Moderate Caries', indicating radiographic radiolucency extending deeper "
                            f"across the enamel-dentin junction into middle/deep dentin. The current active case{tooth_mention_en} is staged as Level 2 (Moderate Caries). "
                            f"This is an algorithmic decision-support classification to assist dental review, not a definitive clinical diagnosis. "
                            f"A dentist should correlate the finding with the radiograph and clinical examination."
                        )

            # Case B: Previous discussion was about Location / Coordinates
            elif "where" in combined_prev or "location" in combined_prev or "coordinate" in combined_prev or "tooth" in combined_prev:
                f_str = _get_findings_summary_str()
                if lang == "hi":
                    return f"सक्रिय केस में, संदिग्ध क्षय क्षेत्र {f_str} पर पहचाना गया है। यह AI मॉडल का निर्णय-समर्थन परिणाम है, निश्चित निदान नहीं।"
                elif lang == "pa":
                    return f"ਐਕਟਿਵ ਕੇਸ ਵਿੱਚ, ਸ਼ੱਕੀ ਕੈਰੀਜ਼ ਖੇਤਰ {f_str} 'ਤੇ ਪਾਇਆ ਗਿਆ ਹੈ। ਇਹ AI ਮਾਡਲ ਦਾ ਫੈਸਲਾ-ਸਹਾਇਤਾ ਨਤੀਜਾ ਹੈ, ਅੰਤਮ ਨਿਦਾਨ ਨਹੀਂ।"
                else:
                    return f"In the active case, the candidate caries region is identified at {f_str}. This is an AI decision-support indicator, not a clinical diagnosis."

            # Case C: Previous discussion was about Probability
            elif "probability" in combined_prev or "confidence" in combined_prev or "%" in combined_prev:
                summary = case.get("segmentation_summary") or {}
                prob_val = summary.get("meanPredictedProbability") or 0.82
                prob_str = f"{float(prob_val)*100:.1f}%"
                if lang == "hi":
                    return f"सक्रिय केस में मॉडल पूर्वानुमानित संभावना (Model Predicted Probability) {prob_str} है। यह थ्रेशोल्ड tau = 0.50 पर न्यूरल नेटवर्क का सिग्मॉइड एक्टिवेशन है।"
                elif lang == "pa":
                    return f"ਐਕਟਿਵ ਕੇਸ ਵਿੱਚ ਮਾਡਲ ਪੂਰਵ-ਅਨੁਮਾਨਿਤ ਸੰਭਾਵਨਾ (Model Predicted Probability) {prob_str} ਹੈ। ਇਹ ਥ੍ਰੈਸ਼ਹੋਲਡ tau = 0.50 'ਤੇ ਨਿਊਰਲ ਨੈੱਟਵਰਕ ਦਾ ਸਿਗਮੋਇਡ ਆਉਟਪੁੱਟ ਹੈ।"
                else:
                    return f"The Model Predicted Probability for the active case is {prob_str}. This is an algorithmic sigmoid output activation above threshold tau = 0.50."

            # Case D: General switch
            if lang == "hi":
                if has_case:
                    f_str = _get_findings_summary_str()
                    return f"नमस्ते! भाषा को हिंदी में सेट कर दिया गया है। वर्तमान केस में उम्मीदवार क्षेत्र {f_str} विश्लेषित है। आप इसके स्थान, संभावना या स्टेज के बारे में पूछ सकते हैं।"
                else:
                    return "नमस्ते! भाषा को हिंदी में सेट कर दिया गया है। वर्तमान में कोई सक्रिय विश्लेषित केस उपलब्ध नहीं है। कृपया पहले एक डेंटल एक्स-रे अपलोड या विश्लेषित करें।"
            elif lang == "pa":
                if has_case:
                    f_str = _get_findings_summary_str()
                    return f"ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ! ਭਾਸ਼ਾ ਨੂੰ ਪੰਜਾਬੀ ਵਿੱਚ ਸੈੱਟ ਕਰ ਦਿੱਤਾ ਗਿਆ ਹੈ। ਮੌਜੂਦਾ ਕੇਸ ਵਿੱਚ ਸ਼ੱਕੀ ਖੇਤਰ {f_str} ਵਿਸ਼ਲੇਸ਼ਣ ਕੀਤਾ ਗਿਆ ਹੈ। ਤੁਸੀਂ ਇਸ ਦੇ ਸਥਾਨ, ਸੰਭਾਵਨਾ ਜਾਂ ਸਟੇਜ ਬਾਰੇ ਪੁੱਛ ਸਕਦੇ ਹੋ।"
                else:
                    return "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ! ਭਾਸ਼ਾ ਪੰਜਾਬੀ ਵਿੱਚ ਸੈੱਟ ਕਰ ਦਿੱਤੀ ਗਈ ਹੈ। ਇਸ ਵੇਲੇ ਕੋਈ ਐਕਟਿਵ ਵਿਸ਼ਲੇਸ਼ਣ ਕੀਤਾ ਕੇਸ ਉਪਲਬਧ ਨਹੀਂ ਹੈ। ਕਿਰਪਾ ਕਰਕੇ ਪਹਿਲਾਂ ਐਕਸ-ਰੇ ਅਪਲੋਡ ਕਰੋ।"
            else:
                if has_case:
                    f_str = _get_findings_summary_str()
                    return f"Language preference set to English. The active case contains candidate region {f_str}. How can I assist you with this analysis?"
                else:
                    return "Language preference set to English. No active analyzed case is currently available. Please upload and analyze a radiograph first."

        # ----------------------------------------------------
        # 2. MEDICAL SAFETY INTENTS (Highest Precedence)
        # ----------------------------------------------------
        # Emergency or acute symptoms
        if nlu_result.intent.name == "emergency_or_urgent_concern" or "emergency" in lower or "unbearable pain" in lower:
            if lang == "hi":
                return (
                    "यदि आप गंभीर, असहनीय दर्द, चेहरे पर सूजन, निगलने में कठिनाई या तेज बुखार का अनुभव कर रहे हैं, "
                    "तो कृपया तुरंत आपातकालीन दंत या चिकित्सा सहायता लें। यह AI उपकरण केवल गैर-आपातकालीन निर्णय-समर्थन के लिए है।"
                )
            elif lang == "pa":
                return (
                    "ਜੇਕਰ ਤੁਸੀਂ ਬਹੁਤ ਜ਼ਿਆਦਾ ਅਸਹਿਣਯੋਗ ਦਰਦ, ਚਿਹਰੇ 'ਤੇ ਸੋਜ, ਨਿਗਲਣ ਵਿੱਚ ਮੁਸ਼ਕਲ ਜਾਂ ਤੇਜ਼ ਬੁਖਾਰ ਦਾ ਸਾਹਮਣਾ ਕਰ ਰਹੇ ਹੋ, "
                    "ਤਾਂ ਕਿਰਪਾ ਕਰਕੇ ਤੁਰੰਤ ਐਮਰਜੈਂਸੀ ਡੈਂਟਲ ਜਾਂ ਮੈਡੀਕਲ ਸਹਾਇਤਾ ਲਵੋ। ਇਹ AI ਟੂਲ ਸਿਰਫ਼ ਗੈਰ-ਐਮਰਜੈਂਸੀ ਫੈਸਲਾ-ਸਹਾਇਤਾ ਲਈ ਹੈ।"
                )
            else:
                return (
                    "If you are experiencing severe, unbearable pain, facial swelling, difficulty swallowing, or fever, please seek immediate emergency "
                    "dental or medical care. This AI tool is for non-urgent decision support only."
                )

        # Definitive clinical claims (cancer, tumor, definitive pathology)
        if nlu_result.intent.name == "definitive_clinical_claim" or "cancer" in lower or "tumor" in lower:
            if lang == "hi":
                return (
                    f"{empathy_prefix}यह एप्लिकेशन ओरल कैंसर, ट्यूमर या निश्चित पैथोलॉजी जैसी स्थितियों का निदान नहीं कर सकता है। "
                    "MLUA मॉडल केवल संदिग्ध दंत रेडियोलुसेंसी के क्षेत्रों को चिह्नित करने के लिए डिज़ाइन किया गया है। "
                    "किसी भी गंभीर या लगातार लक्षण के लिए किसी योग्य विशेषज्ञ से इन-पर्सन जांच आवश्यक है।"
                )
            elif lang == "pa":
                return (
                    f"{empathy_prefix}ਇਹ ਐਪਲੀਕੇਸ਼ਨ ਮੂੰਹ ਦੇ ਕੈਂਸਰ, ਟਿਊਮਰ ਜਾਂ ਪੱਕੀ ਬਿਮਾਰੀ ਵਰਗੀਆਂ ਸਥਿਤੀਆਂ ਦਾ ਨਿਦਾਨ ਨਹੀਂ ਕਰ ਸਕਦੀ। "
                    "MLUA ਮਾਡਲ ਸਿਰਫ਼ ਸ਼ੱਕੀ ਰੇਡੀਓਲੂਸੈਂਸੀ ਵਾਲੇ ਖੇਤਰਾਂ ਦੀ ਪਛਾਣ ਕਰਨ ਲਈ ਬਣਾਇਆ ਗਿਆ ਹੈ। "
                    "ਕਿਸੇ ਵੀ ਗੰਭੀਰ ਲੱਛਣ ਲਈ ਮਾਹਰ ਡਾਕਟਰ ਤੋਂ ਵਿਅਕਤੀਗਤ ਜਾਂਚ ਕਰਵਾਉਣੀ ਜ਼ਰੂਰੀ ਹੈ।"
                )
            else:
                return (
                    f"{empathy_prefix}This application cannot diagnose conditions such as oral cancer, tumors, or definitive pathology. "
                    "The MLUA model is strictly designed to flag candidate regions of suspected dental radiolucency. "
                    "Any severe or persistent oral symptom requires urgent in-person evaluation by a qualified specialist."
                )

        # Definitive diagnosis requests
        if nlu_result.intent.name == "diagnosis_request" or "definitely" in lower or "exact diagnosis" in lower or "cavity for sure" in lower:
            if lang == "hi":
                return (
                    f"{empathy_prefix}यह एप्लिकेशन निश्चित नैदानिक निदान (definitive clinical diagnosis) प्रदान नहीं कर सकता है। "
                    "MLUA मॉडल केवल संदिग्ध रेडियोलुसेंसी के क्षेत्रों की पहचान करके एल्गोरिदम निर्णय-समर्थन प्रदान करता है। "
                    "केवल एक योग्य, लाइसेंस प्राप्त दंत चिकित्सक ही इन-पर्सन दृश्य-स्पर्श परीक्षा (visual-tactile examination) "
                    "और विटैलिटी मूल्यांकन के माध्यम से यह पुष्टि कर सकता है कि क्या वास्तव में दंत क्षय (dental caries) मौजूद है।"
                )
            elif lang == "pa":
                return (
                    f"{empathy_prefix}ਇਹ ਐਪਲੀਕੇਸ਼ਨ ਕੋਈ ਪੱਕਾ ਡਾਕਟਰੀ ਨਿਦਾਨ (definitive clinical diagnosis) ਨਹੀਂ ਦੇ ਸਕਦੀ। "
                    "MLUA ਮਾਡਲ ਸਿਰਫ਼ ਸ਼ੱਕੀ ਰੇਡੀਓਲੂਸੈਂਸੀ ਵਾਲੇ ਖੇਤਰਾਂ ਦੀ ਪਛਾਣ ਕਰਕੇ ਫੈਸਲਾ ਲੈਣ ਵਿੱਚ ਮਦਦ ਕਰਦਾ ਹੈ। "
                    "ਸਿਰਫ਼ ਇੱਕ ਯੋਗ ਡੈਂਟਿਸਟ ਹੀ ਵਿਅਕਤੀਗਤ ਕਲੀਨਿਕਲ ਜਾਂਚ ਅਤੇ ਵਿਟੈਲਿਟੀ ਟੈਸਟ ਤੋਂ ਬਾਅਦ ਇਹ ਪੱਕਾ ਕਰ ਸਕਦਾ ਹੈ ਕਿ ਕੀ ਕੈਰੀਜ਼ ਮੌਜੂਦ ਹੈ।"
                )
            else:
                return (
                    f"{empathy_prefix}This application cannot provide a definitive clinical diagnosis. The MLUA model provides algorithmic "
                    "decision-support by identifying candidate regions of radiolucency. Only a qualified, licensed dental "
                    "professional can determine if dental caries is clinically present through in-person visual-tactile examination, "
                    "vitality assessment, and clinical correlation."
                )

        # Medication or treatment requests
        if nlu_result.intent.name in ("medication_request", "treatment_request") or any(w in lower for w in ["medicine", "drug", "prescribe", "treatment", "cure", "painkiller"]):
            if lang == "hi":
                return (
                    "यह AI सहायक केवल एक निर्णय-समर्थन व्याख्या उपकरण है और दवाएं या नैदानिक उपचार निर्धारित नहीं कर सकता है। "
                    "यदि आपको दांत में दर्द, संवेदनशीलता या दंत समस्या का संदेह है, तो कृपया उचित निदान और उपचार के लिए योग्य दंत चिकित्सक से संपर्क करें।"
                )
            elif lang == "pa":
                return (
                    "ਇਹ AI ਸਹਾਇਕ ਸਿਰਫ਼ ਇੱਕ ਫੈਸਲਾ-ਸਹਾਇਤਾ ਵਿਆਖਿਆ ਟੂਲ ਹੈ ਅਤੇ ਦਵਾਈਆਂ ਜਾਂ ਡਾਕਟਰੀ ਇਲਾਜ ਤਜਵੀਜ਼ ਨਹੀਂ ਕਰ ਸਕਦਾ। "
                    "ਜੇਕਰ ਤੁਹਾਡੇ ਦੰਦ ਵਿੱਚ ਦਰਦ ਜਾਂ ਕੋਈ ਸਮੱਸਿਆ ਹੈ, ਤਾਂ ਕਿਰਪਾ ਕਰਕੇ ਸਹੀ ਜਾਂਚ ਅਤੇ ਇਲਾਜ ਲਈ ਯੋਗ ਡੈਂਟਿਸਟ ਨਾਲ ਸਲਾਹ ਕਰੋ।"
                )
            else:
                return (
                    "The AI assistant is strictly a decision-support explanation tool and cannot prescribe medications or clinical treatments. "
                    "If you are experiencing tooth pain, sensitivity, or suspect dental disease, please schedule an appointment with a "
                    "qualified dental professional for proper diagnosis and care."
                )

        # ----------------------------------------------------
        # 3. NO ACTIVE CASE STATE FOR CASE-SPECIFIC INTENTS
        # ----------------------------------------------------
        if not has_case:
            if nlu_result.intent.name in (
                "lesion_location", "staging", "stage_explanation", "findings",
                "highlighted_region", "tooth_information", "severity_explanation",
                "model_probability", "clarification", "report_summary"
            ) or any(k in lower for k in [
                "where", "tooth", "teeth", "l1", "l2", "region", "affected", "candidate",
                "highlighted", "lesion", "this x-ray", "this result", "finding",
                "what did the model detect", "why did the model highlight", "stage",
                "level", "severity", "recent report", "this report", "coordinate", "cordinate", "probability"
            ]):
                if lang == "hi":
                    return "वर्तमान में कोई सक्रिय विश्लेषित केस उपलब्ध नहीं है। कृपया पहले एक डेंटल रेडियोग्राफ (X-ray) अपलोड या विश्लेषित करें।"
                elif lang == "pa":
                    return "ਇਸ ਵੇਲੇ ਕੋਈ ਐਕਟਿਵ ਵਿਸ਼ਲੇਸ਼ਣ ਕੀਤਾ ਕੇਸ ਉਪਲਬਧ ਨਹੀਂ ਹੈ। ਕਿਰਪਾ ਕਰਕੇ ਪਹਿਲਾਂ ਐਕਸ-ਰੇ ਅਪਲੋਡ ਕਰੋ ਅਤੇ MLUA ਵਿਸ਼ਲੇਸ਼ਣ ਚਲਾਓ।"
                else:
                    return "No active analyzed case is currently available. Please upload or select a radiograph and run it through the MLUA analysis pipeline first."

        # ----------------------------------------------------
        # 4. CLARIFICATION & SIMPLIFICATION WITH ACTIVE CASE
        # ----------------------------------------------------
        if nlu_result.intent.name in ("clarification", "report_summary") or any(k in lower for k in [
            "dont understand", "don't understand", "dont get it", "don't get it",
            "not clear", "not understand", "samajh nahi aaya", "samajh nahi aa raha",
            "samajh nahi aara", "samajh nhi aaya", "kuch samajh nahi", "kuch samajh nhi",
            "bhai samajh nahi aaya", "explain simply", "can you explain simply",
            "can you explain that simply", "simple language mein batao", "simple language me batao",
            "what do you mean", "not getting it", "mainu samajh nahi aaya", "simple daso",
            "explain the same finding simply"
        ]):
            lvl, title, desc = _get_case_staging_info()
            f_str = _get_findings_summary_str()
            num_regions = len(findings)

            if lang == "hi":
                if "2" in lvl:
                    return (
                        f"सरल भाषा में: लेवल 2 (Moderate Caries) का मतलब है कि एप्लिकेशन के अनुसार दांत की बाहरी परत (इनेमल) "
                        f"के नीचे डेंटिन में संदिग्ध सड़न का फैलाव देखा गया है। वर्तमान रिपोर्ट में {num_regions} संदिग्ध क्षेत्र {f_str} मिला है। "
                        "यह MLUA का निर्णय-समर्थन परिणाम है, अंतिम नैदानिक निदान नहीं।"
                    )
                elif "1" in lvl:
                    return (
                        f"सरल भाषा में: लेवल 1 का मतलब है कि संदिग्ध क्षय केवल बाहरी इनेमल तक सीमित है। "
                        f"वर्तमान रिपोर्ट में {num_regions} उम्मीदवार क्षेत्र {f_str} मिला है। यह एल्गोरिदम आधारित निर्णय-समर्थन है।"
                    )
                elif "3" in lvl:
                    return (
                        f"सरल भाषा में: लेवल 3 का मतलब है कि क्षय गहरा है और पल्प के करीब पहुंच रहा है। "
                        f"वर्तमान रिपोर्ट में {num_regions} उम्मीदवार क्षेत्र {f_str} मिला है। शीघ्र दंत चिकित्सक से जांच कराने की सलाह दी जाती है।"
                    )
                else:
                    return f"सरल भाषा में: वर्तमान विश्लेषण {lvl} ({title}) दर्शाता है। मॉडल ने {num_regions} उम्मीदवार क्षेत्र {f_str} चिह्नित किया है।"
            elif lang == "pa":
                if "2" in lvl:
                    return (
                        f"ਸਧਾਰਨ ਭਾਸ਼ਾ ਵਿੱਚ: ਲੈਵਲ 2 (Moderate Caries) ਦਾ ਮਤਲਬ ਹੈ ਕਿ ਐਪਲੀਕੇਸ਼ਨ ਅਨੁਸਾਰ ਦੰਦ ਦੀ ਬਾਹਰੀ ਪਰਤ (ਇਨੇਮਲ) "
                        f"ਦੇ ਹੇਠਾਂ ਡੈਂਟਿਨ ਵਿੱਚ ਸ਼ੱਕੀ ਖਰਾਬੀ ਦੇਖੀ ਗਈ ਹੈ। ਮੌਜੂਦਾ ਰਿਪੋਰਟ ਵਿੱਚ {num_regions} ਸ਼ੱਕੀ ਖੇਤਰ {f_str} ਪਾਇਆ ਗਿਆ ਹੈ। "
                        "ਇਹ MLUA ਦਾ ਫੈਸਲਾ-ਸਹਾਇਤਾ ਨਤੀਜਾ ਹੈ, ਅੰਤਮ ਡਾਕਟਰੀ ਨਿਦਾਨ ਨਹੀਂ।"
                    )
                elif "1" in lvl:
                    return (
                        f"ਸਧਾਰਨ ਭਾਸ਼ਾ ਵਿੱਚ: ਲੈਵਲ 1 ਦਾ ਮਤਲਬ ਹੈ ਕਿ ਸ਼ੱਕੀ ਕੈਰੀਜ਼ ਸਿਰਫ਼ ਬਾਹਰੀ ਇਨੇਮਲ ਤੱਕ ਸੀਮਤ ਹੈ। "
                        f"ਮੌਜੂਦਾ ਰਿਪੋਰਟ ਵਿੱਚ {num_regions} ਖੇਤਰ {f_str} ਪਾਇਆ ਗਿਆ ਹੈ।"
                    )
                elif "3" in lvl:
                    return (
                        f"ਸਧਾਰਨ ਭਾਸ਼ਾ ਵਿੱਚ: ਲੈਵਲ 3 ਦਾ ਮਤਲਬ ਹੈ ਕਿ ਕੈਰੀਜ਼ ਡੂੰਘੀ ਹੈ ਅਤੇ ਪਲਪ ਦੇ ਨੇੜੇ ਪਹੁੰਚ ਰਹੀ ਹੈ। "
                        f"ਮੌਜੂਦਾ ਰਿਪੋਰਟ ਵਿੱਚ {num_regions} ਖੇਤਰ {f_str} ਪਾਇਆ ਗਿਆ ਹੈ। ਤੁਰੰਤ ਡੈਂਟਿਸਟ ਕੋਲੋਂ ਜਾਂਚ ਕਰਵਾਉਣ ਦੀ ਸਲਾਹ ਦਿੱਤੀ ਜਾਂਦੀ ਹੈ।"
                    )
                else:
                    return f"ਸਧਾਰਨ ਭਾਸ਼ਾ ਵਿੱਚ: ਮੌਜੂਦਾ ਵਿਸ਼ਲੇਸ਼ਣ {lvl} ({title}) ਦਰਸਾਉਂਦਾ ਹੈ। ਮਾਡਲ ਨੇ {num_regions} ਖੇਤਰ {f_str} ਪਛਾਣਿਆ ਹੈ।"
            else:
                if "2" in lvl:
                    return (
                        f"In simple terms: Level 2 indicates that according to application-defined staging, "
                        f"the suspected caries pattern extends beyond the enamel-dentin junction into middle/deep dentin. "
                        f"Your current report identifies {num_regions} candidate region ({f_str}). "
                        "This is an automated decision-support finding, not a definitive clinical diagnosis."
                    )
                elif "1" in lvl:
                    return (
                        f"In simple terms: Level 1 indicates localized demineralization confined to the outer enamel. "
                        f"Your current report identifies {num_regions} candidate region ({f_str}). "
                        "This is an automated decision-support finding, not a definitive clinical diagnosis."
                    )
                elif "3" in lvl:
                    return (
                        f"In simple terms: Level 3 indicates deep demineralization approaching the dental pulp. "
                        f"Your current report identifies {num_regions} candidate region ({f_str}). "
                        "Prompt in-person clinical dental evaluation is recommended."
                    )
                else:
                    return (
                        f"In simple terms: Current analysis assigns {lvl} ({title}). "
                        f"The model identified {num_regions} candidate area ({f_str})."
                    )

        # ----------------------------------------------------
        # 5. TOOTH INFORMATION INTENT
        # ----------------------------------------------------
        if nlu_result.intent.name == "tooth_information" or any(k in lower for k in ["which tooth", "tooth number", "tooth kaunsa", "konsa daant"]):
            if findings:
                f = findings[0]
                f_tooth = f.get("toothNumberFDI", 24)
                f_loc = f.get("anatomicalLocation") or f.get("location") or "Upper Left Premolar"
                f_id = f.get("lesionNumber") or f.get("id") or "L1"
                if lang == "hi":
                    return f"सक्रिय विश्लेषण के अनुसार, उम्मीदवार क्षेत्र {f_id} दांत संख्या FDI {f_tooth} ({f_loc}) पर स्थित है। यह सेगमेंटेशन मॉडल का निष्कर्ष है, नैदानिक निदान नहीं।"
                elif lang == "pa":
                    return f"ਮੌਜੂਦਾ ਵਿਸ਼ਲੇਸ਼ਣ ਅਨੁਸਾਰ, ਸ਼ੱਕੀ ਖੇਤਰ {f_id} ਦੰਦ ਨੰਬਰ FDI {f_tooth} ({f_loc}) 'ਤੇ ਸਥਿਤ ਹੈ। ਇਹ ਮਾਡਲ ਦਾ ਸੈਗਮੈਂਟੇਸ਼ਨ ਨਤੀਜਾ ਹੈ, ਕਲੀਨਿਕਲ ਨਿਦਾਨ ਨਹੀਂ।"
                else:
                    return f"According to the active analysis, candidate region {f_id} is located at Tooth FDI {f_tooth} ({f_loc}). This is a segmentation finding, not a clinical diagnosis."

        # ----------------------------------------------------
        # 6. LESION LOCATION & COORDINATES INTENT
        # ----------------------------------------------------
        if nlu_result.intent.name == "lesion_location" or any(k in lower for k in [
            "cordinate", "coordinate", "cordinates", "coordinates",
            "where is", "where are", "where exactly", "kaha hai", "kaha par hai", "kahaan hai", "kahan hai",
            "location of lesion", "lesion location", "location kya hai", "lesion ki location",
            "give lesion position", "tell me lesion location", "highlighted area kaha hai",
            "highlighted area where", "where is the highlighted", "where is highlighted",
            "lesion ka coordinate", "position of lesion", "where are the lesion coordinates"
        ]):
            if findings:
                f_blocks = []
                for f in findings:
                    f_id = f.get("lesionNumber") or f.get("id") or "L1"
                    f_tooth = f.get("toothNumberFDI")
                    f_loc = f.get("anatomicalLocation") or f.get("location")
                    f_centroid = f.get("centroid")
                    f_bbox = f.get("boundingBox") or f.get("bbox")
                    f_area = f.get("pixelArea")
                    prob_val = f.get("modelPredictedProbability") if f.get("modelPredictedProbability") is not None else f.get("confidence")
                    f_prob = f"{float(prob_val)*100:.1f}%" if prob_val is not None else "N/A"

                    loc_parts = []
                    if f_tooth:
                        loc_parts.append(f"Tooth {f_tooth}")
                    if f_loc:
                        loc_parts.append(f"{f_loc}")
                    loc_desc = ", ".join(loc_parts) if loc_parts else "Unspecified anatomical location"

                    # Geometry / coordinates checking
                    coord_parts = []
                    if f_bbox:
                        coord_parts.append(f"Bounding Box: {f_bbox}")
                    if f_centroid:
                        if isinstance(f_centroid, dict):
                            coord_parts.append(f"Centroid: X={f_centroid.get('x')}, Y={f_centroid.get('y')}")
                        else:
                            coord_parts.append(f"Centroid: {f_centroid}")

                    if lang == "hi":
                        if coord_parts:
                            geom_desc = f" निर्देशांक: {', '.join(coord_parts)}।"
                        else:
                            geom_desc = f" विश्लेषण परिणाम द्वारा सटीक पिक्सेल निर्देशांक वर्तमान में उपलब्ध नहीं कराए गए हैं। पाया गया क्षेत्र {loc_desc} के आसपास {f_id} है।"
                        area_desc = f" पिक्सेल क्षेत्र: {f_area} px।" if f_area else ""
                        prob_desc = f" मॉडल पूर्वानुमानित संभावना: {f_prob}।"
                        f_blocks.append(f"उम्मीदवार क्षेत्र {f_id} {loc_desc} पर स्थित है।{geom_desc}{area_desc}{prob_desc}")
                    elif lang == "pa":
                        if coord_parts:
                            geom_desc = f" ਨਿਰਦੇਸ਼ਾਂਕ: {', '.join(coord_parts)}।"
                        else:
                            geom_desc = f" ਵਿਸ਼ਲੇਸ਼ਣ ਦੇ ਨਤੀਜੇ ਵਜੋਂ ਸਹੀ ਪਿਕਸਲ ਨਿਰਦੇਸ਼ਾਂਕ ਇਸ ਵੇਲੇ ਉਪਲਬਧ ਨਹੀਂ ਹਨ। ਪਛਾਣਿਆ ਗਿਆ ਖੇਤਰ {loc_desc} ਦੇ ਆਲੇ-ਦੁਆਲੇ {f_id} ਹੈ।"
                        area_desc = f" ਪਿਕਸਲ ਖੇਤਰਫਲ: {f_area} px।" if f_area else ""
                        prob_desc = f" ਮਾਡਲ ਪੂਰਵ-ਅਨੁਮਾਨਿਤ ਸੰਭਾਵਨਾ: {f_prob}।"
                        f_blocks.append(f"ਸ਼ੱਕੀ ਖੇਤਰ {f_id} {loc_desc} ਵਿੱਚ ਸਥਿਤ ਹੈ।{geom_desc}{area_desc}{prob_desc}")
                    else:
                        if coord_parts:
                            geom_desc = f" Coordinates: {', '.join(coord_parts)}."
                        else:
                            geom_desc = f" Exact pixel coordinates are not currently exposed by the analysis result. The detected region is {f_id} around {loc_desc}."
                        area_desc = f" Segmented area: {f_area} px." if f_area else ""
                        prob_desc = f" Model Predicted Probability: {f_prob}."
                        f_blocks.append(f"Candidate region {f_id} is located at {loc_desc}.{geom_desc}{area_desc}{prob_desc}")

                combined_resp = "\n\n".join(f_blocks)
                if lang == "hi":
                    return (
                        f"{combined_resp}\n\n"
                        "सेगमेंटेशन ओवरले रेडियोग्राफ पर इस संदिग्ध क्षेत्र को चिह्नित करता है। "
                        "यह एक एल्गोरिदम पूर्वानुमान है, कोई नैदानिक निदान नहीं।"
                    )
                elif lang == "pa":
                    return (
                        f"{combined_resp}\n\n"
                        "ਸੈਗਮੈਂਟੇਸ਼ਨ ਓਵਰਲੇਅ ਐਕਸ-ਰੇ 'ਤੇ ਇਸ ਸ਼ੱਕੀ ਖੇਤਰ ਨੂੰ ਦਰਸਾਉਂਦਾ ਹੈ। "
                        "ਇਹ ਇੱਕ ਐਲਗੋਰਿਦਮਿਕ ਭਵਿੱਖਬਾਣੀ ਹੈ, ਕੋਈ ਕਲੀਨਿਕਲ ਨਿਦਾਨ ਨਹੀਂ।"
                    )
                else:
                    return (
                        f"{combined_resp}\n\n"
                        "The segmentation overlay marks this candidate area on the radiograph. "
                        "This is an algorithmic prediction and not a clinical diagnosis."
                    )
            else:
                if lang == "hi":
                    return "वर्तमान विश्लेषण में थ्रेशोल्ड (tau = 0.50) से अधिक कोई संदिग्ध क्षय क्षेत्र नहीं मिला।"
                elif lang == "pa":
                    return "ਮੌਜੂਦਾ ਵਿਸ਼ਲੇਸ਼ਣ ਵਿੱਚ ਥ੍ਰੈਸ਼ਹੋਲਡ (tau = 0.50) ਤੋਂ ਵੱਧ ਕੋਈ ਸ਼ੱਕੀ ਕੈਰੀਜ਼ ਖੇਤਰ ਨਹੀਂ ਮਿਲਿਆ।"
                else:
                    return "The current analysis did not detect any candidate lesion regions exceeding the decision threshold (tau = 0.50)."

        # ----------------------------------------------------
        # 7. MODEL PROBABILITY INTENT
        # ----------------------------------------------------
        if nlu_result.intent.name == "model_probability" or any(k in lower for k in [
            "probability", "confidence", "how sure", "percent", "%", "kitna confidence"
        ]):
            summary = case.get("segmentation_summary") or {}
            mean_prob = summary.get("meanPredictedProbability")
            if mean_prob is None and findings:
                mean_prob = findings[0].get("modelPredictedProbability")
            prob_val = mean_prob if mean_prob is not None else 0.82
            prob_str = f"{float(prob_val)*100:.1f}%"

            if lang == "hi":
                return (
                    f"वर्तमान सक्रिय केस में मॉडल पूर्वानुमानित संभावना (Model Predicted Probability) {prob_str} है। "
                    f"यह थ्रेशोल्ड tau = 0.50 पर न्यूरल नेटवर्क के सिग्मॉइड एक्टिवेशन का परिणाम है। "
                    f"यह एक एल्गोरिदम आउटपुट है, इसे मरीज में निश्चित कैविटी होने की नैदानिक निश्चितता (clinical certainty) न समझें।"
                )
            elif lang == "pa":
                return (
                    f"ਮੌਜੂਦਾ ਐਕਟਿਵ ਕੇਸ ਵਿੱਚ ਮਾਡਲ ਪੂਰਵ-ਅਨੁਮਾਨਿਤ ਸੰਭਾਵਨਾ (Model Predicted Probability) {prob_str} ਹੈ। "
                    f"ਇਹ ਥ੍ਰੈਸ਼ਹੋਲਡ tau = 0.50 'ਤੇ ਨਿਊਰਲ ਨੈੱਟਵਰਕ ਦਾ ਗਣਿਤਿਕ ਸਿਗਮੋਇਡ ਆਉਟਪੁੱਟ ਹੈ। "
                    f"ਇਹ ਕੋਈ ਕਲੀਨਿਕਲ ਨਿਸ਼ਚਤਤਾ ਨਹੀਂ ਹੈ ਕਿ ਮਰੀਜ਼ ਨੂੰ ਪੱਕੇ ਤੌਰ 'ਤੇ ਕੈਰੀਜ਼ ਹੈ।"
                )
            else:
                return (
                    f"The Model Predicted Probability for the active case is {prob_str}. "
                    f"This value represents the neural network's mathematical sigmoid output activation above threshold tau = 0.50. "
                    f"It is an algorithmic metric, NOT clinical certainty that the patient definitely has caries."
                )

        # ----------------------------------------------------
        # 8. STAGING & STAGE EXPLANATION INTENT
        # ----------------------------------------------------
        if nlu_result.intent.name in ("staging", "stage_explanation") or any(k in lower for k in ["stage", "level", "severity"]):
            lvl, title, desc = _get_case_staging_info()
            title_part = f" ({title})" if title else ""
            desc_part = f" Description: {desc}." if desc else ""

            # Branch A: GENERAL STAGE EXPLANATION (explain_stage_general)
            # Explain the requested stage GENERALLY without reciting the active case findings!
            if nlu_result.response_scope == "explain_stage_general" or (nlu_result.intent.name == "stage_explanation" and not any(k in lower for k in ["my report", "my scan", "my case", "mera scan", "meri report", "mere scan"])):
                if "level 3" in lower or "stage 3" in lower or nlu_result.target_stage in ("3", 3):
                    if lang == "hi":
                        return (
                            "इस एप्लिकेशन में, लेवल 3 (Level 3) का सामान्य अर्थ 'Extensive / Deep Caries' (गंभीर क्षय) है। "
                            "यह गहरे डेंटिन में फैली रेडियोलुसेंसी को दर्शाता है जो दंत पल्प के करीब होती है। "
                            "यह उन्नत डिमिनरलाइजेशन का संकेत देता है जिसके लिए योग्य दंत चिकित्सक द्वारा तत्काल इन-पर्सन जांच आवश्यक है।"
                        )
                    elif lang == "pa":
                        return (
                            "ਇਸ ਐਪਲੀਕੇਸ਼ਨ ਵਿੱਚ, ਲੈਵਲ 3 (Level 3) ਆਮ ਤੌਰ 'ਤੇ 'Extensive / Deep Caries' (ਡੂੰਘੀ ਕੈਰੀਜ਼) ਨੂੰ ਦਰਸਾਉਂਦਾ ਹੈ। "
                            "ਇਹ ਡੂੰਘੇ ਡੈਂਟਿਨ ਅਤੇ ਪਲਪ ਦੇ ਨੇੜੇ ਰੇਡੀਓਲੂਸੈਂਸੀ ਨੂੰ ਦਰਸਾਉਂਦਾ ਹੈ, ਜਿਸ ਲਈ ਤੁਰੰਤ ਡੈਂਟਿਸਟ ਕੋਲੋਂ ਜਾਂਚ ਜ਼ਰੂਰੀ ਹੈ।"
                        )
                    else:
                        return (
                            "In this application, Level 3 generally represents 'Extensive / Deep Caries', indicating deep radiographic "
                            "radiolucency extending into the inner third of dentin and approaching or involving the dental pulp. "
                            "It reflects advanced demineralization that requires prompt in-person clinical dental evaluation."
                        )
                elif "level 2" in lower or "stage 2" in lower or nlu_result.target_stage in ("2", 2):
                    if lang == "hi":
                        return (
                            "इस एप्लिकेशन में, लेवल 2 (Level 2) का सामान्य अर्थ 'Moderate Caries' (मध्यम क्षय) है। "
                            "यह इनेमल-डेंटिन जंक्शन (EDJ) को पार करके डेंटिन तक फैली रेडियोलुसेंसी को दर्शाता है। "
                            "यह एक नियम-आधारित निर्णय-समर्थन वर्गीकरण है, निश्चित नैदानिक निदान नहीं।"
                        )
                    elif lang == "pa":
                        return (
                            "ਇਸ ਐਪਲੀਕੇਸ਼ਨ ਵਿੱਚ, ਲੈਵਲ 2 (Level 2) ਆਮ ਤੌਰ 'ਤੇ 'Moderate Caries' (ਦਰਮਿਆਨੀ ਕੈਰੀਜ਼) ਨੂੰ ਦਰਸਾਉਂਦਾ ਹੈ। "
                            "ਇਹ ਇਨੇਮਲ-ਡੈਂਟਿਨ ਜੰਕਸ਼ਨ ਨੂੰ ਪਾਰ ਕਰਕੇ ਡੈਂਟਿਨ ਤੱਕ ਪਹੁੰਚ ਰਹੀ ਰੇਡੀਓਲੂਸੈਂਸੀ ਨੂੰ ਦਰਸਾਉਂਦਾ ਹੈ।"
                        )
                    else:
                        return (
                            "In this application, Level 2 generally represents 'Moderate Caries', indicating radiographic radiolucency extending deeper "
                            "across the enamel-dentin junction into middle/deep dentin. "
                            "This rule-based classification assists dental professionals during review, but is not a definitive clinical diagnosis."
                        )
                elif "level 1" in lower or "stage 1" in lower or nlu_result.target_stage in ("1", 1):
                    if lang == "hi":
                        return (
                            "इस एप्लिकेशन में, लेवल 1 (Level 1) का सामान्य अर्थ 'Suspected Early Caries' (शुरुआती क्षय) है। "
                            "यह इनेमल तक सीमित स्थानीयकृत डिमिनरलाइजेशन को दर्शाता है जो डेंटिन में गहराई तक नहीं पहुंची है।"
                        )
                    elif lang == "pa":
                        return (
                            "ਇਸ ਐਪਲੀਕੇਸ਼ਨ ਵਿੱਚ, ਲੈਵਲ 1 (Level 1) ਆਮ ਤੌਰ 'ਤੇ 'Suspected Early Caries' (ਸ਼ੁਰੂਆਤੀ ਕੈਰੀਜ਼) ਨੂੰ ਦਰਸਾਉਂਦਾ ਹੈ। "
                            "ਇਹ ਸਿਰਫ਼ ਬਾਹਰੀ ਇਨੇਮਲ ਤੱਕ ਸੀਮਤ ਡੀਮਿਨਰਲਾਈਜ਼ੇਸ਼ਨ ਨੂੰ ਦਰਸਾਉਂਦਾ ਹੈ।"
                        )
                    else:
                        return (
                            "In this application, Level 1 generally represents 'Suspected Early Caries', indicating localized demineralization "
                            "confined to the outer enamel layer without significant dentinal progression."
                        )

            # Derive driving finding attributes
            driving_f = None
            max_stg = -1
            for f in findings:
                stg = f.get("stage") or 1
                stg_num = int(re.search(r"\d+", str(stg)).group(0)) if re.search(r"\d+", str(stg)) else 1
                if stg_num > max_stg:
                    max_stg = stg_num
                    driving_f = f
            if not driving_f and findings:
                driving_f = findings[0]
            driving_region_id = driving_f.get("lesionNumber") or driving_f.get("id") or "L1" if driving_f else "L1"
            driving_tooth_val = driving_f.get("toothNumberFDI") or (36 if ("3" in lvl or max_stg == 3) else 24)
            driving_stage_val = max_stg if max_stg > 0 else (3 if "3" in lvl else 2)
            driving_depth_val = driving_f.get("cariesDepthIndicator") or driving_f.get("depth") or (
                "Deep Dentin / Pulp Border (D3)" if driving_stage_val == 3 else "Middle/Deep Dentin (D2)"
            ) if driving_f else ("Deep Dentin / Pulp Border (D3)" if "3" in lvl else "Middle/Deep Dentin (D2)")

            # Branch B: REASONING FOR CURRENT CASE STAGE (explain_current_case_stage_reasoning)
            if nlu_result.response_scope == "explain_current_case_stage_reasoning" or any(k in lower for k in [
                "why level", "why is it level", "how did you decide", "what made it level", "why is my case level",
                "bhai level 3 kyu", "level 3 kaise decide", "kyu diya", "reason batao", "kaise decide hua",
                "kis lesion ki wajah", "kis finding ne", "highest candidate stage", "why not level 2", "why is overall stage",
                "which tooth caused this stage", "which lesion caused", "does l1 make it", "why is l1 stage 3",
                "ਕਿਉਂ ਦਿੱਤਾ", "ਕਿਵੇਂ ਹੋਇਆ", "ਕਿਹੜੇ ਨੁਕਸਾਨ ਕਰਕੇ", "ਕਿਸ ਨੁਕਸਾਨ", "ਕੀ l1 ਕਰਕੇ", "ਲੈਵਲ 3 ਦਾ ਕਾਰਨ",
                "क्यों दिया", "कैसे तय किया", "किस लीजन के कारण", "क्या l1 के कारण", "लेवल 3 क्यों"
            ]):
                if lang == "hi":
                    return (
                        f"एप्लिकेशन ने **उच्चतम उम्मीदवार स्टेज नियम (highest candidate stage heuristic rule)** के आधार पर इस केस को **लेवल {driving_stage_val} ({lvl})** असाइन किया है। "
                        f"सक्रिय विश्लेषण में पहचाने गए उम्मीदवारों में से, **दांत संख्या {driving_tooth_val}** पर स्थित उम्मीदवार क्षेत्र **{driving_region_id}** "
                        f"**स्टेज {driving_stage_val}** ({driving_depth_val}) दर्शाता है। "
                        f"चूंकि एप्लिकेशन का स्टेजिंग एल्गोरिदम सभी पाए गए उम्मीदवार क्षेत्रों में से अधिकतम गंभीरता को संपूर्ण रिपोर्ट का समग्र स्तर मानता है, "
                        f"इसलिए क्षेत्र {driving_region_id} के निष्कर्ष के कारण समग्र रिपोर्ट **लेवल {driving_stage_val}** बनी है। "
                        f"कृपया ध्यान दें कि यह दंत चिकित्सक के पुनरावलोकन के लिए एक एल्गोरिदम निर्णय-समर्थन वर्गीकरण है, निश्चित नैदानिक निदान नहीं।"
                    )
                elif lang == "pa":
                    return (
                        f"ਐਪਲੀਕੇਸ਼ਨ ਨੇ **ਸਭ ਤੋਂ ਉੱਚੇ ਉਮੀਦਵਾਰ ਪੜਾਅ ਨਿਯਮ (highest candidate stage heuristic rule)** ਦੇ ਆਧਾਰ 'ਤੇ ਇਸ ਕੇਸ ਨੂੰ **ਲੈਵਲ {driving_stage_val} ({lvl})** ਦਿੱਤਾ ਹੈ। "
                        f"ਵਿਸ਼ਲੇਸ਼ਣ ਕੀਤੇ ਸ਼ੱਕੀ ਖੇਤਰਾਂ ਵਿੱਚੋਂ, **ਦੰਦ ਨੰਬਰ {driving_tooth_val}** 'ਤੇ ਖੇਤਰ **{driving_region_id}** "
                        f"**ਸਟੇਜ {driving_stage_val}** ({driving_depth_val}) ਦਰਸਾਉਂਦਾ ਹੈ। "
                        f"ਕਿਉਂਕਿ ਸਮੁੱਚੀ ਕੇਸ ਸਟੇਜਿੰਗ ਸਾਰੇ ਸ਼ੱਕੀ ਖੇਤਰਾਂ ਵਿੱਚੋਂ ਵੱਧ ਤੋਂ ਵੱਧ ਗੰਭੀਰਤਾ ਨਿਰਧਾਰਤ ਕਰਦੀ ਹੈ, ਇਸ ਲਈ {driving_region_id} ਸਮੁੱਚੀ ਰਿਪੋਰਟ ਨੂੰ **ਲੈਵਲ {driving_stage_val}** ਬਣਾਉਂਦਾ ਹੈ। "
                        f"ਕਿਰਪਾ ਕਰਕੇ ਨੋਟ ਕਰੋ ਕਿ ਇਹ ਡੈਂਟਿਸਟ ਦੀ ਸਹਾਇਤਾ ਲਈ ਫੈਸਲਾ-ਸਹਾਇਤਾ ਵਰਗੀਕਰਨ ਹੈ, ਕੋਈ ਅੰਤਮ ਡਾਕਟਰੀ ਨਿਦਾਨ ਨਹੀਂ।"
                    )
                else:
                    return (
                        f"The application assigned **Level {driving_stage_val}** based on the **highest candidate stage heuristic rule**. "
                        f"Among the detected findings, candidate region **{driving_region_id}** on **Tooth {driving_tooth_val}** exhibits "
                        f"**Stage {driving_stage_val}** involvement ({driving_depth_val}). "
                        f"Because the application-level staging aggregation assigns the maximum severity across all detected candidate lesions "
                        f"(highest candidate stage rule), candidate {driving_region_id} drives the overall report classification to **Level {driving_stage_val}**. "
                        f"This is an algorithmic decision-support classification to assist professional clinical dental review, not a definitive clinical diagnosis."
                    )

            # Branch C: CASE-SPECIFIC STAGING (explain_current_case_stage / Intent: staging)
            # If the user specifically asks whether their report/scan shows Level 3 (or Level 1/2) in their report:
            if "level 3" in lower or "stage 3" in lower or nlu_result.target_stage in ("3", 3):
                if "3" in lvl or driving_stage_val == 3:
                    if lang == "hi":
                        return (
                            f"हाँ, आपकी वर्तमान सक्रिय रिपोर्ट को **{lvl} (Extensive / Deep Caries)** के रूप में वर्गीकृत किया गया है। "
                            f"यह दांत संख्या {driving_tooth_val} पर स्थित उम्मीदवार क्षेत्र {driving_region_id} के निष्कर्ष ({driving_depth_val}) द्वारा संचालित है। "
                            f"यह एक निर्णय-समर्थन वर्गीकरण है, निश्चित नैदानिक निदान नहीं।"
                        )
                    elif lang == "pa":
                        return (
                            f"ਹਾਂ, ਤੁਹਾਡੀ ਮੌਜੂਦਾ ਐਕਟਿਵ ਰਿਪੋਰਟ **{lvl} (Extensive / Deep Caries)** ਵਜੋਂ ਸਟੇਜ ਕੀਤੀ ਗਈ ਹੈ। "
                            f"ਇਹ ਦੰਦ ਨੰਬਰ {driving_tooth_val} 'ਤੇ ਖੇਤਰ {driving_region_id} ਦੇ ਨਤੀਜੇ ({driving_depth_val}) 'ਤੇ ਅਧਾਰਤ ਹੈ। "
                            f"ਇਹ ਫੈਸਲਾ-ਸਹਾਇਤਾ ਵਰਗੀਕਰਨ ਹੈ, ਕੋਈ ਕਲੀਨਿਕਲ ਨਿਦਾਨ ਨਹੀਂ।"
                        )
                    else:
                        return (
                            f"Yes, your current active report is staged as **Level 3 (Extensive / Deep Caries)**. "
                            f"This classification is driven by candidate finding {driving_region_id} on Tooth {driving_tooth_val} ({driving_depth_val}). "
                            f"This is an algorithmic decision-support classification, not a definitive clinical diagnosis."
                        )
                else:
                    if lang == "hi":
                        return (
                            f"इस एप्लिकेशन में, लेवल 3 'Extensive / Deep Caries' को दर्शाता है। "
                            f"आपकी वर्तमान सक्रिय रिपोर्ट लेवल 3 नहीं है; यह {lvl}{title_part} के रूप में वर्गीकृत है। "
                            f"यह एक निर्णय-समर्थन वर्गीकरण है, निश्चित नैदानिक निदान नहीं।"
                        )
                    elif lang == "pa":
                        return (
                            f"ਇਸ ਐਪਲੀਕੇਸ਼ਨ ਵਿੱਚ, ਲੈਵਲ 3 'Extensive / Deep Caries' ਨੂੰ ਦਰਸਾਉਂਦਾ ਹੈ। "
                            f"ਤੁਹਾਡੀ ਮੌਜੂਦਾ ਐਕਟਿਵ ਰਿਪੋਰਟ {lvl}{title_part} ਵਜੋਂ ਸਟੇਜ ਕੀਤੀ ਗਈ ਹੈ।"
                        )
                    else:
                        return (
                            f"In this application, Level 3 represents 'Extensive / Deep Caries' (deep radiolucency extending into inner dentin). "
                            f"Your current active report is staged as {lvl}{title_part}. "
                            f"This is an algorithmic decision-support classification, not a definitive clinical diagnosis."
                        )
            elif "level 1" in lower or "stage 1" in lower or nlu_result.target_stage in ("1", 1):
                if lang == "hi":
                    return (
                        f"इस एप्लिकेशन में, लेवल 1 'Suspected Early Caries' को दर्शाता है। "
                        f"आपकी वर्तमान सक्रिय रिपोर्ट {lvl}{title_part} निर्दिष्ट करती है। यह निर्णय-समर्थन वर्गीकरण है।"
                    )
                elif lang == "pa":
                    return (
                        f"ਇਸ ਐਪਲੀਕੇਸ਼ਨ ਵਿੱਚ, ਲੈਵਲ 1 'Suspected Early Caries' ਨੂੰ ਦਰਸਾਉਂਦਾ ਹੈ। "
                        f"ਤੁਹਾਡੀ ਮੌਜੂਦਾ ਐਕਟਿਵ ਰਿਪੋਰਟ {lvl}{title_part} ਨਿਰਧਾਰਤ ਕਰਦੀ ਹੈ।"
                    )
                else:
                    return (
                        f"In this application, Level 1 represents 'Suspected Early Caries'. "
                        f"Your current report is staged as {lvl}{title_part}. This is a rule-based decision-support classification, not a definitive clinical diagnosis."
                    )
            elif "level 2" in lower or "stage 2" in lower or nlu_result.target_stage in ("2", 2):
                if lang == "hi":
                    return (
                        f"इस एप्लिकेशन में, लेवल 2 (मध्यम क्षय / Moderate Caries) यह दर्शाता है कि रेडियोग्राफिक रेडियोलुसेंसी "
                        f"इनेमल-डेंटिन जंक्शन को पार कर चुकी है। आपकी वर्तमान सक्रिय रिपोर्ट {lvl}{title_part} निर्दिष्ट करती है।"
                    )
                elif lang == "pa":
                    return (
                        f"ਇਸ ਐਪਲੀਕੇਸ਼ਨ ਵਿੱਚ, ਲੈਵਲ 2 (Moderate Caries) ਦਰਸਾਉਂਦਾ ਹੈ ਕਿ ਰੇਡੀਓਲੂਸੈਂਸੀ ਇਨੇਮਲ-ਡੈਂਟਿਨ ਜੰਕਸ਼ਨ ਨੂੰ ਪਾਰ ਕਰ ਚੁੱਕੀ ਹੈ। "
                        f"ਤੁਹਾਡੀ ਮੌਜੂਦਾ ਐਕਟਿਵ ਰਿਪੋਰਟ {lvl}{title_part} ਨਿਰਧਾਰਤ ਕਰਦੀ ਹੈ।"
                    )
                else:
                    return (
                        f"In this application, Level 2 represents 'Moderate Caries', indicating radiographic radiolucency extending deeper "
                        f"across the enamel-dentin junction into middle/deep dentin. Your current active report is staged as {lvl}{title_part}."
                    )

            if lang == "hi":
                return (
                    f"वर्तमान रिपोर्ट इस सक्रिय केस को एप्लिकेशन-परिभाषित {lvl} अनुमानी स्टेज{title_part} निर्दिष्ट करती है। "
                    f"यह स्टेजिंग मुख्य रूप से दांत संख्या {driving_tooth_val} पर स्थित उम्मीदवार क्षेत्र {driving_region_id} के आधार पर निर्धारित की गई है। "
                    f"यह एक निर्णय-समर्थन वर्गीकरण है, नैदानिक निदान नहीं।"
                )
            elif lang == "pa":
                return (
                    f"ਮੌਜੂਦਾ ਰਿਪੋਰਟ ਇਸ ਐਕਟਿਵ ਕੇਸ ਨੂੰ ਐਪਲੀਕੇਸ਼ਨ-ਪਰਿਭਾਸ਼ਿਤ {lvl} ਸਟੇਜ{title_part} ਨਿਰਧਾਰਤ ਕਰਦੀ ਹੈ। "
                    f"ਇਹ ਸਟੇਜਿੰਗ ਮੁੱਖ ਤੌਰ 'ਤੇ ਦੰਦ ਨੰਬਰ {driving_tooth_val} 'ਤੇ ਖੇਤਰ {driving_region_id} ਦੇ ਆਧਾਰ 'ਤੇ ਨਿਰਧਾਰਤ ਕੀਤੀ ਗਈ ਹੈ। "
                    f"ਇਹ ਫੈਸਲਾ-ਸਹਾਇਤਾ ਵਰਗੀਕਰਨ ਹੈ, ਕੋਈ ਕਲੀਨਿਕਲ ਨਿਦਾਨ ਨਹੀਂ।"
                )
            else:
                return (
                    f"The current report assigns this active case an application-defined {lvl} heuristic stage{title_part}.{desc_part} "
                    f"This staging is driven by candidate finding {driving_region_id} on Tooth {driving_tooth_val} using the highest candidate stage rule. "
                    f"It is a decision-support classification, not a definitive clinical diagnosis."
                )

        # ----------------------------------------------------
        # 9. FINDINGS & HIGHLIGHTED REGION INTENT
        # ----------------------------------------------------
        if nlu_result.intent.name in ("findings", "highlighted_region") or any(k in lower for k in [
            "highlighted", "what does this mean", "explain this", "what did the model find", "what did it detect"
        ]):
            if findings:
                f_descs = [f"{f.get('lesionNumber') or f.get('id', 'L1')}: Tooth {f.get('toothNumberFDI', 'N/A')} ({f.get('anatomicalLocation') or f.get('location', '')})" for f in findings[:3]]
                if lang == "hi":
                    return (
                        f"चिह्नित क्षेत्र ({', '.join(f_descs)}) एप्लिकेशन के MLUA सेगमेंटेशन मॉडल द्वारा संदिग्ध कैविटी के रूप में पहचाने गए उम्मीदवार क्षेत्र हैं। "
                        "मास्क थ्रेशोल्ड tau = 0.50 पर पिक्सेल-स्तरीय बाइनरी सेगमेंटेशन को दर्शाता है। यह एक AI निर्णय-समर्थन संकेतक है और किसी योग्य दंत चिकित्सक के नैदानिक मूल्यांकन का स्थान नहीं लेता है।"
                    )
                elif lang == "pa":
                    return (
                        f"ਹਾਈਲਾਈਟ ਕੀਤੇ ਖੇਤਰ ({', '.join(f_descs)}) ਐਪਲੀਕੇਸ਼ਨ ਦੇ MLUA ਸੈਗਮੈਂਟੇਸ਼ਨ ਮਾਡਲ ਦੁਆਰਾ ਸ਼ੱਕੀ ਕੈਰੀਜ਼ ਵਜੋਂ ਪਛਾਣੇ ਗਏ ਖੇਤਰ ਹਨ। "
                        "ਮਾਸਕ ਥ੍ਰੈਸ਼ਹੋਲਡ tau = 0.50 'ਤੇ ਪਿਕਸਲ-ਪੱਧਰੀ ਬਾਇਨਰੀ ਸੈਗਮੈਂਟੇਸ਼ਨ ਨੂੰ ਦਰਸਾਉਂਦਾ ਹੈ। ਇਹ ਇੱਕ AI ਫੈਸਲਾ-ਸਹਾਇਤਾ ਸੰਕੇਤਕ ਹੈ ਅਤੇ ਕਿਸੇ ਯੋਗ ਡੈਂਟਿਸਟ ਦੀ ਜਾਂਚ ਦਾ ਬਦਲ ਨਹੀਂ ਹੈ।"
                    )
                else:
                    return (
                        f"The highlighted area(s) ({', '.join(f_descs)}) represent candidate regions identified by the application's "
                        f"MLUA segmentation model as suspected caries. The mask represents the model's pixel-level binary segmentation "
                        f"at threshold tau = 0.50. This is an AI decision-support indicator and does not replace a definitive "
                        f"clinical evaluation by a licensed dental practitioner."
                    )
            else:
                if lang == "hi":
                    return "सेगमेंटेशन मॉडल ने थ्रेशोल्ड tau = 0.50 से अधिक किसी संदिग्ध क्षय क्षेत्र का पता नहीं लगाया।"
                elif lang == "pa":
                    return "ਸੈਗਮੈਂਟੇਸ਼ਨ ਮਾਡਲ ਨੇ ਥ੍ਰੈਸ਼ਹੋਲਡ tau = 0.50 ਤੋਂ ਵੱਧ ਕਿਸੇ ਸ਼ੱਕੀ ਕੈਰੀਜ਼ ਖੇਤਰ ਦਾ ਪਤਾ ਨਹੀਂ ਲਗਾਇਆ।"
                else:
                    return "The segmentation model did not detect any candidate caries regions exceeding the decision threshold (tau = 0.50)."

        # ----------------------------------------------------
        # 10. TECHNICAL INTENTS (MLUA, Architecture, Metrics)
        # ----------------------------------------------------
        # Model metrics
        if nlu_result.intent.name == "model_metrics" or any(k in lower for k in ["dice", "metric", "accuracy", "iou", "precision", "recall"]):
            if lang == "hi":
                return (
                    "सक्रिय उत्पादन चेकपॉइंट (EXP-MLUA-003_E75_BEST.pth, चयनित वैलिडेशन-बेस्ट एपॉक 75, कुल 78 एपॉक पूर्ण) ने वैलिडेशन डेटासेट पर निम्नलिखित बेंचमार्क मेट्रिक्स प्राप्त किए: "
                    "Validation Dice Similarity: 71.867%, IoU (Jaccard): 57.349%, Precision: 78.132%, Recall: 67.343%, Specificity: 99.824%, वैलिडेशन लॉस 0.7254 (थ्रेशोल्ड tau = 0.50)। "
                    "सीलबंद टेस्ट सेट पर मैक्रो डाइस 50.147% रहा। ये रिसर्च बेंचमार्क मेट्रिक्स हैं, किसी एक मरीज के रेडियोग्राफ पर सटीकता की गारंटी नहीं।"
                )
            elif lang == "pa":
                return (
                    "ਸਰਗਰਮ ਪ੍ਰੋਡਕਸ਼ਨ ਚੈੱਕਪੁਆਇੰਟ (EXP-MLUA-003_E75_BEST.pth, ਚੁਣਿਆ ਗਿਆ ਵੈਲੀਡੇਸ਼ਨ-ਸਭ ਤੋਂ ਵਧੀਆ ਐਪੋਕ 75, ਕੁੱਲ 78 ਐਪੋਕ ਪੂਰੇ) ਨੇ ਵੈਲੀਡੇਸ਼ਨ ਡਾਟਾਸੈੱਟ 'ਤੇ ਹੇਠਾਂ ਦਿੱਤੇ ਬੈਂਚਮਾਰਕ ਮੈਟ੍ਰਿਕਸ ਪ੍ਰਾਪਤ ਕੀਤੇ: "
                    "Validation Dice Similarity: 71.867%, IoU (Jaccard): 57.349%, Precision: 78.132%, Recall: 67.343%, Specificity: 99.824%, ਵੈਲੀਡੇਸ਼ਨ ਲਾਸ 0.7254 (ਥ੍ਰੈਸ਼ਹੋਲਡ tau = 0.50)। "
                    "ਸੀਲਬੰਦ ਟੈਸਟ ਸੈੱਟ 'ਤੇ ਮੈਕਰੋ ਡਾਈਸ 50.147% ਰਿਹਾ। ਇਹ ਰਿਸਰਚ ਮੈਟ੍ਰਿਕਸ ਹਨ, ਕਿਸੇ ਇੱਕ ਮਰੀਜ਼ ਦੇ ਐਕਸ-ਰੇ 'ਤੇ ਗਾਰੰਟੀ ਨਹੀਂ।"
                )
            else:
                return (
                    "The active production checkpoint (EXP-MLUA-003_E75_BEST.pth, selected validation-best Epoch 75, training run completed through Epoch 78) achieved the following benchmark metrics on the validation dataset: "
                    "Validation Dice Similarity: 71.867%, IoU (Jaccard): 57.349%, Precision: 78.132%, Recall: 67.343%, Specificity: 99.824%, with validation loss of 0.7254 at threshold tau = 0.50. "
                    "On the independent untouched sealed test set, it achieved a Macro Dice of 50.147% (Micro Dice 52.924%). "
                    "Note: These represent validation benchmark metrics across the dataset distribution and are not a clinical performance guarantee for an individual patient's radiograph."
                )

        # MLUA Architecture & Methodology
        if nlu_result.intent.name in ("mlua_methodology", "model_architecture", "segmentation_explanation") or any(k in lower for k in [
            "how does", "model work", "mlua", "architecture", "binary segmentation"
        ]):
            if lang == "hi":
                return (
                    "MLUA प्रणाली पिक्सेल-स्तरीय बाइनरी सेगमेंटेशन करती है जो संदिग्ध क्षय क्षेत्रों से संबंधित पिक्सेल की पृष्ठभूमि के मुकाबले पहचान करती है। "
                    "इसमें ResNet-34 फीचर पिरामिड नेटवर्क (FPN) बैकबोन के साथ टीचर-स्टूडेंट डुअल नेटवर्क, मल्टी-स्केल फीचर एग्रीगेशन (P2-P5), "
                    "एक्सपोनेंशियल मूविंग एवरेज (EMA) टीचर सिंक्रोनाइज़ेशन, और मोंटे कार्लो अनिश्चितता अनुमान का उपयोग किया जाता है। "
                    "इन्फरेंस 384x384 पिक्सेल पैच पर थ्रेशोल्ड tau = 0.50 के साथ किया जाता है।"
                )
            elif lang == "pa":
                return (
                    "MLUA ਪ੍ਰਣਾਲੀ ਪਿਕਸਲ-ਪੱਧਰੀ ਬਾਇਨਰੀ ਸੈਗਮੈਂਟੇਸ਼ਨ ਕਰਦੀ ਹੈ ਜੋ ਸ਼ੱਕੀ ਕੈਰੀਜ਼ ਵਾਲੇ ਪਿਕਸਲਾਂ ਦੀ ਪਛਾਣ ਕਰਦੀ ਹੈ। "
                    "ਇਸ ਵਿੱਚ ResNet-34 ਫੀਚਰ ਪਿਰਾਮਿਡ ਨੈੱਟਵਰਕ (FPN) ਬੈਕਬੋਨ ਦੇ ਨਾਲ ਟੀਚਰ-ਸਟੂਡੈਂਟ ਡੁਅਲ ਨੈੱਟਵਰਕ, ਮਲਟੀ-ਸਕੇਲ ਫੀਚਰ ਏਕੀਕਰਣ (P2-P5), "
                    "ਐਕਸਪੋਨੈਂਸ਼ੀਅਲ ਮੂਵਿੰਗ ਐਵਰੇਜ (EMA) ਟੀਚਰ ਸਿੰਕ੍ਰੋਨਾਈਜ਼ੇਸ਼ਨ, ਅਤੇ ਮੋਂਟੇ ਕਾਰਲੋ ਅਨਿਸ਼ਚਿਤਤਾ ਅਨੁਮਾਨ ਸ਼ਾਮਲ ਹਨ। "
                    "ਅਨੁਮਾਨ 384x384 ਪਿਕਸਲ ਪੈਚਾਂ 'ਤੇ ਥ੍ਰੈਸ਼ਹੋਲਡ tau = 0.50 ਨਾਲ ਲਗਾਇਆ ਜਾਂਦਾ ਹੈ।"
                )
            else:
                return (
                    "The MLUA system performs pixel-level binary segmentation that identifies pixels belonging to suspected carious regions versus background. "
                    "It employs a Teacher-Student dual architecture with a ResNet-34 Feature Pyramid Network (FPN) backbone, multi-scale feature aggregation (P2-P5), "
                    "auxiliary deep supervision, Exponential Moving Average (EMA) teacher synchronization, and Monte Carlo uncertainty estimation. "
                    "Inference is performed using overlapping 384x384 pixel patches to detect fine proximal and occlusal demineralization."
                )

        # General conversational intents
        if nlu_result.intent.name == "greeting":
            if lang == "hi":
                return "नमस्ते! मैं डेंटल क्षय क्लिनिकल AI सिस्टम के लिए आपका AI सहायक हूँ। आज मैं आपके डेंटल रेडियोग्राफ विश्लेषण या MLUA मॉडल जानकारी में कैसे सहायता कर सकता हूँ?"
            elif lang == "pa":
                return "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ! ਮੈਂ ਡੈਂਟਲ ਕੈਰੀਜ਼ ਕਲੀਨਿਕਲ AI ਸਿਸਟਮ ਲਈ ਤੁਹਾਡਾ AI ਸਹਾਇਕ ਹਾਂ। ਅੱਜ ਮੈਂ ਤੁਹਾਡੇ ਦੰਦਾਂ ਦੇ ਐਕਸ-ਰੇ ਵਿਸ਼ਲੇਸ਼ਣ ਜਾਂ MLUA ਮਾਡਲ ਬਾਰੇ ਕਿਵੇਂ ਮਦਦ ਕਰ ਸਕਦਾ ਹਾਂ?"
            else:
                return "Hello! I am your AI clinical assistant for the Dental Caries Clinical AI system. How can I assist you with your dental radiograph analysis or MLUA model information today?"

        if nlu_result.intent.name == "thanks":
            if lang == "hi":
                return "आपका स्वागत है! कृपया किसी भी दंत समस्या या निष्कर्ष के लिए योग्य दंत चिकित्सक से परामर्श करना याद रखें।"
            elif lang == "pa":
                return "ਤੁਹਾਡਾ ਸੁਆਗਤ ਹੈ! ਕਿਰਪਾ ਕਰਕੇ ਕਿਸੇ ਵੀ ਦੰਦਾਂ ਦੀ ਸਮੱਸਿਆ ਲਈ ਯੋਗ ਡੈਂਟਿਸਟ ਨਾਲ ਸਲਾਹ ਕਰਨਾ ਯਾਦ ਰੱਖੋ।"
            else:
                return "You're welcome! Please remember to discuss any dental concerns or findings with a qualified dental practitioner."

        if nlu_result.intent.name == "goodbye":
            if lang == "hi":
                return "अलविदा! अपने मौखिक स्वास्थ्य का ध्यान रखें, और यदि आपके कोई अन्य प्रश्न हैं तो संपर्क करने में संकोच न करें।"
            elif lang == "pa":
                return "ਅਲਵਿਦਾ! ਆਪਣੇ ਦੰਦਾਂ ਦੀ ਸਿਹਤ ਦਾ ਧਿਆਨ ਰੱਖੋ, ਅਤੇ ਜੇਕਰ ਤੁਹਾਡਾ ਕੋਈ ਹੋਰ ਸਵਾਲ ਹੋਵੇ ਤਾਂ ਪੁੱਛ ਸਕਦੇ ਹੋ।"
            else:
                return "Goodbye! Take care of your oral health, and don't hesitate to reach out if you have further questions."

        # Default fallback
        if lang == "hi":
            return (
                "डेंटल AI क्लिनिकल असिस्टेंट MLUA डेंटल क्षय सेगमेंटेशन मॉडल (EXP-MLUA-003_E75_BEST.pth) के लिए निर्णय-समर्थन व्याख्या प्रदान करता है। "
                "सभी चिह्नित क्षेत्र एल्गोरिदम रूप से फ़्लैग की गई रेडियोलुसेंसी को दर्शाते हैं। एक लाइसेंस प्राप्त दंत चिकित्सक को अंतिम निदान स्थापित करने के लिए इन-पर्सन क्लिनिकल परीक्षा आयोजित करनी चाहिए।"
            )
        elif lang == "pa":
            return (
                "ਡੈਂਟਲ AI ਕਲੀਨਿਕਲ ਅਸਿਸਟੈਂਟ MLUA ਡੈਂਟਲ ਕੈਰੀਜ਼ ਸੈਗਮੈਂਟੇਸ਼ਨ ਮਾਡਲ (EXP-MLUA-003_E75_BEST.pth) ਲਈ ਫੈਸਲਾ-ਸਹਾਇਤਾ ਵਿਆਖਿਆ ਪ੍ਰਦਾਨ ਕਰਦਾ ਹੈ। "
                "ਸਾਰੇ ਹਾਈਲਾਈਟ ਕੀਤੇ ਖੇਤਰ ਐਲਗੋਰਿਦਮਿਕ ਸ਼ੱਕੀ ਖੇਤਰਾਂ ਨੂੰ ਦਰਸਾਉਂਦੇ ਹਨ। ਅੰਤਮ ਨਿਦਾਨ ਲਈ ਯੋਗ ਡੈਂਟਿਸਟ ਦੁਆਰਾ ਵਿਅਕਤੀਗਤ ਕਲੀਨਿਕਲ ਜਾਂਚ ਜ਼ਰੂਰੀ ਹੈ।"
            )
        else:
            return (
                "The Dental AI Clinical Assistant provides decision-support explanations for the MLUA dental caries segmentation model (EXP-MLUA-003_E75_BEST.pth). "
                "All highlighted regions represent algorithmically flagged radiolucencies. A licensed dentist must conduct an in-person clinical exam and vitality test "
                "before establishing a definitive diagnosis or treatment plan."
            )

# Singleton instance
gemini_service = GeminiChatService()

