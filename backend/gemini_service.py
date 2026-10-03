"""
Gemini Context-Aware Clinical Assistant Service
Embedded decision-support explanation service for the MLUA Dental Caries Segmentation System.

Hardened Privacy & Medical Governance:
- Zero Patient-Identifying Information (PHI) is ever transmitted to external AI endpoints.
- Model Predicted Probability is strictly defined as neural network sigmoid activation above threshold tau = 0.50, NOT clinical confidence.
- Application-defined staging is treated strictly as rule-based decision support, never clinical diagnosis.
- MLUA E64 inference is the sole source of truth for lesion candidate localization.
- Gemini NEVER acts as an autonomous diagnostic model or treatment prescriber.
- Compatible with Google GenAI Interactions API and Models API.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, AsyncGenerator
from dotenv import load_dotenv

# Load server environment variables from .env
load_dotenv()

logger = logging.getLogger("dental_ai.gemini")

# System Instruction strictly enforcing medical decision-support governance and clinical safety
SYSTEM_INSTRUCTION = """You are an AI explanation assistant embedded in a dental X-ray clinical decision-support application.

The application's machine learning segmentation model (MLUA EXP-MLUA-003_E64_BEST at threshold tau = 0.50) is the sole source of truth for detected regions, lesion coordinates, and candidate segmentation masks.

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
2. CASE STAGING:
   - Queries: "What is the stage?", "What is the stage of this report?", "What is stage of this recent report?", "What stage is this lesion?", "What level is this?", "What does Level 2 mean?", "What does Level 2 indicate?", "What does Level 1 indicate?", "What does Level 3 indicate?", "What severity stage did the model assign?", "What is the application-defined stage?", "Explain the stage.", "What does the current stage indicate?"
   - Action: Check 'CURRENT ANALYZED CASE' first. Explain the application-defined heuristic staging (Level/Stage number, title, depth description) assigned to this active case. Clarify that staging is rule-based decision support from segmentation findings, not a definitive clinical diagnosis.
3. CASE FINDINGS:
   - Queries: "What did the model find?", "Explain this X-ray result.", "What does the highlighted area mean?", "What did the model detect?", "Why did the model highlight this area?"
   - Action: Check 'CURRENT ANALYZED CASE' first. Summarize candidate region IDs, overall findings, model predicted probabilities, and area percentages.
4. TECHNICAL:
   - Queries: "How does MLUA work?", "What is Dice?", "What is binary segmentation?", "Explain the architecture."
   - Action: Use MLUA technical context and validation benchmark specifications (EXP-MLUA-003_E64_BEST.pth, Epoch 64, tau = 0.50).
5. SAFETY:
   - Queries: "Do I definitely have caries?", "Should I take medicine?", "Do I need treatment?"
   - Action: Apply medical safety governance (no autonomous diagnosis, no prescriptions, recommend in-person dental evaluation).

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
- Model: MLUA (Multi-level Uncertainty-Aware / Multi-scale Semi-supervised Caries Segmentation).
- Production Checkpoint: EXP-MLUA-003_E64_BEST.pth (Epoch 64, Step 8,448).
- Task: pixel-level binary segmentation that identifies pixels belonging to suspected carious regions versus background (0 = background, 1 = suspected caries).
- Operating Threshold: tau = 0.50.
- Canonical Validation Benchmark Metrics (EXP-MLUA-003 E64 on validation set):
  * Validation Dice Similarity Coefficient: 69.386%
  * Validation IoU (Jaccard Index): 54.326%
  * Validation Precision: 74.689%
  * Validation Recall (Sensitivity): 66.415%
  * Peak Validation Loss: 0.7471
  * Specificity: 99.784%
  * Zero-prediction ratio: 0%
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
        if not case_context or not case_context.get("case") or not case_context["case"].get("image_available"):
            return (
                f"CURRENT EXPLANATION MODE: {mode.upper()}\n"
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
                f_bbox = f.get("bbox")
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

        lines = [
            f"CURRENT EXPLANATION MODE: {mode.upper()}",
            "CURRENT ANALYZED CASE",
            "---------------------",
            f"Candidate sites: {candidate_sites_str}",
            f"Region details:\n{region_details_str}",
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

        # Build sanitized dynamic context (PHI-free)
        context_prompt = self._format_case_context(case_context, mode)
        
        if not self.is_available():
            # Return high-quality, grounded clinical fallback without breaking user flow
            fallback_text = self._generate_offline_fallback(message, case_context, mode)
            return {
                "text": fallback_text,
                "session_id": session_id,
                "status": "fallback",
                "notice": "Gemini API key is not configured on the server. Showing built-in clinical decision-support response."
            }

        try:
            full_system_instruction = f"{SYSTEM_INSTRUCTION}\n\n==================================================\n{context_prompt}\n=================================================="

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
                reply_text = self._generate_offline_fallback(message, case_context, mode)

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
                "model": self.model_name
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

            fallback_text = self._generate_offline_fallback(message, case_context, mode)
            return {
                "text": fallback_text,
                "session_id": session_id,
                "status": "error",
                "error": user_err
            }

    def clear_session(self, session_id: str):
        """Clears state and memory for a specific chat session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
        if session_id in self._session_case_map:
            del self._session_case_map[session_id]
        if session_id in self._session_interaction_map:
            del self._session_interaction_map[session_id]

    def _generate_offline_fallback(self, message: str, case_context: Optional[Dict[str, Any]], mode: str) -> str:
        """Deterministic, grounded clinical decision-support response when Gemini is offline."""
        lower = message.lower()
        has_case = bool(case_context and case_context.get("case") and case_context["case"].get("image_available"))
        case = case_context.get("case", {}) if has_case else {}
        findings = case.get("findings", [])

        # Safety: definitive diagnosis requests
        if "definitely" in lower or "exact diagnosis" in lower or "do i have" in lower or "is this a cavity" in lower or "have caries" in lower:
            return (
                "This application cannot provide a definitive clinical diagnosis. The MLUA model provides algorithmic "
                "decision-support by identifying candidate regions of radiolucency. Only a qualified, licensed dental "
                "professional can determine if dental caries is clinically present through in-person visual-tactile examination, "
                "vitality assessment, and clinical correlation."
            )

        # Safety: medicine/prescription requests
        if "medicine" in lower or "drug" in lower or "prescribe" in lower or "treat" in lower or "take" in lower:
            return (
                "The AI assistant is strictly a decision-support explanation tool and cannot prescribe medications or clinical treatments. "
                "If you are experiencing tooth pain, sensitivity, or suspect dental disease, please schedule an appointment with a "
                "qualified dental professional for proper diagnosis and care."
            )

        # Location, staging, candidate region, or tooth questions without an active case
        if not has_case:
            if any(k in lower for k in ["where", "tooth", "teeth", "l1", "l2", "region", "affected", "candidate", "highlighted", "lesion", "this x-ray", "this result", "finding", "what did the model detect", "why did the model highlight", "stage", "level", "severity", "recent report", "this report"]):
                return (
                    "No active analyzed case is currently available. Please upload or select a radiograph and run it through the MLUA analysis pipeline first."
                )

        # Case Staging questions WITH an active case
        if any(k in lower for k in ["stage", "level", "severity"]):
            staging = case.get("application_staging") or case.get("stage") or {}
            if isinstance(staging, dict):
                lvl = staging.get("level", "Application-Defined Heuristic Staging")
                title = staging.get("title", "")
                desc = staging.get("description", "")
            else:
                lvl = str(staging)
                title = ""
                desc = ""

            title_part = f" ({title})" if title else ""
            desc_part = f" Description: {desc}." if desc else ""

            if "what does level 1" in lower or "explain level 1" in lower or "what does stage 1" in lower:
                return (
                    f"In this application, Level 1 represents 'Suspected Early Caries', indicating localized demineralization confined to enamel or the outer dentino-enamel junction. "
                    f"The current active case is staged as {lvl}{title_part}. "
                    f"This is a rule-based decision-support classification from segmentation findings, not a definitive clinical diagnosis. "
                    f"A dentist should correlate the finding with the radiograph and clinical examination."
                )
            elif "what does level 2" in lower or "explain level 2" in lower or "what does stage 2" in lower:
                return (
                    f"In this application, Level 2 represents 'Moderate Caries', indicating radiographic radiolucency extending deeper across the enamel-dentin junction into middle/deep dentin. "
                    f"The current active case is staged as {lvl}{title_part}. "
                    f"This is an algorithmic decision-support classification to assist dental review, not a definitive clinical diagnosis. "
                    f"A dentist should correlate the finding with the radiograph and clinical examination."
                )
            elif "what does level 3" in lower or "explain level 3" in lower or "what does stage 3" in lower:
                return (
                    f"In this application, Level 3 represents 'Extensive / Deep Caries', indicating deep radiolucency extending into inner dentin approaching or involving the dental pulp. "
                    f"The current active case is staged as {lvl}{title_part}. "
                    f"This is a decision-support classification, not a definitive clinical diagnosis. Prompt in-person clinical and vitality examination is advised."
                )

            if findings:
                return (
                    f"The current report assigns this candidate region an application-defined {lvl} heuristic stage{title_part}.{desc_part} "
                    f"This staging is generated from the application's segmentation findings and associated lesion indicators. "
                    f"It is a decision-support classification, not a definitive clinical diagnosis. "
                    f"A dentist should correlate the finding with the radiograph and clinical examination."
                )
            else:
                return (
                    f"The current report assigns an application-defined {lvl} heuristic stage{title_part}.{desc_part} "
                    f"The segmentation model detected no candidate caries regions exceeding threshold tau = 0.50. "
                    f"This is a decision-support indicator, not a definitive clinical diagnosis. Routine dental examination remains recommended."
                )

        # Location / candidate region / tooth questions WITH an active case
        if any(k in lower for k in ["where is", "which tooth", "tooth is affected", "tooth is involved", "what does l1 mean", "what is l1", "candidate region", "what did the model detect", "why did the model highlight", "where are", "what did the model find", "what did it find"]):
            if findings:
                if len(findings) == 1:
                    f = findings[0]
                    f_id = f.get("lesionNumber") or f.get("id") or "L1"
                    f_tooth = f.get("toothNumberFDI")
                    f_loc = f.get("anatomicalLocation") or f.get("location")
                    f_depth = f.get("cariesDepthIndicator") or f.get("depth")
                    f_area = f.get("pixelArea")
                    prob_val = f.get("modelPredictedProbability") if f.get("modelPredictedProbability") is not None else f.get("confidence")
                    f_prob = f"{float(prob_val)*100:.1f}%" if prob_val is not None else "N/A"

                    parts = [f"The current analysis identifies one candidate region, {f_id}"]
                    if f_tooth:
                        parts.append(f"associated with tooth {f_tooth}")
                    if f_loc:
                        parts.append(f"in the {f_loc} region")
                    desc = ", ".join(parts) + "."
                    extra = []
                    if f_depth:
                        extra.append(f"Depth indicator: {f_depth}")
                    if f_area:
                        extra.append(f"segmented area: {f_area} px")
                    extra_str = f" ({', '.join(extra)})" if extra else ""

                    return (
                        f"{desc}{extra_str} The segmentation overlay marks this candidate area. "
                        f"The model-predicted probability for the segmented pixels is {f_prob}. "
                        f"This is an algorithmic prediction and not a clinical diagnosis."
                    )
                else:
                    items = []
                    for f in findings:
                        f_id = f.get("lesionNumber") or f.get("id") or "Region"
                        f_tooth = f.get("toothNumberFDI") or "Unspecified"
                        f_loc = f.get("anatomicalLocation") or f.get("location") or "Unspecified"
                        prob_val = f.get("modelPredictedProbability") if f.get("modelPredictedProbability") is not None else f.get("confidence")
                        f_prob = f"{float(prob_val)*100:.1f}%" if prob_val is not None else "N/A"
                        items.append(f"{f_id} (Tooth {f_tooth}, {f_loc}, Model Predicted Probability: {f_prob})")
                    return (
                        f"The current analysis identifies {len(findings)} candidate regions: {'; '.join(items)}. "
                        f"The segmentation overlay marks these candidate areas on the radiograph. "
                        f"These are algorithmic predictions and not clinical diagnoses."
                    )
            else:
                return (
                    "The current analysis did not detect any candidate lesion regions exceeding the decision threshold (tau = 0.50). "
                    "The segmentation overlay shows no highlighted caries regions."
                )

        # Highlighted region explanation
        if "highlighted" in lower or "what does this mean" in lower or "explain this" in lower or "simple" in lower:
            if findings:
                f_descs = [f"{f.get('lesionNumber') or f.get('id', 'L1')}: Tooth {f.get('toothNumberFDI', 'N/A')} ({f.get('anatomicalLocation') or f.get('location', '')})" for f in findings[:3]]
                return (
                    f"The highlighted area(s) ({', '.join(f_descs)}) represent candidate regions identified by the application's "
                    f"MLUA segmentation model as suspected caries. The mask represents the model's pixel-level binary segmentation "
                    f"at threshold tau = 0.50. This is an AI decision-support indicator and does not replace a definitive "
                    f"clinical evaluation by a licensed dental practitioner."
                )
            else:
                return (
                    "The segmentation model did not detect any candidate caries regions exceeding the decision threshold (tau = 0.50). "
                    "This indicates no radiographically evident caries was segmented by the model. A routine clinical dental exam remains recommended."
                )

        # Benchmark metrics explanation
        if "dice" in lower or "metric" in lower or "accuracy" in lower or "iou" in lower or "precision" in lower or "recall" in lower:
            return (
                "The selected production checkpoint (EXP-MLUA-003_E64_BEST.pth) achieved the following benchmark metrics on the validation dataset: "
                "Validation Dice Similarity: 69.386%, IoU (Jaccard): 54.326%, Precision: 74.689%, Recall: 66.415%, with validation loss of 0.7471 at threshold tau = 0.50. "
                "Note: These represent validation benchmark metrics across the training distribution and are not a guarantee of accuracy for an individual patient's radiograph."
            )

        # Technical MLUA architecture
        if "how does" in lower or "model work" in lower or "mlua" in lower or "architecture" in lower or "binary segmentation" in lower:
            return (
                "The MLUA system performs pixel-level binary segmentation that identifies pixels belonging to suspected carious regions versus background. "
                "It employs a Teacher-Student dual architecture with a ResNet-34 Feature Pyramid Network (FPN) backbone, multi-scale feature aggregation (P2-P5), "
                "auxiliary deep supervision, Exponential Moving Average (EMA) teacher synchronization, and Monte Carlo uncertainty estimation. "
                "Inference is performed using overlapping 384x384 pixel patches to detect fine proximal and occlusal demineralization."
            )

        # Limitations
        if "limitation" in lower or "false positive" in lower or "burnout" in lower:
            return (
                "Key limitations of panoramic radiographic AI segmentation include: (1) Cervical burnout artifacts at the tooth neck that can mimic caries, "
                "(2) 15-30% geometric distortion inherent to panoramic OPGs, (3) Difficulty segmenting very early enamel lesions (E1) without intraoral bitewings, "
                "and (4) Potential false positives from radiolucent fillings or overlapping contacts. Clinical tactile and visual examination is always required."
            )

        return (
            "The Dental AI Clinical Assistant provides decision-support explanations for the MLUA dental caries segmentation model (EXP-MLUA-003_E64_BEST.pth). "
            "All highlighted regions represent algorithmically flagged radiolucencies. A licensed dentist must conduct an in-person clinical exam and vitality test "
            "before establishing a definitive diagnosis or treatment plan."
        )

# Singleton instance
gemini_service = GeminiChatService()

