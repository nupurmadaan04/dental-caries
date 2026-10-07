"""
Pydantic Schemas for Dental Caries Clinical AI Natural Language Understanding (NLU).
Strictly separates emotion/intent classification confidence from clinical/medical confidence.
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field

# Valid emotion categories specified in requirements
EmotionLabel = Literal[
    "neutral",
    "positive",
    "confused",
    "anxious",
    "worried",
    "fearful",
    "frustrated",
    "sad",
    "curious",
    "relieved",
    "urgent/concerned",
]

SentimentLabel = Literal[
    "positive",
    "negative",
    "neutral",
    "positive/relieved",
    "mixed",
]

ToneLabel = Literal[
    "informational",
    "concerned",
    "casual",
    "frustrated",
    "urgent",
    "reassuring_needed",
]

UrgencyLevel = Literal[
    "low",
    "normal",
    "high",
    "critical",
]

class EmotionResult(BaseModel):
    """
    Structured outcome of Layer 1 Emotion & Sentiment Analysis.
    Confidence is NLP classification confidence, NOT diagnostic/clinical confidence.
    """
    sentiment: SentimentLabel = Field(
        ...,
        description="High-level sentiment orientation: positive, negative, neutral, positive/relieved"
    )
    emotion: EmotionLabel = Field(
        ...,
        description="Fine-grained affective state: neutral, positive, confused, anxious, worried, fearful, frustrated, sad, curious, relieved, urgent/concerned"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Emotion classification confidence (0.0 - 1.0). NOTE: This is linguistic model confidence, NOT clinical confidence."
    )
    tone: ToneLabel = Field(
        ...,
        description="Communicative tone: informational, concerned, casual, frustrated, urgent, reassuring_needed"
    )

class IntentResult(BaseModel):
    """
    Structured outcome of Layer 2 NLP Intent Classification.
    """
    name: str = Field(
        ...,
        description="Classified semantic intent identifier (e.g. lesion_location, staging, diagnosis_request)"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Intent classification confidence (0.0 - 1.0). Linguistic model confidence only."
    )
    category: str = Field(
        default="general_conversation",
        description="Intent domain: case_specific, model_technical, general_conversation, medical_safety, system"
    )
    requires_case_context: bool = Field(
        default=False,
        description="Flag indicating if the intent pertains to an active dental X-ray analysis case"
    )
    requires_safety_guard: bool = Field(
        default=False,
        description="Flag indicating if the intent requires high-priority medical safety disclaimers (e.g. diagnosis, treatment, medications)"
    )
    canonical_v2_name: Optional[str] = Field(
        default=None,
        description="Canonical 31-intent identifier from the hardened conversational benchmark taxonomy"
    )

class NLUAnalysisResult(BaseModel):
    """
    Combined Three-Layer NLU analysis outcome passed into Gemini context and safety routers.
    """
    intent: IntentResult
    sentiment: EmotionResult
    tone: ToneLabel = Field(..., description="Overall conversational tone indicator")
    urgency: UrgencyLevel = Field(default="normal", description="Urgency assessment: low, normal, high, critical")
    requires_case_context: bool = Field(default=False, description="Whether active case context must be prioritized")
    requires_safety_guard: bool = Field(default=False, description="Whether strict clinical safety guard is triggered")
    detected_language: Optional[str] = Field(default=None, description="Explicit requested language code if detected in the current message ('en', 'hi', 'pa')")
    response_language: str = Field(default="en", description="Target language code for the response ('en', 'hi', 'pa')")
    response_scope: str = Field(default="general_info", description="Constrained response scope: explain_stage_general, explain_current_case_stage, clinical_safety, case_location, etc.")
    target_stage: Optional[str] = Field(default=None, description="Explicit target stage/level number if specified in prompt ('1', '2', '3')")

    def to_context_block(self) -> str:
        """
        Formats the structured NLU metadata block for Gemini reasoning context.
        Zero PHI is contained in this block.
        """
        lang_map = {"en": "English", "hi": "Hindi", "pa": "Punjabi"}
        req_lang_name = lang_map.get(self.detected_language, "None") if self.detected_language else "None"
        resp_lang_name = lang_map.get(self.response_language, "English")

        return (
            "CURRENT USER LANGUAGE ANALYSIS\n"
            "------------------------------\n"
            f"Intent: {self.intent.name}\n"
            f"Intent Confidence: {self.intent.confidence:.2f}\n"
            f"Response Scope: {self.response_scope}\n"
            f"Target Stage: {self.target_stage or 'N/A'}\n"
            f"Emotion: {self.sentiment.emotion}\n"
            f"Emotion Confidence: {self.sentiment.confidence:.2f}\n"
            f"Tone: {self.tone}\n"
            f"Urgency: {self.urgency}\n"
            f"Requested Language: {req_lang_name}\n"
            f"Response Language: {resp_lang_name}\n"
            f"Requires Case Context: {str(self.requires_case_context).lower()}\n"
            f"Safety Guard: {str(self.requires_safety_guard).lower()}\n"
            "------------------------------"
        )
