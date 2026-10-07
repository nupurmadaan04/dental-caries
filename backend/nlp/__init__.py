"""
Natural Language Understanding (NLU) Module for Dental Caries Clinical AI Assistant.
Provides three-layer conversational intelligence:
1. Emotion / Sentiment Analysis
2. Semantic Intent Classification
3. Structured NLU Context for LLM Reasoning & Clinical Governance
"""

from backend.nlp.schemas import EmotionResult, IntentResult, NLUAnalysisResult
from backend.nlp.emotion_analyzer import emotion_analyzer, EmotionAnalyzer
from backend.nlp.intent_classifier import intent_classifier, IntentClassifier
from backend.nlp.nlu_router import nlu_router, NLURouter

__all__ = [
    "EmotionResult",
    "IntentResult",
    "NLUAnalysisResult",
    "emotion_analyzer",
    "EmotionAnalyzer",
    "intent_classifier",
    "IntentClassifier",
    "nlu_router",
    "NLURouter",
]
