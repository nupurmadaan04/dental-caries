"""
Layer 1: Sentiment & Emotion Analysis Engine
Fine-grained multi-class emotion and sentiment classification.
Supports 11 clinical & conversational affective states:
- neutral
- positive
- confused
- anxious
- worried
- fearful
- frustrated
- sad
- curious
- relieved
- urgent/concerned

Designed for:
- Lightweight local CPU execution (<2ms latency)
- Emoji, informal English, slang, and Hinglish recognition
- Deterministic routing and independent testability
- Strict isolation from MLUA image segmentation weights
"""

import re
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backend.nlp.schemas import EmotionResult, EmotionLabel, SentimentLabel, ToneLabel

# Emotion definitions and representative linguistic anchors (English, slang, Hinglish, emojis)
EMOTION_CORPUS: Dict[EmotionLabel, list[str]] = {
    "fearful": [
        "scared", "terrified", "frightened", "fear", "horrified", "panicking",
        "i am really scared", "scared after seeing this report", "screwed", "am i screwed",
        "dar lag raha hai", "bahut dar lag raha hai", "phat rahi hai", "freaking out",
        "terrifying", "scared to death", "crying scared", "😭", "😨", "😱", "😰"
    ],
    "anxious": [
        "anxious", "nervous", "stressed", "uneasy", "tense", "dread",
        "so nervous about this", "anxiety", "freaking out a bit", "super anxious",
        "ghabrahat", "tension ho rahi hai", "bahut tension hai", "dar lag raha",
        "can't sleep thinking about this", "anxious about cavity", "overthinking"
    ],
    "worried": [
        "worried", "worry", "concerned", "should i be worried", "is this bad",
        "is it serious", "will i lose my tooth", "chinta", "chinta ho rahi hai",
        "kahi kharab toh nahi", "worried about this lesion", "bad news",
        "is something wrong", "should i panic", "worried about teeth", "😟", "🥺"
    ],
    "confused": [
        "confused", "confusing", "don't understand", "dont get it", "what do you mean",
        "what does this mean", "makes no sense", "unclear", "lost", "baffled",
        "samajh nahi aaya", "samajh nahi aa raha", "ye kya bol rahe ho",
        "kuch samajh nahi", "what does that even mean", "huh", "puzzled", "🤔", "🤨"
    ],
    "curious": [
        "curious", "what does this mean", "what is this highlighted area",
        "bro what does this mean", "bro what is this", "what did you find",
        "tell me more", "how come", "interested to know", "want to understand",
        "ye kya hai", "bhai ye kya hai", "kaha par hai", "batao", "dekhna hai",
        "what is the model seeing", "can you explain", "what is this"
    ],
    "frustrated": [
        "frustrated", "annoyed", "irritated", "why isn't this working", "why is this not working",
        "useless", "broken", "terrible app", "waste of time", "not working", "stupid",
        "bakwas", "kaam kyu nahi kar raha", "dimag kharab", "pareshan ho gaya",
        "giving such a weird result", "ridiculous", "hate this", "😠", "😡", "😤"
    ],
    "relieved": [
        "thank god", "thank goodness", "what a relief", "relieved", "so glad",
        "glad to hear", "phew", "feeling better", "sukr hai", "bhagwan ka shukr",
        "bach gaya", "peace of mind", "that is a relief", "blessed",
        "so this doesn't necessarily mean i have a cavity", "grateful", "😌", "🙏"
    ],
    "sad": [
        "sad", "depressed", "unhappy", "heartbroken", "crying", "miserable",
        "feeling down", "dukhi", "udas", "bura lag raha hai", "rula diya",
        "sad about this diagnosis", "terrible day", "😢", "😞", "💔"
    ],
    "urgent/concerned": [
        "urgent", "emergency", "immediate", "terrible pain", "severe pain",
        "swelling", "bleeding", "fever", "infection", "hurts so bad",
        "unbearable pain", "turant", "jaldi", "dard bardasht nahi ho raha",
        "emergency help", "severe toothache", "puss", "jaw swollen", "🚨", "⚠️"
    ],
    "positive": [
        "great", "awesome", "excellent", "good", "amazing", "cool", "wonderful",
        "nice job", "helpful", "perfect", "thank you", "thanks bro", "appreciated",
        "badhiya", "achha hai", "mast", "shukriya", "dhanyawad", "happy", "👍", "😊", "✨"
    ],
    "neutral": [
        "where is it", "show me tooth 24", "what is the stage", "explain level 2",
        "what is dice score", "give me the report", "how does mlua work",
        "what is binary segmentation", "teeth information", "patient report",
        "summary please", "is tooth 16 affected", "threshold value", "tau 0.50",
        "what are coordinates", "coordinates of lesion", "where are coordinates",
        "tell me coordinates", "what is the location", "lesion coordinates",
        "what are coordinates of lesion", "what are the coordinates of the lesion"
    ]
}

# Emotion to Sentiment mapping
EMOTION_TO_SENTIMENT: Dict[EmotionLabel, SentimentLabel] = {
    "fearful": "negative",
    "anxious": "negative",
    "worried": "negative",
    "frustrated": "negative",
    "sad": "negative",
    "urgent/concerned": "negative",
    "confused": "neutral",
    "curious": "neutral",
    "neutral": "neutral",
    "relieved": "positive/relieved",
    "positive": "positive",
}

# Emotion to Communicative Tone mapping
EMOTION_TO_TONE: Dict[EmotionLabel, ToneLabel] = {
    "fearful": "concerned",
    "anxious": "concerned",
    "worried": "concerned",
    "urgent/concerned": "urgent",
    "frustrated": "frustrated",
    "sad": "concerned",
    "confused": "reassuring_needed",
    "curious": "informational",
    "relieved": "casual",
    "positive": "casual",
    "neutral": "informational",
}


class EmotionAnalyzer:
    """
    Lightweight, deterministic Emotion and Sentiment Analyzer.
    Pre-vectorized corpus for sub-millisecond execution.
    """

    def __init__(self):
        self.labels = list(EMOTION_CORPUS.keys())
        # Build training corpus from predefined profiles
        self.documents = []
        self.doc_labels = []
        for label, phrases in EMOTION_CORPUS.items():
            combined = " ".join(phrases)
            self.documents.append(combined)
            self.doc_labels.append(label)

        # Character and word n-gram TF-IDF vectorizer for typo and subword tolerance
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            analyzer="word",
            lowercase=True,
            sublinear_tf=True
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)

    def analyze(self, text: str) -> EmotionResult:
        """
        Analyzes the emotional and sentimental orientation of a user's input.
        Returns an EmotionResult with linguistics-specific confidence.
        """
        clean_text = text.strip()
        lower = clean_text.lower()

        # 1. High-precision rule & emoji overrides for explicit emotional signals
        explicit_override = self._check_explicit_rules(clean_text, lower)
        if explicit_override is not None:
            emotion, conf, sentiment, tone = explicit_override
            return EmotionResult(
                sentiment=sentiment,
                emotion=emotion,
                confidence=round(conf, 2),
                tone=tone
            )

        # 2. Vector semantic similarity against emotion profiles
        try:
            query_vec = self.vectorizer.transform([lower])
            scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]
            max_idx = int(np.argmax(scores))
            max_score = float(scores[max_idx])
            
            if max_score > 0.12:
                matched_emotion: EmotionLabel = self.doc_labels[max_idx]
                # Scale confidence gracefully into [0.65, 0.96]
                confidence = min(0.96, max(0.65, 0.60 + max_score * 0.45))
            else:
                matched_emotion = "neutral"
                confidence = 0.75
        except Exception:
            matched_emotion = "neutral"
            confidence = 0.70

        sentiment = EMOTION_TO_SENTIMENT.get(matched_emotion, "neutral")
        tone = EMOTION_TO_TONE.get(matched_emotion, "informational")

        return EmotionResult(
            sentiment=sentiment,
            emotion=matched_emotion,
            confidence=round(confidence, 2),
            tone=tone
        )

    def _check_explicit_rules(self, text: str, lower: str) -> Tuple[EmotionLabel, float, SentimentLabel, ToneLabel] | None:
        """
        High-precision pattern matcher for overt emotional expressions, slang, and emojis.
        """
        # Fearful / Terrified
        if any(w in lower for w in [
            "scared", "terrified", "i'm really scared", "im really scared",
            "am i screwed", "dar lag raha", "freaking out", "phat rahi"
        ]) or any(e in text for e in ["😭", "😱", "😨"]):
            if "not scared" not in lower:
                return ("fearful", 0.93, "negative", "concerned")

        # Relieved / Grateful
        if any(w in lower for w in [
            "thank god", "thank goodness", "what a relief", "so this doesn't necessarily mean",
            "phew", "sukr hai", "shukr hai", "bach gaya"
        ]) or "😌" in text or "🙏" in text:
            return ("relieved", 0.92, "positive/relieved", "casual")

        # Frustrated / Broken / Annoyed
        if any(w in lower for w in [
            "why isn't this working", "why is this not working", "not working",
            "giving such a weird result", "useless", "bakwas", "dimag kharab",
            "stupid app", "irritating"
        ]) or ("??" in text and ("why" in lower or "not" in lower)) or any(e in text for e in ["😠", "😡"]):
            return ("frustrated", 0.91, "negative", "frustrated")

        # Urgent / Severe pain / Emergency
        if any(w in lower for w in [
            "emergency", "urgent", "unbearable pain", "severe pain", "bleeding",
            "jaw swollen", "turant", "dard bardasht", "severe toothache"
        ]) or "🚨" in text:
            return ("urgent/concerned", 0.94, "negative", "urgent")

        # Anxious / Nervous
        if any(w in lower for w in [
            "anxious", "nervous", "ghabrahat", "tension ho rahi", "freaked out", "panicking"
        ]):
            return ("anxious", 0.88, "negative", "concerned")

        # Worried / Concerned
        if any(w in lower for w in [
            "should i be worried", "is this bad", "is it bad", "chinta", "is this serious",
            "worried about this", "worried"
        ]):
            return ("worried", 0.89, "negative", "concerned")

        # Confused / Unclear / Not understanding
        if any(w in lower for w in [
            "confused", "makes no sense", "samajh nahi aa raha", "samajh nahi aaya",
            "samajh nahi aara", "samajh nhi aaya", "kuch samajh nahi", "kuch samajh nhi",
            "what do you mean", "unclear", "puzzled", "dont understand", "don't understand",
            "dont get it", "don't get it", "not clear", "not getting it", "didnt understand",
            "didn't understand", "explain simply", "can you explain simply", "can you explain that simply",
            "simple language mein batao", "simple language me batao", "mainu samajh nahi aaya",
            "mainu samajh nahi", "samjha nahi", "samjh nahi", "thoda easy batao", "simple mein batao",
            "simple me batao", "simple daso", "simple samjhao", "simple language please"
        ]) or "🤔" in text:
            return ("confused", 0.88, "neutral", "reassuring_needed")

        # Curious / Asking for explanations
        if any(w in lower for w in [
            "what does this mean", "what is this highlighted", "bro what does this",
            "bro what is this", "bhai ye kya hai", "what did your model find",
            "tell me what this is", "curious"
        ]):
            return ("curious", 0.88, "neutral", "informational")

        # Sad
        if any(w in lower for w in ["sad", "depressed", "dukhi", "crying"]):
            return ("sad", 0.86, "negative", "concerned")

        # Positive / Appreciation
        if any(w in lower for w in [
            "thank you", "thanks bro", "great job", "awesome", "shukriya", "dhanyawad",
            "excellent", "good work", "super helpful"
        ]) and "not" not in lower:
            return ("positive", 0.89, "positive", "casual")

        return None


# Singleton instance
emotion_analyzer = EmotionAnalyzer()
