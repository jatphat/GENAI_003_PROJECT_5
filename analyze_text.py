# ==========================================
# File: analyze_text.py
# (new) moved BERT/TextBlob risk analysis
# ==========================================

import streamlit as st
from transformers import pipeline
from textblob import TextBlob
from typing import Dict, List
import torch

# Global model cache
BERT_MODEL = None

# Risk patterns
risk_patterns = {
    'high': [
        'sell your data', 'share your data', 'third party sharing',
        'track location', 'biometric data', 'facial recognition',
        'sell personal information', 'share personal information',
        'data broker', 'surveillance', 'data mining',
        'right to publish', 'use your content',
        'without consent', 'without explanation',
        'reject without notice', 'remove without notice',
        'share with third parties', 'transfer your data',
        'unlimited rights', 'perpetual license',
        'misuse of personal data', 'unauthorized access'
    ],
    'medium': [
        'cookie tracking', 'analytics', 'advertising',
        'marketing email', 'promotional content', 'user profiling',
        'targeted ads', 'tracking pixels', 'social media integration',
        'collect information', 'store information', 'process data',
        'usage statistics', 'behavioral data'
    ],
    'low': [
        'contact information', 'email address', 'basic account info',
        'security measure', 'encryption', 'necessary data',
        'customer service', 'basic analytics', 'technical data',
        'improve service', 'maintain service', 'essential cookies'
    ]
}

risk_amplifiers = [
    'without', 'unlimited', 'any purpose', 'all rights',
    'may', 'could', 'might', 'reserve the right',
    'at our discretion', 'not responsible', 'no liability'
]

# Helper

def get_bert_model():
    global BERT_MODEL
    if BERT_MODEL is None:
        try:
            BERT_MODEL = pipeline(
                "text-classification",
                model="nlptown/bert-base-multilingual-uncased-sentiment",
                tokenizer="nlptown/bert-base-multilingual-uncased-sentiment",
                device=-1
            )
        except Exception as e:
            print(f"Error loading BERT model: {str(e)}")
            return None
    return BERT_MODEL

# Chunker

def chunk_text(text: str, max_length: int = 512) -> List[str]:
    sentences = text.split('.')
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_length:
            current_chunk += sentence + "."
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence + "."

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

# Analyzer

def analyze_privacy_policy(text: str) -> Dict[str, List[str]]:
    try:
        classifier = get_bert_model()
        if classifier is None:
            return analyze_with_textblob(text)

        risks = {'high': [], 'medium': [], 'low': []}
        chunks = chunk_text(text)

        progress_bar = st.progress(0)
        progress_text = st.empty()
        total_chunks = len(chunks)

        for i, chunk in enumerate(chunks):
            progress_bar.progress((i+1)/total_chunks)
            progress_text.text(f"Analyzing chunk {i+1}/{total_chunks}")

            result = classifier(chunk[:512])[0]
            score = int(result['label'][0])

            sentences = [s.strip() for s in chunk.split('.') if len(s.strip()) > 20]
            for sentence in sentences:
                sentence_lower = sentence.lower()
                for risk_level, patterns in risk_patterns.items():
                    for pattern in patterns:
                        if pattern.lower() in sentence_lower:
                            final_risk_level = risk_level
                            if any(amp in sentence_lower for amp in risk_amplifiers):
                                if risk_level == 'low':
                                    final_risk_level = 'medium'
                                elif risk_level == 'medium':
                                    final_risk_level = 'high'

                            formatted_sentence = sentence.strip()
                            if score <= 2:
                                formatted_sentence = "⚠️ " + formatted_sentence
                            elif score >= 4:
                                formatted_sentence = "✓ " + formatted_sentence

                            risks[final_risk_level].append(formatted_sentence)

        progress_bar.empty()
        progress_text.empty()

        for risk_level in risks:
            seen = set()
            risks[risk_level] = [x for x in risks[risk_level] if not (x in seen or seen.add(x))]
            risks[risk_level] = [(s[:150] + '...') if len(s) > 150 else s for s in risks[risk_level]]
            if not risks[risk_level]:
                risks[risk_level].append(f"No {risk_level}-risk items identified")

        return risks

    except Exception as e:
        print(f"BERT analysis failed, fallback to TextBlob: {str(e)}")
        return analyze_with_textblob(text)

# TextBlob fallback

def analyze_with_textblob(text: str) -> Dict[str, List[str]]:
    try:
        blob = TextBlob(text.lower())
        risks = {'high': [], 'medium': [], 'low': []}

        progress_bar = st.progress(0)
        progress_text = st.empty()
        total_sentences = len(blob.sentences)

        for i, sentence in enumerate(blob.sentences):
            progress_bar.progress((i+1)/total_sentences)
            progress_text.text(f"Analyzing sentence {i+1}/{total_sentences}")

            sentence_text = str(sentence)
            if len(sentence_text) < 20:
                continue

            sentiment = sentence.sentiment.polarity

            for risk_level, patterns in risk_patterns.items():
                for pattern in patterns:
                    if pattern.lower() in sentence_text:
                        risk_entry = sentence_text.strip()
                        if sentiment < -0.1:
                            risk_entry = "⚠️ " + risk_entry
                        elif sentiment > 0.1:
                            risk_entry = "✓ " + risk_entry
                        risks[risk_level].append(risk_entry)

        progress_bar.empty()
        progress_text.empty()

        for risk_level in risks:
            risks[risk_level] = list(set(risks[risk_level]))
            risks[risk_level] = [(s[:150] + '...') if len(s) > 150 else s for s in risks[risk_level]]
            if not risks[risk_level]:
                risks[risk_level].append(f"No {risk_level}-risk items identified")

        return risks

    except Exception as e:
        print(f"TextBlob fallback error: {str(e)}")
        return {
            'high': ['Analysis Error'],
            'medium': ['Analysis Error'],
            'low': ['Analysis Error']
        }
