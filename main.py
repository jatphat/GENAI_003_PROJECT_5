


# ==========================
# Section 0: Import libraries
# ==========================
import torch_patch  
import streamlit as st
import os
from dotenv import load_dotenv
from typing import Dict, List, Tuple
import pandas as pd
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
import re
from get_url import get_privacy_policy_url
from summarize_text import get_summary_for_tos
from check_cache import LLMCacheTool
from openai import OpenAI

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import numpy as np
from textblob import TextBlob

# ✅ Patch to prevent Streamlit from inspecting torch.classes
import sys
import types
torch.classes = types.SimpleNamespace()
sys.modules['torch.classes'] = torch.classes

# Global variables
BERT_MODEL = None
MAX_TEXT_LENGTH = 10000
CACHE_DAYS = 7
MIN_INPUT_LENGTH = 3

def check_requirements():
    try:
        import transformers
        import torch
        import textblob
    except ImportError as e:
        st.error(f"Missing required package: {str(e)}")
        st.info("Please install required packages: pip install transformers torch textblob")
        st.stop()

# def configure():
#     """Load environment variables and initialize API keys"""
#     # load_dotenv()
#     # openai.api_key = os.getenv("OPENAI_API_KEY")
#     openai.api_key = st.secrets["OPENAI_API_KEY"]
#     try:
#         response = openai.ChatCompletion.create(...)
#     except Exception as e:
#         st.error(f"OpenAI error: {e}")
 
      
def configure():
    """Initialize OpenAI API key using new SDK (v1.x) with local/Streamlit fallback"""
    from dotenv import load_dotenv
    load_dotenv()

    try:
        # ✅ Use Streamlit secret if present, fallback to local .env
        api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
        client = OpenAI(api_key=api_key)
         # client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

        # ✅ Optional: test API connectivity
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"}
            ],
            max_tokens=5,
        )
        st.write("✅ OpenAI connectivity check passed.")
        st.session_state.openai_client = client  # 🔐 Store client globally in session
    except Exception as e:
        st.error(f"OpenAI error: {e}")
        

def get_bert_model():
    """Initialize or retrieve cached BERT model"""
    global BERT_MODEL
    if BERT_MODEL is None:
        try:
            BERT_MODEL = pipeline(
                "text-classification",
                model="nlptown/bert-base-multilingual-uncased-sentiment",
                tokenizer="nlptown/bert-base-multilingual-uncased-sentiment",
                device=-1  # Use CPU if GPU not available
            )
        except Exception as e:
            print(f"Error loading BERT model: {str(e)}")
            return None
    return BERT_MODEL

def chunk_text(text: str, max_length: int = 512) -> List[str]:
    """Split text into manageable chunks"""
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

# ==========================
# Section 1: Configure page
# ==========================
st.set_page_config(
    page_title="LegalLens - Privacy Policy Analyzer",
    page_icon="⚖️",
    layout="wide"
)

# ✅ Unified CSS for compact layout and alignment
st.markdown("""
    <style>
    /* Shrink block container padding */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }

    /* Tighten column content spacing */
    .stColumn > div {
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
    }

    /* Reduce margin above/below headings */
    h1, h2, h3, h4 {
        margin-top: 0.25rem;
        margin-bottom: 0.25rem;
    }

    /* Reduce space in expanders */
    details summary {
        font-size: 16px;
        margin-bottom: 0;
    }

    /* Optional: Compact metrics */
    div[data-testid="metric-container"] {
        padding: 0.25rem;
        margin: 0;
    }

    /* Align elements inside columns to the top */
    .stColumn {
        vertical-align: top;
    }
    /* Fix expander height so both columns align */
    .stExpander {
        min-height: 180px;
    }
    /* Optional: Prevent large gaps inside markdown boxes */
    .stMarkdown {
        margin-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)


# Custom CSS for risk colors
st.markdown("""
    <style>
    .high-risk { color: #FF0000; }
    .medium-risk { color: #FFA500; }
    .low-risk { color: #008000; }
    .risk-header-high { color: #FF0000; font-size: 20px; font-weight: bold; }
    .risk-header-medium { color: #FFA500; font-size: 20px; font-weight: bold; }
    .risk-header-low { color: #008000; font-size: 20px; font-weight: bold; }
    .risk-box {
        padding: 10px;
        border-radius: 5px;
        margin: 5px;
        background-color: rgba(255, 255, 255, 0.1);
    }
    .summary-high { color: #FF0000; }
    .summary-medium { color: #FFA500; }
    .summary-low { color: #008000; }
    .progress-bar {
        height: 20px;
        background-color: #f0f0f0;
        border-radius: 10px;
        margin: 10px 0;
    }
    .progress-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.3s ease;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================
# Section 2: Helper functions
# ==========================
def get_root_url(url: str) -> str:
    """Extract the root URL from a given URL"""
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"

def scrape_text_direct(url: str) -> str:
    """Directly scrape text from a given URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Find main content (usually in specific tags)
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content') or soup
        
        text = main_content.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
        
    except Exception as e:
        raise Exception(f"Error scraping {url}: {str(e)}")

def scrape_text(url: str) -> str:
    """Main function to scrape text from URL with improved error handling"""
    try:
        # Try direct URL first
        text = scrape_text_direct(url)
        if len(text) > MIN_INPUT_LENGTH:
            return text

        # Try root URL if direct URL fails
        root_url = get_root_url(url)
        if root_url != url:
            text = scrape_text_direct(root_url)
            if len(text) > MIN_INPUT_LENGTH:
                return text

        # Try common privacy policy paths
        common_paths = [
            '/privacy',
            '/privacy-policy',
            '/privacy_policy',
            '/legal/privacy',
            '/about/privacy'
        ]
        
        for path in common_paths:
            try:
                policy_url = f"{root_url}{path}"
                text = scrape_text_direct(policy_url)
                if len(text) > MIN_INPUT_LENGTH:
                    return text
            except:
                continue

        raise Exception("Could not find privacy policy text")
    except Exception as e:
        raise Exception(f"Error scraping text: {str(e)}")

def calculate_risk_score(risks: Dict[str, List[str]]) -> int:
    """Calculate overall risk score based on findings"""
    score = 100
    # Deduct points for each risk
    score -= len([r for r in risks['high'] if not r.startswith('No')]) * 15
    score -= len([r for r in risks['medium'] if not r.startswith('No')]) * 7
    score -= len([r for r in risks['low'] if not r.startswith('No')]) * 3
    return max(0, min(score, 100))

def generate_recommendations(risks: Dict[str, List[str]], score: int) -> List[str]:
    """Generate contextual recommendations based on risk keywords"""
    recommendations = []

    # 🚨 High-risk triggers
    if any("sell" in r.lower() or "share" in r.lower() for r in risks['high']):
        recommendations.append("🚨 Consider limiting data sharing by checking your privacy settings.")

    if any("track" in r.lower() or "location" in r.lower() for r in risks['high']):
        recommendations.append("📍 Disable location tracking in your account and device settings.")

    if any("biometric" in r.lower() or "facial recognition" in r.lower() for r in risks['high']):
        recommendations.append("🧬 Reconsider using services requiring biometric data without consent.")

    # ⚠️ Medium-risk triggers
    if any("cookie" in r.lower() or "ads" in r.lower() for r in risks['medium']):
        recommendations.append("🍪 Install a cookie manager browser extension to reduce tracking.")

    if any("analytics" in r.lower() for r in risks['medium']):
        recommendations.append("📊 Use privacy-focused browsers or analytics blockers like uBlock Origin.")

    # 📉 Score-based
    if score < 50:
        recommendations.append("⚠️ Overall privacy score is low. Consider using an alternative service.")

    # ✅ Default fallback
    if not recommendations:
        recommendations.append("✅ This privacy policy seems reasonable. No immediate action needed.")

    return recommendations



# ==========================
# Section 3: Analysis Functions
# ==========================


def analyze_privacy_policy_with_bert(text: str) -> Dict[str, List[str]]:
    """
    Analyze privacy policy text using BERT with improved error handling and progress tracking
    """
    try:
        # Get or initialize BERT model
        classifier = get_bert_model()
        if classifier is None:
            return analyze_privacy_policy_with_textblob(text)

        # Define comprehensive risk patterns
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

        # Add context words that increase risk level
        risk_amplifiers = [
            'without', 'unlimited', 'any purpose', 'all rights',
            'may', 'could', 'might', 'reserve the right',
            'at our discretion', 'not responsible', 'no liability'
        ]

        # Initialize risks dictionary
        risks = {
            'high': [],
            'medium': [],
            'low': []
        }

        # Split text into manageable chunks
        chunks = chunk_text(text)
        total_chunks = len(chunks)
        
        # Initialize progress tracking
        progress_bar = st.progress(0)
        progress_text = st.empty()

        # Analyze each chunk
        for i, chunk in enumerate(chunks):
            try:
                # Update progress
                progress = (i + 1) / total_chunks
                progress_bar.progress(progress)
                progress_text.text(f"Analyzing section {i+1} of {total_chunks}")
                
                # Get BERT sentiment
                result = classifier(chunk[:512])[0]
                score = int(result['label'][0])
                
                # Analyze sentences within chunk
                sentences = [s.strip() for s in chunk.split('.') if len(s.strip()) > 20]
                
                for sentence in sentences:
                    sentence_lower = sentence.lower()
                    
                    # Check for risk patterns
                    for risk_level, patterns in risk_patterns.items():
                        for pattern in patterns:
                            if pattern.lower() in sentence_lower:
                                # Check for risk amplifiers
                                has_amplifier = any(amp in sentence_lower for amp in risk_amplifiers)
                                
                                # Determine final risk level
                                final_risk_level = risk_level
                                if has_amplifier:
                                    if risk_level == 'low':
                                        final_risk_level = 'medium'
                                    elif risk_level == 'medium':
                                        final_risk_level = 'high'
                                
                                # Add sentiment indicators
                                formatted_sentence = sentence.strip()
                                if score <= 2:  # Negative sentiment
                                    formatted_sentence = "⚠️ " + formatted_sentence
                                elif score >= 4:  # Positive sentiment
                                    formatted_sentence = "✓ " + formatted_sentence
                                
                                risks[final_risk_level].append(formatted_sentence)

            except Exception as e:
                print(f"Error processing chunk {i}: {str(e)}")
                continue

        # Clean up progress indicators
        progress_bar.empty()
        progress_text.empty()

        # Process and clean up results
        for risk_level in risks:
            # Remove duplicates while preserving order
            seen = set()
            risks[risk_level] = [x for x in risks[risk_level] 
                               if not (x in seen or seen.add(x))]
            
            # Limit length and clean up
            risks[risk_level] = [
                (s[:150] + '...') if len(s) > 150 else s 
                for s in risks[risk_level]
            ]
            
            # Add default message if no risks found
            if not risks[risk_level]:
                risks[risk_level].append(f"No {risk_level}-risk items identified")

        return risks

    except Exception as e:
        print(f"BERT analysis failed, falling back to TextBlob: {str(e)}")
        return analyze_privacy_policy_with_textblob(text)
    
def explain_risk_categorization():
    """Explain how risks are categorized"""
    st.markdown("""
    ### How Risks Are Categorized

    🔴 **High Risk Items**:
    - Data selling or sharing with third parties
    - Collection of sensitive data (biometric, location)
    - Unlimited rights to user content
    - Lack of user consent or notification
    - Potential for data misuse or unauthorized access

    🟡 **Medium Risk Items**:
    - Cookie tracking and analytics
    - Marketing and advertising practices
    - User profiling and behavioral tracking
    - Data collection for non-essential purposes
    - Social media integration

    🟢 **Low Risk Items**:
    - Basic account information
    - Essential service functionality
    - Security measures and encryption
    - Customer service communication
    - Technical maintenance data

    **Note**: Risk levels may be elevated if the policy:
    - Uses vague or permissive language ("may", "could", "at our discretion")
    - Lacks clear user controls or consent mechanisms
    - Contains broad or unlimited rights claims
    """)

def calculate_privacy_score(risks: Dict[str, List[str]]) -> int:
    """Calculate proportional privacy score favoring no or minor risks."""
    total_risks = sum(
        len([r for r in risks[level] if not r.startswith('No')])
        for level in ['high', 'medium', 'low']
    )
    if total_risks == 0:
        return 100  # No risks = perfect score
    weighted_sum = (
        len([r for r in risks['high'] if not r.startswith('No')]) * 0.5 +
        len([r for r in risks['medium'] if not r.startswith('No')]) * 0.3 +
        len([r for r in risks['low'] if not r.startswith('No')]) * 0.2
    )
    penalty_ratio = weighted_sum / total_risks
    adjusted_score = 100 - (penalty_ratio * 100)
    return max(0, int(adjusted_score))

        


def analyze_privacy_policy_with_textblob(text: str) -> Dict[str, List[str]]:
    """
    Fallback analysis using TextBlob with improved pattern matching
    """
    try:
        # Define comprehensive risk patterns (same as BERT)
        risk_patterns = {
            'high': [
                'sell your data', 'share your data', 'third party', 
                'track location', 'biometric data', 'facial recognition',
                'sell personal information', 'share personal information',
                'data broker', 'surveillance', 'data mining', 'behavioral tracking'
            ],
            'medium': [
                'cookie', 'analytic', 'advertising', 'marketing email', 
                'promotional', 'profiling', 'targeted ads', 'tracking pixel',
                'social media', 'device information', 'usage data'
            ],
            'low': [
                'contact information', 'email address', 'basic account',
                'security measure', 'encryption', 'necessary data',
                'customer service', 'basic analytics', 'technical data'
            ]
        }
        
        risks = {
            'high': [],
            'medium': [],
            'low': []
        }

        # Create TextBlob object and analyze
        blob = TextBlob(text.lower())
        
        # Show progress
        progress_bar = st.progress(0)
        progress_text = st.empty()
        total_sentences = len(blob.sentences)

        for i, sentence in enumerate(blob.sentences):
            # Update progress
            progress = (i + 1) / total_sentences
            progress_bar.progress(progress)
            progress_text.text(f"Analyzing sentence {i+1} of {total_sentences}")

            sentence_text = str(sentence)
            if len(sentence_text) < 20:
                continue

            sentiment = sentence.sentiment.polarity
            
            for risk_level, patterns in risk_patterns.items():
                for pattern in patterns:
                    if pattern.lower() in sentence_text.lower():
                        risk_entry = sentence_text.strip()
                        if sentiment < -0.1:  # Negative sentiment
                            risk_entry = "⚠️ " + risk_entry
                        elif sentiment > 0.1:  # Positive sentiment
                            risk_entry = "✓ " + risk_entry
                        risks[risk_level].append(risk_entry)

        # Clean up progress indicators
        progress_bar.empty()
        progress_text.empty()

        # Clean up results
        for risk_level in risks:
            risks[risk_level] = list(set(risks[risk_level]))
            risks[risk_level] = [
                (s[:150] + '...') if len(s) > 150 else s 
                for s in risks[risk_level]
            ]
            if not risks[risk_level]:
                risks[risk_level].append(f"No {risk_level}-risk items identified")

        return risks

    except Exception as e:
        print(f"Error in TextBlob analysis: {str(e)}")
        return {
            'high': ['Analysis Error - Please try again'],
            'medium': ['Analysis Error - Please try again'],
            'low': ['Analysis Error - Please try again']
        }

# def analyze_privacy_policy(text: str) -> Dict[str, List[str]]:
#     """
#     Main analysis function with improved error handling
#     """
#     try:
#         with st.spinner("Analyzing with BERT..."):
#             return analyze_privacy_policy_with_bert(text)
#     except Exception as e:
#         st.warning("BERT analysis failed, falling back to TextBlob")
#         with st.spinner("Analyzing with TextBlob..."):
#             return analyze_privacy_policy_with_textblob(text)
        

# ==========================
# Section 4: Display Functions
# ==========================

def format_summary_with_risks(summary: str, risks: Dict[str, List[str]]) -> str:
    """Enhanced summary formatting with better risk highlighting and context"""
    formatted_summary = summary
    
    # Create a risk dictionary for quick lookup
    risk_terms = {}
    for risk_level, items in risks.items():
        for item in items:
            # Remove the warning symbols if present
            clean_item = item.replace('⚠️ ', '').replace('✓ ', '')
            # Take only the first part if there's an ellipsis
            clean_item = clean_item.split('...')[0]
            risk_terms[clean_item.lower()] = risk_level

    # Split summary into sentences
    sentences = formatted_summary.split('. ')
    formatted_sentences = []

    for sentence in sentences:
        sentence_lower = sentence.lower()
        risk_level = None
        
        # Check if sentence contains any risk terms
        for term, level in risk_terms.items():
            if term in sentence_lower:
                risk_level = level
                break
        
        # Format sentence based on risk level
        if risk_level:
            formatted_sentences.append(
                f"<span class='summary-{risk_level}'>{sentence}</span>"
            )
        else:
            formatted_sentences.append(sentence)

    return '. '.join(formatted_sentences)

def display_risk_meter(score: int, risks: Dict[str, List[str]]):
    """Display a consistent visual risk meter based purely on score."""
    st.markdown("### Overall Privacy Risk Score")
    
    if score >= 70:
        color = "green"
        message = "Low Risk"
        context = "This policy appears to have good privacy practices."
    elif score >= 40:
        color = "orange"
        message = "Medium Risk"
        context = "This policy has some concerning elements that should be reviewed."
    else:
        color = "red"
        message = "High Risk"
        context = "This policy has significant privacy concerns that require attention."
    
    col1, col2 = st.columns([2, 3])

    with col1:
        st.progress(score / 100)
        st.markdown(
            f"<h2 style='color: {color}; text-align: center;'>{score}/100</h2>",
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(f"<h3 style='color: {color};'>{message}</h3>", unsafe_allow_html=True)
        st.write(context)


def display_analysis(risks: Dict[str, List[str]], text: str):
    """Enhanced display with comprehensive risk analysis and recommendations"""
    try:
        # Calculate and display risk score
        # score = calculate_risk_score(risks)
        score = calculate_privacy_score(risks)

        
        
        display_risk_meter(score, risks)

        
        # Display categorized risks
        st.markdown("### Detailed Risk Analysis")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("<div class='risk-header-high'>🔴 High Risk Items</div>", 
                       unsafe_allow_html=True)
            high_risks = [r for r in risks['high'] if not r.startswith('No')]
            if high_risks:
                for item in high_risks:
                    st.markdown(f"<div class='risk-box high-risk'>• {item}</div>", 
                               unsafe_allow_html=True)
            else:
                st.markdown("<div class='risk-box'>✓ No high-risk items found</div>", 
                           unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='risk-header-medium'>🟡 Medium Risk Items</div>", 
                       unsafe_allow_html=True)
            medium_risks = [r for r in risks['medium'] if not r.startswith('No')]
            if medium_risks:
                for item in medium_risks:
                    st.markdown(f"<div class='risk-box medium-risk'>• {item}</div>", 
                               unsafe_allow_html=True)
            else:
                st.markdown("<div class='risk-box'>✓ No medium-risk items found</div>", 
                           unsafe_allow_html=True)
        
        with col3:
            st.markdown("<div class='risk-header-low'>🟢 Low Risk Items</div>", 
                       unsafe_allow_html=True)
            low_risks = [r for r in risks['low'] if not r.startswith('No')]
            if low_risks:
                for item in low_risks:
                    st.markdown(f"<div class='risk-box low-risk'>• {item}</div>", 
                               unsafe_allow_html=True)
            else:
                st.markdown("<div class='risk-box'>✓ No low-risk items found</div>", 
                           unsafe_allow_html=True)

        # # Add recommendations
        # st.markdown("### Recommendations")
        # recommendations = generate_recommendations(risks, score)
        # for rec in recommendations:
        #     st.info(rec)
        
        # Add recommendations inside an expander for alignment and compact view
        with st.expander("📌 Recommendations"):
            # Add a wrapper div to enforce consistent height for alignment
            st.markdown('<div style="min-height: 150px;">', unsafe_allow_html=True)

            recommendations = generate_recommendations(risks, score)
            if recommendations:
                for rec in recommendations:
                    st.info(rec)
            else:
                st.success("✅ No additional recommendations found.")

            st.markdown('</div>', unsafe_allow_html=True)



        # Add privacy impact summary
        st.markdown("### Privacy Impact Summary")
        impact_summary = generate_impact_summary(risks, score)
        st.write(impact_summary)

    except Exception as e:
        st.error(f"Error displaying analysis: {str(e)}")
        st.info("Falling back to simple display")
        # Fallback display
        for risk_level, items in risks.items():
            st.markdown(f"### {risk_level.title()} Risk Items")
            for item in items:
                st.write(f"• {item}")

def generate_impact_summary(risks: Dict[str, List[str]], score: int) -> str:
    """Generate a comprehensive privacy impact summary"""
    high_count = len([r for r in risks['high'] if not r.startswith('No')])
    medium_count = len([r for r in risks['medium'] if not r.startswith('No')])
    low_count = len([r for r in risks['low'] if not r.startswith('No')])
    
    summary = []
    
    if score >= 70:
        summary.append("🟢 This privacy policy demonstrates good privacy practices overall.")
    elif score >= 40:
        summary.append("🟡 This privacy policy has some areas that could be improved.")
    else:
        summary.append("🔴 This privacy policy raises significant privacy concerns.")
    
    if high_count > 0:
        summary.append(f"Found {high_count} high-risk practices that could significantly impact your privacy.")
    if medium_count > 0:
        summary.append(f"Identified {medium_count} medium-risk items that warrant attention.")
    if low_count > 0:
        summary.append(f"Noted {low_count} low-risk items that are generally acceptable but should be monitored.")
    
    return "\n\n".join(summary)

# ==========================
# Section 5: Main App
# ==========================
def main():
    st.title("Welcome to LegalLens! ⚖️")
    st.subheader("Your AI-Powered Legal Assistant")
    st.write("This app provides easy-to-understand summaries of "
    "privacy policies and terms of service (ToS) agreements.")

    # Sidebar with metrics and information
    with st.sidebar:
        st.header("Privacy Metrics")

        with st.expander("🔐 Your Privacy Rights"):
            st.markdown("""
            **You have the right to:**
            - Access your personal data
            - Delete your data
            - Opt out of data sales
            - Request data portability

            Regularly review privacy settings and use strong, unique passwords for better protection.
            """)

        with st.expander("📊 Risk Categories"):
            st.markdown("""
            **Risk Types:**
            - 🔴 **High Risk**: Major threats like data selling, location tracking without consent.
            - 🟡 **Medium Risk**: Moderate concerns like analytics tracking, cookie profiling.
            - 🟢 **Low Risk**: Minor items like account creation details, necessary services.

            Risks are assigned based on detected practices in the policy text.
            """)

        st.header("🔎 Privacy Scoring Approach")
        st.markdown("""
        - High Risk = **50%** impact
        - Medium Risk = **30%** impact
        - Low Risk = **20%** impact

        The final score adjusts based on the **proportion and severity** of identified risks.
        """)

        st.header("📈 Score Range Interpretation")
        st.markdown("""
        - 🟢 **≥ 70**: Low Risk (Good Privacy Practices)
        - 🟠 **40–69**: Medium Risk (Needs Attention)
        - 🔴 **< 40**: High Risk (Significant Concerns)
        """)

    
        st.subheader("Analytics")
        # st.metric("Documents Analyzed", "100+")
        # st.metric("Average Privacy Score", "75/100")
        
        with st.expander("Learn About Your Privacy Rights"):
            st.markdown("""
            ### Key Privacy Rights
            - Right to Access Your Data
            - Right to Delete Your Data
            - Right to Opt-Out
            - Right to Data Portability
            
            ### Privacy Best Practices
            - Regularly review privacy settings
            - Use strong, unique passwords
            - Enable two-factor authentication
            - Be cautious with third-party integrations
            """)
        
        with st.expander("About Risk Levels"):
            st.markdown("""
            🔴 **High Risk**: Practices that could significantly impact your privacy
            - Data selling
            - Extensive tracking
            - Broad data sharing
            
            🟡 **Medium Risk**: Practices that warrant attention
            - Cookie usage
            - Analytics
            - Marketing
            
            🟢 **Low Risk**: Generally acceptable practices
            - Basic account info
            - Essential services
            - Security measures
            """)
        with st.expander("How Are Risks Categorized?"):
            explain_risk_categorization()

    # Main content
    tab1, tab2 = st.tabs(["Single Analysis", "Compare Policies"])

    with tab1:
        input_type = st.selectbox(
            "To get started, select your preferred input method:",
            ["Company Name", "URL"],
            index=None,
            placeholder="Select input type",
        )

        if input_type:
            input_text = st.text_input(f"Enter {input_type}:").strip()
            
            if input_text:
                if len(input_text) < MIN_INPUT_LENGTH:
                    st.error(f"Please enter at least {MIN_INPUT_LENGTH} characters")
                    return

                try:
                    with st.spinner("Fetching policy..."):
                        if input_type == "Company Name":
                            privacy_policy_url = get_privacy_policy_url(os.getenv("API_KEY"), input_text)
                        else:
                            privacy_policy_url = input_text

                        if privacy_policy_url:
                            st.write("✅ URL found:", privacy_policy_url)
                            text = scrape_text(privacy_policy_url)

                            if text:
                                with st.spinner("Analyzing risks..."):
                                    cache = LLMCacheTool()
                                    
                                    # summary, risks, from_cache = get_summary_for_tos(privacy_policy_url, text, cache)
                                    summary, risks, from_cache = get_summary_for_tos(privacy_policy_url, text, cache, client=st.session_state.openai_client
)

                                    
                                    if from_cache:
                                        st.info("✅ Retrieved analysis from cache")
                                    else:
                                        st.success("✅ Analysis complete")

                                # Display risk analysis
                                display_analysis(risks, text)
                                
                                # Display color-coded summary
                                with st.expander("📄 Full Summary"):
                                    formatted_summary = format_summary_with_risks(summary, risks)
                                    st.markdown(formatted_summary, unsafe_allow_html=True)
                                
                                with st.expander("📄 Full Policy Text"):
                                    st.write(text)

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

    with tab2:
        col1, col2 = st.columns(2)
        
        # Initialize variables
        risks1 = None
        risks2 = None
        text1 = None
        text2 = None
        score1 = 0
        score2 = 0
        
        # with col1:
        #     company1 = st.text_input("First Company/URL")
        #     if company1:
        #         try:
        #             with st.spinner("Analyzing first policy..."):
        #                 url1 = get_privacy_policy_url(os.getenv("API_KEY"), company1) if not company1.startswith('http') else company1
        #                 text1 = scrape_text(url1)
        #                 cache = LLMCacheTool()
                        
        #                 summary1, risks1, from_cache1 = get_summary_for_tos(url1, text1, cache)

        #                 risks1 = analyze_privacy_policy(text1)
        #                 st.success("✅ First policy analyzed")
        #         except Exception as e:
        #             st.error(f"Error analyzing first policy: {str(e)}")
                    
        # with col2:
        #     company2 = st.text_input("Second Company/URL")
        #     if company2:
        #         try:
        #             with st.spinner("Analyzing second policy..."):
        #                 url2 = get_privacy_policy_url(os.getenv("API_KEY"), company2) if not company2.startswith('http') else company2
        #                 text2 = scrape_text(url2)
        #                 cache = LLMCacheTool()
                        
        #                 summary2, risks2, from_cache2 = get_summary_for_tos(url2, text2, cache)

        #                 risks2 = analyze_privacy_policy(text2)
        #                 st.success("✅ Second policy analyzed")
        #         except Exception as e:
        #             st.error(f"Error analyzing second policy: {str(e)}")
        
        # Analyze First Company
        with col1:
            company1 = st.text_input("First Company/URL")
            if company1:
                try:
                    with st.spinner("Analyzing first policy..."):
                        url1 = get_privacy_policy_url(os.getenv("API_KEY"), company1) if not company1.startswith('http') else company1
                        text1 = scrape_text(url1)
                        cache = LLMCacheTool()
                        
                        # summary1, risks1, from_cache1 = get_summary_for_tos(url1, text1, cache)
                        summary1, risks1, from_cache1 = get_summary_for_tos(url1, text1, cache, client=st.session_state.openai_client
)
                        st.success("✅ First policy analyzed")
                except Exception as e:
                    st.error(f"Error analyzing first policy: {str(e)}")

        # Analyze Second Company
        with col2:
            company2 = st.text_input("Second Company/URL")
            if company2:
                try:
                    with st.spinner("Analyzing second policy..."):
                        url2 = get_privacy_policy_url(os.getenv("API_KEY"), company2) if not company2.startswith('http') else company2
                        text2 = scrape_text(url2)
                        cache = LLMCacheTool()

                        # summary2, risks2, from_cache2 = get_summary_for_tos(url2, text2, cache)
                        summary2, risks2, from_cache2 = get_summary_for_tos(url2, text2, cache, client=st.session_state.openai_client
)
                        if risks1 and risks2:
                            score1 = calculate_privacy_score(risks1)
                            score2 = calculate_privacy_score(risks2)
                        st.success("✅ Second policy analyzed")
                except Exception as e:
                    st.error(f"Error analyzing second policy: {str(e)}")

        
        
          
        if company1 and company2 and risks1 and risks2:
            st.subheader("Comparison Results")

            comparison_cols = st.columns(2)

            with comparison_cols[0]:
                st.markdown(f"### 🔍 [{company1}]({url1})")
                display_analysis(risks1, text1)

            with comparison_cols[1]:
                st.markdown(f"### 🔍 [{company2}]({url2})")
                display_analysis(risks2, text2)

            # Key Differences
            st.markdown("### Key Differences")
            diff_col1, diff_col2, diff_col3 = st.columns(3)

            with diff_col1:
                diff_high = len(risks1['high']) - len(risks2['high'])
                st.metric("High Risk Items", f"{len(risks1['high'])} vs {len(risks2['high'])}", diff_high, delta_color="inverse")

            with diff_col2:
                diff_medium = len(risks1['medium']) - len(risks2['medium'])
                st.metric("Medium Risk Items", f"{len(risks1['medium'])} vs {len(risks2['medium'])}", diff_medium, delta_color="inverse")

            with diff_col3:
                diff_low = len(risks1['low']) - len(risks2['low'])
                st.metric("Low Risk Items", f"{len(risks1['low'])} vs {len(risks2['low'])}", diff_low, delta_color="inverse")

            # Overall Conclusion
            st.markdown("### Overall Comparison")
            if score1 > score2:
                st.success(f"🏆 {company1} has better privacy practices (Score: {score1} vs {score2})")
            elif score2 > score1:
                st.success(f"🏆 {company2} has better privacy practices (Score: {score2} vs {score1})")
            else:
                st.info("Both policies have similar privacy practices.")


if __name__ == "__main__":
    try:
        check_requirements()
        configure()
        main()
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.info("Please refresh the page and try again")
