# ==========================
# Section 0: Import Libraries
# ==========================
import streamlit as st
import os
import textwrap
from dotenv import load_dotenv
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import requests
from typing import Dict, List, Tuple
import openai
from get_url import get_privacy_policy_url
from check_cache import LLMCacheTool
from summarize_text import get_summary_for_tos, extract_risks_from_summary

# ==========================
# Section 1: Global Variables
# ==========================
FEW_SHOT_PROMPT = """
You are a world-class Data Privacy Risk Auditor AI.

Task: Analyze the provided privacy policy and extract **privacy-related risks**, classifying them into:
- High Risk
- Medium Risk
- Low Risk

Always return **all three sections**, even if no risks are found (e.g., say "No high risk items identified").

Classification Guide:
---
**High Risk**: Unauthorized data sharing/selling, biometric/location data without consent, unlimited content rights, indefinite data retention, missing user notification, surveillance potential.

**Medium Risk**: Cookie tracking, marketing profiling, analytics services, non-essential data collection, third-party integrations.

**Low Risk**: Basic account creation data, customer support interactions, encryption/security mentions, essential cookies, maintenance logs.
---

### Output Template:
High Risk:
- [bullet 1]
- [bullet 2]

Medium Risk:
- [bullet 1]
- [bullet 2]

Low Risk:
- [bullet 1]
- [bullet 2]

---

Now carefully review and classify the following policy text:
"""

# ==========================
# Section 2: Configuration
# ==========================

def configure():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env")
    openai.api_key = api_key

# ==========================
# Section 3: Helper Functions
# ==========================

def scrape_text(url: str) -> str:
    """Scrape the text content from the provided URL."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for tag in soup(["script", "style"]):
            tag.decompose()
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content') or soup
        text = ' '.join([line.strip() for line in main_content.get_text().splitlines() if line.strip()])
        return text
    except Exception as e:
        raise Exception(f"Error scraping {url}: {e}")

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


# ==========================
# Section 4: Streamlit App
# ==========================

def main():
    st.set_page_config(page_title="LegalLens - Privacy Policy Analyzer", page_icon="⚖️", layout="wide")
    st.title("LegalLens ⚖️")
    st.subheader("Analyze Privacy Policies for Risks")

   
    
    with st.sidebar:
        st.header("Privacy Metrics")
        
        with st.expander("Your Privacy Rights"):
            st.markdown("""
            - Right to Access
            - Right to Delete
            - Right to Opt-Out
            - Right to Data Portability
            """)

        with st.expander("Risk Categories"):
            st.markdown("""
            🔴 **High Risk**: Data selling, location tracking without consent  
            🟡 **Medium Risk**: Analytics, cookies  
            🟢 **Low Risk**: Necessary account details
            """)

        st.header("Privacy Scoring Approach")
        st.markdown("""
        - **High Risk**: Highest weight (50%)
        - **Medium Risk**: Medium weight (30%)
        - **Low Risk**: Lightest weight (20%)

        Overall score is based on the proportion and severity of identified risks.
        Policies with fewer and lower risks get higher scores.
        """)


    tab1, tab2 = st.tabs(["Single Analysis", "Compare Policies"])

    with tab1:
        input_type = st.selectbox("Select input type:", ["Company Name", "URL"])
        input_text = st.text_input(f"Enter {input_type}:")

        if input_text:
            if st.button("Analyze"):
                try:
                    with st.spinner("Fetching policy..."):
                        if input_type == "Company Name":
                            privacy_policy_url = get_privacy_policy_url(os.getenv("API_KEY"), input_text)
                        else:
                            privacy_policy_url = input_text

                        text = scrape_text(privacy_policy_url)
                        cache = LLMCacheTool()
                        summary, risks, from_cache = get_summary_for_tos(privacy_policy_url, text, cache)

                    st.success("✅ Analysis Complete")

                    st.header("Privacy Risk Summary")
                    privacy_score = calculate_privacy_score(risks)
                    st.metric("Overall Privacy Score", f"{privacy_score}/100")

                    for level in ["high", "medium", "low"]:
                        st.subheader(f"{level.capitalize()} Risk:")
                        for item in risks[level]:
                            st.markdown(f"- {item}")

                    st.header("Full Summary")
                    st.markdown(summary)

                except Exception as e:
                    st.error(f"Error: {e}")

    with tab2:
        col1, col2 = st.columns(2)

        risks1 = risks2 = None

        with col1:
            company1 = st.text_input("First Company/URL")
            if company1:
                try:
                    with st.spinner("Analyzing first policy..."):
                        url1 = get_privacy_policy_url(os.getenv("API_KEY"), company1) if not company1.startswith('http') else company1
                        text1 = scrape_text(url1)
                        cache = LLMCacheTool()
                        summary1, risks1, from_cache1 = get_summary_for_tos(url1, text1, cache)
                except Exception as e:
                    st.error(f"Error analyzing first policy: {e}")

        with col2:
            company2 = st.text_input("Second Company/URL")
            if company2:
                try:
                    with st.spinner("Analyzing second policy..."):
                        url2 = get_privacy_policy_url(os.getenv("API_KEY"), company2) if not company2.startswith('http') else company2
                        text2 = scrape_text(url2)
                        cache = LLMCacheTool()
                        summary2, risks2, from_cache2 = get_summary_for_tos(url2, text2, cache)
                except Exception as e:
                    st.error(f"Error analyzing second policy: {e}")

        if risks1 and risks2:
            st.subheader("Comparison Results")

            score1 = calculate_privacy_score(risks1)
            score2 = calculate_privacy_score(risks2)

            comparison_score_cols = st.columns(2)
            with comparison_score_cols[0]:
                st.metric(f"{company1} Privacy Score", f"{score1}/100")
            with comparison_score_cols[1]:
                st.metric(f"{company2} Privacy Score", f"{score2}/100")

            comparison_cols = st.columns(2)
            with comparison_cols[0]:
                st.markdown(f"### 🔍 {company1} — Score: {score1}/100")
                for level in ["high", "medium", "low"]:
                    st.subheader(f"{level.capitalize()} Risk:")
                    for item in risks1[level]:
                        st.markdown(f"- {item}")

            with comparison_cols[1]:
                st.markdown(f"### 🔍 {company2} — Score: {score2}/100")
                for level in ["high", "medium", "low"]:
                    st.subheader(f"{level.capitalize()} Risk:")
                    for item in risks2[level]:
                        st.markdown(f"- {item}")

# ==========================
# Section 6: App Entry Point
# ==========================

if __name__ == "__main__":
    try:
        configure()
        main()
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.info("Please refresh the page and try again.")
