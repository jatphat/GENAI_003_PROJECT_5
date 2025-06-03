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
# Section 3: Core Functions
# ==========================

def scrape_text_direct(url: str) -> str:
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for tag in soup(['script', 'style']):
            tag.decompose()
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content') or soup
        text = ' '.join([line.strip() for line in main_content.get_text().splitlines() if line.strip()])
        return text
    except Exception as e:
        raise Exception(f"Error scraping {url}: {e}")

def get_root_url(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"

def scrape_text(url: str) -> str:
    text = scrape_text_direct(url)
    if len(text) > 100:
        return text
    root_url = get_root_url(url)
    if root_url != url:
        text = scrape_text_direct(root_url)
    return text

def summarize_policy(text: str, model="gpt-4o", chunk_size=2800, max_tokens=800) -> str:
    paragraphs = textwrap.wrap(text, width=chunk_size)
    all_summaries = []
    for idx, paragraph in enumerate(paragraphs):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert privacy risk auditor."},
                    {"role": "user", "content": FEW_SHOT_PROMPT + f"\n\nPolicy Text:\n\n{paragraph}"}
                ],
                max_tokens=max_tokens,
                temperature=0.2
            )
            all_summaries.append(response.choices[0].message['content'].strip())
        except Exception as e:
            all_summaries.append(f"Error: {e}")
    combined = "\n".join(all_summaries)
    final_response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are refining a privacy policy audit."},
            {"role": "user", "content": FEW_SHOT_PROMPT + f"\n\nCombined Summaries:\n\n{combined}"}
        ],
        max_tokens=max_tokens,
        temperature=0.2
    )
    return final_response.choices[0].message['content'].strip()

def extract_risks(summary_text: str) -> Dict[str, List[str]]:
    risks = {'high': [], 'medium': [], 'low': []}
    current = None
    for line in summary_text.splitlines():
        line = line.strip()
        if line.lower().startswith("high risk:"):
            current = 'high'
        elif line.lower().startswith("medium risk:"):
            current = 'medium'
        elif line.lower().startswith("low risk:"):
            current = 'low'
        elif line.startswith("- ") or line.startswith("\u2022 "):
            if current:
                risks[current].append(line.lstrip("-\u2022 ").strip())
    for risk_level in risks:
        if not risks[risk_level]:
            risks[risk_level].append(f"No {risk_level} risk items identified.")
    return risks

def get_summary_and_risks(tos_url: str, tos_text: str, cache_tool) -> Tuple[str, Dict[str, List[str]], bool]:
    cached = cache_tool.get(tos_url)
    if cached:
        summary = cached["llm_summary"]
        risks = extract_risks(summary)
        return summary, risks, True
    summary = summarize_policy(tos_text)
    risks = extract_risks(summary)
    cache_tool.add(tos_url, tos_text, summary)
    return summary, risks, False

# ==========================
# Section 4: Utility Functions
# ==========================

def calculate_privacy_score(risks: Dict[str, List[str]]) -> int:
    score = 100
    score -= len([r for r in risks['high'] if not r.startswith('No')]) * 15
    score -= len([r for r in risks['medium'] if not r.startswith('No')]) * 7
    score -= len([r for r in risks['low'] if not r.startswith('No')]) * 3
    return max(0, min(score, 100))

# ==========================
# Section 5: Streamlit App
# ==========================

def main():
    st.set_page_config(page_title="LegalLens - Privacy Policy Analyzer", page_icon="⚖️", layout="wide")
    st.title("LegalLens ⚖️")
    st.subheader("Analyze Privacy Policies for Risks")

    with st.sidebar:
        st.header("Privacy Metrics")
        st.info("🔍 This app evaluates privacy policies based on common risk factors.")
        st.metric("High Risk Impact", "-15 points per item")
        st.metric("Medium Risk Impact", "-7 points per item")
        st.metric("Low Risk Impact", "-3 points per item")
        with st.expander("Your Privacy Rights"):
            st.markdown("""
            - Right to Access
            - Right to Delete
            - Right to Opt-Out
            - Right to Data Portability
            """)
        with st.expander("Risk Categories"):
            st.markdown("""
            🔴 High Risk: Data selling, location tracking without consent
            🟡 Medium Risk: Analytics, cookies
            🟢 Low Risk: Necessary account details
            """)

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
                    summary, risks, from_cache = get_summary_and_risks(privacy_policy_url, text, cache)

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

# ==========================
# Section 6: Run App
# ==========================

if __name__ == "__main__":
    configure()
    main()
