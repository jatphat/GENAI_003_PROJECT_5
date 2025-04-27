# ================================
# summarize_text_llama.py
# LLaMA version of summarization
# ================================

import os
import textwrap
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

if not TOGETHER_API_KEY:
    raise ValueError("TOGETHER_API_KEY not found in .env")

# Few-shot prompt
FEW_SHOT_EXAMPLES = """
You are a highly specialized Data Privacy Risk Auditor AI.

Your task is to read the provided privacy policy text, and **strictly extract privacy-related risks only**. You must classify the extracted risks into **three risk levels**:
- High Risk
- Medium Risk
- Low Risk

You must always return **all three categories**, even if there are no findings (e.g., say "No high risk items identified").

### How to Classify Risks:
**High Risk**:
- Sharing or selling personal data to third parties without consent
- Collecting sensitive data (biometric, location, financial)
- Claiming unlimited rights to user content, waiving moral rights
- Retaining data after account deletion without user control
- Lack of user notification for changes to policies
- Potential for data misuse, unauthorized access, surveillance

**Medium Risk**:
- Use of cookie tracking and analytics services
- Profiling users for marketing and advertising
- Collection of information for non-essential purposes
- Use of tracking pixels or social media integration
- Retention of usage statistics and behavior data

**Low Risk**:
- Collection of basic account information (email, username)
- Customer service communications
- Implementation of encryption and security measures
- Use of essential cookies for website functionality
- Collection of technical maintenance data (device info)

---

### Output Format (Strict):

High Risk:
- [list all high risk items found]

Medium Risk:
- [list all medium risk items found]

Low Risk:
- [list all low risk items found]

---
"""

# Helper function to call LLaMA via Together.ai API
def llama_chat(prompt: str, model="togethercomputer/llama-2-13b-chat", temperature=0.3, max_tokens=800):
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    response = requests.post(url, json=body, headers=headers)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# ---------- LLM Summary Function ----------
def summarize_long_text(text, model="togethercomputer/llama-2-13b-chat", chunk_size=3000, max_tokens=800):
    chunks = textwrap.wrap(text, chunk_size)
    chunk_summaries = []

    for i, chunk in enumerate(chunks):
        try:
            print(f"Summarizing chunk {i+1} of {len(chunks)}...")
            prompt = FEW_SHOT_EXAMPLES + f"\n\nHere is the policy text:\n\n{chunk}"
            chunk_summary = llama_chat(prompt, model=model, max_tokens=max_tokens)
            chunk_summaries.append(chunk_summary)
        except Exception as e:
            print(f"Error summarizing chunk {i+1}: {e}")
            chunk_summaries.append("")

    combined_summary_text = "\n".join(chunk_summaries)

    print("Creating final summary...")
    try:
        final_prompt = FEW_SHOT_EXAMPLES + f"\n\nHere are the extracted summaries from different parts of the policy:\n\n{combined_summary_text}"
        final_summary = llama_chat(final_prompt, model=model, max_tokens=max_tokens)
    except Exception as e:
        print(f"Error creating final summary: {e}")
        final_summary = "An error occurred during summarization. Please try again later."

    return final_summary

# ---------- Risk Extraction Function ----------
def extract_risks_from_summary(summary: str) -> dict:
    risks = {'high': [], 'medium': [], 'low': []}
    current_risk = None

    for line in summary.splitlines():
        line = line.strip()

        if line.lower().startswith("high risk:"):
            current_risk = 'high'
        elif line.lower().startswith("medium risk:") or line.lower().startswith("warnings:"):
            current_risk = 'medium'
        elif line.lower().startswith("low risk:"):
            current_risk = 'low'
        elif line.startswith("- ") or line.startswith("\u2022 "):
            if current_risk:
                risks[current_risk].append(line.lstrip("-\u2022 ").strip())

    for level in risks:
        if not risks[level]:
            risks[level].append(f"No {level}-risk items identified")

    return risks

# ---------- Cached Summary Retrieval ----------
def get_summary_for_tos(tos_url, tos_txt, cache_tool):
    cached = cache_tool.get(tos_url)
    if cached:
        summary = cached["llm_summary"]
        risks = extract_risks_from_summary(summary)
        return summary, risks, True

    summary = summarize_long_text(tos_txt)
    risks = extract_risks_from_summary(summary)
    cache_tool.add(tos_url, tos_txt, summary)
    return summary, risks, False

