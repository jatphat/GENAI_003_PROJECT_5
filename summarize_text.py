import os
import textwrap
from dotenv import load_dotenv
# from openai import OpenAI
import openai

# Load both Serper API key and OpenAI key from .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set up OpenAI client
# load_dotenv()
# api_key = os.getenv("OPENAI_API_KEY")
# if not api_key:
#     raise ValueError("OPENAI_API_KEY not found in .env")
# openai.api_key = api_key
# # no need to assign `client = api_key` (remove that line)

# Load environment variables once
# load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Few-shot prompt to guide GPT behavior
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
- [list *all* high risk items found, one per line, even if many]
- [continue listing all detected high risk summaries]
(...or say "No high risk items identified")

Medium Risk:
- [list *all* medium risk items found, one per line, even if many]
(...or say "No medium risk items identified")

Low Risk:
- [list *all* low risk items found, one per line, even if many]
(...or say "No low risk items identified")

---

### Examples:

High Risk:
- The company shares user data with third parties without explicit consent.
- Biometric data like facial recognition information is collected without opt-out options.
- Sensitive location data is tracked continuously without clear purpose.
- User-generated content is granted perpetual and unlimited license rights.

Medium Risk:
- The service uses cookies and analytics tools to track user behavior.
- User profiles are built for targeted advertising.
- Usage statistics are retained for unspecified periods.

Low Risk:
- Basic account information like email addresses is collected for account creation.
- Essential security measures like data encryption are mentioned.
- Device information is collected for technical troubleshooting.

---

Now, summarize and classify the following privacy policy text:
"""

# ---------- LLM Summary Function ----------
# def summarize_long_text(text, model="gpt-4o-mini", chunk_size=3000, max_tokens=800):
#     """
#     Summarizes long policy text into structured privacy risks using OpenAI API.
#     """
#     chunks = textwrap.wrap(text, chunk_size)
#     chunk_summaries = []

#     for i, chunk in enumerate(chunks):
#         try:
#             print(f"Summarizing chunk {i+1} of {len(chunks)}...")
#             response = client.chat.completions.create(
#                 model=model,
#                 messages=[
#                     {"role": "system", "content": "You are a legal risk auditor specializing in data privacy."},
#                     {"role": "user", "content": FEW_SHOT_EXAMPLES + f"\n\nHere is the policy text:\n\n{chunk}"}
#                 ],
#                 max_tokens=max_tokens,
#                 temperature=0.3
#             )
#             chunk_summary = response.choices[0].message.content
#             chunk_summaries.append(chunk_summary)
#         except Exception as e:
#             print(f"Error summarizing chunk {i+1}: {e}")
#             chunk_summaries.append("")

#     combined_summary_text = "\n".join(chunk_summaries)

#     print("Creating final summary...")
#     try:
#         final_response = client.chat.completions.create(
#             model=model,
#             messages=[
#                 {"role": "system", "content": "You are a legal risk auditor summarizing data privacy risks."},
#                 {"role": "user", "content": FEW_SHOT_EXAMPLES + f"\n\nHere are the extracted summaries from different parts of the policy:\n\n{combined_summary_text}"}
#             ],
#             max_tokens=800,
#             temperature=0.3
#         )
#         final_summary = final_response.choices[0].message.content
#     except Exception as e:
#         print(f"Error creating final summary: {e}")
#         final_summary = "An error occurred while generating the final summary."

#     return final_summary

# def summarize_long_text(text, model="gpt-4o-mini", chunk_size=3000, max_tokens=800):
#     """
#     Summarizes long policy text into structured privacy risks using LLM API.
#     """
#     # ✂️ Auto-trim long text to avoid explosion
#     max_text_length = 8000
#     if len(text) > max_text_length:
#         print(f"Text too long ({len(text)} chars). Trimming to {max_text_length} chars...")
#         text = text[:max_text_length]

#     chunks = textwrap.wrap(text, chunk_size)
#     print(f"Original text length: {len(text)} characters")

#     chunk_summaries = []

#     for i, chunk in enumerate(chunks):
#         try:
#             print(f"Summarizing chunk {i+1} of {len(chunks)}...")
#             print(f"Length of combined summary being sent: {len(combined_summary_text)} characters")

#             response = openai.ChatCompletion.create(
#                 model=model,
#                 messages=[
#                     {"role": "system", "content": "You are a legal risk auditor specializing in data privacy."},
#                     {"role": "user", "content": FEW_SHOT_EXAMPLES + f"\n\nHere is the policy text:\n\n{chunk}"}
#                 ],
#                 max_tokens=max_tokens,
#                 temperature=0.3
#             )
#             chunk_summary = response['choices'][0]['message']['content']
#             chunk_summaries.append(chunk_summary)
#         except Exception as e:
#             print(f"Error summarizing chunk {i+1}: {e}")
#             chunk_summaries.append("")

#     # ✅ Combine summaries safely
#     combined_summary_text = "\n".join(chunk_summaries)
#     max_combined_length = 6000
#     if len(combined_summary_text) > max_combined_length:
#         print(f"Final combined text too big ({len(combined_summary_text)}), trimming to {max_combined_length} characters")
#         combined_summary_text = combined_summary_text[:max_combined_length]
#         print("Creating final summary...")

#     try:
#         final_response = openai.ChatCompletion.create(
#             model=model,
#             messages=[
#                 {"role": "system", "content": "You are a legal risk auditor summarizing data privacy risks."},
#                 {"role": "user", "content": FEW_SHOT_EXAMPLES + f"\n\nHere are the extracted summaries:\n\n{combined_summary_text}"}
#             ],
#             max_tokens=max_tokens,
#             temperature=0.3
#         )
#         final_summary = final_response['choices'][0]['message']['content']
#     except Exception as e:
#         print(f"Error creating final summary: {e}")
#         final_summary = "An error occurred while generating the final summary."

#     return final_summary

def summarize_long_text(text, model="gpt-4o-mini", chunk_size=3000, max_tokens=800):
    max_text_length = 8000
    if len(text) > max_text_length:
        print(f"Trimming input text from {len(text)} to {max_text_length}")
        text = text[:max_text_length]

    print(f"Input text after trimming: {len(text)} characters")

    chunks = textwrap.wrap(text, chunk_size)
    chunk_summaries = []

    for i, chunk in enumerate(chunks):
        try:
            print(f"Summarizing chunk {i+1}/{len(chunks)}...")
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a legal risk auditor specializing in data privacy."},
                    {"role": "user", "content": FEW_SHOT_EXAMPLES + f"\n\nHere is the policy text:\n\n{chunk}"}
                ],
                max_tokens=max_tokens,
                temperature=0.3
            )
            chunk_summary = response['choices'][0]['message']['content']
            chunk_summaries.append(chunk_summary)
        except Exception as e:
            print(f"Error summarizing chunk {i+1}: {e}")
            chunk_summaries.append("")

    # Now combine chunk summaries
    combined_summary_text = "\n".join(chunk_summaries)

    # ✂️ VERY IMPORTANT: TRIM before final summarization
    max_final_input_length = 5000  # characters, adjust if needed
    if len(combined_summary_text) > max_final_input_length:
        print(f"Combined summary text too large ({len(combined_summary_text)}), trimming to {max_final_input_length} characters")
        combined_summary_text = combined_summary_text[:max_final_input_length]

    print(f"Final summary input size: {len(combined_summary_text)} characters")

    try:
        final_response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a legal risk auditor summarizing data privacy risks."},
                {"role": "user", "content": FEW_SHOT_EXAMPLES + f"\n\nHere are the extracted summaries:\n\n{combined_summary_text}"}
            ],
            max_tokens=max_tokens,
            temperature=0.3
        )
        final_summary = final_response['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error creating final summary: {e}")
        final_summary = "An error occurred while generating the final summary."

    return final_summary


# def extract_risks_from_summary(summary: str) -> dict:
#     """
#     Extract high, medium, and low risk items from the LLM summary text.
#     Returns a dictionary with keys: 'high', 'medium', and 'low'.
#     """
#     risks = {'high': [], 'medium': [], 'low': []}
#     current_risk = None

#     for line in summary.splitlines():
#         line = line.strip()

#         if line.lower().startswith("high risk:"):
#             current_risk = 'high'
#         elif line.lower().startswith("warnings:") or line.lower().startswith("medium risk:"):
#             current_risk = 'medium'
#         elif line.lower().startswith("low risk:"):
#             current_risk = 'low'
#         elif line.startswith("- ") or line.startswith("• "):
#             if current_risk:
#                 risks[current_risk].append(line.lstrip("-• ").strip())

#     for level in risks:
#         if not risks[level]:
#             risks[level].append(f"No {level}-risk items identified")

#     return risks
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
        elif line.startswith("- ") or line.startswith("• "):
            if current_risk:
                risks[current_risk].append(line.lstrip("-• ").strip())

    for level in risks:
        if not risks[level]:
            risks[level].append(f"No {level}-risk items identified")

    return risks



# # ---------- Cached Summary Retrieval Function ----------
# def get_summary_for_tos(tos_url, tos_txt, cache_tool):
#     cached = cache_tool.get(tos_url)
#     if cached:
#         summary = cached["llm_summary"]
#         risks = extract_risks_from_summary(summary)
#         return summary, risks, True

#     summary = summarize_long_text(tos_txt)
#     risks = extract_risks_from_summary(summary)
#     cache_tool.add(tos_url, tos_txt, summary)
#     return summary, risks, False

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