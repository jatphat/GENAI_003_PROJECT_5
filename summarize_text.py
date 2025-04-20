import os
import textwrap
from dotenv import load_dotenv
from openai import OpenAI

# Load both Serper API key and OpenAI key from .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set up OpenAI client
client = OpenAI(api_key=openai_api_key)

# Few-shot prompt to guide GPT behavior
FEW_SHOT_EXAMPLES = """
You are a data privacy auditor AI. Summarize the privacy risks mentioned in the text below, and group them into three categories: High Risk, Warnings, and Low Risk.

Only extract risks related to **data privacy** and **user data usage**. Do not include general terms, contact info, or formatting instructions.

Format the output as:

High Risk:
- Risks where there is a threat to the user's data being used or held without consent, including loss, misuse, or alteration, data being sold to third parties and the identity of the user being compromised. 

Warnings:
- Risks where the service collects personal data and very limited rights for the user, in case of disputes.

Low Risk:
- Risks of the onus of the data is only with the user, and any other risks not included in the above categories.

Examples:

High Risk:
- You waive your moral rights 
- This service still tracks you even if you opted out from tracking
- This service may keep personal data after a request for erasure for business interests or legal obligations

Warnings:
- You must provide your identifiable information
- This service can share your personal information to third parties
- This service forces users into binding arbitration in the case of disputes

Low Risk:
- Users agree not to use the service for illegal purposes
- Third parties may be involved in operating the service
- The service may change its terms at any time
"""

def summarize_long_text(text, model="gpt-4o", chunk_size=3000, max_tokens=500):
    """
    Summarizes long policy text into structured privacy risks using OpenAI API.
    """
    chunks = textwrap.wrap(text, chunk_size)
    chunk_summaries = []

    for i, chunk in enumerate(chunks):
        try:
            print(f"Summarizing chunk {i+1} of {len(chunks)}...")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a legal risk auditor specializing in data privacy."},
                    {"role": "user", "content": FEW_SHOT_EXAMPLES + f"\n\nHere is the policy text:\n\n{chunk}"}
                ],
                max_tokens=max_tokens,
                temperature=0.3
            )
            chunk_summary = response.choices[0].message.content
            chunk_summaries.append(chunk_summary)
        except Exception as e:
            print(f"Error summarizing chunk {i+1}: {e}")
            chunk_summaries.append("")

    combined_summary_text = "\n".join(chunk_summaries)

    print("Creating final summary...")
    try:
        final_response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a legal risk auditor summarizing data privacy risks."},
                {"role": "user", "content": FEW_SHOT_EXAMPLES + f"\n\nHere are the extracted summaries from different parts of the policy:\n\n{combined_summary_text}"}
            ],
            max_tokens=800,
            temperature=0.3
        )
        final_summary = final_response.choices[0].message.content
    except Exception as e:
        print(f"Error creating final summary: {e}")
        final_summary = "An error occurred while generating the final summary."

    return final_summary
