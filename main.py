# ==========================
# Section 0: Import libraries
# ==========================
import streamlit as st
import os
from dotenv import load_dotenv
from get_url import get_privacy_policy_url
from scrape_text import scrape_text
from summarize_text import summarize_long_text

# ==========================
# Section 1: Set up streamlit app
# ==========================

st.title("Welcome to LegalLens!")
st.subheader("Your AI-Powered Legal Assistant")
st.write("This app provides easy-to-understand summaries of "
"privacy policies and terms of service (ToS) agreements.")


input_type = st.session_state.input_type = st.selectbox(
    "To get started, select your preferred input method:",
    ["Company Name", "URL"],
    index = None,
    placeholder = "Select input type",
    )

input = st.text_input(f"Enter {input_type}:")

# ==========================
# Section 2: Import and call functions to extract URL and scrape text
# ==========================
def configure():
    load_dotenv()
configure()

api_key = os.getenv("API_KEY")

if input:
    if input_type == "Company Name":
        privacy_policy_url = get_privacy_policy_url(api_key, input)
    elif input_type == "URL":
        privacy_policy_url = input
else:
    privacy_policy_url = ""

if privacy_policy_url:
    st.write("✅ Here is the URL I found:", privacy_policy_url)
    text = scrape_text(privacy_policy_url)

    if text:
        with st.spinner("Analyzing risks using GPT..."):
            summary = summarize_long_text(text)
        st.subheader("⚠️ Privacy Risk Summary")
        st.write(summary)

        with st.expander("📄 Full Extracted Policy Text"):
            st.write(text)



