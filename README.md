# GENAI_003_PROJECT_5 - LegalLens

# 🛡️ Terms & Conditions Risk Analyzer

An LLM-powered web app that summarizes and highlights risk factors hidden within lengthy Terms & Conditions (T&Cs). Built to empower users with clear, digestible insights into what they're agreeing to—before clicking "Accept."

---

## 🚀 Project Overview

Terms & Conditions are often too long and complex for the average user to read. This project leverages large language models (LLMs) to parse, summarize, and score T&C clauses by risk level. Our tool scrapes publicly available policy documents, breaks them down into digestible parts, and categorizes each clause as **Low (Safe)**, **Medium (Caution)**, or **High Risk (Red Flag)**.

---

## 📅 Timeline & Phases

| Phase | Dates | Key Activities | Deliverables |
|-------|-------|----------------|--------------|
| **Phase 1** | Mar 31 – Apr 2 | Requirements gathering, competitor research | Competitor analysis, user personas, risk rubric |
| **Phase 2** | Apr 14 – Apr 18 | MVP development (scraper, LLM testing) | Scraper, preprocessing pipeline, early summaries |
| **Phase 3** | Apr 20 – Apr 25 | Model integration & UI | Interactive frontend, integrated LLM output |
| **Phase 4** | Apr 25 – Apr 27 | Deployment & testing | Full working prototype on AWS, fallback solutions |
| **Final Demo** | Apr 28 | Presentation & walkthrough | Final pitch deck and demo |

---

## 🧠 Features

- 🔍 **T&C Scraper** – Automatically pulls Terms & Conditions from public sources using Serper API
- 📄 **Clause Segmentation** – Breaks text into clean, meaningful legal clauses
- 🤖 **LLM-Based Summarization** – Uses GPT-style models to generate plain-English summaries
- ⚠️ **Risk Classification** – Labels clauses as Safe, Medium Risk, or High Risk
- 🎨 **User-Friendly UI** – Color-coded, mobile-friendly design with tooltip explanations
- 📚 **Educational Popups & Quizzes** – Helps users learn about legal language and red flags

---

## 🧩 Tech Stack

| Component | Tool |
|----------|------|
| **Frontend** | Streamlit |
| **Backend** | Python |
| **Scraping** | Serper API, BeautifulSoup |
| **LLM** | OpenAI GPT, Llama3 and open-source alternatives |
| **Cloud** | TBD |
| **Data Storage** | SQLite / JSON (MVP phase) |
| **Collaboration** | GitHub, Google Docs |

---

## 🔍 Risk Scoring Rubric

| Example Clause | Risk Level | Reason |
|----------------|------------|--------|
| “We do not sell your personal information.” | ✅ Safe | Privacy-preserving |
| “We use cookies for personalization and marketing.” | ⚠️ Medium Risk | May track user behavior |
| “By continuing, you consent to all tracking and data collection.” | ❌ High Risk | Full consent without clarity |

---

## 🔧 Setup Instructions

> ⚠️ MVP may require environment variables for API keys

1. Clone the repository:
   ```bash
   git clone https://github.com/NivethithaP-Rajan/GENAI_003_PROJECT_5.git
   cd GENAI_003_PROJECT_5
    ```
2. Install dependencies:
   ```bash
    pip install -r requirements.txt
    ```

    Add API keys in .env:
    ```bash
    SERPER_API_KEY=your-key
    OPENAI_API_KEY=your-key
    ```

3. 🧪 Testing



---

## 🛠️ Known Issues / Contingencies




---

## 👥 Team
| Name | Role |
|----------------|------------|
Jehoshaphat | Scraping & Integration
Mark | Frontend & UI
Nivethitha | Preprocessing & Prompt Dev
Pavithra | LLM Fine-Tuning
Venkatesh | LLM, Features & Frontend

---

## 📌 Future Enhancements

    📂 Support for more legal document types (Privacy Policy, Cookie Policy)

    🌍 Multi-language support

    🧑‍⚖️ Legal expert review integration

    📊 Risk dashboard for comparing multiple sites

---

## 💎 License

This project is licensed under the proprietary License.

---

## 💬 Contact

For feedback or collaboration, please reach out to your-email@example.com or open an issue in this repo.