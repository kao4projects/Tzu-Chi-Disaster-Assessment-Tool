import streamlit as st
import google.generativeai as genai
import json
import pandas as pd
import importlib.metadata

# --- 1. CONFIGURATION & SCORING FRAMEWORK ---
SCORING_FRAMEWORK = {
    "1. IMPACT": {
        "1.1 People Affected": 0.25,
        "1.2 Fatalities": 0.25,
        "1.3 People in Need": 0.30,
        "1.4 Housing & Building Damage": 0.10,
        "1.5 Land Mass Affected": 0.10
    },
    "2. HUMANITARIAN CONDITIONS": {
        "2.1 Food Security (IPC Score)": 0.35,
        "2.2 WASH / NFI Needs": 0.20,
        "2.3 Displacement": 0.20,
        "2.4 Vulnerable Groups Proportion": 0.10,
        "2.5 Health System": 0.15
    },
    "3. COMPLEXITY": {
        "3.1 Access (roads/airports)": 0.30,
        "3.2 Security": 0.30,
        "3.3 Government Capacity": 0.20,
        "3.4 Communications": 0.20
    },
    "4. STAKEHOLDER ATTENTION": {
        "4.1 Media Intensity": 0.25,
        "4.2 UN/INGO Activation": 0.20,
        "4.3 Internal Interest (Tzu Chi)": 0.55
    },
    "5. FEASIBILITY & PARTNERSHIPS": {
        "5.1 Local Partnerships": 0.40,
        "5.2 Legal & Financing": 0.40,
        "5.3 Culture & Faith Alignment": 0.20
    }
}

FLAT_WEIGHTS = {}
for cat, indicators in SCORING_FRAMEWORK.items():
    for name, weight in indicators.items():
        FLAT_WEIGHTS[name] = weight

# --- 2. SYSTEM PROMPT ---
SYSTEM_PROMPT = """
You are the assessment engine for the 'Tzu Chi Global Disaster Assessment Tool'.
Your goal is to research a humanitarian disaster and output structured JSON data based on the user's input.

STRICT SOURCE RULES:
- Use ONLY: OCHA, ReliefWeb, UN bodies (WHO, UNICEF, WFP), EU/ECHO, ACAPS, CNN, BBC, Reuters.
- Prioritise data from the last 7-30 days.
- If numbers conflict, choose the conservative/high-risk estimate.

OUTPUT FORMAT:
Return ONLY valid JSON with this structure (no markdown formatting):
{
  "summary": {
    "title": "Crisis Name",
    "country": "Country",
    "type": "Type",
    "date": "Date Range",
    "description": "2-4 sentence narrative."
  },
  "key_figures": {
    "affected": "Value (Source)",
    "fatalities": "Value (Source)",
    "displaced": "Value (Source)",
    "in_need": "Value (Source)",
    "housing_damage": "Value",
    "land_area": "Value"
  },
  "sources": [
    {"name": "Source Name", "date": "Date", "url": "URL"}
  ],
  "scores": {
    "1.1 People Affected": {"score": 1, "justification": "Reasoning..."},
    "1.2 Fatalities": {"score": 1, "justification": "Reasoning..."},
    "1.3 People in Need": {"score": 1, "justification": "Reasoning..."},
    "1.4 Housing & Building Damage": {"score": 1, "justification": "Reasoning..."},
    "1.5 Land Mass Affected": {"score": 1, "justification": "Reasoning..."},
    "2.1 Food Security (IPC Score)": {"score": 1, "justification": "Reasoning..."},
    "2.2 WASH / NFI Needs": {"score": 1, "justification": "Reasoning..."},
    "2.3 Displacement": {"score": 1, "justification": "Reasoning..."},
    "2.4 Vulnerable Groups Proportion": {"score": 1, "justification": "Reasoning..."},
    "2.5 Health System": {"score": 1, "justification": "Reasoning..."},
    "3.1 Access (roads/airports)": {"score": 1, "justification": "Reasoning..."},
    "3.2 Security": {"score": 1, "justification": "Reasoning..."},
    "3.3 Government Capacity": {"score": 1, "justification": "Reasoning..."},
    "3.4 Communications": {"score": 1, "justification": "Reasoning..."},
    "4.1 Media Intensity": {"score": 1, "justification": "Reasoning..."},
    "4.2 UN/INGO Activation": {"score": 1, "justification": "Reasoning..."},
    "4.3 Internal Interest (Tzu Chi)": {"score": 1, "justification": "Reasoning..."},
    "5.1 Local Partnerships": {"score": 1, "justification": "Reasoning..."},
    "5.2 Legal & Financing": {"score": 1, "justification": "Reasoning..."},
    "5.3 Culture & Faith Alignment": {"score": 1, "justification": "Reasoning..."}
  }
}
"""

# --- 3. HELPER FUNCTIONS ---

def calculate_final_metrics(scores_dict):
    raw_weighted_sum = 0.0
    for indicator, weight in FLAT_WEIGHTS.items():
        score = scores_dict.get(indicator, 3)
        raw_weighted_sum += score * weight
        
    severity_score = raw_weighted_sum / 5.0
    inform_score = severity_score * 2.0
    
    if severity_score >= 4.0:
        category = "A"
        label = "Major International"
        action = "Immediately initiate assessment, actively contact potential partners, review emergency budget, and submit internal proposal."
        color = "red"
    elif severity_score >= 2.5:
        category = "B"
        label = "Medium"
        action = "Maintain close communication with partners, track safety, and monitor for escalation."
        color = "orange"
    else:
        category = "C"
        label = "Minimal"
        action = "No proactive large-scale mobilisation; maintain observation and offer prayers."
        color = "green"
        
    return {
        "raw_sum": round(raw_weighted_sum, 2),
        "severity": round(severity_score, 1),
        "inform": round(inform_score, 1),
        "category": category,
        "cat_label": label,
        "action": action,
        "color": color
    }

def fetch_ai_assessment(api_key, query):
    """Calls Gemini API with Google Search Grounding."""
    try:
        genai.configure(api_key=api_key)
        
        # 1. USE STABLE MODEL:
        # We use "gemini-1.5-flash" because it is the most stable version for Search.
        # (gemini-2.5 does not exist yet).
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # 2. DICTIONARY TOOL CONFIGURATION (Version Safe):
        # Instead of importing the 'Tool' class (which causes errors on different versions),
        # we pass the configuration as a pure dictionary. 
        # Note: We use "google_search_retrieval" which is the correct key for version 0.8.5+.
        tools_payload = [
            {'google_search_retrieval': {
                'dynamic_retrieval_config': {
                    'mode': 'dynamic',
                    'dynamic_threshold': 0.7,
                }
            }}
        ]

        full_prompt = f"{SYSTEM_PROMPT}\n\nUSER QUERY: {query}"
        
        # 3. GENERATION
        response = model.generate_content(
            full_prompt,
            tools=tools_payload
        )
        
        text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
        
    except Exception as e:
        st.error(f"
