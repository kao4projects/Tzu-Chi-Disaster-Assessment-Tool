import streamlit as st
from google import genai
from google.genai import types
import json
import pandas as pd
import re
import time

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Tzu Chi Disaster Tool", layout="wide")
st.title("Tzu Chi Global Disaster Assessment Tool (Deep Search v3)")

# --- 2. API KEY SETUP ---
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("Missing GOOGLE_API_KEY. Please add it to Streamlit Secrets.")
    st.stop()

# --- 3. SCORING FRAMEWORK ---
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

# --- 4. SYSTEM PROMPT (AGGRESSIVE RESEARCH MODE) ---
SYSTEM_PROMPT = """
You are the Lead Researcher for the 'Tzu Chi Disaster Assessment Unit'.
Your task is to find SPECIFIC NUMBERS and SOURCES for a humanitarian disaster.


### RESEARCH PROTOCOL:
STRICT SOURCE RULES:
- Use ONLY: OCHA, ReliefWeb, UN bodies (WHO, UNICEF, WFP), EU/ECHO, ACAPS, CNN, BBC, Reuters.
- Prioritise data from the last 7-30 days.
- If numbers conflict, choose the conservative/high-risk estimate.
**Handle Uncertain Data:** If exact official figures are not yet released, YOU MUST REPORT THE ESTIMATES found in news (e.g., "Reports indicate >300 dead" is better than "Unknown").
**Source Triangulation:** For every data point, try to find 2-3 sources (e.g., ReliefWeb, Major News Outlet, UN).

### OUTPUT FORMAT (JSON ONLY):
{
  "summary": {
    "title": "Official Crisis Name",
    "country": "Country",
    "date": "Date of Event",
    "description": "3-4 sentence summary of the situation."
  },
  "key_figures": {
    "affected": {"value": "e.g. 1.2 million", "source_url": "url..."},
    "fatalities": {"value": "e.g. 350 confirmed", "source_url": "url..."},
    "displaced": {"value": "e.g. 15,000 in shelters", "source_url": "url..."},
    "in_need": {"value": "e.g. 300,000", "source_url": "url..."}
  },
  "scores": {
    "1.1 People Affected": {
      "score": 1-5, 
      "extracted_value": "The specific number found (e.g., '1.4 million')",
      "justification": "Why this score? (e.g., >1M is Critical)",
      "source_urls": ["url1", "url2", "url3"]
    },
    ... (Repeat for ALL 19 indicators) ...
  }
}

### SCORING RUBRIC:
- **Fatalities:** 1(<10), 2(10-100), 3(100-500), 4(500-1k), 5(>1k)
- **Affected:** 1(<10k), 3(100k-1M), 5(>1M)
- **Displaced:** 1(<1k), 3(10k-50k), 5(>100k)
"""

# --- 5. CORE LOGIC ---

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
        action = "IMMEDIATE MOBILISATION: Initiate assessment team, release emergency funds, alert international partners."
        color = "#ff4b4b" # Red
    elif severity_score >= 2.5:
        category = "B"
        label = "Medium Scale"
        action = "WATCH LIST: Maintain contact with local partners, prepare standby funds, monitor for 48h."
        color = "#ffa421" # Orange
    else:
        category = "C"
        label = "Local / Minimal"
        action = "MONITORING: Local volunteers to assess; no HQ deployment likely needed."
        color = "#09ab3b" # Green
        
    return {
        "severity": round(severity_score, 1),
        "inform": round(inform_score, 1),
        "category": category,
        "cat_label": label,
        "action": action,
        "color": color
    }

def robust_json_extractor(text):
    """
    Robustly extracts JSON from a string using raw_decode to ignore trailing text.
    """
    try:
        start_idx = text.find('{')
        if start_idx == -1: return None
        text_trimmed = text[start_idx:]
        obj, _ = json.JSONDecoder().raw_decode(text_trimmed)
        return obj
    except Exception as e:
        return None

def fetch_ai_assessment(api_key, query):
    """Calls Gemini API with Re-Prompting logic for deeper search."""
    try:
        client = genai.Client(api_key=api_key)
        
        # 1. Primary Search Prompt
        full_prompt = f"{SYSTEM_PROMPT}\n\nUSER QUERY: {query}\n\nIMPORTANT: If the event is in the future (e.g. 2025), search for it as if it is happening NOW. Do not say 'it hasn't happened yet'."
        
        # Tools configuration (Google Search enabled)
        tool_config = types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            response_mime_type="application/json"
        )

        # 2. Execute Call
        # We prefer gemini-2.0-flash for better search capabilities
        try:
            response = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=full_prompt,
                config=tool_config
            )
        except Exception:
            st.warning("Gemini 2.0 Flash busy, falling back to 1.5 Flash...")
            response = client.models.generate_content(
                model='gemini-1.5-flash',
                contents=full_prompt,
                config=tool_config
            )

        if not response.text:
            return None

        # 3. Parse Data
        data = robust_json_extractor(response.text)
        
        # 4. (Optional) Validation - Check if we got "Unknown"
        # If 'fatalities' is unknown, we could theoretically trigger a second specific prompt here.
        # For this version, we trust the "Aggressive Search" prompt.
        
        return data
        
    except Exception as e:
        st.error(f"API Error details: {e}")
        return None

# --- 6. UI RENDER ---

query = st.text_area("Describe the disaster (Location, Date, Type):", 
                     placeholder="e.g., Cyclone Ditwah, Sri Lanka, Nov 2025")
run_btn = st.button("Start Deep Research", type="primary")

if "assessment_data" not in st.session_state:
    st.session_state.assessment_data = None
if "current_scores" not in st.session_state:
    st.session_state.current_scores = {}

if run_btn and query:
    with st.spinner("🔍 Accessing ReliefWeb, OCHA & Local News... (15-20s)"):
        data = fetch_ai_assessment(api_key, query)
        
        if data:
            st.session_state.assessment_data = data
            # Pre-load scores
            for key, val in data.get("scores", {}).items():
                st.session_state.current_scores[key] = val.get("score", 3)
        else:
            st.error("Failed to retrieve data. Please try again.")

if st.session_state.assessment_data:
    data = st.session_state.assessment_data
    
    # --- HEADER ---
    st.divider()
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.title(f"{data['summary']['title']}")
        st.caption(f"📍 {data['summary']['country']} | 📅 {data['summary']['date']}")
        st.success(data['summary']['description'])
        
    # --- KEY FIGURES SECTION ---
    with col2:
        st.subheader("Key Figures")
        kf = data.get('key_figures', {})
        
        # Helper to format key figure display
        def fmt_kf(kf_data):
            if isinstance(kf_data, dict):
                val = kf_data.get('value', 'Unknown')
                url = kf_data.get('source_url', '#')
                return val, url
            return kf_data, '#'

        aff_val, aff_url = fmt_kf(kf.get('affected', {}))
        fat_val, fat_url = fmt_kf(kf.get('fatalities', {}))
        disp_val, disp_url = fmt_kf(kf.get('displaced', {}))
        need_val, need_url = fmt_kf(kf.get('in_need', {}))

        # Custom metrics
        c1, c2 = st.columns(2)
        c1.metric("Affected", aff_val)
        if aff_url != '#': c1.markdown(f"[Source]({aff_url})")
        
        c2.metric("Fatalities", fat_val)
        if fat_url != '#': c2.markdown(f"[Source]({fat_url})")
        
        c3, c4 = st.columns(2)
        c3.metric("Displaced", disp_val)
        if disp_url != '#': c3.markdown(f"[Source]({disp_url})")
        
        c4.metric("In Need", need_val)
        if need_url != '#': c4.markdown(f"[Source]({need_url})")

    # --- TABS FOR DETAILED SCORING ---
    st.divider()
    st.subheader("Detailed Assessment & Evidence")
    
    tabs = st.tabs(list(SCORING_FRAMEWORK.keys()))
    
    for i, (dim_name, indicators) in enumerate(SCORING_FRAMEWORK.items()):
        with tabs[i]:
            for indicator_name, weight in indicators.items():
                # Get AI Data
                ai_data = data["scores"].get(indicator_name, {})
                ai_score = ai_data.get("score", 3)
                ai_value = ai_data.get("extracted_value", "No specific data point found.")
                ai_just = ai_data.get("justification", "No justification provided.")
                ai_urls = ai_data.get("source_urls", [])

                # UI Layout for each Row
                with st.container():
                    c1, c2, c3 = st.columns([2, 4, 1])
                    
                    with c1:
                        st.markdown(f"**{indicator_name}**")
                        st.caption(f"Weight: {weight}")
                        
                    with c2:
                        # Highlight the extracted value
                        if "Unknown" in str(ai_value):
                             st.markdown(f"**Evidence:** `{ai_value}`", help="AI could not find exact number")
                        else:
                             st.markdown(f"**Evidence:** `{ai_value}`")
                        
                        st.write(f"_{ai_just}_")
                        
                        # Render Sources as multiple links
                        if ai_urls and isinstance(ai_urls, list):
                            links_md = " | ".join([f"[Source {j+1}]({url})" for j, url in enumerate(ai_urls) if url])
                            st.markdown(f"🔗 Found in: {links_md}")
                        elif isinstance(ai_urls, str):
                            st.markdown(f"[Source]({ai_urls})")
                        else:
                            st.caption("No direct links returned.")

                    with c3:
                        # Slider for manual override
                        current_val = st.session_state.current_scores.get(indicator_name, ai_score)
                        new_val = st.slider(
                            "Score", 1, 5, int(current_val),
                            key=f"slider_{indicator_name}",
                            label_visibility="collapsed"
                        )
                        st.session_state.current_scores[indicator_name] = new_val
                    
                    st.divider()

    # --- FINAL RESULTS ---
    metrics = calculate_final_metrics(st.session_state.current_scores)
    
    st.header("Final Assessment")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Tzu Chi Severity Score", f"{metrics['severity']} / 5.0")
    m2.metric("INFORM Equivalent", f"{metrics['inform']} / 10.0")
    m3.metric("Category", f"{metrics['category']}")
    
    st.markdown(f"""
    <div style="padding: 20px; border-radius: 10px; background-color: {metrics['color']}15; border: 2px solid {metrics['color']}; text-align: center;">
        <h2 style="color:{metrics['color']}; margin:0;">{metrics['cat_label']}</h2>
        <p style="font-size: 1.2em; margin-top: 10px;"><strong>ACTION: {metrics['action']}</strong></p>
    </div>
    """, unsafe_allow_html=True)
