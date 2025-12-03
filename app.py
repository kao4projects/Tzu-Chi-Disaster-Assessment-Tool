import streamlit as st
from google import genai
from google.genai import types
import json
import pandas as pd
import re

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Tzu Chi Disaster Tool", layout="wide")
st.title("Tzu Chi Global Disaster Assessment Tool (Deep Search v2)")

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

# --- 4. SYSTEM PROMPT (ENHANCED) ---
SYSTEM_PROMPT = """
You are the Lead Assessment Engine for the 'Tzu Chi Global Disaster Assessment Tool'.
Your goal is to conduct a DEEP DIVE research on a humanitarian disaster and output granular, evidence-based data.

RESEARCH STRATEGY:
1. Search specifically for "Situation Report", "Flash Update", or "OCHA Snapshot" + [Disaster Name].
2. Cross-reference at least 3 distinct sources (e.g., local media, UN OCHA, International News) to verify numbers.
3. If figures are "unknown", search for "preliminary damage assessment [location]".

OUTPUT FORMAT:
Return ONLY valid JSON. Structure:
{
  "summary": {
    "title": "Official Crisis Name",
    "country": "Country",
    "date": "Date of Event",
    "description": "Concise executive summary (3-4 sentences)."
  },
  "key_figures": {
    "affected": {"value": "e.g. 1.2M people", "source_url": "url..."},
    "fatalities": {"value": "e.g. 450 confirmed", "source_url": "url..."},
    "displaced": {"value": "e.g. 15,000 in shelters", "source_url": "url..."},
    "in_need": {"value": "e.g. 300,000", "source_url": "url..."}
  },
  "scores": {
    "1.1 People Affected": {
      "score": 1-5, 
      "extracted_value": "Exact number or % found (e.g., '150k affected')",
      "justification": "Why this score? (e.g. 'High impact relative to population...')",
      "source_urls": ["url1", "url2", "url3"]
    },
    ... (Repeat for ALL 19 indicators in the framework) ...
  }
}

SCORING GUIDE (1=Low, 5=Critical):
- 1.1 Affected: 1(<10k), 3(100k-500k), 5(>1M)
- 1.2 Fatalities: 1(<10), 3(100-1000), 5(>1000)
- 1.3 In Need: 1(<10% pop), 5(>50% pop)
- 2.1 Food: 1(IPC 1-2), 3(IPC 3), 5(IPC 4-5 Famine)
- 3.1 Access: 1(Open), 3(Restricted), 5(Inaccessible)

IMPORTANT: 
- Populate "extracted_value" with the RAW data point you found.
- Populate "source_urls" with the DIRECT links where you found that specific data point.
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
        st.error(f"Data Extraction Error: {e}")
        with st.expander("Show Raw Output"):
            st.code(text)
        return None

def fetch_ai_assessment(api_key, query):
    """Calls Gemini API using the NEW v1.0 SDK (google-genai)."""
    try:
        client = genai.Client(api_key=api_key)
        full_prompt = f"{SYSTEM_PROMPT}\n\nUSER QUERY: {query}"
        
        # Use gemini-2.0-flash for speed + search capability
        # Fallback to 1.5-flash if 2.0 is not active in the region
        models_to_try = ['gemini-2.0-flash', 'gemini-1.5-flash']
        
        response = None
        for model_name in models_to_try:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        tools=[types.Tool(google_search=types.GoogleSearch())],
                        response_mime_type="application/json"
                    )
                )
                break # If successful, exit loop
            except Exception:
                continue

        if response and response.text:
            return robust_json_extractor(response.text)
        else:
            st.error("Could not retrieve data from AI models.")
            return None
        
    except Exception as e:
        st.error(f"API Error details: {e}")
        return None

# --- 6. UI RENDER ---

query = st.text_area("Describe the disaster (Location, Date, Type):", 
                     placeholder="e.g., Cyclone Fengal, Sri Lanka/India, Nov-Dec 2024")
run_btn = st.button("Start Deep Research", type="primary")

if "assessment_data" not in st.session_state:
    st.session_state.assessment_data = None
if "current_scores" not in st.session_state:
    st.session_state.current_scores = {}

if run_btn and query:
    with st.spinner("🔍 Deep Diving into UN OCHA, ReliefWeb, and Local News... (This may take 15s)"):
        data = fetch_ai_assessment(api_key, query)
        if data:
            st.session_state.assessment_data = data
            # Pre-load scores
            for key, val in data.get("scores", {}).items():
                st.session_state.current_scores[key] = val.get("score", 3)

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

        # Custom metrics with links
        c1, c2 = st.columns(2)
        c1.metric("Affected", aff_val)
        c1.markdown(f"[Source]({aff_url})" if aff_url != '#' else "")
        
        c2.metric("Fatalities", fat_val)
        c2.markdown(f"[Source]({fat_url})" if fat_url != '#' else "")
        
        c3, c4 = st.columns(2)
        c3.metric("Displaced", disp_val)
        c3.markdown(f"[Source]({disp_url})" if disp_url != '#' else "")
        
        c4.metric("In Need", need_val)
        c4.markdown(f"[Source]({need_url})" if need_url != '#' else "")

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
                        st.markdown(f"**Evidence:** `{ai_value}`")
                        st.write(f"_{ai_just}_")
                        
                        # Render Sources as small pills or links
                        if ai_urls:
                            links = " | ".join([f"[Source {j+1}]({url})" for j, url in enumerate(ai_urls) if url])
                            st.markdown(f"🔗 {links}")
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
