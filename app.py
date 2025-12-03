import streamlit as st
from google import genai
from google.genai import types
import json
import pandas as pd
import re
import time

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Tzu Chi Disaster Tool", layout="wide")
st.title("Tzu Chi Global Disaster Assessment Tool (v7: Comprehensive)")

# --- 2. API KEY SETUP ---
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("Missing GOOGLE_API_KEY. Please add it to Streamlit Secrets.")
    st.stop()

# --- 3. SIDEBAR: SOURCE CONTROL ---
with st.sidebar:
    st.header("Research Settings")
    st.caption("Control search strictness.")
    
    DEFAULT_DOMAINS = [
        "reliefweb.int", "unocha.org", "bbc.com", "reuters.com", 
        "aljazeera.com", "news.un.org", "cnn.com", "apnews.com"
    ]
    
    selected_domains = st.multiselect(
        "Preferred Sources (for highlighting):",
        options=DEFAULT_DOMAINS,
        default=DEFAULT_DOMAINS[:5]
    )
    

# --- 4. SCORING FRAMEWORK ---
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

# --- 5. SYSTEM PROMPT (ENHANCED FOR QUALITATIVE INFERENCE) ---
SYSTEM_PROMPT = """
You are the Senior Analyst for the 'Tzu Chi Disaster Assessment Unit'. 
Your goal is to provide a COMPREHENSIVE Situation Analysis, not just numbers.

### CORE INSTRUCTIONS:
1. **QUALITATIVE INFERENCE (CRITICAL):** - Breaking news often lacks exact numbers. You MUST infer severity from descriptive text.
   - Example: "Villages washed away" -> Implies High Housing Damage (Score 5) even if no house count exists.
   - Example: "People trapped on roofs" -> Implies Critical Displacement (Score 5).
   - **DO NOT default to 'Unknown' or Score 1 if the text describes chaos.**

2. **DATA EXTRACTION:**
   - If exact figures (Fatalities, Affected) are missing, look for *estimates* like "dozens", "hundreds", "thousands".
   - Use the highest reported credible estimate (Conservative Risk Principle).

3. **OUTPUT:**
   - **Justification:** Must be a full sentence explaining *why* you gave that score based on the text found.
   - **Extracted Value:** If no number, put the *descriptive phrase* found (e.g., "Widespread destruction reported").

### OUTPUT FORMAT (JSON ONLY):
{
  "summary": {
    "title": "Crisis Name",
    "country": "Country",
    "date": "Date",
    "description": "Detailed 4-5 sentence executive summary."
  },
  "key_figures": {
    "affected": {"value": "e.g. >50,000 (est)", "source_url": "url"},
    "fatalities": {"value": "e.g. 12 confirmed", "source_url": "url"},
    "displaced": {"value": "e.g. 2,000 in camps", "source_url": "url"},
    "in_need": {"value": "e.g. Unknown (Assessment ongoing)", "source_url": "url"}
  },
  "scores": {
    "1.1 People Affected": {
      "score": 1-5, 
      "extracted_value": "e.g. 'Whole district flooded'",
      "justification": "e.g. Reports indicate entire district under water, implying high affected count.",
      "source_urls": ["url1"]
    },
    ... (Repeat for ALL indicators) ...
  }
}
"""

# --- 6. CORE LOGIC ---

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
        action = "IMMEDIATE MOBILISATION: Initiate assessment, Stocktake on Inventory & Emergency funds, Contact international partners."
        color = "#ff4b4b" # Red
    elif severity_score >= 2.5:
        category = "B"
        label = "Medium Scale"
        action = "WATCH LIST: Maintain contact with local partners, Monitor developments for 72h."
        color = "#ffa421" # Orange
    else:
        category = "C"
        label = "Local / Minimal"
        action = "MONITORING: No HQ deployment likely needed, Pray & Monitor."
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
    try:
        start_idx = text.find('{')
        if start_idx == -1: return None
        text_trimmed = text[start_idx:]
        obj, _ = json.JSONDecoder().raw_decode(text_trimmed)
        return obj
    except Exception:
        return None

def fetch_ai_assessment(api_key, query, domains, strict):
    try:
        client = genai.Client(api_key=api_key)
        
        # --- SMART SEARCH QUERY CONSTRUCTION ---
        if strict:
            # STRICT: Forces Google to ONLY look at specific sites. 
            # Risk: Returns 0 results if event is too new.
            site_operators = " OR ".join([f"site:{d}" for d in domains])
            search_query = f"{query} ({site_operators})"
            bias_instruction = "Strictly use data from the search results from these domains."
        else:
            # OPEN (Recommended): Searches the whole web for keywords.
            # This finds local news, Twitter aggregators, etc.
            search_query = f"{query} latest humanitarian impact death toll damage assessment"
            bias_instruction = f"Prioritize sources like {', '.join(domains[:3])}, but use ANY credible local news source if official reports are missing."

        full_prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"USER QUERY: {query}\n"
            f"SEARCH CONTEXT: {bias_instruction}\n"
        )
        
        tool_config = types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            response_mime_type="application/json"
        )

        try:
            # Pass the constructed 'search_query' implicitly via the prompt context
            # The model uses the tool with its own formulated query, but we guide it.
            response = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=full_prompt,
                config=tool_config
            )
        except Exception:
            response = client.models.generate_content(
                model='gemini-1.5-flash',
                contents=full_prompt,
                config=tool_config
            )

        # Extract Sources for Debugging
        debug_sources = []
        try:
            if (response.candidates and 
                response.candidates[0].grounding_metadata and 
                response.candidates[0].grounding_metadata.grounding_chunks):
                for chunk in response.candidates[0].grounding_metadata.grounding_chunks:
                    if chunk.web:
                        debug_sources.append(f"{chunk.web.title}: {chunk.web.uri}")
        except Exception:
            pass

        if not response.text:
            return None, debug_sources

        data = robust_json_extractor(response.text)
        return data, debug_sources
        
    except Exception as e:
        st.error(f"API Error details: {e}")
        return None, []

# --- 7. UI RENDER ---

query = st.text_area("Describe the disaster (Location, Date, Type):", 
                     placeholder="e.g., Cyclone Ditwah, Sri Lanka, Dec 2025")
run_btn = st.button("Start Deep Research", type="primary")

if "assessment_data" not in st.session_state:
    st.session_state.assessment_data = None
if "debug_sources" not in st.session_state:
    st.session_state.debug_sources = []
if "current_scores" not in st.session_state:
    st.session_state.current_scores = {}

if run_btn and query:
    with st.spinner("🔍 Scanning Global & Local Sources..."):
        data, sources = fetch_ai_assessment(api_key, query, selected_domains, strict_mode)
        
        if data:
            st.session_state.assessment_data = data
            st.session_state.debug_sources = sources
            for key, val in data.get("scores", {}).items():
                st.session_state.current_scores[key] = val.get("score", 3)
        else:
            st.error("Failed to retrieve data. Please check connection or try a broader query.")

if st.session_state.assessment_data:
    data = st.session_state.assessment_data
    
    # --- HEADER ---
    st.divider()
    col1, col2 = st.columns([2, 1])
    with col1:
        st.title(f"{data['summary']['title']}")
        st.caption(f"📍 {data['summary']['country']} | 📅 {data['summary']['date']}")
        st.info(data['summary']['description'])
        
    with col2:
        st.subheader("Key Figures")
        kf = data.get('key_figures', {})
        
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
                ai_data = data["scores"].get(indicator_name, {})
                ai_score = ai_data.get("score", 3)
                ai_value = ai_data.get("extracted_value", "No specific data point found.")
                ai_just = ai_data.get("justification", "No justification provided.")
                ai_urls = ai_data.get("source_urls", [])

                with st.container():
                    c1, c2, c3 = st.columns([2, 4, 1])
                    with c1:
                        st.markdown(f"**{indicator_name}**")
                        st.caption(f"Weight: {weight}")
                    with c2:
                        # Logic to display text even if no number
                        st.markdown(f"**Evidence:** `{ai_value}`")
                        st.write(f"_{ai_just}_")
                        
                        if ai_urls and isinstance(ai_urls, list):
                            links_md = " | ".join([f"[Source {j+1}]({url})" for j, url in enumerate(ai_urls) if url])
                            st.markdown(f"🔗 {links_md}")
                        elif isinstance(ai_urls, str):
                            st.markdown(f"[Source]({ai_urls})")
                    with c3:
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

    with st.expander("Search Debugger (View Sources)"):
        if st.session_state.debug_sources:
            st.write("Sources used:")
            for src in st.session_state.debug_sources:
                st.write(f"- {src}")
        else:
            st.info("No direct links returned. AI inferred data from general search context.")
