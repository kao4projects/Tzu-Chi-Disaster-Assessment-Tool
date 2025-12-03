import streamlit as st
from google import genai
from google.genai import types
import json
import pandas as pd
import re
import time
import datetime

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Tzu Chi Disaster Tool", layout="wide")
st.title("Tzu Chi Global Disaster Assessment Tool (v10: Matrix Calculation)")

# --- 2. API KEY SETUP ---
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("Missing GOOGLE_API_KEY. Please add it to Streamlit Secrets.")
    st.stop()

# --- 3. SIDEBAR: SOURCE CONTROL ---
with st.sidebar:
    st.header("Research Settings")
    st.caption("The AI will prioritize these sources.")
    
    DEFAULT_DOMAINS = [
        "reliefweb.int",
        "unocha.org",
        "bbc.com",
        "reuters.com",
        "aljazeera.com",
        "news.un.org",    
        "cnn.com", 
        "euronews.com",
        "apnews.com",
    ]
    
    selected_domains = st.multiselect(
        "Target Sources:",
        options=DEFAULT_DOMAINS,
        default=DEFAULT_DOMAINS[:6]
    )
    
    custom_domain = st.text_input("Add Custom Domain:")
    if custom_domain and custom_domain not in selected_domains:
        selected_domains.append(custom_domain)

# --- 4. SCORING FRAMEWORK (EXACTLY FROM YOUR CSV) ---
# Structure: {Dimension: {Indicator: {Weight: float, Rubric: dict}}}
SCORING_FRAMEWORK = {
    "1. IMPACT": {
        "1.1 People Affected": {"weight": 0.25, "rubric": "1=<10k; 2=10k-50k; 3=50k-100k; 4=100k-500k; 5=≥500k"},
        "1.2 Fatalities": {"weight": 0.25, "rubric": "1=<50; 2=50-199; 3=200-499; 4=500-2,999; 5=≥3,000"},
        "1.3 People in Need": {"weight": 0.30, "rubric": "1=<10k; 2=10k-50k; 3=50k-100k; 4=100k-500k; 5=≥500k"},
        "1.4 Housing & Building Damage": {"weight": 0.10, "rubric": "1=Minimal; 3=Moderate(10-30%); 5=Extreme(>60%/Washed away)"},
        "1.5 Land Mass Affected": {"weight": 0.10, "rubric": "1=Small localized; 3=District level; 5=Provincial/National"}
    },
    "2. HUMANITARIAN CONDITIONS": {
        "2.1 Food Security (IPC Score)": {"weight": 0.35, "rubric": "1=IPC1/2; 3=IPC3(Crisis); 5=IPC4/5(Famine)"},
        "2.2 WASH / NFI Needs": {"weight": 0.20, "rubric": "1=Access normal; 3=Limited water/hygiene; 5=Critical shortage/Disease risk"},
        "2.3 Displacement": {"weight": 0.20, "rubric": "1=<1k; 2=1k-10k; 3=10k-50k; 4=50k-100k; 5=≥100k"},
        "2.4 Vulnerable Groups Proportion": {"weight": 0.10, "rubric": "1=<10%; 2=10-19%; 3=20-34%; 4=35-49%; 5=≥50%"},
        "2.5 Health System": {"weight": 0.15, "rubric": "1=Functioning; 3=Regional disruption/Disease; 5=Collapsed/High mortality"}
    },
    "3. COMPLEXITY": {
        "3.1 Access (roads/airports)": {"weight": 0.30, "rubric": "1=Open; 3=Blocked routes; 5=Inaccessible"},
        "3.2 Security": {"weight": 0.30, "rubric": "1=Low risk; 3=Unstable but safe; 5=High violence/No go"},
        "3.3 Government Capacity": {"weight": 0.20, "rubric": "1=Strong capacity; 3=Asking for help; 5=Overwhelmed/Failed state"},
        "3.4 Communications": {"weight": 0.20, "rubric": "1=Normal; 3=Intermittent; 5=Total blackout"}
    },
    "4. STAKEHOLDER ATTENTION": {
        "4.1 Media Intensity": {"weight": 0.25, "rubric": "1=Low local; 3=National news; 5=Global headlines"},
        "4.2 UN/INGO Activation": {"weight": 0.20, "rubric": "1=None; 3=Regional response; 5=L3 Emergency declared"},
        "4.3 Internal Interest (Tzu Chi)": {"weight": 0.55, "rubric": "1=Low; 3=Moderate; 5=High strategic priority"}
    },
    "5. FEASIBILITY & PARTNERSHIPS": {
        "5.1 Local Partnerships": {"weight": 0.40, "rubric": "1=Strong existing partners; 3=Potential partners; 5=No partners"},
        "5.2 Legal & Financing": {"weight": 0.40, "rubric": "1=Easy transfer; 3=Some restrictions; 5=Sanctioned/Blocked"},
        "5.3 Culture & Faith Alignment": {"weight": 0.20, "rubric": "1=Aligned; 3=Neutral; 5=Sensitive/Hostile"}
    }
}

# --- 5. SYSTEM PROMPT (DYNAMICALLY BUILT FROM FRAMEWORK) ---
# We build the rubric text dynamically so the AI always has the latest rules from the dict above.
rubric_text = ""
for dim, indicators in SCORING_FRAMEWORK.items():
    rubric_text += f"\n**{dim}**:\n"
    for ind, details in indicators.items():
        rubric_text += f"- {ind}: {details['rubric']}\n"

SYSTEM_PROMPT = f"""
You are the Lead Researcher for the 'Tzu Chi Disaster Assessment Unit'.
Your task is to populate a disaster matrix with EXACT DATA and SCORING based on the specific rubric below.

### 1. KEY FIGURES EXTRACTION:
For 'Affected', 'Fatalities', 'Displaced', 'In Need', you MUST extract:
- **Value:** The number (e.g., "1,200"). If a range, take the HIGHER number.
- **Date:** The date of the report (e.g., "Dec 03").
- **Source:** The publisher (e.g., "Reuters").

### 2. SCORING RUBRIC (STRICT):
Map the data you find to these scores (1-5).
{rubric_text}

### 3. QUALITATIVE INFERENCE:
If exact numbers are missing (e.g., Breaking News), infer the score from text severity:
- "Catastrophic" / "Widespread" -> Score 5
- "Severe" / "Significant" -> Score 3-4
- "Minor" / "Localized" -> Score 1-2
**NEVER DEFAULT TO 1 IF NEWS IS BAD.**

### OUTPUT FORMAT (JSON ONLY):
{{
  "summary": {{
    "title": "Crisis Name",
    "country": "Country",
    "date": "Date of Event",
    "description": "Executive summary."
  }},
  "key_figures": {{
    "affected":   {{"value": "e.g. 1.2M", "date": "YYYY-MM-DD", "source": "Source Name", "url": "http..."}},
    "fatalities": {{"value": "e.g. 350", "date": "YYYY-MM-DD", "source": "Source Name", "url": "http..."}},
    "displaced":  {{"value": "e.g. 15k", "date": "YYYY-MM-DD", "source": "Source Name", "url": "http..."}},
    "in_need":    {{"value": "e.g. 300k", "date": "YYYY-MM-DD", "source": "Source Name", "url": "http..."}}
  }},
  "scores": {{
    "1.1 People Affected": {{
      "score": 1-5, 
      "extracted_value": "e.g. '1.2 million reported'",
      "justification": "e.g. Matches Rubric 5 (>=500k).",
      "source_urls": ["url1"]
    }},
    ... (Repeat for ALL indicators) ...
  }}
}}
"""

# --- 6. CORE LOGIC (CALCULATION METHOD) ---

def calculate_final_metrics(scores_dict):
    """
    Step 1: Weighted Score = Score(1-5) * Weight
    Step 2: Tzu Chi Severity = Sum of ALL Weighted Scores
    Step 3: INFORM Equivalent = Severity * 2
    """
    total_severity_score = 0.0
    
    # Iterate through the framework to get weights
    for dim, indicators in SCORING_FRAMEWORK.items():
        for ind_name, details in indicators.items():
            weight = details['weight']
            # Get user score (default to 3 if missing)
            user_score = scores_dict.get(ind_name, 3)
            
            # Step 1: Weighted Score
            weighted_score = user_score * weight
            
            # Step 2: Sum
            total_severity_score += weighted_score
            
    # Step 3: INFORM Equivalent (Scale 0-10)
    # Note: Max Severity Score is 5 (if all weights per dim sum to 1.0, and there are 5 dims? No wait.)
    # Let's check weights sum:
    # Dim 1 Sum = 1.0. Dim 2 Sum = 1.0. Dim 3 = 1.0. Dim 4 = 1.0. Dim 5 = 1.0.
    # So Max Total Severity = 5 * 1.0 * 5(score) = 25?
    # Wait, usually the severity score is normalized to 1-5.
    # The previous logic was `sum / 5`. 
    # IF the excel says "Score >= 4.0 is Major", then the final score must be 1-5.
    # So we divide the total sum by the number of dimensions (5).
    
    final_severity_index = total_severity_score / 5.0
    inform_score = final_severity_index * 2.0 
    
    if final_severity_index >= 4.0:
        category = "A"
        label = "MAJOR International"
        action = "IMMEDIATE MOBILISATION: Initiate assessment, Stocktake on Inventory & Emergency funds, Contact international partners."
        color = "#ff4b4b" # Red
    elif final_severity_index >= 2.5:
        category = "B"
        label = "Medium Scale"
        action = "WATCH LIST: Maintain contact with local partners, Monitor developments for 72h."
        color = "#ffa421" # Orange
    else:
        category = "C"
        label = "Minimal / Local"
        action = "MONITORING: No HQ deployment likely needed, Pray & Monitor."
        color = "#09ab3b" # Green
        
    return {
        "severity": round(final_severity_index, 1),
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

def fetch_ai_assessment(api_key, query, domains):
    try:
        client = genai.Client(api_key=api_key)
        
        domain_list_str = ", ".join(domains)
        full_prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"USER QUERY: {query}\n"
            f"TARGET SOURCES: {domain_list_str}\n"
            "INSTRUCTION: Find the LATEST data. If official numbers are missing, use descriptive text from these sources to estimate severity according to the Rubric."
        )
        
        tool_config = types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            response_mime_type="application/json"
        )

        try:
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

        debug_sources = []
        try:
            if response.candidates[0].grounding_metadata.grounding_chunks:
                for chunk in response.candidates[0].grounding_metadata.grounding_chunks:
                    if chunk.web:
                        debug_sources.append(f"{chunk.web.title}: {chunk.web.uri}")
        except:
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
    if selected_domains:
        st.caption(f"Searching: {', '.join(selected_domains)}")
        
    with st.spinner("🔍 Researching Sources & Scoring against Rubric..."):
        data, sources = fetch_ai_assessment(api_key, query, selected_domains)
        
        if data:
            st.session_state.assessment_data = data
            st.session_state.debug_sources = sources
            for dim, inds in SCORING_FRAMEWORK.items():
                for k in inds.keys():
                    # Set default to what AI found, or 3
                    val = data["scores"].get(k, {}).get("score", 3)
                    st.session_state.current_scores[k] = val
        else:
            st.error("Failed to retrieve data.")

if st.session_state.assessment_data:
    data = st.session_state.assessment_data
    
    # --- HEADER ---
    st.divider()
    col1, col2 = st.columns([2, 1])
    with col1:
        st.title(f"{data['summary']['title']}")
        st.caption(f"📍 {data['summary']['country']} | 📅 {data['summary']['date']}")
        st.info(data['summary']['description'])
        
    # --- IMPROVED KEY FIGURES ---
    with col2:
        st.subheader("Key Figures")
        kf = data.get('key_figures', {})
        
        def render_kf_card(label, kf_item):
            val = kf_item.get('value', 'Unknown')
            date = kf_item.get('date', '-')
            src = kf_item.get('source', 'Unknown')
            url = kf_item.get('url', '#')
            
            st.markdown(f"""
            <div style="border:1px solid #444; padding:10px; border-radius:5px; margin-bottom:10px;">
                <div style="font-size:0.8em; color:#888;">{label}</div>
                <div style="font-size:1.4em; font-weight:bold;">{val}</div>
                <div style="font-size:0.7em; margin-top:5px;">
                    📅 {date}<br>
                    📰 <a href="{url}" target="_blank">{src}</a>
                </div>
            </div>
            """, unsafe_allow_html=True)

        k1, k2 = st.columns(2)
        with k1:
            render_kf_card("Affected", kf.get('affected', {}))
            render_kf_card("Displaced", kf.get('displaced', {}))
        with k2:
            render_kf_card("Fatalities", kf.get('fatalities', {}))
            render_kf_card("In Need", kf.get('in_need', {}))

    # --- SCORES ---
    st.divider()
    st.subheader("Detailed Assessment & Evidence")
    
    tabs = st.tabs(list(SCORING_FRAMEWORK.keys()))
    
    for i, (dim_name, indicators) in enumerate(SCORING_FRAMEWORK.items()):
        with tabs[i]:
            for indicator_name, details in indicators.items():
                ai_data = data["scores"].get(indicator_name, {})
                ai_score = ai_data.get("score", 3)
                ai_value = ai_data.get("extracted_value", "No data")
                ai_just = ai_data.get("justification", "-")
                ai_urls = ai_data.get("source_urls", [])
                
                weight = details['weight']
                rubric = details['rubric']

                with st.container():
                    c1, c2, c3 = st.columns([2, 4, 1])
                    with c1:
                        st.markdown(f"**{indicator_name}**")
                        st.caption(f"Weight: {weight}")
                        st.caption(f"Rule: {rubric}") # Show Rubric to user
                    with c2:
                        st.markdown(f"**Evidence:** `{ai_value}`")
                        st.write(f"_{ai_just}_")
                        if ai_urls:
                            links = " | ".join([f"[Link {j+1}]({u})" for j, u in enumerate(ai_urls) if isinstance(u, str)])
                            st.markdown(f"🔗 {links}")
                    with c3:
                        current_val = st.session_state.current_scores.get(indicator_name, ai_score)
                        new_val = st.slider(
                            "Score", 1, 5, int(current_val),
                            key=f"slider_{indicator_name}",
                            label_visibility="collapsed"
                        )
                        st.session_state.current_scores[indicator_name] = new_val
                    st.divider()

    # --- RESULTS ---
    metrics = calculate_final_metrics(st.session_state.current_scores)
    st.header("Final Assessment")
    m1, m2, m3 = st.columns(3)
    m1.metric("Severity Score", f"{metrics['severity']} / 5.0")
    m2.metric("INFORM Equivalent", f"{metrics['inform']} / 10.0")
    m3.metric("Category", f"{metrics['category']}")
    
    st.markdown(f"""
    <div style="padding: 20px; border-radius: 10px; background-color: {metrics['color']}15; border: 2px solid {metrics['color']}; text-align: center;">
        <h2 style="color:{metrics['color']}; margin:0;">{metrics['cat_label']}</h2>
        <p style="font-size: 1.2em; margin-top: 10px;"><strong>ACTION: {metrics['action']}</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("Search Debugger"):
         if st.session_state.debug_sources:
            st.write(st.session_state.debug_sources)
         else:
            st.write("No external sources linked.")
