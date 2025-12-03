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
st.title("Tzu Chi Global Disaster Assessment Tool (v9: Rubric Integrated)")

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

# --- 4. SCORING FRAMEWORK (MATCHING YOUR EXCEL) ---
# Each Dimension sums to 1.0 weight. Total score is average of 5 dimensions.
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

# --- 5. SYSTEM PROMPT (WITH RUBRIC & DATE EXTRACTION) ---
SYSTEM_PROMPT = """
You are the Lead Researcher for the 'Tzu Chi Disaster Assessment Unit'.
Your task is to populate a disaster matrix with EXACT DATA and SCORING based on a strict rubric.

### 1. KEY FIGURES EXTRACTION RULE:
You must extract 3 components for every key figure:
- **Value:** The number (e.g., "1,200").
- **Date:** The date of that specific report (e.g., "2025-12-02"). If today, write "Live".
- **Source:** The publisher (e.g., "Reuters").

### 2. SCORING RUBRIC (STRICTLY FOLLOW THIS):
Use these thresholds to determine scores (1=Low, 5=Critical):

**1. IMPACT:**
- **Fatalities:** 1(<10), 2(10-100), 3(100-500), 4(500-1,000), 5(>1,000).
- **Affected:** 1(<10k), 2(10k-100k), 3(100k-500k), 4(500k-1M), 5(>1M).
- **Housing:** 1(Minor), 3(Significant/Partial), 5(Total Destruction/Washed Away).

**2. CONDITIONS:**
- **Displacement:** 1(<1k), 3(10k-50k), 5(>100k displaced).
- **Food/IPC:** 1(IPC 1/2), 3(IPC 3 Crisis), 5(IPC 4/5 Famine).

**3. COMPLEXITY:**
- **Access:** 1(Accessible), 3(Hampered), 5(Totally Inaccessible/Cut off).

*If exact numbers are missing, infer the score from descriptive text (e.g., "Catastrophic loss of life" implies Score 5).*

### OUTPUT FORMAT (JSON ONLY):
{
  "summary": {
    "title": "Crisis Name",
    "country": "Country",
    "date": "Date of Event",
    "description": "Executive summary."
  },
  "key_figures": {
    "affected":   {"value": "e.g. 1.2M", "date": "YYYY-MM-DD", "source": "Source Name", "url": "http..."},
    "fatalities": {"value": "e.g. 350", "date": "YYYY-MM-DD", "source": "Source Name", "url": "http..."},
    "displaced":  {"value": "e.g. 15k", "date": "YYYY-MM-DD", "source": "Source Name", "url": "http..."},
    "in_need":    {"value": "e.g. 300k", "date": "YYYY-MM-DD", "source": "Source Name", "url": "http..."}
  },
  "scores": {
    "1.1 People Affected": {
      "score": 1-5, 
      "extracted_value": "e.g. '1.2 million reported'",
      "justification": "e.g. Score 5 because >1M affected matches rubric.",
      "source_urls": ["url1"]
    },
    ... (Repeat for ALL 19 indicators) ...
  }
}
"""

# --- 6. CORE LOGIC ---

def calculate_final_metrics(scores_dict):
    # Calculate weighted sum per dimension first to ensure 50/50 balance if needed
    # But sticking to the user's flat weights which sum to 1.0 per dimension.
    
    total_weighted_sum = 0.0
    
    # Check: Does the user want a straight sum of all weighted indicators?
    # Dim 1 Sum = 1.0
    # Dim 2 Sum = 1.0
    # ...
    # Total Max Score = 5 (Dimensions) * 1.0 (Max Weight) * 5 (Max Score) = 25?
    # No, typically: Score = (Sum(w * s)) / 5
    
    for indicator, weight in FLAT_WEIGHTS.items():
        score = scores_dict.get(indicator, 3)
        total_weighted_sum += score * weight
        
    # Since there are 5 dimensions, and each dimension's weights sum to 1.0:
    # The 'total_weighted_sum' will effectively range from 1 to 5.
    severity_score = total_weighted_sum  # No division needed if weights sum to 1.0 per dim and we sum them up?
    # Wait, if we have 5 dimensions, the max total_weighted_sum is 5 * 5 = 25? 
    # Let's re-verify:
    # If all scores are 5: 
    # Dim 1 = 5 * (0.25+0.25+0.3+0.1+0.1) = 5 * 1 = 5.
    # Total for 5 dims = 25.
    # We want a 1-5 scale. So we divide by 5.
    
    final_score = total_weighted_sum / 5.0
    inform_score = final_score * 2.0 # Scale to 10
    
    if final_score >= 4.0:
        category = "A"
        label = "Major International"
        action = "IMMEDIATE MOBILISATION: Initiate assessment, Stocktake on Inventory & Emergency funds, Contact international partners."
        color = "#ff4b4b" # Red
    elif final_score >= 2.5:
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
        "severity": round(final_score, 1),
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
            # Fallback
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
            for key, val in data.get("scores", {}).items():
                st.session_state.current_scores[key] = val.get("score", 3)
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
        
        # Grid layout for figures
        # Using a container to make it look like a dashboard card
        
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
            for indicator_name, weight in indicators.items():
                ai_data = data["scores"].get(indicator_name, {})
                ai_score = ai_data.get("score", 3)
                ai_value = ai_data.get("extracted_value", "No data")
                ai_just = ai_data.get("justification", "-")
                ai_urls = ai_data.get("source_urls", [])

                with st.container():
                    c1, c2, c3 = st.columns([2, 4, 1])
                    with c1:
                        st.markdown(f"**{indicator_name}**")
                        st.caption(f"Weight: {weight}")
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
