import streamlit as st
import google.generativeai as genai
import json
import pandas as pd
import importlib.metadata

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Tzu Chi Disaster Tool", layout="wide")
st.title("Tzu Chi Global Disaster Assessment Tool")

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

# --- 4. SYSTEM PROMPT ---
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
    """Calls Gemini API using the specific tool syntax requested by the error."""
    try:
        genai.configure(api_key=api_key)
        
        # 1. MODEL SELECTION
        # We try Gemini 2.5 Flash as requested.
        # If it doesn't exist yet, we fall back to 1.5 Flash automatically.
        target_model = "gemini-2.5-flash"
        
        # 2. TOOL CONFIGURATION
        # The error explicitly asked for 'google_search', NOT 'google_search_retrieval'
        # This dictionary syntax is the most robust way to pass it.
        tools_config = [
            {'google_search': {}} 
        ]

        model = genai.GenerativeModel(target_model)
        
        full_prompt = f"{SYSTEM_PROMPT}\n\nUSER QUERY: {query}"
        
        # 3. GENERATION
        response = model.generate_content(
            full_prompt,
            tools=tools_config
        )
        
        text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
        
    except Exception as e:
        # Fallback Logic: If 2.5 doesn't exist, try 1.5
        if "404" in str(e) and "models/" in str(e):
            st.warning(f"Gemini 2.5 Flash not found. Falling back to Gemini 1.5 Flash...")
            try:
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(
                    f"{SYSTEM_PROMPT}\n\nUSER QUERY: {query}",
                    tools=[{'google_search': {}}]
                )
                text = response.text.replace("```json", "").replace("```", "").strip()
                return json.loads(text)
            except Exception as e2:
                st.error(f"Fallback failed: {e2}")
                return None
        else:
            st.error(f"Error details: {e}")
            return None

# --- 6. UI RENDER ---

query = st.text_area("Describe the disaster (Location, Date, Type):", 
                     placeholder="e.g., Floods in Southern Brazil, May 2024")
run_btn = st.button("Run Assessment", type="primary")

if "assessment_data" not in st.session_state:
    st.session_state.assessment_data = None
if "current_scores" not in st.session_state:
    st.session_state.current_scores = {}

if run_btn and query:
    with st.spinner("Researching trusted sources..."):
        data = fetch_ai_assessment(api_key, query)
        if data:
            st.session_state.assessment_data = data
            for key, val in data["scores"].items():
                st.session_state.current_scores[key] = val["score"]

if st.session_state.assessment_data:
    data = st.session_state.assessment_data
    
    st.divider()
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header(f"{data['summary']['title']}")
        st.markdown(f"**Location:** {data['summary']['country']} | **Date:** {data['summary']['date']}")
        st.info(data['summary']['description'])
        
    with col2:
        st.subheader("Key Figures")
        kf = data['key_figures']
        df_figs = pd.DataFrame([
            ["Affected", kf.get('affected', '-')],
            ["Fatalities", kf.get('fatalities', '-')],
            ["Displaced", kf.get('displaced', '-')],
            ["In Need", kf.get('in_need', '-')]
        ], columns=["Metric", "Estimate"])
        st.table(df_figs)

    st.divider()
    st.subheader("Indicator Scoring & Manual Override")
    
    tabs = st.tabs(list(SCORING_FRAMEWORK.keys()))
    
    for i, (dim_name, indicators) in enumerate(SCORING_FRAMEWORK.items()):
        with tabs[i]:
            for indicator_name, weight in indicators.items():
                ai_score_obj = data["scores"].get(indicator_name, {"score": 3, "justification": "No data"})
                ai_val = ai_score_obj["score"]
                justification = ai_score_obj["justification"]
                
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(f"**{indicator_name}** (Weight: {weight})")
                    st.caption(f"AI Justification: {justification}")
                with c2:
                    new_val = st.slider(
                        "Score", 
                        1, 5, 
                        int(st.session_state.current_scores.get(indicator_name, ai_val)),
                        key=f"slider_{indicator_name}"
                    )
                    st.session_state.current_scores[indicator_name] = new_val

    metrics = calculate_final_metrics(st.session_state.current_scores)
    
    st.divider()
    st.header("Assessment Results")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Tzu Chi Severity Score (1-5)", f"{metrics['severity']}")
    m2.metric("INFORM Equivalent (0-10)", f"{metrics['inform']}")
    m3.metric("Category", f"{metrics['category']} - {metrics['cat_label']}")
    
    st.markdown(f"""
    <div style="padding: 20px; border-radius: 10px; background-color: {metrics['color']}20; border: 2px solid {metrics['color']};">
        <h3 style="color:{metrics['color']}">Recommended Action</h3>
        <p><strong>{metrics['action']}</strong></p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("View Trusted Sources Used"):
        for src in data.get("sources", []):
            st.markdown(f"- [{src['name']}]({src['url']}) ({src['date']})")
