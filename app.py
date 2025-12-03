import streamlit as st
import google.generativeai as genai
import json
import pandas as pd
from google.generativeai.types import Tool, GoogleSearchRetrieval


# --- 1. CONFIGURATION & SCORING FRAMEWORK ---
# We define the indicators and weights exactly as specified in your framework.

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

# Flattens the dictionary for easier calculation logic
FLAT_WEIGHTS = {}
for cat, indicators in SCORING_FRAMEWORK.items():
    for name, weight in indicators.items():
        FLAT_WEIGHTS[name] = weight

# --- 2. SYSTEM PROMPT FOR THE AI RESEARCHER ---
# This prompts the LLM to act as the research engine and return structured JSON.

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

Use the following SCORING RUBRIC to determine scores (1-5):
[Insert Abbreviated Rubric Here - The AI knows the rubric from the context context, but strictly mapped to the keys above]
For 'Unknown' data, default to score 1 but note it in justification.
"""

# --- 3. HELPER FUNCTIONS ---

def calculate_final_metrics(scores_dict):
    """
    Calculates the Weighted Sum, Normalized Score, and Category
    based on the current scores (which might be edited by the user).
    """
    raw_weighted_sum = 0.0
    
    for indicator, weight in FLAT_WEIGHTS.items():
        # Get score, default to 0 if missing (shouldn't happen)
        score = scores_dict.get(indicator, 3)
        raw_weighted_sum += score * weight
        
    # Tzu Chi Severity Score (Normalised)
    # Total weight sum is 5.0
    severity_score = raw_weighted_sum / 5.0
    
    # INFORM Equivalent
    inform_score = severity_score * 2.0
    
    # Category Determination
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
    """Calls Gemini API with Google Search Grounding to get real-time data."""
    try:
        genai.configure(api_key=api_key)
        
        # 1. Use the correct model name (Likely 1.5-flash or 2.0-flash-exp)
        # We use 'gemini-1.5-flash-002' as it is the stable production version.
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        # 2. Define the Search Tool explicitly using the class constructor
        # This fixes the "Unknown field" / "FunctionDeclaration" error
        search_tool = Tool(
            google_search_retrieval=GoogleSearchRetrieval(
                dynamic_retrieval_config=None 
            )
        )
        
        full_prompt = f"{SYSTEM_PROMPT}\n\nUSER QUERY: {query}"
        
        # 3. Pass the tool object
        response = model.generate_content(
            full_prompt,
            tools=[search_tool]
        )
        
        # 4. Clean up response
        text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
        
    except Exception as e:
        st.error(f"Error details: {e}")
        return None

# --- 4. STREAMLIT UI ---

st.set_page_config(page_title="Tzu Chi Disaster Tool", layout="wide")

# Header
st.title("Tzu Chi Global Disaster Assessment Tool")
st.subheader("Research humanitarian crises and calculate the Tzu Chi Severity Score.")

# Sidebar for API Key
api_key = st.secrets["GOOGLE_API_KEY"]

# Input Panel
query = st.text_area("Describe the disaster (Location, Date, Type):", 
                     placeholder="e.g., Floods in Southern Brazil, May 2024")
run_btn = st.button("Run Assessment", type="primary")

# Initialize Session State
if "assessment_data" not in st.session_state:
    st.session_state.assessment_data = None
if "current_scores" not in st.session_state:
    st.session_state.current_scores = {}

# Logic when button is clicked
if run_btn and query and api_key:
    with st.spinner("Researching trusted sources and calculating scores..."):
        data = fetch_ai_assessment(api_key, query)
        if data:
            st.session_state.assessment_data = data
            # Initialize current scores from AI data
            for key, val in data["scores"].items():
                st.session_state.current_scores[key] = val["score"]
        else:
            st.error("Failed to generate assessment. Please check API Key or try again.")

# --- 5. RESULTS DISPLAY ---

if st.session_state.assessment_data:
    data = st.session_state.assessment_data
    
    # 5A. Disaster Summary
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

    # 5B. Scoring & Manual Override
    st.divider()
    st.subheader("Indicator Scoring & Manual Override")
    st.markdown("Adjust the sliders below to override the AI's estimated scores. The final recommendation will update in real-time.")

    # Create tabs for dimensions to keep UI clean
    tabs = st.tabs(list(SCORING_FRAMEWORK.keys()))
    
    # We iterate through tabs/dimensions
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
                    # SLIDER for Manual Override
                    # Key is unique per indicator to track state
                    new_val = st.slider(
                        "Score", 
                        1, 5, 
                        int(st.session_state.current_scores.get(indicator_name, ai_val)),
                        key=f"slider_{indicator_name}"
                    )
                    # Update session state immediately
                    st.session_state.current_scores[indicator_name] = new_val

    # 5C. Final Calculation (Real-time)
    metrics = calculate_final_metrics(st.session_state.current_scores)
    
    st.divider()
    st.header("Assessment Results")
    
    # Metrics Columns
    m1, m2, m3 = st.columns(3)
    m1.metric("Tzu Chi Severity Score (1-5)", f"{metrics['severity']}")
    m2.metric("INFORM Equivalent (0-10)", f"{metrics['inform']}")
    m3.metric("Category", f"{metrics['category']} - {metrics['cat_label']}")
    
    # Recommendation Box
    st.markdown(f"""
    <div style="padding: 20px; border-radius: 10px; background-color: {metrics['color']}20; border: 2px solid {metrics['color']};">
        <h3 style="color:{metrics['color']}">Recommended Action</h3>
        <p><strong>{metrics['action']}</strong></p>
    </div>
    """, unsafe_allow_html=True)

    # 5D. Sources
    with st.expander("View Trusted Sources Used"):
        for src in data.get("sources", []):
            st.markdown(f"- [{src['name']}]({src['url']}) ({src['date']})")
