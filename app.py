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
st.title("Tzu Chi Global Disaster Assessment Tool (v16: Production)")

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
        "adaderana.lk", 
        "dailymirror.lk", 
        "newsfirst.lk"
    ]
    
    selected_domains = st.multiselect(
        "Target Sources:",
        options=DEFAULT_DOMAINS,
        default=DEFAULT_DOMAINS[:6]
    )
    
    custom_domain = st.text_input("Add Custom Domain:")
    if custom_domain and custom_domain not in selected_domains:
        selected_domains.append(custom_domain)

# --- 4. SCORING FRAMEWORK ---
SCORING_FRAMEWORK = {
    "1. IMPACT": {
        "1.1 People Affected": {"weight": 0.25, "rubric": "1=<10k; 2=10k-50k; 3=50k-100k; 4=100k-500k; 5=≥500k"},
        "1.2 Fatalities": {"weight": 0.25, "rubric": "1=<50; 2=50-199; 3=200-499; 4=500-2,999; 5=≥3,000"},
        "1.3 People in Need": {"weight": 0.30, "rubric": "1=<10k; 2=10k-50k; 3=50k-100k; 4=100k-500k; 5=≥500k"},
        "1.4 Housing & Building Damage": {"weight": 0.10, "rubric": "1=Minor; 3=Moderate(10-30%); 5=Extreme(>60%)"},
        "1.5 Land Mass Affected": {"weight": 0.10, "rubric": "1=Small; 3=District; 5=Provincial/National"}
    },
    "2. HUMANITARIAN CONDITIONS": {
        "2.1 Food Security (IPC Score)": {"weight": 0.35, "rubric": "1=IPC1/2; 3=IPC3(Crisis); 5=IPC4/5(Famine)"},
        "2.2 WASH / NFI Needs": {"weight": 0.20, "rubric": "1=Normal; 3=Limited; 5=Critical/Disease Risk"},
        "2.3 Displacement": {"weight": 0.20, "rubric": "1=<1k; 2=1k-10k; 3=10k-50k; 4=50k-100k; 5=≥100k"},
        "2.4 Vulnerable Groups Proportion": {"weight": 0.10, "rubric": "1=<10%; 2=10-19%; 3=20-34%; 4=35-49%; 5=≥50%"},
        "2.5 Health System": {"weight": 0.15, "rubric": "1=Functioning; 3=Disrupted; 5=Collapsed"}
    },
    "3. COMPLEXITY": {
        "3.1 Access (roads/airports)": {"weight": 0.30, "rubric": "1=Open; 3=Blocked; 5=Inaccessible"},
        "3.2 Security": {"weight": 0.30, "rubric": "1=Low; 3=Unstable; 5=High Violence"},
        "3.3 Government Capacity": {"weight": 0.20, "rubric": "1=Strong; 3=Requesting Help; 5=Failed State"},
        "3.4 Communications": {"weight": 0.20, "rubric": "1=Normal; 3=Intermittent; 5=Blackout"}
    },
    "4. STAKEHOLDER ATTENTION": {
        "4.1 Media Intensity": {"weight": 0.25, "rubric": "1=Low; 3=National; 5=Global"},
        "4.2 UN/INGO Activation": {"weight": 0.20, "rubric": "1=None; 3=Regional; 5=L3 Emergency"},
        "4.3 Internal Interest (Tzu Chi)": {"weight": 0.55, "rubric": "1=Low; 3=Moderate; 5=High"}
    },
    "5. FEASIBILITY & PARTNERSHIPS": {
        "5.1 Local Partnerships": {"weight": 0.40, "rubric": "1=Strong; 3=Potential; 5=None"},
        "5.2 Legal & Financing": {"weight": 0.40, "rubric": "1=Easy; 3=Restricted; 5=Blocked"},
        "5.3 Culture & Faith Alignment": {"weight": 0.20, "rubric": "1=Aligned; 3=Neutral; 5=Hostile"}
    }
}

# --- 5. SYSTEM PROMPT ---
rubric_text = ""
for dim, indicators in SCORING_FRAMEWORK.items():
    rubric_text += f"\n**{dim}**:\n"
    for ind, details in indicators.items():
        rubric_text += f"- {ind}: {details['rubric']}\n"

SYSTEM_PROMPT = f"""
You are the Lead Researcher for the 'Tzu Chi Disaster Assessment Unit'.
Your task is to populate a disaster matrix with EXACT DATA and SCORING.

### 1. SEARCH & RECOVERY PROTOCOL:
- **Primary:** Search the 'Target Sources' for real-time data.
- **Strict Data Requirement:** You must find real-world reports. Do not simulate data. If zero information is found after a thorough search, state "No data found".

### 2. KEY FIGURES EXTRACTION:
Extract specific fields.
- **Value:** The number (e.g. "1,200").
- **Date:** The date of report (e.g. "2025-12-02").
- **Source:** The publisher (e.g. "Reuters").
- **Url:** The direct link.

### 3. SCORING RUBRIC (STRICT):
Map found data to these scores (1-5).
{rubric_text}

### 4. QUALITATIVE INFERENCE:
If exact numbers are missing, score based on text severity (e.g. "Catastrophic" = 5). 
**DO NOT DEFAULT TO 3.** If data implies severity, score high.

### OUTPUT FORMAT (JSON ONLY):
{{
  "summary": {{ "title": "...", "country": "...", "date": "...", "description": "..." }},
  "key_figures": {{
    "affected":   {{"value": "...", "date": "...", "source": "...", "url": "..."}},
    "fatalities": {{"value": "...", "date": "...", "source": "...", "url": "..."}},
    "displaced":  {{"value": "...", "date": "...", "source": "...", "url": "..."}},
    "in_need":    {{"value": "...", "date": "...", "source": "...", "url": "..."}}
  }},
  "scores": {{
    "1.1 People Affected": {{ "score": 1-5, "extracted_value": "...", "justification": "...", "source_urls": ["..."] }},
    ...
  }}
}}
"""

# --- 6. HELPER FUNCTIONS ---

def match_score_key(ai_key, framework_keys):
    ai_key_clean = ai_key.lower().replace(".", "").strip()
    if ai_key in framework_keys: return ai_key
    for fk in framework_keys:
        if ai_key_clean in fk.lower().replace(".", ""): return fk
    for fk in framework_keys:
        if ai_key.lower() in fk.lower(): return fk
    return None

def calculate_final_metrics(scores_dict):
    total_weighted_sum = 0.0
    for dim, indicators in SCORING_FRAMEWORK.items():
        for ind_name, details in indicators.items():
            weight = details['weight']
            score = scores_dict.get(ind_name, 3)
            total_weighted_sum += score * weight
            
    # Normalize to 1-5 Scale (Assuming 5 Dimensions with weights summing to 1.0 each)
    # Total sum will be approx 0-5 range already if weights are 0.25, 0.25 etc.
    # Actually, let's just use the sum directly if the weights per dimension sum to 1.0
    # and we sum across 5 dimensions, we need to divide by 5.
    
    final_severity_index = total_weighted_sum / 5.0
    inform_score = final_severity_index * 2.0 
    
    if final_severity_index >= 4.0:
        category = "A"
        label = "MAJOR International"
        action = "IMMEDIATE MOBILISATION: Initiate assessment, Stocktake on Inventory & Emergency funds, Contact international partners."
        color = "#ff4b4b" 
    elif final_severity_index >= 2.5:
        category = "B"
        label = "Medium Scale"
        action = "WATCH LIST: Maintain contact with local partners, Monitor developments for 72h."
        color = "#ffa421"
    else:
        category = "C"
        label = "Minimal / Local"
        action = "MONITORING: No HQ deployment likely needed, Pray & Monitor."
        color = "#09ab3b"
        
    return {
        "severity": round(final_severity_index, 1),
        "inform": round(inform_score, 1),
        "category": category,
        "cat_label": label,
        "action": action,
        "color": color
    }

def safe_get_response_text(response):
    """
    Safely extract a string payload from GenerateContentResponse.

    Tries several shapes to be compatible with different google-genai versions:
    - response.output_text
    - response.text
    - first candidate.content.parts[i].text
    """
    # 1) Newer SDK property: output_text
    for attr in ("output_text", "text"):
        if hasattr(response, attr):
            try:
                txt = getattr(response, attr)
                if isinstance(txt, str) and txt.strip():
                    return txt
            except Exception:
                # Some SDK versions throw if not present
                pass

    # 2) Fallback: walk candidates/parts
    try:
        for cand in getattr(response, "candidates", []) or []:
            content = getattr(cand, "content", None)
            if not content:
                continue
            for part in getattr(content, "parts", []) or []:
                t = getattr(part, "text", None)
                if isinstance(t, str) and t.strip():
                    return t
    except Exception:
        pass

    return None


def robust_json_extractor(text: str):
    """
    Try to pull a JSON object out of a model response.

    Handles:
    - Raw JSON
    - ```json ... ``` fenced blocks
    - JSON embedded in extra commentary
    """
    try:
        if not text:
            return None

        text = text.strip()

        # If it already looks like a pure JSON object
        if text.startswith("{") and text.endswith("}"):
            return json.loads(text)

        # Strip markdown fenced block if present
        if "```" in text:
            fenced = re.search(
                r"```(?:json)?\s*(\{.*\})\s*```",
                text,
                re.DOTALL | re.IGNORECASE,
            )
            if fenced:
                return json.loads(fenced.group(1))

        # Fallback: first {...} block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))

        return None
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
            "INSTRUCTION: Find the LATEST data. Use descriptive text to infer scores if numbers are missing."
        )

        tool_config = types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            # Ask Gemini to respond with JSON
            response_mime_type="application/json",
        )

        # Try Gemini 2.0 first, then fall back to 1.5 if needed
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=full_prompt,
                config=tool_config,
            )
        except Exception:
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=full_prompt,
                config=tool_config,
            )

        # ---------- Extract URLs from grounding metadata (defensive) ----------
        valid_urls = []
        try:
            for cand in getattr(response, "candidates", []) or []:
                gm = getattr(cand, "grounding_metadata", None)
                if not gm:
                    continue
                chunks = getattr(gm, "grounding_chunks", None)
                if not chunks:
                    continue
                for chunk in chunks:
                    web = getattr(chunk, "web", None)
                    if web and getattr(web, "uri", None):
                        valid_urls.append(web.uri)
        except Exception:
            # If the structure changes, we just skip URLs – don't fail the whole call
            pass

        # ---------- Safely get raw text from the response ----------
        raw_text_debug = safe_get_response_text(response)
        if not raw_text_debug:
            # This is what will show in the Debugger expander
            return None, valid_urls, "Model response contained no text (possibly tool-only call)."

        # ---------- Parse JSON ----------
        data = robust_json_extractor(raw_text_debug)
        if data is None:
            # Surface the first chunk of text so you can see what Gemini actually returned
            snippet = raw_text_debug[:1200]
            debug_msg = (
                "Could not parse JSON from model. "
                "Here is the first part of the raw response:\n\n"
                f"{snippet}"
            )
            return None, valid_urls, debug_msg

        # Success
        return data, valid_urls, raw_text_debug

    except Exception as e:
        # Any unexpected error returns a debug string instead of crashing the app
        return None, [], f"Exception in fetch_ai_assessment: {e!r}"



# --- 7. UI RENDER ---

query = st.text_area("Describe the disaster (Location, Date, Type):", 
                     placeholder="e.g., Cyclone Ditwah, Sri Lanka, Dec 2025")
run_btn = st.button("Start Deep Research", type="primary")

if "assessment_data" not in st.session_state:
    st.session_state.assessment_data = None
if "valid_urls" not in st.session_state:
    st.session_state.valid_urls = []
if "current_scores" not in st.session_state:
    st.session_state.current_scores = {}
if "raw_debug" not in st.session_state:
    st.session_state.raw_debug = ""

if run_btn and query:
    with st.spinner("🔍 Researching Sources & Scoring against Rubric..."):
        data, urls, raw_debug = fetch_ai_assessment(api_key, query, selected_domains)
        
        st.session_state.raw_debug = raw_debug # Store raw text for debugging
        
        if data:
            st.session_state.assessment_data = data
            st.session_state.valid_urls = urls
            
            framework_keys = []
            for d in SCORING_FRAMEWORK.values():
                framework_keys.extend(d.keys())
                
            for ai_key, ai_val_obj in data.get("scores", {}).items():
                matched_key = match_score_key(ai_key, framework_keys)
                if matched_key:
                    try:
                        # Extract score, ensure it's int
                        val_str = str(ai_val_obj.get("score", 3))
                        # Handle cases where AI returns "3 (Moderate)"
                        score_val = int(re.search(r'\d+', val_str).group())
                        st.session_state.current_scores[matched_key] = score_val
                    except:
                        pass # Keep default if parsing fails
        else:
            st.error("Failed to retrieve data. Check Debugger below.")

if st.session_state.assessment_data:
    data = st.session_state.assessment_data
    
    # --- HEADER ---
    st.divider()
    col1, col2 = st.columns([2, 1])
    with col1:
        st.title(f"{data.get('summary', {}).get('title', 'Assessment')}")
        st.caption(f"📍 {data.get('summary', {}).get('country', '-')} | 📅 {data.get('summary', {}).get('date', '-')}")
        st.info(data.get('summary', {}).get('description', 'No description available.'))
        
    # --- KEY FIGURES ---
    with col2:
        st.subheader("Key Figures")
        kf = data.get('key_figures', {})
        
        def render_kf_card(label, kf_item):
            if not kf_item: kf_item = {}
            val = kf_item.get('value', 'Unknown')
            date = kf_item.get('date', '-')
            src = kf_item.get('source', 'Unknown')
            url = kf_item.get('url', '#')
            
            # Use valid_urls fallback if AI url is empty/broken
            if (not url or url == "#" or "..." in url) and st.session_state.valid_urls:
                url = st.session_state.valid_urls[0]

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
                # Fuzzy match to get data from AI response
                ai_data = {}
                for k, v in data.get("scores", {}).items():
                    if match_score_key(k, [indicator_name]) == indicator_name:
                        ai_data = v
                        break
                
                ai_score = ai_data.get("score", 3)
                ai_value = ai_data.get("extracted_value", "No specific data")
                ai_just = ai_data.get("justification", "-")
                
                weight = details['weight']
                rubric = details['rubric']

                with st.container():
                    c1, c2, c3 = st.columns([2, 4, 1])
                    with c1:
                        st.markdown(f"**{indicator_name}**")
                        st.caption(f"Weight: {weight}")
                        with st.expander("Rubric"):
                            st.write(rubric)
                    with c2:
                        st.markdown(f"**Evidence:** `{ai_value}`")
                        st.write(f"_{ai_just}_")
                        
                        # Display valid URLs from session state if AI ones are generic
                        if st.session_state.valid_urls:
                            unique_urls = list(set(st.session_state.valid_urls))[:3]
                            links = " | ".join([f"[Source {j+1}]({u})" for j, u in enumerate(unique_urls)])
                            st.markdown(f"🔗 {links}")
                            
                    with c3:
                        # Use the session state value which was updated in the button callback
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
    
    # --- DEEP DEBUGGER ---
    with st.expander("🛠️ Developer Debugger (Click if Data is Missing)"):
        st.write("### 1. Extracted URLs (Grounding Metadata)")
        if st.session_state.valid_urls:
            st.write(st.session_state.valid_urls)
        else:
            st.warning("No URLs found in grounding metadata.")
            
        st.write("### 2. Raw JSON Response from AI")
        st.code(st.session_state.raw_debug, language='json')
