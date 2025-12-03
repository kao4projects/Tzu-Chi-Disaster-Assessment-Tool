import ast
import streamlit as st
from google import genai
from google.genai import types
import json
import re

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
        "reliefweb.int", "unocha.org", "bbc.com", "reuters.com",
        "aljazeera.com", "news.un.org", "cnn.com", "euronews.com",
        "apnews.com", "adaderana.lk", "dailymirror.lk", "newsfirst.lk"
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
        # global weight = 0.375
        "1.1 People Affected": {
            "weight": 0.375,
            "rubric": "1=<10,000; 2=10,000-50,000; 3=50,000-100,000; 4=100,000-499,999; 5=≥500,000"
        },
        # global weight = 0.375
        "1.2 Fatalities": {
            "weight": 0.375,
            "rubric": "1=<50; 2=50-199; 3=200-499; 4=500-2,999; 5=≥3,000"
        },
        # global weight = 0.45
        "1.3 People in Need": {
            "weight": 0.45,
            "rubric": "1=<10,000; 2=10,000-49,999; 3=50,000-199,999; 4=200,000-999,999; 5=≥1,000,000"
        },
        # global weight = 0.15
        "1.4 Housing & Building Damage": {
            "weight": 0.15,
            "rubric": "1=<1,000 houses; 2=1,000-9,999; 3=10,000-49,999; 4=50,000-199,999; 5=≥200,000"
        },
        # global weight = 0.15
        "1.5 Land Mass Affected": {
            "weight": 0.15,
            "rubric": "1=<100 km²; 2=100-999 km²; 3=1,000-4,999 km² or 2–3 districts; "
                      "4=5,000-19,999 km² or ≥50% of 2–3 districts; 5=≥20,000 km² or multi-country large-scale"
        },
    },

    "2. HUMANITARIAN CONDITIONS": {
        # global weight = 0.525
        "2.1 Food Security (IPC Score)": {
            "weight": 0.525,
            "rubric": "1=IPC Phase 1–2; 2=IPC3+ <10k people; 3=widespread IPC3+/IPC4; "
                      "4=widespread IPC4; 5=IPC5 or IPC3+ ≥1M people"
        },
        # global weight = 0.30
        "2.2 WASH / NFI Needs": {
            "weight": 0.30,
            "rubric": "1=<1k need WASH/NFI; 2=1–9.9k; 3=10–49k; 4=50–199k; 5=≥200k"
        },
        # global weight = 0.30
        "2.3 Displacement": {
            "weight": 0.30,
            "rubric": "1=<1,000 displaced; 2=1,000-9,999; 3=10,000-49,999; 4=50,000-199,999; 5=≥200,000"
        },
        # global weight = 0.15
        "2.4 Vulnerable Groups Proportion": {
            "weight": 0.15,
            "rubric": "1=<10%; 2=10-19%; 3=20-34%; 4=35-49%; 5=≥50%"
        },
        # global weight = 0.225
        "2.5 Health System": {
            "weight": 0.225,
            "rubric": "1=functioning health system, referral possible; 2=medicine shortages; "
                      "3=regional hospitals closed, infectious disease present; "
                      "4=most hospitals closed, large-scale infectious disease; "
                      "5=external health actors lead / very high mortality from infectious disease"
        },
    },

    "3. COMPLEXITY": {
        # global weight = 0.15
        "3.1 Access (roads/airports)": {
            "weight": 0.15,
            "rubric": "1=free access; 2=localised disruption, can detour; 3=most roads blocked; "
                      "4=severe access issues, time/corridor restrictions; 5=no road access"
        },
        # global weight = 0.15
        "3.2 Security": {
            "weight": 0.15,
            "rubric": "1=low risk; 2=isolated incidents, predictable/controllable; "
                      "3=frequent incidents, generally not life-threatening; "
                      "4=high risk, frequent violent incidents; "
                      "5=extreme, operations require heavy security or suspension"
        },
        # global weight = 0.10
        "3.3 Government Capacity": {
            "weight": 0.10,
            "rubric": "1=adequate resources, strong coordination; 2=can manage most needs; "
                      "3=formally requests international assistance; "
                      "4=highly dependent on external support; 5=loss of governance/coordination"
        },
        # global weight = 0.10
        "3.4 Communications": {
            "weight": 0.10,
            "rubric": "1=internet & video stable; 2=slow/unstable video; 3=intermittent outages; "
                      "4=severe degradation, text-only; 5=large-scale blackout, only satellite works"
        },
    },

    "4. STAKEHOLDER ATTENTION": {
        # global weight = 0.25
        "4.1 Media Intensity": {
            "weight": 0.25,
            "rubric": "1=ReliefWeb only; 2=local news only, no intl; 3=3+ major international outlets; "
                      "4=widespread coverage (domestic & intl); 5=front-page headline level"
        },
        # global weight = 0.20
        "4.2 UN/INGO Activation": {
            "weight": 0.20,
            "rubric": "1=monitoring only, no formal activation; 2=local NGOs responding; "
                      "3=international response teams deployed; 4=OCHA formally activated/present; "
                      "5=system-wide activation (e.g. HRP or similar)"
        },
        # global weight = 0.55
        "4.3 Internal Interest (Tzu Chi)": {
            "weight": 0.55,
            "rubric": "1=low inquiry; 2=featured in daily intl updates; "
                      "3=Religious Affairs/volunteers raising; "
                      "4=Master/first-tier leadership engaged; "
                      "5=Board interest, major fundraising / mobilisation"
        },
    },

    "5. FEASIBILITY & PARTNERSHIPS": {
        # global weight = 0.20
        "5.1 Local Partnerships": {
            "weight": 0.20,
            "rubric": "1=no known partners; 2=contact only, untested; 3=≥1 reliable organisation; "
                      "4=≥2 organisations with successful past collaborations; "
                      "5=multiple mature, reliable partners"
        },
        # global weight = 0.20
        "5.2 Legal & Financing": {
            "weight": 0.20,
            "rubric": "1=very high uncertainty / risk; 2=complex requirements; "
                      "3=first-time but likely feasible; 4=feasible, clear procedures; "
                      "5=smooth, established financial/legal channels"
        },
        # global weight = 0.10
        "5.3 Culture & Faith Alignment": {
            "weight": 0.10,
            "rubric": "1=strong resistance; 2=requires heavy dialogue/negotiation; "
                      "3=basic compatibility; 4=good collaborative environment; "
                      "5=long-term trust, strong values alignment"
        },
    },
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
You MUST use the full 1–5 range. Scores 2 and 4 are allowed when the data clearly sits between two levels.
Map found data to these scores (1-5).
{rubric_text}

{rubric_text}

### 4. QUALITATIVE INFERENCE:
If exact numbers are missing, score based on text severity (e.g. "Catastrophic" = 5). 
**DO NOT DEFAULT TO 3.** If data implies severity, score high.

### 5. OUTPUT FORMAT (STRICT JSON):
Return a single JSON object with this shape.
Do NOT wrap it in backticks or Markdown.
Do NOT include comments or ellipses.

{{
  "summary": {{
    "title": "…",
    "country": "…",
    "date": "…",
    "description": "…"
  }},
  "key_figures": {{
    "affected":   {{"value": "…", "date": "…", "source": "…", "url": "…"}},
    "fatalities": {{"value": "…", "date": "…", "source": "…", "url": "…"}},
    "displaced":  {{"value": "…", "date": "…", "source": "…", "url": "…"}},
    "in_need":    {{"value": "…", "date": "…", "source": "…", "url": "…"}}
  }},
  "scores": {{
    "1.1 People Affected":         {{"score": 1, "extracted_value": "…", "justification": "…", "source_urls": ["…"]}},
    "1.2 Fatalities":              {{"score": 1, "extracted_value": "…", "justification": "…", "source_urls": ["…"]}},
    "1.3 People in Need":          {{"score": 1, "extracted_value": "…", "justification": "…", "source_urls": ["…"]}},
    "1.4 Housing & Building Damage": {{"score": 1, "extracted_value": "…", "justification": "…", "source_urls": ["…"]}},
    "1.5 Land Mass Affected":      {{"score": 1, "extracted_value": "…", "justification": "…", "source_urls": ["…"]}},
    "2.1 Food Security (IPC Score)": {{"score": 1, "extracted_value": "…", "justification": "…", "source_urls": ["…"]}},
    "2.2 WASH / NFI Needs":        {{"score": 1, "extracted_value": "…", "justification": "…", "source_urls": ["…"]}},
    "2.3 Displacement":            {{"score": 1, "extracted_value": "…", "justification": "…", "source_urls": ["…"]}},
    "2.4 Vulnerable Groups Proportion": {{"score": 1, "extracted_value": "…", "justification": "…", "source_urls": ["…"]}},
    "2.5 Health System":           {{"score": 1, "extracted_value": "…", "justification": "…", "source_urls": ["…"]}},
    "3.1 Access (roads/airports)": {{"score": 1, "extracted_value": "…", "justification": "…", "source_urls": ["…"]}},
    "3.2 Security":                {{"score": 1, "extracted_value": "…", "justification": "…", "source_urls": ["…"]}},
    "3.3 Government Capacity":     {{"score": 1, "extracted_value": "…", "justification": "…", "source_urls": ["…"]}},
    "3.4 Communications":          {{"score": 1, "extracted_value": "…", "justification": "…", "source_urls": ["…"]}},
    "4.1 Media Intensity":         {{"score": 1, "extracted_value": "…", "justification": "…", "source_urls": ["…"]}},
    "4.2 UN/INGO Activation":      {{"score": 1, "extracted_value": "…", "justification": "…", "source_urls": ["…"]}},
    "4.3 Internal Interest (Tzu Chi)": {{"score": 1, "extracted_value": "…", "justification": "…", "source_urls": ["…"]}},
    "5.1 Local Partnerships":      {{"score": 1, "extracted_value": "…", "justification": "…", "source_urls": ["…"]}},
    "5.2 Legal & Financing":       {{"score": 1, "extracted_value": "…", "justification": "…", "source_urls": ["…"]}},
    "5.3 Culture & Faith Alignment": {{"score": 1, "extracted_value": "…", "justification": "…", "source_urls": ["…"]}}
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
    """
    Compute severity exactly like the Excel:
    Severity = sum( (score_i / 5) * global_weight_i )
    Global weights (Weighted Score column) already sum to 5.
    """
    total = 0.0

    for dim, indicators in SCORING_FRAMEWORK.items():
        for ind_name, details in indicators.items():
            weight = details["weight"]          # this IS the global weight (e.g. 0.375)
            score = scores_dict.get(ind_name, 3)
            # normalise score 1–5 to 0–1, then multiply by global weight
            total += (score / 5.0) * weight

    final_severity_index = total          # already in 0–5 range
    inform_score = final_severity_index * 2.0

    if final_severity_index >= 4.0:
        category = "A"
        label = "MAJOR International"
        action = ("IMMEDIATE MOBILISATION: Initiate assessment, "
                  "stocktake inventory & emergency funds, contact international partners.")
        color = "#ff4b4b"
    elif final_severity_index >= 2.5:
        category = "B"
        label = "Medium Scale"
        action = ("WATCH LIST: Maintain contact with local partners, "
                  "monitor developments for 72h.")
        color = "#ffa421"
    else:
        category = "C"
        label = "Minimal / Local"
        action = "MONITORING: No HQ deployment likely needed. Pray & monitor."
        color = "#09ab3b"

    return {
        "severity": round(final_severity_index, 2),
        "inform": round(inform_score, 2),
        "category": category,
        "cat_label": label,
        "action": action,
        "color": color,
    }


def safe_get_response_text(response):
    """Safely extract a string payload from GenerateContentResponse."""
    # 1) Newer SDK property: output_text
    for attr in ("output_text", "text"):
        if hasattr(response, attr):
            try:
                txt = getattr(response, attr)
                if isinstance(txt, str) and txt.strip():
                    return txt
            except Exception:
                pass

    # 2) Fallback: walk candidates/parts
    try:
        for cand in getattr(response, "candidates", []) or []:
            content = getattr(cand, "content", None)
            if not content: continue
            for part in getattr(content, "parts", []) or []:
                t = getattr(part, "text", None)
                if isinstance(t, str) and t.strip():
                    return t
    except Exception:
        pass
    return None

def robust_json_extractor(text: str):
    """
    Try (very) hard to pull a JSON object out of a model response.

    Handles:
    - ```json ... ``` fenced blocks
    - Extra commentary before/after the JSON
    - Trailing commas and null/true/false using ast.literal_eval
    """
    if not text:
        return None

    # Always work on a plain string
    text = str(text).strip()

    # --- Strip markdown fences if present ---
    # e.g. ```json\n{...}\n```  or  ```\n{...}\n```
    if text.startswith("```"):
        # remove the first ``` line
        parts = text.split("```", 2)
        if len(parts) >= 2:
            # everything after the first ``` block opener
            text = parts[1]
        text = text.strip()

    # Now find the first '{' and the last '}'
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    candidate = text[start : end + 1]

    # --- First attempt: strict JSON ---
    try:
        return json.loads(candidate)
    except Exception:
        pass

    # --- Fallback: Python literal via ast.literal_eval (more forgiving) ---
    # Convert JSON literal keywords to Python
    candidate_py = re.sub(r"\bnull\b", "None", candidate)
    candidate_py = re.sub(r"\btrue\b", "True", candidate_py, flags=re.I)
    candidate_py = re.sub(r"\bfalse\b", "False", candidate_py, flags=re.I)

    try:
        obj = ast.literal_eval(candidate_py)
        if isinstance(obj, dict):
            return obj
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
        )

        # Try 2.5 then fall back to 2.0
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=full_prompt,
                config=tool_config,
            )
        except Exception:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=full_prompt,
                config=tool_config,
            )

        # ---------- Extract URLs from grounding metadata ----------
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
            pass

        # ---------- Safely get raw text ----------
        raw_text_debug = safe_get_response_text(response)
        if not raw_text_debug:
            return None, valid_urls, "Model response contained no text (possibly tool-only call)."

        # ---------- Parse JSON (lenient) ----------
        data = robust_json_extractor(raw_text_debug)
        if data is None:
            snippet = raw_text_debug[:1200]
            debug_msg = (
                "Could not parse JSON from model, even after lenient parsing.\n\n"
                "First part of response:\n\n"
                f"{snippet}"
            )
            return None, valid_urls, debug_msg

        return data, valid_urls, raw_text_debug

    except Exception as e:
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

        st.session_state.raw_debug = raw_debug
        st.session_state.valid_urls = urls or []

        if data is not None:
            st.session_state.assessment_data = data

            framework_keys = []
            for d in SCORING_FRAMEWORK.values():
                framework_keys.extend(d.keys())

            for ai_key, ai_val_obj in data.get("scores", {}).items():
                matched_key = match_score_key(ai_key, framework_keys)
                if matched_key:
                    try:
                        val_str = str(ai_val_obj.get("score", 3))
                        score_val = int(re.search(r'\d+', val_str).group())
                        st.session_state.current_scores[matched_key] = score_val
                    except Exception:
                        pass
        else:
            st.error("Failed to retrieve data. See reason below in the Debugger.")
            if raw_debug:
                st.code(str(raw_debug), language="text")

if st.session_state.assessment_data:
    data = st.session_state.assessment_data
    
    st.divider()
    col1, col2 = st.columns([2, 1])
    with col1:
        st.title(f"{data.get('summary', {}).get('title', 'Assessment')}")
        st.caption(f"📍 {data.get('summary', {}).get('country', '-')} | 📅 {data.get('summary', {}).get('date', '-')}")
        st.info(data.get('summary', {}).get('description', 'No description available.'))
        
    with col2:
        st.subheader("Key Figures")
        kf = data.get('key_figures', {})
        
        def render_kf_card(label, kf_item):
            if not kf_item: kf_item = {}
            val = kf_item.get('value', 'Unknown')
            date = kf_item.get('date', '-')
            src = kf_item.get('source', 'Unknown')
            url = kf_item.get('url', '#')
            
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

    st.divider()
    st.subheader("Detailed Assessment & Evidence")
    
    tabs = st.tabs(list(SCORING_FRAMEWORK.keys()))
    
    for i, (dim_name, indicators) in enumerate(SCORING_FRAMEWORK.items()):
        with tabs[i]:
            for indicator_name, details in indicators.items():
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
                        if st.session_state.valid_urls:
                            unique_urls = list(set(st.session_state.valid_urls))[:3]
                            links = " | ".join([f"[Source {j+1}]({u})" for j, u in enumerate(unique_urls)])
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
    
    with st.expander("🛠️ Developer Debugger"):
        st.write("### 1. Extracted URLs")
        st.write(st.session_state.valid_urls)
        st.write("### 2. Raw JSON Response")
        st.code(st.session_state.raw_debug, language='json')
