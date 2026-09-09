import streamlit as st
import pandas as pd
import anthropic
import os
import re

# ──────────────────────────────────────────────────────────────────────────────
# 1. INITIAL APP CONFIGURATION & STYLING
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IntelliPrice",
    page_icon="🎯",
    layout="wide"
)

st.markdown("""
    <style>
    .main .block-container { padding-top: 1.5rem; }
    .chat-bubble { padding: 1rem; border-radius: 8px; margin-bottom: 1rem; }
    </style>
""", unsafe_allow_html=True)

st.title("🎯 IntelliPrice")
st.caption("Interactive exact-match cross-referencing and conversational analysis engine using Claude Sonnet 5.")

# ──────────────────────────────────────────────────────────────────────────────
# 2. FILE SYSTEM DATA LOADER
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR = r"C:\Users\rpattis\Python_Programs\pricing_agent\pricing_data"

@st.cache_data
def load_internal_knowledge_base():
    files = {
        "costs": os.path.join(DATA_DIR, "Cost.csv"),
        "specs": os.path.join(DATA_DIR, "Digikey_market_feed.csv"),
        "tiers": os.path.join(DATA_DIR, "Price List.csv"),
        "opps": os.path.join(DATA_DIR, "Quote History.csv")
    }
    try:
        df1 = pd.read_csv(files["costs"])
        df2 = pd.read_csv(files["specs"])
        df3 = pd.read_csv(files["tiers"])
        df4 = pd.read_csv(files["opps"])
        
        for df in [df1, df2, df3, df4]:
            for col in df.columns:
                if df[col].dtype == "object":
                    df[col] = df[col].astype(str).str.strip()
        return df1, df2, df3, df4, None
    except Exception as e:
        return None, None, None, None, str(e)

df_costs, df_specs, df_tiers, df_opps, load_error = load_internal_knowledge_base()

if load_error:
    st.error(f"⚠️ **Data Loading Error:** Path not resolved: `{DATA_DIR}`. Details: {load_error}")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# 3. TEXT SANITIZATION ENGINE (FIXES THE COUPLING/SQUISHING ISSUE)
# ──────────────────────────────────────────────────────────────────────────────
def sanitize_markdown_spacing(text):
    """
    Cleans up rendering compilation compressions by explicitly identifying and separating
    words and spaces from bold or italic wrappers before UI compilation.
    """
    if not text:
        return text
        
    # Step A: Push spaces out of bold formatting boundaries
    text = re.sub(r'\*\* +', ' **', text)
    text = re.sub(r' +\*\*', '** ', text)
    
    # Step B: Push spaces out of italic formatting boundaries
    text = re.sub(r'\* +', ' *', text)
    text = re.sub(r' +\*', '* ', text)
    
    # Step C: Fix tightly coupled inline boundary phrases explicitly
    text = re.sub(r'(\w)\*\*([a-zA-Z])', r'\1 **\2', text)
    text = re.sub(r'([a-zA-Z])\*\*(\w)', r'\1** \2', text)
    text = re.sub(r'(\w)\*([a-zA-Z])', r'\1 *\2', text)
    text = re.sub(r'([a-zA-Z])\*(\w)', r'\1* \2', text)
    
    # Step D: Fix common squished phrases patterns explicitly
    text = text.replace("perunit", "per unit")
    text = text.replace("targetprice", "target price")
    text = text.replace("approvedfloor", "approved floor")
    text = text.replace("pricingdiscussions", "pricing discussions")

    return text

# ──────────────────────────────────────────────────────────────────────────────
# 4. EXACT STRUCTURAL DETERMINISTIC LOOKUP FUNCTIONS (THE RAG COMPONENT)
# ──────────────────────────────────────────────────────────────────────────────
def extract_part_intelligence(part_number):
    part_number = str(part_number).strip()
    
    cost_matches = df_costs[df_costs['Part Number'].astype(str) == part_number] if 'Part Number' in df_costs.columns else pd.DataFrame()
    spec_matches = df_specs[df_specs['Mfr Part #'].astype(str) == part_number] if 'Mfr Part #' in df_specs.columns else pd.DataFrame()
    tier_matches = df_tiers[df_tiers['Orderable Part Number'].astype(str) == part_number] if 'Orderable Part Number' in df_tiers.columns else pd.DataFrame()
    opp_matches = df_opps[df_opps['Part Number'].astype(str) == part_number] if 'Part Number' in df_opps.columns else pd.DataFrame()
    
    if cost_matches.empty and not df_costs.empty and 'Part Number' not in df_costs.columns:
        cost_matches = df_costs[df_costs.iloc[:, 0].astype(str) == part_number]
    if spec_matches.empty and not df_specs.empty and 'Mfr Part #' not in df_specs.columns:
        spec_matches = df_specs[df_specs.iloc[:, 0].astype(str) == part_number]
    if tier_matches.empty and not df_tiers.empty and 'Orderable Part Number' not in df_tiers.columns:
        tier_matches = df_tiers[df_tiers.iloc[:, 0].astype(str) == part_number]
    if opp_matches.empty and not df_opps.empty and 'Part Number' not in df_opps.columns:
        opp_matches = df_opps[df_opps.iloc[:, 0].astype(str) == part_number]

    if cost_matches.empty and spec_matches.empty and tier_matches.empty and opp_matches.empty:
        return None

    markdown_context = f"### RAG RECOVERY LEDGER FOR PART: {part_number}\n\n"
    if not cost_matches.empty:
        markdown_context += "#### [Source: Cost.csv] Production Baseline Cost:\n```\n" + cost_matches.to_string(index=False) + "\n```\n\n"
    if not spec_matches.empty:
        markdown_context += "#### [Source: Digikey_market_feed.csv] Distribution Market Specs & Benchmark:\n```\n" + spec_matches.to_string(index=False) + "\n```\n\n"
    if not tier_matches.empty:
        markdown_context += "#### [Source: Price List.csv] Approved Contractual Quantity Tier Prices:\n```\n" + tier_matches.to_string(index=False) + "\n```\n\n"
    if not opp_matches.empty:
        markdown_context += "#### [Source: Quote History.csv] Historic CRM Deals & Pipeline Activity Transactions:\n```\n" + opp_matches.to_string(index=False) + "\n```\n\n"
        
    return markdown_context

# ──────────────────────────────────────────────────────────────────────────────
# 5. SIDEBAR CONTROLS & STATE INITIALIZATION
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.header("🔑 Session Configurations")
anthropic_api_key = st.sidebar.text_input("Anthropic API Key", type="password")
selected_model = st.sidebar.selectbox("Claude Variant", ["claude-sonnet-5", "claude-haiku-4-5-20251001"])

if "current_part" not in st.session_state:
    st.session_state.current_part = ""
if "current_qty" not in st.session_state:
    st.session_state.current_qty = 0
if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_context" not in st.session_state:
    st.session_state.active_context = ""
if "ui_display_context" not in st.session_state:
    st.session_state.ui_display_context = ""

if st.sidebar.button("Clear Chat Conversation"):
    st.session_state.messages = []
    st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# 6. INPUT PARAMETERS CONTROL MATRIX
# ──────────────────────────────────────────────────────────────────────────────
col_input, col_config = st.columns(2)
with col_input:
    input_part = st.text_input("🔍 Target Part Number", placeholder="e.g., ADP....").strip()
with col_config:
    input_qty = st.number_input("📦 Target Deal Volume (Quantity)", min_value=1, value=150, step=1)

if st.button("Initialize / Rerun Pricing Analysis") and input_part:
    with st.spinner("Extracting records from tables..."):
        context_payload = extract_part_intelligence(input_part)
        
        if context_payload is None:
            st.error(f"❌ Part number **'{input_part}'** was not resolved in any linked files.")
            st.session_state.active_context = ""
            st.session_state.ui_display_context = ""
        else:
            st.session_state.current_part = input_part
            st.session_state.current_qty = input_qty
            st.session_state.active_context = context_payload
            st.session_state.messages = []  
            
            sections = context_payload.split("#### ")
            cleaned_ui_context = ""
            for section in sections:
                if section and "Quote History.csv" not in section:
                    prefix = "#### " if cleaned_ui_context else ""
                    cleaned_ui_context += prefix + section
            st.session_state.ui_display_context = cleaned_ui_context
            
            system_instruction = (
                "You are a highly skilled Senior Pricing Analyst specializing in industrial electronics components. "
                "CRITICAL OUTPUT INSTRUCTION: You must strictly use normal spacing between every single word. Never output words "
                "mashed together in a continuous string. For example, write 'is the officially approved floor' instead of "
                "'is_the_officially_approved_floor' or 'istheofficiallyapprovedfloor'. "
                "When formatting text in bold or italics, ensure you place standard spaces BEFORE and AFTER the formatting asterisks."
            )
            
            prompt_template = """
Analyze the financial position for part identifier: {part} for a prospective customer transaction order size of {qty} units.

Here is the verified exact-match telemetry from our backend systems data ledgers (including all historic context details):
{context}

Please generate a structured analysis report covering:
1. **Cost & Baseline Matrix:** Evaluate real internal manufacturing costs vs standard marketplace benchmarks shown in the data.
2. **Target Quantity Pricing Tier Alignment:** Identify the closest catalog discount break tier for a request size of {qty} units.
3. **Historical Pipeline Insights:** Review the historic opportunities provided in the context file data. Summarize previous won vs lost trends and the historical pricing thresholds without outputting the raw data table rows.
4. **Definitive Price Recommendation:** Propose a calculated floor, target, and ceiling net unit price. Explicitly state the calculated gross margin percentage based on your suggested target price.
"""
            initial_user_prompt = prompt_template.format(part=input_part, qty=str(input_qty), context=context_payload)
            
            if not anthropic_api_key:
                st.warning("⚠️ Enter your Anthropic API Key in the sidebar to complete initialization.")
            else:
                try:
                    client = anthropic.Anthropic(api_key=anthropic_api_key)
                    message = client.messages.create(
                        model=selected_model,
                        max_tokens=8000,
                        system=system_instruction,
                        messages=[{"role": "user", "content": initial_user_prompt}]
                    )
                    
                    raw_response = "".join([block.text for block in message.content if hasattr(block, "text")])
                    sanitized_response = sanitize_markdown_spacing(raw_response)
                    
                    st.session_state.messages.append({"role": "user", "content": f"Analyze part {input_part} at quantity {input_qty}."})
                    st.session_state.messages.append({"role": "assistant", "content": sanitized_response})
                    st.rerun()
                except Exception as e:
                    st.error(f"API Connection failure: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# 7. CHAT HISTORY RENDERING & CONVERSATIONAL STREAM INTERACTION
# ──────────────────────────────────────────────────────────────────────────────
if st.session_state.active_context:
    st.markdown("### 💬 Strategic Conversation Stream")
    
    with st.expander("👁️ View Extracted Context Fed to Claude (Historical Table Hidden in UI)"):
        st.markdown(st.session_state.ui_display_context)
        
    st.markdown("---")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if follow_up_query := st.chat_input("Ask a follow-up question regarding this part...", key="pricing_chat_input"):
        st.session_state.messages.append({"role": "user", "content": follow_up_query})
        
        with st.chat_message("user"):
            st.markdown(follow_up_query)
            
        with st.spinner("Claude is analyzing pricing data context..."):
            try:
                conversation_payload = [
                    {"role": "user", "content": f"Here is the context data: {st.session_state.active_context}"}
                ]
                for msg in st.session_state.messages:
                    conversation_payload.append(msg)
                    
                client = anthropic.Anthropic(api_key=anthropic_api_key)
                follow_up_response = client.messages.create(
                    model=selected_model,
                    max_tokens=8000,
                    system="You are an expert Pricing Co-Pilot. Answer questions concisely and maintain strict word spacing guidelines.",
                    messages=conversation_payload
                )
                
                raw_reply = "".join([block.text for block in follow_up_response.content if hasattr(block, "text")])
                sanitized_reply = sanitize_markdown_spacing(raw_reply)
                
                st.session_state.messages.append({"role": "assistant", "content": sanitized_reply})
                with st.chat_message("assistant"):
                    st.markdown(sanitized_reply)
                st.rerun()
                
            except Exception as api_err:
                st.error(f"Chat execution interface failure: {api_err}")
else:
    st.info("💡 Input a part number and volume, then click 'Initialize / Rerun Pricing Analysis' to establish your workspace.")
