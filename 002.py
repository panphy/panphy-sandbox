import streamlit as st
from openai import OpenAI
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import gspread
from google.oauth2.service_account import Credentials
import io, base64, json, pandas as pd, numpy as np
from datetime import datetime

st.set_page_config(page_title="Physics Examiner Pro", layout="wide")

# --- AUTHENTICATION HELPER ---
@st.cache_resource
def get_gspread_client():
    try:
        # Step 1: Get the raw string from secrets
        raw_json_str = st.secrets["connections"]["gsheets"]["service_account_info"]
        
        # Step 2: Clean potential "control characters" (tabs/newlines)
        # This fixes the "Invalid control character" error
        clean_json_str = raw_json_str.strip().replace('\n', '\\n').replace('\r', '\\r')
        
        # Step 3: Parse and Authenticate
        info = json.loads(raw_json_str) # standard json.loads
        scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        creds = Credentials.from_service_account_info(info, scopes=scope)
        return gspread.authorize(creds)
    except Exception as e:
        st.error(f"❌ Authentication Failed: {e}")
        st.info("Ensure your Secrets JSON has no literal line breaks inside the private_key quotes.")
        return None

gc = get_gspread_client()
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- APP LOGIC ---
if "canvas_key" not in st.session_state: st.session_state["canvas_key"] = 0
if "feedback" not in st.session_state: st.session_state["feedback"] = None

QUESTIONS = {
    "Q1: Forces": {"question": "5kg box, 20N force, 4N friction. Acceleration?", "marks": 3, "mark_scheme": "1. F=16N. 2. F=ma. 3. a=3.2m/s²."},
    "Q2: Refraction": {"question": "Draw ray diagram: air to glass block.", "marks": 2, "mark_scheme": "1. Bends to normal. 2. Labels."}
}

def save_to_cloud(name, set_name, q_name, score, max_m, summary):
    if not gc: return False
    try:
        url = st.secrets["connections"]["gsheets"]["spreadsheet"]
        s_id = url.split("/d/")[1].split("/")[0] if "/d/" in url else url
        sheet = gc.open_by_key(s_id).get_worksheet(0)
        sheet.append_row([datetime.now().strftime("%Y-%m-%d %H:%M"), name, set_name, q_name, int(score), int(max_m), str(summary)])
        return True
    except Exception as e:
        st.error(f"Save error: {e}")
        return False

# --- UI TABS ---
t1, t2 = st.tabs(["✍️ Student Portal", "📊 Teacher Dashboard"])

with t1:
    st.title("⚛️ Physics Examiner")
    with st.expander("👤 Identity"):
        c1, c2, c3 = st.columns(3)
        name = f"{c1.text_input('First')} {c2.text_input('Last')}"
        cl_set = c3.selectbox("Set", ["11Y/Ph1", "11X/Ph2", "Teacher Test"])

    col_l, col_r = st.columns(2)
    with col_l:
        q_key = st.selectbox("Question", list(QUESTIONS.keys()))
        q_data = QUESTIONS[q_key]
        st.info(q_data['question'])
        
        mode = st.radio("Mode", ["Type", "Draw"], horizontal=True)
        if mode == "Type":
            ans = st.text_area("Working:")
            if st.button("Submit"):
                # Simplified GPT call for brevity
                res = client.chat.completions.create(
                    model="gpt-5-nano",
                    messages=[{"role": "user", "content": f"Mark: {ans}. Scheme: {q_data['mark_scheme']}"}],
                    response_format={"type": "json_object"}
                )
                data = json.loads(res.choices[0].message.content)
                # Ensure the keys match what's in the save_to_cloud call
                # GPT-5 is instructed to return 'score' and 'summary'
                st.session_state["feedback"] = data
                save_to_cloud(name, cl_set, q_key, data.get('score', 0), q_data['marks'], data.get('summary', ''))
        else:
            # Toolbar
            tool = st.toggle("Eraser")
            canvas = st_canvas(stroke_width=20 if tool else 2, stroke_color="#f8f9fa" if tool else "#000", background_color="#f8f9fa", height=300, width=400, key=f"c_{st.session_state['canvas_key']}")
            if st.button("Submit Drawing"):
                st.write("Analyzing...") # Placeholder for full vision logic

    with col_r:
        if st.session_state["feedback"]:
            f = st.session_state["feedback"]
            st.metric("Score", f.get('score', 0))
            st.write(f.get('summary', ''))

with t2:
    if st.text_input("Password", type="password") == "Newton2025":
        if gc:
            url = st.secrets["connections"]["gsheets"]["spreadsheet"]
            s_id = url.split("/d/")[1].split("/")[0] if "/d/" in url else url
            rows = gc.open_by_key(s_id).get_worksheet(0).get_all_records()
            st.dataframe(pd.DataFrame(rows))
