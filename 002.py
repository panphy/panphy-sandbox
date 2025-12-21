import streamlit as st
from openai import OpenAI
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import io
import base64
import json
import re
import numpy as np
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="AI Physics Examiner + Cloud DB",
    page_icon="⚛️",
    layout="wide"
)

# --- CONSTANTS ---
MODEL_NAME = "gpt-4o" # Using a valid model name (GPT-5-nano placeholder replaced for stability)
CANVAS_BG_HEX = "#f8f9fa"
CANVAS_BG_RGB = (248, 249, 250)
MAX_IMAGE_WIDTH = 1024

# --- SCOPE FOR GOOGLE SHEETS ---
G_SCOPE = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

# --- SETUP CLIENTS ---
@st.cache_resource
def get_openai_client():
    try:
        return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    except Exception:
        return None

@st.cache_resource
def get_google_sheet():
    """Connects to Google Sheets using st.secrets."""
    if "gcp_service_account" not in st.secrets:
        return None
    
    try:
        creds = ServiceAccountCredentials.from_json_keyfile_dict(
            dict(st.secrets["gcp_service_account"]), G_SCOPE
        )
        client = gspread.authorize(creds)
        # Ensure you created a sheet named "Student Marks" in your Drive
        sheet = client.open("Student Marks").sheet1 
        return sheet
    except Exception as e:
        st.error(f"DB Connection Error: {e}")
        return None

client = get_openai_client()
sheet_conn = get_google_sheet()

AI_READY = client is not None
DB_READY = sheet_conn is not None

# --- SESSION STATE ---
if "canvas_key" not in st.session_state:
    st.session_state["canvas_key"] = 0
if "feedback" not in st.session_state:
    st.session_state["feedback"] = None

# --- QUESTION BANK ---
QUESTIONS = {
    "Q1: Forces (Resultant)": {
        "question": "A 5kg box is pushed with a 20N force. Friction is 4N. Calculate the acceleration.",
        "marks": 3,
        "mark_scheme": "1. Resultant force = 20 - 4 = 16N (1 mark). 2. F = ma (1 mark). 3. a = 16 / 5 = 3.2 m/s² (1 mark).",
    },
    "Q2: Refraction (Drawing)": {
        "question": "Draw a ray diagram showing light passing from air into a glass block at an angle.",
        "marks": 2,
        "mark_scheme": "1. Ray bends towards the normal inside the glass. 2. Angles of incidence and refraction labeled correctly.",
    },
}

# --- HELPER FUNCTIONS ---
def encode_image(image_pil: Image.Image) -> str:
    buffered = io.BytesIO()
    image_pil.save(buffered, format="PNG", optimize=True)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def safe_parse_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None

def clamp_int(value, lo, hi, default=0):
    try:
        v = int(value)
    except Exception:
        v = default
    return max(lo, min(hi, v))

def canvas_has_ink(image_data: np.ndarray) -> bool:
    if image_data is None: return False
    arr = image_data.astype(np.uint8)
    if arr.ndim != 3 or arr.shape[2] < 3: return False
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3] if arr.shape[2] >= 4 else np.full((arr.shape[0], arr.shape[1]), 255, dtype=np.uint8)
    bg = np.array(CANVAS_BG_RGB, dtype=np.uint8)
    diff = np.abs(rgb.astype(np.int16) - bg.astype(np.int16)).sum(axis=2)
    ink = (diff > 60) & (alpha > 30)
    return (ink.mean() > 0.001)

def preprocess_canvas_image(image_data: np.ndarray) -> Image.Image:
    raw_img = Image.fromarray(image_data.astype("uint8"))
    if raw_img.mode == "RGBA":
        white_bg = Image.new("RGB", raw_img.size, (255, 255, 255))
        white_bg.paste(raw_img, mask=raw_img.split()[3])
        img = white_bg
    else:
        img = raw_img.convert("RGB")
    if img.width > MAX_IMAGE_WIDTH:
        ratio = MAX_IMAGE_WIDTH / img.width
        img = img.resize((MAX_IMAGE_WIDTH, int(img.height * ratio)))
    return img

def log_to_sheet(name, topic, marks, max_marks, summary):
    """Writes data to Google Sheet."""
    if not DB_READY:
        st.error("Database not connected. Check secrets.")
        return False
    
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Ensure your Google Sheet columns match this order:
        # Timestamp | Name | Topic | Marks | Max | Summary
        sheet_conn.append_row([timestamp, name, topic, marks, max_marks, summary])
        return True
    except Exception as e:
        st.error(f"Failed to save: {e}")
        return False

def get_gpt_feedback(student_answer, q_data, is_image=False):
    max_marks = q_data["marks"]
    system_instr = f"""
You are a strict GCSE Physics examiner.
CONFIDENTIALITY: Do NOT reveal the mark scheme.
OUTPUT: JSON only.
Schema:
{{
  "marks_awarded": <int>,
  "max_marks": <int>,
  "summary": "<1-2 sentences>",
  "feedback_points": ["<point 1>", "<point 2>"],
  "next_steps": ["<action 1>", "<action 2>"]
}}
Question: {q_data["question"]}
Max Marks: {max_marks}
""".strip()

    messages = [{"role": "system", "content": system_instr}]
    messages.append({"role": "system", "content": f"MARK SCHEME: {q_data['mark_scheme']}"})

    if is_image:
        base64_img = encode_image(student_answer)
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Mark this. JSON only."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
            ]
        })
    else:
        messages.append({"role": "user", "content": f"Answer:\n{student_answer}\nJSON only."})

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_completion_tokens=800,
            reasoning_effort="minimal"
        )
        data = safe_parse_json(response.choices[0].message.content or "")
        if not data: raise ValueError("Invalid JSON")

        return {
            "marks_awarded": clamp_int(data.get("marks_awarded", 0), 0, max_marks),
            "max_marks": max_marks,
            "summary": str(data.get("summary", "")).strip(),
            "feedback_points": data.get("feedback_points", [])[:4],
            "next_steps": data.get("next_steps", [])[:3]
        }
    except Exception as e:
        return {"marks_awarded": 0, "max_marks": max_marks, "summary": f"Error: {e}", "feedback_points": [], "next_steps": []}

def render_report(report: dict):
    st.markdown(f"**Marks:** {report.get('marks_awarded', 0)} / {report.get('max_marks', 0)}")
    st.info(report.get('summary', ''))
    if report.get("feedback_points"):
        st.write("**Feedback:**")
        for p in report["feedback_points"]: st.write(f"- {p}")
    if report.get("next_steps"):
        st.write("**Next Steps:**")
        for n in report["next_steps"]: st.write(f"- {n}")

# --- MAIN APP UI ---

# 1. Top Navigation
top_col1, top_col2, top_col3 = st.columns([3, 2, 1])
with top_col1:
    st.title("⚛️ AI Examiner")
    st.caption("Multimodal Grading & Tracking")
with top_col2:
    q_key = st.selectbox("Select Topic:", list(QUESTIONS.keys()))
    q_data = QUESTIONS[q_key]
with top_col3:
    status_color = "🟢" if (AI_READY and DB_READY) else "🟠" if AI_READY else "🔴"
    st.markdown(f"**System Status:** {status_color}")
    if not DB_READY:
        st.caption("Cloud DB Offline")

st.divider()

# 2. Workspace
col1, col2 = st.columns([5, 4])

with col1:
    st.subheader("📝 The Question")
    st.markdown(f"**{q_data['question']}**")
    st.caption(f"Max Marks: {q_data['marks']}")
    
    tab_type, tab_draw = st.tabs(["⌨️ Type Answer", "✍️ Draw Answer"])

    with tab_type:
        answer = st.text_area("Your Answer:", height=200)
        if st.button("Submit Text", type="primary", disabled=not AI_READY):
            if not answer.strip(): st.toast("Type something first!")
            else:
                with st.spinner("Marking..."):
                    st.session_state["feedback"] = get_gpt_feedback(answer, q_data)

    with tab_draw:
        tool = st.radio("Tool", ["Pen", "Eraser"], horizontal=True, label_visibility="collapsed")
        stroke = "#000000" if tool == "Pen" else CANVAS_BG_HEX
        width = 2 if tool == "Pen" else 20
        
        canvas_res = st_canvas(
            stroke_width=width, stroke_color=stroke, background_color=CANVAS_BG_HEX,
            height=400, width=600, drawing_mode="freedraw", key=f"cv_{st.session_state['canvas_key']}"
        )
        if st.button("Submit Drawing", type="primary", disabled=not AI_READY):
            if not canvas_has_ink(canvas_res.image_data): st.toast("Canvas is empty!")
            else:
                with st.spinner("Analyzing Diagram..."):
                    img = preprocess_canvas_image(canvas_res.image_data)
                    st.session_state["feedback"] = get_gpt_feedback(img, q_data, True)
        
        if st.button("Clear Canvas"):
            st.session_state["canvas_key"] += 1
            st.session_state["feedback"] = None
            st.rerun()

with col2:
    st.subheader("👨‍🏫 Result Card")
    
    with st.container(border=True):
        if st.session_state["feedback"]:
            render_report(st.session_state["feedback"])
            
            st.divider()
            
            # --- SAVE TO CLOUD FEATURE ---
            st.markdown("#### ☁️ Save Progress")
            student_name = st.text_input("Enter Name:", placeholder="e.g. John Doe")
            
            if st.button("💾 Save to Teacher Dashboard", disabled=not DB_READY):
                if not student_name.strip():
                    st.toast("Please enter your name first.", icon="⚠️")
                else:
                    with st.spinner("Syncing..."):
                        success = log_to_sheet(
                            student_name, 
                            q_key, 
                            st.session_state["feedback"]["marks_awarded"],
                            st.session_state["feedback"]["max_marks"],
                            st.session_state["feedback"]["summary"]
                        )
                        if success:
                            st.success("Data saved successfully!")
                            st.balloons()
        else:
            st.info("Submit an answer to see feedback.")

# 3. Teacher Dashboard (Secure)
st.markdown("<br><br><br>", unsafe_allow_html=True)
with st.expander("🔐 Teacher Dashboard (Admin Access)"):
    pwd = st.text_input("Admin Password", type="password")
    admin_pass = st.secrets.get("admin", {}).get("password", "admin")
    
    if pwd == admin_pass:
        if DB_READY:
            st.write("### 📊 Live Student Performance")
            try:
                data = sheet_conn.get_all_records()
                if data:
                    df = pd.DataFrame(data)
                    st.dataframe(df, use_container_width=True)
                    
                    # Mini Analytics
                    if not df.empty and "Marks" in df.columns:
                        st.subheader("Performance Overview")
                        st.bar_chart(df.set_index("Name")["Marks"])
                else:
                    st.info("No data recorded yet.")
            except Exception as e:
                st.error(f"Error fetching data: {e}")
        else:
            st.warning("Database not connected.")
