import streamlit as st
from openai import OpenAI
from streamlit_drawable_canvas import st_canvas
from PIL import Image
from streamlit_gsheets import GSheetsConnection
import io
import base64
import json
import re
import numpy as np
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(page_title="Physics AI Examiner & Dashboard", page_icon="⚛️", layout="wide")

# --- CONNECTIONS & CLIENTS ---
@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

try:
    client = get_openai_client()
    # Google Sheets Connection
    conn = st.connection("gsheets", type=GSheetsConnection)
    AI_READY = True
except Exception as e:
    st.error(f"Configuration Error: {e}")
    AI_READY = False

# --- SESSION STATE ---
if "canvas_key" not in st.session_state: st.session_state["canvas_key"] = 0
if "feedback" not in st.session_state: st.session_state["feedback"] = None

# --- DATA BANK ---
QUESTIONS = {
    "Q1: Forces": {"question": "A 5kg box is pushed with a 20N force. Friction is 4N. Calculate the acceleration.", "marks": 3, "mark_scheme": "1. Resultant force = 16N. 2. F=ma. 3. a=3.2 m/s²."},
    "Q2: Refraction": {"question": "Draw a ray diagram showing light passing from air into glass.", "marks": 2, "mark_scheme": "1. Bends toward normal. 2. Correct labels."}
}
CLASS_SETS = ["11Y/Ph1", "11X/Ph2", "10A/Ph1", "Teacher Test"]

# --- UTILITY FUNCTIONS ---
def save_to_cloud(name, set_name, q_name, score, max_m, summary):
    """Saves student results to your personal Google Sheet on the server side."""
    try:
        existing_data = conn.read(ttl=0)
        new_row = pd.DataFrame([{
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Student Name": name,
            "Class Set": set_name,
            "Question": q_name,
            "Score": score,
            "Max Marks": max_m,
            "Feedback Summary": summary
        }])
        updated_df = pd.concat([existing_data, new_row], ignore_index=True)
        conn.update(data=updated_df)
        return True
    except:
        return False

def get_gpt_feedback(answer, q_data, is_image=False):
    model_name = "gpt-5-nano"
    # We use a JSON-focused prompt to ensure we get structured data back
    system_instr = f"You are a GCSE Examiner. Mark strictly. Return ONLY JSON. Scheme: {q_data['mark_scheme']}"
    
    messages = [{"role": "system", "content": system_instr}]
    if is_image:
        buffered = io.BytesIO()
        answer.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode()
        messages.append({"role": "user", "content": [{"type": "text", "text": "Mark this image."}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}]})
    else:
        messages.append({"role": "user", "content": f"Student Answer: {answer}"})

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=1500,
        reasoning_effort="minimal",
        response_format={ "type": "json_object" } # Forces JSON output
    )
    return json.loads(response.choices[0].message.content)

# --- TEACHER DASHBOARD LOGIC ---
def check_password():
    def password_entered():
        if st.session_state["password"] == "Newton2025": # Change your password here
            st.session_state["password_correct"] = True
        else:
            st.session_state["password_correct"] = False
    if not st.session_state.get("password_correct", False):
        st.text_input("Teacher Access", type="password", on_change=password_entered, key="password")
        return False
    return True

# --- MAIN UI ---
tab1, tab2 = st.tabs(["✍️ Student Portal", "📊 Teacher Dashboard"])

with tab1:
    st.title("⚛️ AI Physics Examiner")
    
    with st.expander("👤 Student Identity", expanded=True):
        col_f, col_l, col_s = st.columns(3)
        fname = col_f.text_input("First Name")
        lname = col_l.text_input("Last Name")
        student_set = col_s.selectbox("Class Set", CLASS_SETS)

    col_q, col_ans = st.columns([1, 1])
    with col_q:
        q_key = st.selectbox("Select Question", list(QUESTIONS.keys()))
        q_data = QUESTIONS[q_key]
        st.info(f"**Question:** {q_data['question']}")
        
        mode = st.radio("Mode:", ["Type", "Draw"], horizontal=True)
        if mode == "Type":
            ans = st.text_area("Your answer:")
            if st.button("Submit Answer"):
                with st.spinner("Marking..."):
                    result = get_gpt_feedback(ans, q_data)
                    st.session_state["feedback"] = result
                    save_to_cloud(f"{fname} {lname}", student_set, q_key, result['marks_awarded'], q_data['marks'], result['summary'])
        else:
            canvas = st_canvas(stroke_width=2, stroke_color="#000", background_color="#f8f9fa", height=300, width=500, key=f"c_{st.session_state['canvas_key']}")
            if st.button("Submit Drawing"):
                raw_img = Image.fromarray(canvas.image_data.astype('uint8'))
                # Pre-processing for AI clarity
                white_bg = Image.new("RGB", raw_img.size, (255, 255, 255))
                white_bg.paste(raw_img, mask=raw_img.split()[3])
                with st.spinner("Analyzing..."):
                    result = get_gpt_feedback(white_bg, q_data, is_image=True)
                    st.session_state["feedback"] = result
                    save_to_cloud(f"{fname} {lname}", student_set, q_key, result['marks_awarded'], q_data['marks'], result['summary'])

    with col_ans:
        if st.session_state["feedback"]:
            res = st.session_state["feedback"]
            st.metric("Score", f"{res['marks_awarded']} / {q_data['marks']}")
            st.write(f"**Feedback:** {res['summary']}")
            if st.button("Try Again"):
                st.session_state["feedback"] = None
                st.session_state["canvas_key"] += 1
                st.rerun()

with tab2:
    if check_password():
        st.title("👩‍🏫 Class Overview")
        try:
            df = conn.read(ttl=0)
            st.dataframe(df, use_container_width=True)
            
            # Simple analytics
            st.subheader("Performance by Set")
            avg_scores = df.groupby("Class Set")["Score"].mean()
            st.bar_chart(avg_scores)
        except:
            st.info("No student data submitted yet.")
