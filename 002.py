import streamlit as st
from openai import OpenAI
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import gspread
from google.oauth2.service_account import Credentials
import io, base64, json, pandas as pd, numpy as np
from datetime import datetime

# --- CONFIG & STYLING ---
st.set_page_config(page_title="Physics Examiner Pro", layout="wide")

# --- AUTHENTICATION HELPER ---
@st.cache_resource
def get_gspread_client():
    """
    Handles parsing of Service Account JSON from st.secrets.
    Fixed to handle both string and dict inputs to avoid 'length 1' errors.
    """
    try:
        # 1. Fetch the secret info
        raw_info = st.secrets["connections"]["gsheets"]["service_account_info"]
        
        # 2. Convert to dictionary if it's currently a string
        if isinstance(raw_info, str):
            info = json.loads(raw_info, strict=False)
        else:
            info = dict(raw_info)
        
        # 3. CRITICAL: Fix the private key formatting for the RSA library
        if "private_key" in info:
            info["private_key"] = info["private_key"].replace("\\n", "\n")
        
        scope = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        
        # 4. Authenticate
        creds = Credentials.from_service_account_info(info, scopes=scope)
        return gspread.authorize(creds)
    except Exception as e:
        st.error(f"❌ Authentication Failed: {e}")
        return None

# Initialize Clients
gc = get_gspread_client()
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- PHYSICS QUESTION BANK ---
QUESTIONS = {
    "Q1: Forces": {
        "question": "A 5kg box is pushed with a 20N force. Friction is 4N. Calculate acceleration.", 
        "marks": 3, 
        "mark_scheme": "1. Resultant Force = 16N. 2. Use F=ma. 3. a = 3.2 m/s²."
    },
    "Q2: Refraction": {
        "question": "Draw a ray diagram showing light traveling from air into a glass block.", 
        "marks": 2, 
        "mark_scheme": "1. Ray bends towards the normal in glass. 2. Correct labels for Incident and Refracted rays."
    }
}

# --- IMPROVED SAVE LOGIC ---
def save_to_cloud(name, set_name, q_name, score, max_m, summary):
    """Appends results and provides specific error feedback for 403/404 errors."""
    if not gc:
        st.error("Google Sheets client is not authenticated.")
        return False
    try:
        url = st.secrets["connections"]["gsheets"]["spreadsheet"]
        # Extract ID from URL
        s_id = url.split("/d/")[1].split("/")[0] if "/d/" in url else url
        
        # Open the sheet
        spreadsheet = gc.open_by_key(s_id)
        sheet = spreadsheet.get_worksheet(0)
        
        # Prepare the row data
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [timestamp, str(name), str(set_name), str(q_name), int(score), int(max_m), str(summary)]
        
        # Append the row
        sheet.append_row(row, value_input_option="USER_ENTERED")
        return True
        
    except gspread.exceptions.APIError as e:
        # This handles the "Save error" by identifying if permissions are missing
        msg = e.response.json().get('error', {}).get('message', str(e))
        st.error(f"❌ Google API Error: {msg}")
        if "permission" in msg.lower():
            email = st.secrets["connections"]["gsheets"]["service_account_info"]["client_email"]
            st.info(f"💡 ACTION REQUIRED: Share your Google Sheet with: `{email}` as an 'Editor'.")
        return False
    except Exception as e:
        st.error(f"⚠️ Unexpected Save Error: {e}")
        return False

# --- SESSION STATE ---
if "feedback" not in st.session_state: st.session_state["feedback"] = None

# --- UI LAYOUT ---
t1, t2 = st.tabs(["✍️ Student Portal", "📊 Teacher Dashboard"])

with t1:
    st.title("⚛️ Physics Examiner Pro")
    
    with st.expander("👤 Student Identity", expanded=True):
        c1, c2, c3 = st.columns(3)
        f_name = c1.text_input("First Name")
        l_name = c2.text_input("Last Name")
        cl_set = c3.selectbox("Physics Set", ["11Y/Ph1", "11X/Ph2", "Teacher Test"])
        full_name = f"{f_name} {l_name}"

    col_l, col_r = st.columns([2, 1])
    
    with col_l:
        q_key = st.selectbox("Select Question", list(QUESTIONS.keys()))
        q_data = QUESTIONS[q_key]
        st.info(f"**Question:** {q_data['question']}")
        
        mode = st.radio("Response Method", ["Type Calculations", "Draw Diagram"], horizontal=True)
        
        if mode == "Type Calculations":
            ans = st.text_area("Show your working:", height=200)
            if st.button("Submit Calculations", use_container_width=True):
                if not full_name.strip():
                    st.warning("Please enter your name first.")
                else:
                    with st.spinner("AI marking in progress..."):
                        prompt = f"GCSE Physics Marker. Q: {q_data['question']}\nAns: {ans}\nScheme: {q_data['mark_scheme']}\nTotal: {q_data['marks']}\nReturn JSON: {{'score': int, 'summary': str}}"
                        res = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[{"role": "user", "content": prompt}],
                            response_format={"type": "json_object"}
                        )
                        feedback = json.loads(res.choices[0].message.content)
                        st.session_state["feedback"] = feedback
                        
                        # Trigger cloud save
                        if save_to_cloud(full_name, cl_set, q_key, feedback.get('score', 0), q_data['marks'], feedback.get('summary', '')):
                            st.success("Result recorded successfully!")

        else:
            st.write("Draw your diagram below:")
            canvas = st_canvas(
                stroke_width=3, stroke_color="#000", background_color="#fff",
                height=350, width=500, drawing_mode="freedraw", key="canvas_phy"
            )
            
            if st.button("Submit Drawing", use_container_width=True):
                if canvas.image_data is not None:
                    with st.spinner("Analyzing diagram with AI..."):
                        # Convert canvas to Base64 image
                        img = Image.fromarray(canvas.image_data.astype('uint8'), 'RGBA').convert('RGB')
                        buffered = io.BytesIO()
                        img.save(buffered, format="PNG")
                        img_str = base64.b64encode(buffered.getvalue()).decode()

                        res = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[{
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": f"Mark this diagram based on: {q_data['mark_scheme']}. Marks: {q_data['marks']}. JSON: {{'score': int, 'summary': str}}"},
                                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}
                                ]
                            }],
                            response_format={"type": "json_object"}
                        )
                        feedback = json.loads(res.choices[0].message.content)
                        st.session_state["feedback"] = feedback
                        save_to_cloud(full_name, cl_set, q_key, feedback.get('score', 0), q_data['marks'], feedback.get('summary', ''))

    with col_r:
        st.subheader("Results & Feedback")
        if st.session_state["feedback"]:
            f = st.session_state["feedback"]
            st.metric("Score", f"{f.get('score', 0)} / {q_data['marks']}")
            st.markdown(f"**Examiner's Note:**\n\n{f.get('summary', 'No feedback provided.')}")
        else:
            st.write("Submit your work to receive instant AI marking.")

with t2:
    st.header("📊 Teacher Results Dashboard")
    if st.text_input("Access Password", type="password") == "Newton2025":
        if gc:
            try:
                url = st.secrets["connections"]["gsheets"]["spreadsheet"]
                s_id = url.split("/d/")[1].split("/")[0] if "/d/" in url else url
                rows = gc.open_by_key(s_id).get_worksheet(0).get_all_records()
                
                if rows:
                    df = pd.DataFrame(rows)
                    st.dataframe(df, use_container_width=True)
                    
                    # CSV Download Option
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Results CSV", data=csv, file_name="physics_results.csv", mime='text/csv')
                else:
                    st.info("The spreadsheet is currently empty.")
            except Exception as e:
                st.error(f"Error fetching dashboard data: {e}")
