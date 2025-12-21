import streamlit as st
import google.generativeai as genai
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="GCSE Physics AI Examiner", page_icon="📝", layout="wide")

# --- API SETUP & SMART MODEL SELECTOR ---
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    API_CONNECTED = True
except Exception:
    st.error("⚠️ API Key missing. Please add 'GEMINI_API_KEY' to your Streamlit Secrets.")
    API_CONNECTED = False

def get_best_model():
    """
    Scans the user's API for available models and picks the best one.
    Priority: Gemini 2.0/3.0 (Newest) -> Gemini 1.5 Flash-8b (Fastest) -> Gemini 1.5 Flash (Standard)
    """
    try:
        # Get list of all models available to this specific API key
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        # Preference List (Try these in order)
        preferences = [
            "models/gemini-2.0-flash-exp",   # Cutting edge (Dec 2024)
            "models/gemini-1.5-flash-8b",    # Ultra-fast, low latency
            "models/gemini-1.5-flash-002",   # Updated stable version
            "models/gemini-1.5-flash",       # Standard fallback
            "models/gemini-pro"              # Old reliable
        ]
        
        # Return the first match found
        for pref in preferences:
            if pref in available_models:
                return pref
        
        # If no preferred match, just return the first available 'gemini' model
        for model in available_models:
            if "gemini" in model:
                return model
                
        return "models/gemini-1.5-flash" # Absolute fallback
    except Exception as e:
        return "models/gemini-1.5-flash"

# Initialize Model on App Load
if API_CONNECTED:
    ACTIVE_MODEL_NAME = get_best_model()
else:
    ACTIVE_MODEL_NAME = "Offline"

# --- QUESTION BANK (GCSE Style) ---
QUESTIONS = {
    "Q1: Forces (Calculation)": {
        "question": "A car of mass 1200 kg accelerates from rest to 20 m/s in 10 seconds.\n\nCalculate the resultant force acting on the car. Show your working.",
        "marks": 3,
        "mark_scheme": "1. Recall F = ma or a = (v-u)/t (1 mark).\n2. Calculate acceleration: a = 20/10 = 2 m/s² (1 mark).\n3. Calculate Force: F = 1200 * 2 = 2400 N (1 mark)."
    },
    "Q2: Energy (Description)": {
        "question": "Describe the energy changes that take place when a student jumps off a chair and lands on the ground.",
        "marks": 3,
        "mark_scheme": "1. GPE decreases as they fall (1 mark).\n2. KE increases as they fall (1 mark).\n3. On impact, KE is transferred to Sound and Thermal/Internal energy (1 mark)."
    },
    "Q3: Waves (Explanation)": {
        "question": "Explain the difference between longitudinal and transverse waves in terms of particle vibration.",
        "marks": 2,
        "mark_scheme": "1. Transverse: Vibration is perpendicular (90 degrees) to direction of energy transfer.\n2. Longitudinal: Vibration is parallel to direction of energy transfer."
    },
     "Q4: Electricity (Circuit)": {
        "question": "A filament lamp breaks in a series circuit. Explain what happens to the other components in the circuit.",
        "marks": 2,
        "mark_scheme": "1. The circuit is broken / incomplete (1 mark).\n2. Current stops flowing, so all other components stop working (1 mark)."
    }
}

# --- HELPER: AI MARKING FUNCTION ---
def get_ai_feedback(student_input, question_data, input_type="text"):
    try:
        model = genai.GenerativeModel(ACTIVE_MODEL_NAME)
        
        system_prompt = f"""
        You are a strict GCSE Physics Examiner. 
        
        Question: {question_data['question']}
        Mark Scheme: {question_data['mark_scheme']}
        Max Marks: {question_data['marks']}
        
        Task:
        1. Review the student's answer (text or image).
        2. Compare strictly against the Mark Scheme.
        3. If handwritten, interpret the handwriting first.
        
        Output Format:
        **Score:** X / {question_data['marks']}
        **Feedback:** [Short, constructive feedback explaining why marks were awarded or lost]
        """

        if input_type == "text":
            response = model.generate_content([system_prompt, f"Student Answer: {student_input}"])
        else:
            # student_input is a PIL Image
            response = model.generate_content([system_prompt, student_input])
            
        return response.text
    except Exception as e:
        return f"❌ Error: {str(e)}\n\n*Tip: Check if your API key is valid.*"

# --- MAIN APP UI ---

st.title("🤖 AI Examiner: GCSE Physics")
st.markdown(f"**Status:** Connected to `{ACTIVE_MODEL_NAME}`")

# Layout: Sidebar for navigation
with st.sidebar:
    st.header("Select Question")
    selected_q_name = st.selectbox("Topic", list(QUESTIONS.keys()))
    
    st.markdown("---")
    st.info("💡 **Tip:** Use the 'Handwriting' mode to practice showing your working out for calculation questions.")

q_data = QUESTIONS[selected_q_name]

# Main Content Area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 The Question")
    st.info(f"**{selected_q_name}**\n\n{q_data['question']}\n\n*(Max Marks: {q_data['marks']})*")
    
    # Input Mode Selection
    mode = st.radio("Answer Mode:", ["⌨️ Type Answer", "✍️ Handwriting"], horizontal=True)

    user_input = None
    input_type = None
    submit = False

    if mode == "⌨️ Type Answer":
        user_text = st.text_area("Type your answer:", height=200)
        if st.button("Submit Text Answer"):
            user_input = user_text
            input_type = "text"
            submit = True

    else: # Handwriting
        st.write("Draw your working below:")
        # Canvas configuration
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=2,
            stroke_color="#000000",
            background_color="#ffffff",
            height=300,
            width=500,
            drawing_mode="freedraw",
            key="canvas",
        )
        
        if st.button("Submit Handwriting"):
            if canvas_result.image_data is not None:
                # Process Image
                img_data = canvas_result.image_data.astype('uint8')
                image = Image.fromarray(img_data)
                
                # Create white background to replace transparency
                bg = Image.new("RGB", image.size, (255, 255, 255))
                bg.paste(image, mask=image.split()[3])
                
                user_input = bg
                input_type = "image"
                submit = True
            else:
                st.warning("Canvas is empty.")

with col2:
    st.subheader("👨‍🏫 Examiner's Report")
    
    if submit and API_CONNECTED:
        if user_input:
            with st.spinner("Marking your work..."):
                feedback = get_ai_feedback(user_input, q_data, input_type)
            
            st.success("Marking Complete!")
            st.markdown(feedback)
            
            with st.expander("👀 Reveal Mark Scheme"):
                st.write(q_data['mark_scheme'])
        else:
            st.warning("Please enter an answer to get feedback.")
            
    elif not API_CONNECTED:
        st.error("Please configure your API Key in Streamlit Secrets to enable marking.")
    else:
        st.info("Submit your answer on the left to see the examiner's feedback here.")

