import streamlit as st
import google.generativeai as genai
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import io
import numpy as np

# --- CONFIGURATION ---
st.set_page_config(page_title="GCSE Physics AI Examiner", page_icon="📝", layout="wide")

# --- API SETUP ---
# We try to grab the key from secrets.
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    AI_AVAILABLE = True
except:
    st.error("⚠️ API Key missing. Please add 'GEMINI_API_KEY' to Streamlit Secrets.")
    AI_AVAILABLE = False

# --- QUESTION BANK ---
# In a real app, this could come from a spreadsheet or database.
QUESTIONS = {
    "Q1: Forces": {
        "question": "A car of mass 1200 kg accelerates from rest to 20 m/s in 10 seconds. Calculate the resultant force acting on the car. Show your working.",
        "marks": 3,
        "mark_scheme": "1. Recall F = ma or a = (v-u)/t (1 mark).\n2. Calculate acceleration: a = 20/10 = 2 m/s² (1 mark).\n3. Calculate Force: F = 1200 * 2 = 2400 N (1 mark)."
    },
    "Q2: Energy": {
        "question": "Describe the energy changes that take place when a student jumps off a chair and lands on the ground.",
        "marks": 3,
        "mark_scheme": "1. GPE decreases as they fall (1 mark).\n2. KE increases as they fall (1 mark).\n3. On impact, KE is transferred to Sound and Thermal/Internal energy (1 mark)."
    },
    "Q3: Waves": {
        "question": "Explain the difference between longitudinal and transverse waves in terms of particle vibration.",
        "marks": 2,
        "mark_scheme": "1. Transverse: Vibration is perpendicular (90 degrees) to direction of energy transfer.\n2. Longitudinal: Vibration is parallel to direction of energy transfer."
    }
}

# --- HELPER FUNCTIONS ---

def get_ai_feedback(student_input, question_data, input_type="text"):
    """
    Sends the student's work to Gemini for marking.
    input_type can be 'text' or 'image'.
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # The Prompt (Instruction to the AI)
    system_prompt = f"""
    You are a strict GCSE Physics Examiner. 
    Question: {question_data['question']}
    Mark Scheme: {question_data['mark_scheme']}
    Max Marks: {question_data['marks']}
    
    Task:
    1. Read the student's answer.
    2. Award marks strictly according to the mark scheme.
    3. If the answer is handwritten, interpret it as best as possible.
    4. Provide a short, constructive comment explaining where marks were gained or lost.
    
    Output Format:
    **Score:** X/{question_data['marks']}
    **Feedback:** [Your feedback here]
    """

    try:
        if input_type == "text":
            response = model.generate_content([system_prompt, f"Student Answer: {student_input}"])
        else:
            # student_input is a PIL Image here
            response = model.generate_content([system_prompt, student_input])
            
        return response.text
    except Exception as e:
        return f"Error connecting to AI Examiner: {e}"

# --- MAIN APP UI ---

st.title("🤖 AI Examiner: GCSE Physics")

# Sidebar: Select Question
selected_q_name = st.sidebar.selectbox("Select a Question", list(QUESTIONS.keys()))
q_data = QUESTIONS[selected_q_name]

# Display Question
st.info(f"**Question ({q_data['marks']} Marks):**\n\n{q_data['question']}")

# Input Mode Selection
mode = st.radio("How do you want to answer?", ["⌨️ Type Answer", "✍️ Handwriting (Tablet/Mouse)"], horizontal=True)

user_answer = None
image_data = None

# --- INPUT AREA ---
if mode == "⌨️ Type Answer":
    user_text = st.text_area("Type your answer here...", height=150)
    if st.button("Submit for Marking"):
        if user_text:
            user_answer = user_text
            input_type = "text"
        else:
            st.warning("Please type an answer first.")

else: # Handwriting Mode
    st.write("Draw your working out below:")
    # Create the canvas
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=2,
        stroke_color="#000000",
        background_color="#ffffff",
        height=300,
        width=600,
        drawing_mode="freedraw",
        key="canvas",
    )

    if st.button("Submit Handwriting"):
        if canvas_result.image_data is not None:
            # Convert the numpy array from the canvas into a PIL Image for the AI
            img_data = canvas_result.image_data.astype('uint8')
            image_from_canvas = Image.fromarray(img_data)
            
            # We must convert RGBA to RGB (remove transparency) for better AI processing
            bg_layer = Image.new("RGB", image_from_canvas.size, (255, 255, 255))
            bg_layer.paste(image_from_canvas, mask=image_from_canvas.split()[3]) # 3 is the alpha channel
            
            user_answer = bg_layer
            input_type = "image"
        else:
            st.warning("Please write something on the canvas.")

# --- MARKING SECTION ---
if user_answer is not None and AI_AVAILABLE:
    with st.spinner("The AI Examiner is marking your work..."):
        # Call the AI function
        feedback = get_ai_feedback(user_answer, q_data, input_type)
        
    st.divider()
    st.subheader("📝 Examiner's Report")
    st.markdown(feedback)
    
    # Show the Mark Scheme for reference
    with st.expander("View Official Mark Scheme"):
        st.write(q_data['mark_scheme'])

elif not AI_AVAILABLE:
    st.warning("AI features disabled. Please configure your API Key.")
