import streamlit as st
from openai import OpenAI
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import io
import base64

# --- PAGE CONFIG ---
st.set_page_config(page_title="OpenAI Physics Examiner", page_icon="🚀", layout="wide")

# --- INITIALIZE OPENAI CLIENT ---
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    AI_READY = True
except Exception:
    st.error("⚠️ OpenAI API Key missing in Streamlit Secrets!")
    AI_READY = False

# --- QUESTION BANK ---
QUESTIONS = {
    "Q1: Forces (Resultant)": {
        "question": "A 5kg box is pushed with a 20N force. Friction is 4N. Calculate the acceleration.",
        "marks": 3,
        "mark_scheme": "1. Resultant force = 20 - 4 = 16N (1 mark). 2. F = ma (1 mark). 3. a = 16 / 5 = 3.2 m/s² (1 mark)."
    },
    "Q2: Refraction (Drawing)": {
        "question": "Draw a ray diagram showing light passing from air into a glass block at an angle.",
        "marks": 2,
        "mark_scheme": "1. Ray bends towards the normal inside the glass. 2. Angles of incidence and refraction labeled correctly."
    }
}

# --- HELPER FUNCTIONS ---
def encode_image(image_pil):
    buffered = io.BytesIO()
    image_pil.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def get_gpt_feedback(student_answer, q_data, is_image=False):
    model_name = "gpt-5.2" 
    system_instr = f"You are a strict GCSE Physics Examiner.\nQuestion: {q_data['question']}\nScheme: {q_data['mark_scheme']}\nMax Marks: {q_data['marks']}"
    messages = [{"role": "system", "content": system_instr}]
    
    if is_image:
        base64_img = encode_image(student_answer)
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Mark this handwritten/drawn answer strictly."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
            ]
        })
    else:
        messages.append({"role": "user", "content": f"Student Answer: {student_answer}"})

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_completion_tokens=600 # Updated parameter for GPT-5.x series
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Examiner Error: {str(e)}"

# --- APP UI ---
st.title("🚀 OpenAI Examiner (GPT-5.2)")

with st.sidebar:
    st.header("Select Task")
    q_key = st.selectbox("Question Topic", list(QUESTIONS.keys()))
    q_data = QUESTIONS[q_key]
    st.divider()
    st.markdown("### Examiner Tips")
    st.caption("Show all working out to gain partial marks even if your final answer is wrong.")

# Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 The Question")
    st.info(f"**{q_key}**\n\n{q_data['question']}\n\n*(Max Marks: {q_data['marks']})*")
    
    # Mode Switcher
    mode = st.radio("How will you answer?", ["⌨️ Type", "✍️ Handwriting/Drawing"], horizontal=True)

    if mode == "⌨️ Type":
        # TYPING MODE
        answer = st.text_area("Type your working and final answer here:", height=250)
        if st.button("Submit Text Answer") and AI_READY:
            with st.spinner("GPT-5.2 is marking..."):
                st.session_state["feedback"] = get_gpt_feedback(answer, q_data)
    
    else:
        # DRAWING MODE
        st.write("Draw/Write your working below:")
        
        # Pen/Eraser Toggle
        tool_col1, tool_col2 = st.columns([1, 4])
        with tool_col1:
            tool = st.radio("Tool:", ["🖊️ Pen", "🧼 Eraser"], label_visibility="collapsed")
        
        # Tool Logic
        current_stroke = "#000000" if tool == "🖊️ Pen" else "#f8f9fa" # Eraser matches BG color
        stroke_width = 2 if tool == "🖊️ Pen" else 20 # Wider stroke for eraser
        
        canvas_result = st_canvas(
            stroke_width=stroke_width,
            stroke_color=current_stroke,
            background_color="#f8f9fa",
            height=350,
            width=550,
            drawing_mode="freedraw",
            key="canvas_marking"
        )
        
        if st.button("Submit Drawing") and AI_READY:
            if canvas_result.image_data is not None:
                with st.spinner("Analyzing handwriting..."):
                    raw_img = Image.fromarray(canvas_result.image_data.astype('uint8'))
                    # Replace transparency with white background for AI vision clarity
                    white_bg = Image.new("RGB", raw_img.size, (255, 255, 255))
                    white_bg.paste(raw_img, mask=raw_img.split()[3]) 
                    st.session_state["feedback"] = get_gpt_feedback(white_bg, q_data, is_image=True)
            else:
                st.warning("Please draw something first.")

with col2:
    st.subheader("👨‍🏫 Examiner's Report")
    if "feedback" in st.session_state:
        st.markdown(st.session_state["feedback"])
        if st.button("Clear Report"):
            del st.session_state["feedback"]
            st.rerun()
    else:
        st.info("Your feedback will appear here once you submit an answer.")
