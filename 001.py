import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- PAGE CONFIGURATION ---
# Changed to 'wide' layout for a more responsive, full-screen feel
st.set_page_config(
    page_title="AQA Physics: Resistance of a Wire",
    page_icon="⚡",
    layout="wide" 
)

# --- SESSION STATE INITIALIZATION ---
if 'student_data' not in st.session_state:
    default_data = {
        "Length (cm)": [10.0, 20.0, 30.0, 40.0, 50.0],
        "Voltage (V)": [0.0, 0.0, 0.0, 0.0, 0.0],
        "Current (A)": [0.0, 0.0, 0.0, 0.0, 0.0]
    }
    st.session_state['student_data'] = pd.DataFrame(default_data)

if 'gradient_verified' not in st.session_state:
    st.session_state['gradient_verified'] = False

if 'form_submitted' not in st.session_state:
    st.session_state['form_submitted'] = False

# --- HELPER FUNCTIONS ---
def check_password():
    def password_entered():
        if st.session_state["password"] == "Newton":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Teacher Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Teacher Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False
    else:
        return True

# --- MAIN APP LOGIC ---

st.title("⚡ Digital Lab Assistant")
st.markdown("**Practical:** Resistance of a Wire ($R = V/I$)")

tab1, tab2 = st.tabs(["👨‍🎓 Student Lab Book", "👩‍🏫 Teacher Dashboard"])

with tab1:
    # --- PHASE 1 & 2: INPUT FORM (This fixes the lag) ---
    # We wrap the inputs in a form. The app will NOT reload until the button is pressed.
    with st.form("lab_entry_form"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("1. Equipment Check")
            ammeter_res = st.number_input("Ammeter Resolution (A)", 0.001, 1.0, 0.01, format="%.3f")
            voltmeter_res = st.number_input("Voltmeter Resolution (V)", 0.01, 20.0, 0.1)
        
        with col2:
            st.subheader("2. Data Collection")
            st.markdown("_Edit values below. Click 'Update Graph' when done._")
            # The data editor is now inside the form
            edited_df = st.data_editor(
                st.session_state['student_data'],
                num_rows="dynamic",
                use_container_width=True,
                height=200
            )

        # The Form Submit Button
        submitted = st.form_submit_button("📊 Update Graph & Calculate")
        
        if submitted:
            st.session_state['student_data'] = edited_df
            st.session_state['form_submitted'] = True

    st.markdown("---")

    # --- PHASE 3: ANALYSIS (Only runs after submit) ---
    if st.session_state['form_submitted']:
        st.subheader("3. Analysis & Graphing")
        
        # Process Data
        plot_df = st.session_state['student_data'].copy()
        plot_df['Resistance (Ω)'] = plot_df.apply(
            lambda row: row['Voltage (V)'] / row['Current (A)'] if row['Current (A)'] > 0 else None, 
            axis=1
        )
        valid_data = plot_df.dropna(subset=['Resistance (Ω)', 'Length (cm)'])

        if len(valid_data) < 3:
            st.warning("⚠️ Please record at least 3 valid data points (non-zero Current) and click 'Update Graph'.")
        else:
            # Layout: Graph on Left, Interaction on Right
            col_graph, col_interact = st.columns([2, 1])
            
            with col_graph:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.scatter(valid_data['Length (cm)'], valid_data['Resistance (Ω)'], color='blue', label='Experimental Data', zorder=5)
                ax.set_xlabel("Length (cm)")
                ax.set_ylabel("Resistance (Ω)")
                ax.grid(True, linestyle='--', alpha=0.6)
                
                # Plot Best Fit ONLY if verified
                if st.session_state['gradient_verified']:
                    x = valid_data['Length (cm)']
                    y = valid_data['Resistance (Ω)']
                    m, c = np.polyfit(x, y, 1)
                    x_line = np.linspace(x.min(), x.max(), 100)
                    y_line = m * x_line + c
                    ax.plot(x_line, y_line, color='red', linestyle='-', linewidth=2, label=f'Best Fit (m={m:.4f})')
                    ax.legend()
                    ax.set_title(f"Graph: R vs L (Gradient = {m:.4f})")
                else:
                    ax.set_title("Graph: R vs L")

                st.pyplot(fig)

            with col_interact:
                st.markdown("#### Calculate Gradient")
                st.info("Calculate $ \\frac{\Delta y}{\Delta x} $ from the points.")
                
                student_gradient = st.number_input("Your Gradient (Ω/cm):", format="%.4f")
                
                if st.button("Verify Gradient"):
                    st.session_state['gradient_verified'] = True
                    # Rerun to update graph immediately
                    st.rerun() 

                if st.session_state['gradient_verified']:
                    # Recalculate m for the check
                    x = valid_data['Length (cm)']
                    y = valid_data['Resistance (Ω)']
                    m, c = np.polyfit(x, y, 1)
                    
                    if student_gradient != 0:
                        diff = abs((student_gradient - m) / m) * 100
                    else:
                        diff = 100.0

                    if diff < 5:
                        st.balloons()
                        st.success(f"✅ Great! Diff: {diff:.1f}%")
                    elif diff < 10:
                        st.warning(f"⚠️ Close. Diff: {diff:.1f}%")
                    else:
                        st.error(f"❌ Try again. Diff: {diff:.1f}%")

with tab2:
    if check_password():
        st.success("🔓 Teacher Access Granted")
        st.dataframe(st.session_state['student_data'])
