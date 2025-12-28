# PanPhy Skill Builder

PanPhy Skill Builder is a Streamlit web app for building, delivering, and marking physics exam-style questions. It supports two question sources:

- **AI generated questions** (Markdown + LaTeX formatted plain text)
- **Teacher uploaded scans** (question image + mark scheme image)

Students can answer using a **drawing canvas** and typed responses, then receive **AI-assisted marking** and feedback. Teachers can curate a question bank and review attempts in a dashboard.

---

## Key features

### Question bank (Teacher)
- Browse, filter, and preview questions (AI generated and teacher uploaded)
- AI question generation with editable output
- Upload scanned question images and mark scheme images
- Store question text as **plain text** that renders as **Markdown + LaTeX**

### Student experience
- Select an assignment and question
- Answer using:
  - a **canvas** for working (diagrams, calculations)
  - optional typed response
- Submit for marking and feedback

### Data + storage
- **Postgres (Supabase)** for structured data (question bank, attempts)
- **Supabase Storage** for scanned images (question + mark scheme)
- App guards against common Streamlit issues (state mismatch in dropdowns, missing storage objects)

---

## Tech stack

- **Streamlit** UI
- **OpenAI API** for question generation and marking
- **Supabase Postgres** for data persistence
- **Supabase Storage** for images
- **SQLAlchemy + psycopg** for database access
- **Pillow** for image processing
- **streamlit-drawable-canvas** for student working

---

## Project structure

This repo is intentionally lightweight.

- `app.py` (single-file Streamlit app)
- `requirements.txt`
- `README.md` (this file)

---


```bash
pip install -r requirements.txt
