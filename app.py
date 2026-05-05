import streamlit as st
import os
import uuid
import warnings

warnings.filterwarnings("ignore")

from utils import extract_text, parse_teacher_answers, split_student_answers, evaluate_all

st.set_page_config(page_title="AI Answer Evaluator")

st.title("🧠 AI Answer Evaluator")

teacher_file = st.file_uploader("📘 Upload Teacher File (.txt)", type=["txt"])
student_file = st.file_uploader("📝 Upload Student Image", type=["png", "jpg", "jpeg"])

if st.button("Evaluate"):

    if teacher_file and student_file:

        os.makedirs("uploads", exist_ok=True)

        # save image
        s_path = os.path.join("uploads", str(uuid.uuid4()) + ".png")
        with open(s_path, "wb") as f:
            f.write(student_file.getbuffer())

        st.image(s_path, caption="Student Image")

        teacher_text = teacher_file.read().decode("utf-8")

        student_text = extract_text(s_path)

        st.subheader("🔍 OCR Output")
        st.text_area("Extracted Text", student_text, height=150)

        teacher_dict = parse_teacher_answers(teacher_text)

        student_dict = split_student_answers(student_text, len(teacher_dict))

        if not student_dict:
            student_dict = {"Q1": student_text}

        # debug
        st.write("Parsed Student:", student_dict)

        result = evaluate_all(teacher_dict, student_dict)

        st.subheader("📊 Results")

        for q, res in result["results"].items():
            st.write(f"{q} → {res['marks']} marks ({res['feedback']})")

        st.success(f"Total: {result['total_marks']} / {result['max_marks']}")
        st.info(f"Percentage: {result['percentage']}%")
        st.success(result["feedback"])

    else:
        st.warning("Upload both files")