import os
# 🔥 suppress torch / gpu warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
os.environ["KMP_WARNINGS"] = "0"

import easyocr
import re
import cv2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob

# silent OCR
reader = easyocr.Reader(['en'], verbose=False)


# ---------------- OCR ----------------
def extract_text(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return ""

    # improve OCR quality
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ✅ SAFE (no join error)
    result = reader.readtext(thresh, detail=0)

    text = " ".join(result)

    # 🔥 fix merging like "1. ... 2."
    text = re.sub(r'(\d+\.)', r' \1 ', text)

    return text


# ---------------- CLEAN ----------------
def clean_text(text):
    text = text.lower()
    text = str(TextBlob(text).correct())
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ---------------- SIMILARITY ----------------
def similarity_score(t, s):
    vec = TfidfVectorizer()
    vectors = vec.fit_transform([t, s])
    return cosine_similarity(vectors[0], vectors[1])[0][0]


# ---------------- KEYWORD ----------------
def keyword_score(t, s):
    t_words = set(t.split())
    s_words = set(s.split())
    return len(t_words & s_words) / max(len(t_words), 1)


# ---------------- EVALUATE ONE ----------------
def evaluate_answer(teacher, student):
    if not student or len(student.strip()) < 3:
        return {"marks": 0, "feedback": "No answer"}

    sim = similarity_score(teacher, student)
    keyword = keyword_score(teacher, student)

    quality = (0.7 * sim) + (0.3 * keyword)

    words = len(student.split())

    if words < 10:
        base_min, base_max = 1, 3
    elif words < 20:
        base_min, base_max = 3, 5
    elif words < 40:
        base_min, base_max = 5, 7
    else:
        base_min, base_max = 8, 10

    marks = base_min + (base_max - base_min) * quality

    if sim < 0.3:
        marks = base_min

    if quality > 0.8:
        feedback = "Excellent answer"
    elif quality > 0.6:
        feedback = "Good answer"
    elif quality > 0.4:
        feedback = "Average answer"
    else:
        feedback = "Poor answer"

    return {
        "marks": round(min(marks, 10), 2),
        "feedback": feedback
    }


# ---------------- PARSE TEACHER ----------------
def parse_teacher_answers(text):
    qa = {}
    text = text.replace("\r", "")

    blocks = re.split(r'(Q\d+\.)', text)

    for i in range(1, len(blocks), 2):
        q_label = blocks[i]
        content = blocks[i + 1]

        q_num = re.findall(r'\d+', q_label)[0]
        q_key = f"Q{q_num}"

        match = re.search(r'A\d*[:\.]?\s*(.*)', content, re.DOTALL)

        if match:
            answer = match.group(1)
            answer = re.split(r'Q\d+\.', answer)[0]
            answer = re.sub(r'\n+', ' ', answer).strip()
            qa[q_key] = answer

    return qa


# ---------------- PARSE STUDENT ----------------
def split_student_answers(text, total_questions):
    answers = {}

    text = text.lower()

    # 🔥 smart split even if OCR merges
    splits = re.split(r'(?=\b\d+\.)', text)

    for part in splits:
        part = part.strip()
        match = re.match(r'(\d+)\.', part)

        if match:
            q_num = match.group(1)
            content = re.sub(r'^\d+\.\s*', '', part)
            answers[f"Q{q_num}"] = content.strip()

    # fallback
    if not answers and total_questions == 1:
        return {"Q1": text.strip()}

    return answers


# ---------------- EVALUATE ALL ----------------
def evaluate_all(teacher_dict, student_dict):
    results = {}
    total_marks = 0
    max_total = 0
    correct = 0

    for q in teacher_dict:
        t = clean_text(teacher_dict[q])
        s = clean_text(student_dict.get(q, ""))

        res = evaluate_answer(t, s)

        results[q] = res
        total_marks += res["marks"]
        max_total += 10

        if res["marks"] >= 7:
            correct += 1

    percent = (total_marks / max_total) * 100 if max_total else 0

    if percent >= 80:
        fb = "Excellent performance"
    elif percent >= 60:
        fb = "Good performance"
    elif percent >= 40:
        fb = "Average performance"
    else:
        fb = "Poor performance"

    return {
        "results": results,
        "total_marks": round(total_marks, 2),
        "max_marks": max_total,
        "percentage": round(percent, 2),
        "correct_answers": correct,
        "total_questions": len(teacher_dict),
        "feedback": fb
    }