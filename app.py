"""Streamlit web app: streamlit run app.py"""
import streamlit as st

from src.predict import analyse_student

st.set_page_config(page_title="Student Performance AI", page_icon="🎓", layout="wide")
st.title("🎓 Student Performance Prediction & Personalised Learning")
st.caption("Enter a student's details to predict their final score and get a tailored learning plan.")

with st.sidebar:
    st.header("Student profile")
    gender = st.selectbox("Gender", ["Female", "Male"])
    study_hours = st.slider("Study hours / day", 0.0, 10.0, 3.0, 0.5)
    attendance = st.slider("Attendance (%)", 30, 100, 80)
    sleep_hours = st.slider("Sleep hours", 3.0, 10.0, 7.0, 0.5)
    participation = st.slider("Participation (0-10)", 0.0, 10.0, 5.0, 0.5)
    internet = st.selectbox("Internet access", ["Yes", "No"])
    extra = st.selectbox("Extra-curricular activities", ["Yes", "No"])
    support = st.selectbox("Parental support", ["Low", "Medium", "High"], index=1)

st.subheader("Academic scores")
c1, c2, c3 = st.columns(3)
previous = c1.slider("Previous exam score", 0, 100, 65)
assignment = c2.slider("Assignment score", 0, 100, 65)
quiz = c3.slider("Quiz score", 0, 100, 65)

st.subheader("Topic-wise mastery")
cols = st.columns(5)
topics = {}
for col, name in zip(cols, ["mathematics", "programming", "data_structures", "databases", "statistics"]):
    topics[f"topic_{name}"] = col.slider(name.replace("_", " ").title(), 0, 100, 60, key=name)

if st.button("Analyse student", type="primary"):
    student = {
        "gender": gender, "study_hours": study_hours, "attendance": attendance,
        "previous_score": previous, "assignment_score": assignment, "quiz_score": quiz,
        "participation": participation, "sleep_hours": sleep_hours,
        "internet_access": internet, "extra_activities": extra,
        "parental_support": support, **topics,
    }
    r = analyse_student(student)
    m1, m2 = st.columns(2)
    m1.metric("Predicted final score", r["predicted_score"])
    m2.metric("Performance level", r["predicted_level"])
    st.bar_chart(r["level_probabilities"])

    st.subheader("Weak topics")
    if r["weak_topics"]:
        for t, s in r["weak_topics"]:
            st.write(f"- **{t.replace('_', ' ').title()}** - {s:.0f}/100")
    else:
        st.success("No weak topics - great work!")

    st.subheader("Habit tips")
    for tip in r["tips"]:
        st.write(f"- {tip}")

    st.subheader("Recommended resources")
    st.dataframe(r["resources"][["title", "topic", "difficulty", "type", "url", "reason"]],
                 use_container_width=True, hide_index=True)

    st.subheader("Weekly study plan")
    for line in r["study_plan"]:
        st.write(f"- {line}")
