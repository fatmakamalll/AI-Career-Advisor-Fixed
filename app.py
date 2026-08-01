
import os
import streamlit as st

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

from rag_job_recommender import get_recommendation, find_matched_skills, MIN_MATCHED_SKILLS

# ================= PAGE CONFIG =================

st.set_page_config(
    page_title="AI Career Advisor",
    page_icon="💼",
    layout="wide"
)

# ================= STYLE =================
st.markdown("""
<style>
.stApp{
background:#f8fafc;
}
.title{
font-size:45px;
font-weight:800;
color:#2563eb;
}
.card{
background:white;
padding:25px;
border-radius:20px;
box-shadow:0 5px 20px #ddd;
}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown(
"""
<div class="title">
💼 AI Career Advisor
</div>
<p>
Find the best Data & AI career path based on your profile
</p>
""",
unsafe_allow_html=True
)
st.divider()

# ================= USER PROFILE =================
st.subheader("👤 Your Profile")
skills = st.text_input(
    "Technical Skills",
    placeholder="Python, SQL, Power BI, Machine Learning"
)
experience = st.selectbox(
    "Experience Level",
    ["Entry Level", "Junior", "Mid", "Senior", "Lead"]
)
education = st.selectbox(
    "Education",
    ["Bachelor's", "Master's", "PhD", "Bootcamp/Self-taught"]
)
industry = st.selectbox(
    "Preferred Industry",
    ["Any", "Technology", "Finance", "Healthcare", "Automotive"]
)
remote = st.selectbox(
    "Work Preference",
    ["Any", "Fully Remote", "Hybrid", "On-site"]
)
st.write("")

# ================= BUTTON =================
if st.button("🔍 Recommend Career", use_container_width=True):
    if skills.strip() == "":
        st.warning("Please enter your skills")
    elif len(find_matched_skills(skills)) < MIN_MATCHED_SKILLS:
        # Same guard the CLI (step_query) already applies -- previously app.py
        # skipped straight to get_recommendation with no input validation at all.
        st.warning("Please enter at least one recognized technical skill (e.g. Python, SQL, Machine Learning).")
    else:
        profile = f"""
        Skills:
        {skills}

        Experience:
        {experience}

        Education:
        {education}

        Industry:
        {industry}

        Remote:
        {remote}
        """

        with st.spinner("Analyzing your career path..."):
            try:
                result = get_recommendation(profile)
            except Exception as e:
                # step_query() in the CLI already wraps this call in try/except;
                # app.py previously had no equivalent, so a Groq API failure
                # (rate limit, network issue, bad key) would crash the whole
                # page with a raw traceback instead of a readable message.
                st.error(f"Something went wrong while generating your recommendation: {e}")
                result = None

        if result:
            st.divider()
            st.subheader("🎯 Recommended Career")
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(result)
            st.markdown('</div>', unsafe_allow_html=True)
