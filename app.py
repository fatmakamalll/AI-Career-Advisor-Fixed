import os
import streamlit as st
from PIL import Image
# =====================================================
# PAGE CONFIG (MUST BE FIRST STREAMLIT COMMAND)
# =====================================================
st.set_page_config(
    page_title="AI Career Advisor",
    page_icon="🚀",
    layout="wide"
)
# =====================================================
# GROQ API KEY
# =====================================================
if "GROQ_API_KEY" in st.secrets:

    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
else:
    st.error(
        "Missing GROQ_API_KEY. Add it in Streamlit Secrets."
    )
    st.stop()
# =====================================================
# BACKEND IMPORT
# =====================================================
from rag_job_recommender import (
    get_recommendation,
    find_matched_skills,
    MIN_MATCHED_SKILLS
)
# =====================================================
# CUSTOM CSS
# =====================================================
st.markdown(
"""
<style>
.stApp {

    background:
    linear-gradient(
        135deg,
        #f8fafc,
        #dbeafe
    );
}
.title {

    font-size:55px;
    font-weight:900;
    color:#1d4ed8;
    text-align:center;
}
.subtitle {

    font-size:22px;
    color:#475569;
    text-align:center;
    line-height:1.6;
}
.card {

    background:white;
    padding:30px;
    border-radius:25px;
    box-shadow:
    0 10px 30px rgba(0,0,0,0.08);
    margin-top:20px;
}
.section-title {
    font-size:30px;
    font-weight:800;
    color:#2563eb;
}
div.stButton > button {
    width:100%;
    height:50px;
    border-radius:15px;
    font-size:18px;
    font-weight:bold;
    background:#2563eb;
    color:white;
}
div.stButton > button:hover {

    background:#1e40af;
}
</style>
""",
unsafe_allow_html=True
)
# =====================================================
# HEADER SECTION
# =====================================================
header_left, header_right = st.columns(
    [1, 2]
)
with header_left:
    try:

        image = Image.open(
            "assets/image.png"
        )
        st.image(
            image,
            use_container_width=True
        )
    except FileNotFoundError:
        st.image(r"C:\Users\Admin\Downloads\project احمد يحيي\files (0)\image.png")
with header_right:
    st.markdown(
    """
    <div class="title">
    🚀 AI Career Advisor
    </div>
    <div class="subtitle">
    Personalized AI Career Guidance
    <br>
    Designed for Digilians Track
    <br>
    Applied AI & Data Analytics Graduates
    </div>
    """,
    unsafe_allow_html=True
    )
st.divider()
# =====================================================
# USER PROFILE SIDEBAR
# =====================================================
with st.sidebar:
    st.header(
        "👤 Your Profile"
    )
    skills = st.text_input(
        "Technical Skills",
        placeholder=
        "Python, SQL, Power BI, Machine Learning"
    )
    experience = st.selectbox(
        "Experience Level",
        [
            "Entry Level",
            "Junior",
            "Mid",
            "Senior",
            "Lead"
        ]
    )
    education = st.selectbox(
        "Education",
        [
            "Bachelor's",
            "Master's",
            "PhD",
            "Bootcamp/Self-taught"
        ]
    )
    industry = st.selectbox(
        "Preferred Industry",
        [
            "Any",
            "Technology",
            "Finance",
            "Healthcare",
            "Automotive"
        ]
    )
    remote = st.selectbox(
        "Work Preference",
        [
            "Any",
            "Fully Remote",
            "Hybrid",
            "On-site"
        ]
    )
    st.divider()
    recommend_button = st.button(
        "🔍 Analyze Career Path"
    )
# =====================================================
# CAREER RECOMMENDATION
# =====================================================
if recommend_button:
    if skills.strip() == "":
        st.warning(
            "Please enter your technical skills."
        )
    else:
        matched_skills = find_matched_skills(
            skills
        )
        if len(matched_skills) < MIN_MATCHED_SKILLS:
            st.warning(
            """
            No recognized technical skills found.
            Example:
            Python, SQL, Power BI, Machine Learning
            """
            )
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
            with st.spinner(
                "🤖 AI is analyzing your career profile..."
            ):
                try:
                    result = get_recommendation(
                        profile
                    )
                except Exception as e:
                    st.error(
                        f"Recommendation failed: {e}"
                    )
                    result = None
            if result:
                st.divider()
                st.markdown(
                """
                <div class="section-title">
                🎯 Recommended Career Path
                </div>
                """,
                unsafe_allow_html=True
                )
                st.markdown(
                """
                <div class="card">
                """,
                unsafe_allow_html=True
                )
                st.markdown(
                    result
                )
                st.markdown(
                """
                </div>
                """,
                unsafe_allow_html=True
                )
# =====================================================
# INITIAL SCREEN
# =====================================================
else:
    st.markdown(

    """
    <div class="card">
    ## 👋 Welcome to AI Career Advisor
    This AI system helps Applied AI & Data Analytics graduates:
    ✅ Find suitable AI/Data career roles
    ✅ Match current skills with job requirements
    ✅ Identify missing skills
    ✅ Build a better career direction
    Enter your skills from the sidebar to start.
    </div>
    """,
    unsafe_allow_html=True
    )