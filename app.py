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
    MIN_MATCHED_SKILLS,
    NO_MATCH_MESSAGE,
)
import re

# =====================================================
# RESULT PARSING
# =====================================================
# build_prompt() enforces a fixed "## Heading" markdown structure so this can
# be parsed reliably. If the model ever deviates from that structure (small
# LLMs don't always follow formatting instructions perfectly), parsing simply
# comes back incomplete and the UI falls back to showing the raw markdown --
# it never hides or breaks on an unexpected response.

def parse_recommendation(markdown_text: str) -> dict:
    sections = {}
    parts = re.split(r"\n##\s+", "\n" + markdown_text.strip())
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if "\n" in part:
            heading, body = part.split("\n", 1)
        else:
            heading, body = part, ""
        sections[heading.strip()] = body.strip()
    return sections


def parse_bullet_list(body: str) -> list:
    items = []
    for line in body.splitlines():
        line = line.strip().lstrip("-*").strip()
        if line:
            items.append(line)
    return items


def parse_key_values(body: str) -> dict:
    info = {}
    for line in body.splitlines():
        if ":" in line:
            key, val = line.split(":", 1)
            info[key.strip()] = val.strip() or "—"
    return info


def split_role_and_source(role_text: str):
    """Splits '(Job Title) - based on Source X' into the clean job title and
    the source number. 'Source X' is an internal retrieval-system identifier
    (which of the N retrieved database rows the answer is grounded in) --
    meaningful for verifying the recommendation is grounded and not
    hallucinated, but meaningless jargon to an end user reading the app.
    Regular users see just the job title; the source reference moves into an
    optional, collapsed detail section instead of the headline."""
    match = re.search(r"^(.*?)\s*-\s*based on Source\s*(\d+)\s*$", role_text.strip(), re.IGNORECASE)
    if match:
        return match.group(1).strip(), match.group(2)
    return role_text.strip(), None


def render_skill_pills(body: str, css_class: str, empty_caption: str):
    """Renders parsed skill items as pills -- but only if they actually look
    like short skill labels. Small LLMs don't always follow the "one skill
    per line" instruction and sometimes write one long sentence instead; if
    that sentence gets split by parse_bullet_list it becomes a single
    "skill" that is really a full paragraph, which looks broken as a pill
    (a giant orange box wrapping a sentence). Falling back to plain text in
    that case keeps the UI readable no matter how the model formats it."""
    items = parse_bullet_list(body)
    items = [i for i in items if i.lower() not in ("none", "n/a", "-")]
    if not items:
        st.caption(empty_caption)
        return
    if all(len(item.split()) <= 5 for item in items):
        st.markdown(
            "".join(f'<span class="pill {css_class}">{s}</span>' for s in items),
            unsafe_allow_html=True,
        )
    else:
        # At least one "item" reads like a sentence, not a skill label --
        # show it as normal text instead of forcing it into pill styling.
        st.markdown(" ".join(items))
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
.role-hero {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: white;
    padding: 18px 24px;
    border-radius: 16px;
    font-size: 22px;
    font-weight: 800;
    margin-bottom: 18px;
}
.pill {
    display: inline-block;
    padding: 5px 14px;
    margin: 4px 6px 4px 0;
    border-radius: 999px;
    font-size: 14px;
    font-weight: 600;
}
.pill-green {
    background: #dcfce7;
    color: #166534;
    border: 1px solid #86efac;
}
.pill-orange {
    background: #ffedd5;
    color: #9a3412;
    border: 1px solid #fdba74;
}
.info-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 12px 16px;
    text-align: center;
}
.info-card .label {
    font-size: 12px;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.info-card .value {
    font-size: 16px;
    font-weight: 700;
    color: #1e293b;
    margin-top: 4px;
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
                st.markdown('<div class="card">', unsafe_allow_html=True)

                if result.strip() == NO_MATCH_MESSAGE:
                    # Distinct treatment for "no good match" so it doesn't look
                    # like a normal recommendation card with empty sections.
                    st.info(
                        "😕 No strong match found for this profile in the "
                        "current job database. Try adding more specific "
                        "technical skills."
                    )
                else:
                    sections = parse_recommendation(result)

                    if "Recommended Role" not in sections:
                        # Model didn't follow the expected format -- fall back
                        # to showing exactly what it returned, unmodified.
                        st.markdown(result)
                    else:
                        role_title, source_num = split_role_and_source(sections["Recommended Role"])
                        st.markdown(
                            f'<div class="role-hero">🎯 {role_title}</div>',
                            unsafe_allow_html=True,
                        )

                        if sections.get("Why this matches"):
                            st.markdown("#### 💡 Why this matches")
                            st.markdown(sections["Why this matches"])

                        col_req, col_missing = st.columns(2)
                        with col_req:
                            st.markdown("#### ✅ Required Skills")
                            render_skill_pills(
                                sections.get("Required Skills", ""),
                                "pill-green",
                                "Not specified",
                            )
                        with col_missing:
                            st.markdown("#### ⚠️ Missing Skills")
                            render_skill_pills(
                                sections.get("Missing Skills", ""),
                                "pill-orange",
                                "None — your listed skills cover this role",
                            )

                        if sections.get("Career Information"):
                            st.markdown("#### 📋 Career Information")
                            info = parse_key_values(sections["Career Information"])
                            info_cols = st.columns(4)
                            for col, key in zip(info_cols, ["Experience", "Industry", "Remote", "Salary"]):
                                with col:
                                    st.markdown(
                                        f'<div class="info-card"><div class="label">{key}</div>'
                                        f'<div class="value">{info.get(key, "—")}</div></div>',
                                        unsafe_allow_html=True,
                                    )

                        if source_num:
                            with st.expander("🔍 How was this determined?"):
                                st.caption(
                                    f"This recommendation is grounded in job listing #{source_num} "
                                    "from our database, selected by an automated retrieval algorithm "
                                    "that ranks listings by how well they match your skills and profile "
                                    "(semantic similarity + keyword relevance) -- not generated freely "
                                    "by the AI model."
                                )

                st.markdown('</div>', unsafe_allow_html=True)
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