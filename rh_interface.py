import streamlit as st
import os
import io
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, PyMongoError
import CV_Ranking.rank_cv as rank

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Table, TableStyle
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER

# ─────────────────────────────────────────────
# MongoDB connection
# ─────────────────────────────────────────────

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB_NAME", "cv_database")
COLLECTION_NAME = os.getenv("MONGO_COLLECTION", "cvs")


@st.cache_resource
def get_mongo_collection():
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        return client[DB_NAME][COLLECTION_NAME]
    except ConnectionFailure as e:
        st.error(f"Could not connect to MongoDB: {e}")
        return None


def get_all_cvs():
    collection = get_mongo_collection()
    if collection is None:
        return []
    try:
        return list(collection.find())
    except PyMongoError as e:
        st.error(f"Failed to get CVs from MongoDB: {e}")
        return []


# ─────────────────────────────────────────────
# PDF generation
# ─────────────────────────────────────────────

def _safe(value, fallback="—"):
    """Return a printable string, replacing empty / None with fallback."""
    if value is None:
        return fallback
    s = str(value).strip()
    return s if s else fallback

def normalize_skills(skills):
    cleaned = []
    for s in skills:
        if isinstance(s, dict):
            cleaned.append(s.get("skill") or s.get("name") or str(s))
        else:
            cleaned.append(str(s))
    return cleaned

def generate_cv_pdf(res: dict) -> bytes:
    """
    Build a clean interview-prep PDF for a single candidate and return the
    raw bytes so Streamlit can serve them as a download.
    """
    cv = res["cv"]
    info = cv.get("personal_information", {})
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    # ── Styles ──────────────────────────────────────────────────────────────
    base = getSampleStyleSheet()
    ACCENT = colors.HexColor("#1E3A5F")
    LIGHT  = colors.HexColor("#EEF2F7")

    h1 = ParagraphStyle("h1", parent=base["Heading1"],
                         fontSize=20, textColor=ACCENT, spaceAfter=2)
    h2 = ParagraphStyle("h2", parent=base["Heading2"],
                         fontSize=12, textColor=ACCENT,
                         spaceBefore=10, spaceAfter=4,
                         borderPad=3, backColor=LIGHT,
                         borderColor=ACCENT, borderWidth=0)
    normal = ParagraphStyle("normal", parent=base["Normal"],
                             fontSize=9, leading=14)
    small  = ParagraphStyle("small", parent=base["Normal"],
                             fontSize=8, textColor=colors.HexColor("#555555"),
                             leading=12)
    bold9  = ParagraphStyle("bold9", parent=normal,
                             fontName="Helvetica-Bold")
    center = ParagraphStyle("center", parent=normal, alignment=TA_CENTER)

    story = []

    # ── Header ──────────────────────────────────────────────────────────────
    story.append(Paragraph(_safe(info.get("full_name"), "Unknown Candidate"), h1))
    contacts = " · ".join(filter(None, [
        info.get("email"), info.get("phone"),
        cv.get("website_and_social_links", {}).get("linkedin"),
    ]))
    if contacts:
        story.append(Paragraph(contacts, small))
    story.append(HRFlowable(width="100%", thickness=1.5,
                             color=ACCENT, spaceAfter=6))

    # ── Ranking scores ───────────────────────────────────────────────────────
    story.append(Paragraph("Ranking scores", h2))

    def _pct(v):
        try:
            return f"{float(v) * 100:.1f}%"
        except Exception:
            return _safe(v)

    score_data = [
        ["Metric", "Score"],
        ["Required skills",  _pct(res.get("required_score",  0))],
        ["Preferred skills", _pct(res.get("preferred_score", 0))],
        ["Experience",       _pct(res.get("experience_score", 0))],
        ["Education",        _pct(res.get("education_score", 0))],
        ["Overall",          _pct(res.get("final_score",     0))],
    ]
    score_table = Table(score_data, colWidths=[10 * cm, 6 * cm])
    score_table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), ACCENT),
        ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID",         (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
    ]))
    story.append(score_table)
    story.append(Spacer(1, 6))

    matched = res.get("matched_required_skills", [])
    if matched:
        story.append(Paragraph(
            "<b>Matched required skills:</b> " + ", ".join(map(str, normalize_skills(matched))), small))
    story.append(Spacer(1, 4))

    # ── Professional summary ─────────────────────────────────────────────────
    summary = cv.get("professional_summary", "")
    if summary:
        story.append(Paragraph("Professional summary", h2))
        story.append(Paragraph(summary, normal))

    # ── Education ────────────────────────────────────────────────────────────
    education = cv.get("education", [])
    if education:
        story.append(Paragraph("Education", h2))
        for edu in education:
            degree = _safe(edu.get("degree"))
            field  = edu.get("field_of_study", "")
            school = _safe(edu.get("school"))
            end    = _safe(edu.get("end_year"))
            gpa    = edu.get("gpa")

            title_parts = [degree]
            if field:
                title_parts.append(f"– {field}")
            story.append(Paragraph(" ".join(title_parts), bold9))

            meta_parts = [school, end]
            if gpa:
                meta_parts.append(f"GPA {gpa}")
            story.append(Paragraph("  |  ".join(meta_parts), small))
            story.append(Spacer(1, 3))

    # ── Work experience ──────────────────────────────────────────────────────
    work = cv.get("work_experience", [])
    if work:
        story.append(Paragraph("Work experience", h2))
        for job in work:
            title   = _safe(job.get("job_title"))
            company = _safe(job.get("company"))
            loc     = job.get("location", "")
            start   = _safe(job.get("start_date"), "")
            end     = _safe(job.get("end_date"), "present")
            period  = f"{start} – {end}" if start else end

            story.append(Paragraph(f"<b>{title}</b> · {company}", bold9))
            story.append(Paragraph(
                "  |  ".join(filter(None, [loc, period])), small))

            for resp in job.get("responsibilities", []):
                story.append(Paragraph(f"• {resp}", normal))
            for ach in job.get("achievements", []):
                story.append(Paragraph(f"★ {ach}", normal))
            story.append(Spacer(1, 4))

    # ── Skills ───────────────────────────────────────────────────────────────
    skills_block = cv.get("skills_and_interests", {})
    tech   = skills_block.get("technical_skills", [])
    soft   = skills_block.get("soft_skills", [])
    langs  = skills_block.get("languages", [])

    if tech or soft or langs:
        story.append(Paragraph("Skills", h2))
        if tech:
            story.append(Paragraph(
                "<b>Technical:</b> " + ", ".join(map(str, tech)), normal))
        if soft:
            story.append(Paragraph(
                "<b>Soft skills:</b> " + ", ".join(map(str, soft)), normal))
        if langs:
            story.append(Paragraph(
                "<b>Languages:</b> " + ", ".join(map(str, langs)), normal))
        story.append(Spacer(1, 4))

    # ── Projects ─────────────────────────────────────────────────────────────
    projects = cv.get("projects", [])
    if projects:
        story.append(Paragraph("Projects", h2))
        for proj in projects:
            story.append(Paragraph(f"<b>{_safe(proj.get('name'))}</b>", bold9))
            desc = proj.get("description", "")
            if desc:
                story.append(Paragraph(desc, normal))
            story.append(Spacer(1, 3))

    # ── Certifications ───────────────────────────────────────────────────────
    certs = cv.get("certifications", [])
    if certs:
        story.append(Paragraph("Certifications", h2))
        for cert in certs:
            story.append(Paragraph(f"• {_safe(cert)}", normal))

    # ── Awards ───────────────────────────────────────────────────────────────
    awards = cv.get("awards_and_achievements", [])
    if awards:
        story.append(Paragraph("Awards & achievements", h2))
        for award in awards:
            title = _safe(award.get("title"))
            date  = award.get("date", "")
            line  = f"• {title}" + (f"  ({date})" if date else "")
            story.append(Paragraph(line, normal))

    doc.build(story)
    return buffer.getvalue()


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────

st.title("CV Ranking System")
st.write("Please fill the job description to rank applicants.")

with st.form("job_form"):
    job_title = st.text_input("Job Title")
    job_description = st.text_area("Job Description")
    submitted = st.form_submit_button("Rank CVs")

# ─────────────────────────────────────────────
# Action
# ─────────────────────────────────────────────

if submitted:
    if not job_description.strip():
        st.warning("Please enter a job description.")
    else:
        with st.spinner("Ranking CVs..."):
            try:
                cvs = get_all_cvs()
                if not cvs:
                    st.warning("No CVs found in database: No one applied yet :(")
                    st.session_state.pop("ranking_results", None)
                else:
                    st.session_state["ranking_results"] = rank.rank(job_description, cvs)
                    st.success("Ranking completed!")
            except Exception as e:
                st.error(f"Error: {e}")

# Render results from session state (survives download-button reruns)
if "ranking_results" in st.session_state:
    for i, res in enumerate(st.session_state["ranking_results"], 1):
        info = res["cv"].get("personal_information", {})

        with st.container():
            col_rank, col_info, col_btn = st.columns([0.5, 5, 2])

            with col_rank:
                st.markdown(f"### #{i}")

            with col_info:
                st.write(f"**{res['name']}**")
                st.write(info.get("email", ""))
                st.write(info.get("phone", ""))
                st.write(f"Score: `{res['final_score']}`")

            with col_btn:
                pdf_bytes = generate_cv_pdf(res)
                safe_name = res["name"].replace(" ", "_")
                st.download_button(
                    label="📄 Download CV (PDF)",
                    data=pdf_bytes,
                    file_name=f"{safe_name}_interview_prep.pdf",
                    mime="application/pdf",
                    key=f"dl_{i}",
                )
        st.divider()