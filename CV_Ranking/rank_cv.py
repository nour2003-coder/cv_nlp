from dotenv import load_dotenv
import os
import requests
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from uuid import uuid4
from json_repair import repair_json
import json
import re

load_dotenv()

model_name = os.getenv("model_name")
CHROMA_PATH = os.getenv("CHROMA_PATH")
api_key2 = os.getenv("api_key2")
api_key = os.getenv("api_key")
api_key3 = os.getenv("api_key3")
API_URL = os.getenv("API_URL")
API_KEY = os.getenv("API_KEY")
API_HOST = os.getenv("API_HOST")

failed_job_details = {
    "required_skills": [],
    "preferred_skills": [],
    "min_experience": None,
    "education": ""
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

model = SentenceTransformer("all-MiniLM-L6-v2")


def clean_text(ch):
    ch = ch.replace('json', "")
    ch = ch.replace('```', "")
    ch = re.sub(r'[^\w\s]', '', ch)
    ch = re.sub(r'\s+', ' ', ch)
    ch = ch.encode('ascii', 'ignore').decode('ascii')
    ch = ch.lower()
    ch = re.sub(r'http\S+', '', ch)
    return ch.strip()


def get_cv_text_features(cv):
    skills = cv["skills_and_interests"]["technical_skills"]
    soft_skills = cv["skills_and_interests"].get("soft_skills", [])
    all_skills = skills + soft_skills

    projects = [p["description"] for p in cv.get("projects", [])]

    experience = []
    for w in cv.get("work_experience", []):
        experience.append(" ".join(w.get("responsibilities", [])))

    education = " ".join([
        e.get("field_of_study", "") for e in cv.get("education", [])
    ])

    return {
        "skills": all_skills,
        "projects": " ".join(projects),
        "experience": " ".join(experience),
        "education": education
    }


def semantic_match(text1, text2):
    if not text1 or not text2:
        return 0
    emb1 = model.encode(text1)
    emb2 = model.encode(text2)
    return cosine_similarity([emb1], [emb2])[0][0]


def skill_match_count(cv_skills, jd_skills, threshold=0.75):
    cv_skills = [s.lower() for s in cv_skills]
    jd_skills = [s.lower() for s in jd_skills]
    cv_set = set(cv_skills)

    exact_matches = cv_set & set(jd_skills)
    matched = set(exact_matches)

    cv_embs = model.encode(cv_skills)
    jd_embs = model.encode(jd_skills)

    for i, jd_skill in enumerate(jd_skills):
        best_score = 0
        best_match = None

        for j, cv_skill in enumerate(cv_skills):
            sim = cosine_similarity([cv_embs[j]], [jd_embs[i]])[0][0]
            if sim > best_score:
                best_score = sim
                best_match = cv_skill

        if best_score >= threshold:
            matched.add(best_match)

    return len(matched), len(jd_skills), matched


def experience_score(cv_exp_years, required_exp):
    if required_exp is None or required_exp == 0:
        return 1.0
    if cv_exp_years >= required_exp:
        return 1.0
    return cv_exp_years / required_exp


def education_score(cv_edu, jd_edu):
    return semantic_match(cv_edu, jd_edu)


def extract_cv_experience_years(cv):
    work_experience = cv.get("work_experience", [])
    total_years = 0

    for job in work_experience:
        start_year = extract_year(job.get("start_date"))
        end_year = extract_year(job.get("end_date"))

        if start_year is None:
            continue
        if end_year is None:
            end_year = 2026

        duration = end_year - start_year
        if duration > 0:
            total_years += duration

    return total_years


def rank_cvs(cvs, jd):
    results = []

    for cv in cvs:
        features = get_cv_text_features(cv)

        matched_req, total_req, matched_skills = skill_match_count(
            features["skills"],
            jd["required_skills"]
        )
        req_score = matched_req / total_req if total_req else 0

        matched_pref, total_pref, _ = skill_match_count(
            features["skills"] + features["projects"].split(),
            jd["preferred_skills"]
        )
        pref_score = matched_pref / total_pref if total_pref else 0

        skills_score = 0.7 * req_score + 0.3 * pref_score

        cv_exp = extract_cv_experience_years(cv)
        if len(cv.get("work_experience", [])) > 0 and cv_exp == 0:
            cv_exp = 1

        min_exp = jd.get("min_experience") or 0
        exp_score = experience_score(cv_exp, min_exp)

        edu_score = education_score(features["education"], jd["education"])

        final_score = (
            0.5 * skills_score +
            0.3 * exp_score +
            0.2 * edu_score
        )

        results.append({
            "name": cv["personal_information"]["full_name"],
            "matched_required_skills": list(matched_skills),
            "required_score": round(req_score, 3),
            "preferred_score": round(pref_score, 3),
            "experience_score": round(exp_score, 3),
            "education_score": round(edu_score, 3),
            "final_score": round(final_score, 3),
            "cv": cv
        })

    return sorted(results, key=lambda x: x["final_score"], reverse=True)


def extract_year(date_str):
    if date_str:
        match = re.search(r'(\d{4})', str(date_str))
        if match:
            return int(match.group(1))
    return None


def parse_with_rapidapi(data_path, api_key, api_host, api_url, timeout=30):
    headers = {
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": api_host
    }

    try:
        with open(data_path, "rb") as f:
            files = {"resume": (data_path, f, "application/pdf")}
            response = requests.post(api_url, headers=headers, files=files, timeout=timeout)

        response.raise_for_status()

        try:
            data = response.json()
            logger.info("RapidAPI parsing succeeded.")
            return data
        except json.JSONDecodeError:
            logger.error("Failed to parse JSON response from RapidAPI.")
            return None

    except FileNotFoundError:
        logger.error(f"File not found: {data_path}")
        return None
    except requests.exceptions.Timeout:
        logger.error("RapidAPI request timed out.")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"RapidAPI request failed: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error with RapidAPI: {e}")
        return None


def extract_info(query, llm_chain, failed_json):
    try:
        raw_results = llm_chain.invoke(query)
        js = json_parser(raw_results, failed_json)
        if js == {}:
            return failed_json
        return js
    except Exception as e:
        logger.error(f"LLM failed: {e}.")
        return failed_json


def setup_llm(api_key, model_name, vector_store):
    llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
        max_retries=3,
        request_timeout=60
    )

    prompt = ChatPromptTemplate.from_template("""
    Answer the question based on the following context:

    Context:
    {context}

    Question:
    {question}

    Answer:
    """)

    retriever = vector_store.as_retriever(search_kwargs={"k": 7})

    qa_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return qa_chain


def json_parser(ch, failed_json):
    try:
        return json.loads(ch)
    except json.JSONDecodeError:
        pass

    try:
        repaired = repair_json(ch)
        return json.loads(repaired)
    except Exception:
        pass

    try:
        cleaned = clean_text(ch)
        return json.loads(cleaned)
    except Exception as e:
        logger.warning(f"json_parser: all strategies failed — {e}")
        return failed_json


def extract_cv_with_fallback(query, primary_chain, fallback_chain, fallback_chain2, failedjson):
    result = extract_info(query, primary_chain, failedjson)
    if result not in (failedjson, {}):
        logger.info("Primary LLM succeeded.")
        return result

    logger.warning("Primary LLM failed. Trying fallback LLM...")
    if fallback_chain:
        result = extract_info(query, fallback_chain, failedjson)
        if result not in (failedjson, {}):
            logger.info("Fallback LLM succeeded.")
            return result
        logger.warning("Fallback LLM also failed.")

    if fallback_chain2:
        result = extract_info(query, fallback_chain2, failedjson)
        if result not in (failedjson, {}):
            logger.info("Fallback LLM 2 succeeded.")
            return result
        logger.warning("Fallback LLM 2 also failed.")

    return failedjson


def rank(job_description, cvs):
    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ── KEY FIX: delete the old collection before adding new documents ──────
    # Without this, every call to rank() appends more chunks to the same
    # collection. The retriever then sees all previous job descriptions mixed
    # with the current one, so the LLM extracts stale/blended details and the
    # scores never change between submissions.
    try:
        stale = Chroma(
            collection_name="job_collection",
            embedding_function=embeddings_model,
            persist_directory=CHROMA_PATH,
        )
        stale.delete_collection()
        logger.info("Cleared stale Chroma collection.")
    except Exception as e:
        logger.warning(f"Could not clear Chroma collection: {e}")

    vector_store = Chroma(
        collection_name="job_collection",
        embedding_function=embeddings_model,
        persist_directory=CHROMA_PATH,
    )

    docs = [Document(page_content=job_description)]
    chunks = text_splitter.split_documents(docs)
    uuids = [str(uuid4()) for _ in range(len(chunks))]
    vector_store.add_documents(documents=chunks, ids=uuids)

    llm_model  = setup_llm(api_key,  model_name, vector_store)
    llm_model2 = setup_llm(api_key2, model_name, vector_store)
    llm_model3 = setup_llm(api_key3, model_name, vector_store)

    query = """
You are an expert HR information extraction assistant.

Extract the following fields from this job description:
- required_skills
- preferred_skills
- min_experience (integer in years)
- education

Return ONLY valid JSON in this format:
{
 "required_skills": [],
 "preferred_skills": [],
 "min_experience": null,
 "education": ""
}

Job Description:

"""

    job_description_details = extract_cv_with_fallback(
        query, llm_model, llm_model2, llm_model3, failed_job_details
    )

    logger.info(f"Extracted JD details: {job_description_details}")

    ranking = rank_cvs(cvs, job_description_details)
    return ranking