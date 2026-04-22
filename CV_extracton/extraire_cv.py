from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from uuid import uuid4
from json_repair import repair_json
import json
import re
import requests
import logging

load_dotenv()

model_name = os.getenv("model_name")
CHROMA_PATH = os.getenv("CHROMA_PATH")
api_key2 = os.getenv("api_key2")
api_key = os.getenv("api_key")
api_key3 = os.getenv("api_key3")
API_URL = os.getenv("API_URL")
API_KEY = os.getenv("API_KEY")
API_HOST = os.getenv("API_HOST")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

cv_failed_json = {
    "personal_information": {"full_name": None, "email": None, "phone": None},
    "education": [],
    "professional_summary": None,
    "work_experience": [],
    "certifications": [],
    "awards_and_achievements": [],
    "projects": [],
    "skills_and_interests": {
        "technical_skills": [],
        "soft_skills": [],
        "languages": [],
        "hobbies_and_interests": []
    },
    "volunteering": [],
    "publications": [],
    "website_and_social_links": {
        "linkedin": None,
        "github": None,
        "portfolio": None
    },
    "full_cv_text": None
}


# ─────────────────────────────────────────────
# CORE HELPERS
# ─────────────────────────────────────────────

def clean_text(ch):
    ch = ch.replace('json', "")
    ch = ch.replace('```', "")
    ch = re.sub(r'\s+', ' ', ch)
    ch = ch.replace("None", "null")
    ch = ch.encode('ascii', 'ignore').decode('ascii')
    return ch.strip()


def json_parser(ch, failed_json):
    try:
        # FIX: was parsing original `ch` after cleaning to `cleaned` but then
        # discarding `cleaned` and passing raw `ch` to json.loads
        cleaned = clean_text(ch)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            repaired = repair_json(ch)
            return json.loads(repaired)
        except Exception:
            return failed_json


# ─────────────────────────────────────────────
# EMBEDDINGS & LLM SETUP
# ─────────────────────────────────────────────

def to_embeddings_pdf(DATA_PATH):
    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = Chroma(
        collection_name="cv_collection",
        embedding_function=embeddings_model,
        persist_directory=CHROMA_PATH,
    )
    loader = PyPDFLoader(DATA_PATH)
    raw_documents = loader.load()

    print(f"Loaded {len(raw_documents)} pages")
    print(raw_documents[0].page_content[:500])

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(raw_documents)
    print(f"Total chunks: {len(chunks)}")

    uuids = [str(uuid4()) for _ in range(len(chunks))]
    vector_store.add_documents(documents=chunks, ids=uuids)
    return vector_store, raw_documents


def setup_llm(api_key, model_name, vector_store):
    # FIX: vector_store was used inside but never passed as a parameter —
    # it was referenced from an outer scope that may not exist at call time
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


# ─────────────────────────────────────────────
# EXTRACTION CORE
# ─────────────────────────────────────────────

def extract_info(query, llm_chain, failed_json):
    # FIX: original signature had an unused `vector_store` parameter
    try:
        raw_result = llm_chain.invoke(query)
        js = json_parser(raw_result, failed_json)
        if js == {}:
            return failed_json
        return js
    except Exception as e:
        logger.error(f"LLM failed: {e}.")
        return failed_json


def extract_cv_with_fallback(query, primary_chain, fallback_chain, fallback_chain2, failed_json):
    # FIX 1: original had `vector_store` param that was passed to extract_info
    #         but extract_info never used it
    # FIX 2: final return referenced `failed_json` as `failed_json` (typo: was `failedjson`)
    result = extract_info(query, primary_chain, failed_json)
    if result != failed_json and result != {}:
        logger.info("Primary LLM succeeded.")
        return result

    logger.warning("Primary LLM failed. Trying fallback LLM...")
    if fallback_chain:
        result = extract_info(query, fallback_chain, failed_json)
        if result != failed_json and result != {}:
            logger.info("Fallback LLM succeeded.")
            return result
        logger.warning("Fallback LLM also failed.")

    if fallback_chain2:
        result = extract_info(query, fallback_chain2, failed_json)
        if result != failed_json and result != {}:
            logger.info("Fallback LLM 2 succeeded.")
            return result
        logger.warning("Fallback LLM 2 also failed.")

    return failed_json


# ─────────────────────────────────────────────
# RAPID API PARSER
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# SECTION EXTRACTORS
# ─────────────────────────────────────────────

def extract_personal_info(llm_model, llm_model2, llm_model3, cv_failed_json):
    query = """
    You are an expert extraction algorithm.
    Extract from the candidate's CV: personal_information.
    Return null for any field you cannot find.
    Return strictly in JSON format:
    {
        "personal_information": {
            "full_name": "",
            "email": "",
            "phone": ""
        }
    }
    """
    failed_json = cv_failed_json['personal_information']
    return extract_cv_with_fallback(query, llm_model, llm_model2, llm_model3, failed_json)


def extract_links(llm_model, llm_model2, llm_model3, cv_failed_json):
    query = """
    You are an expert extraction algorithm.
    Extract from the candidate's CV: website_and_social_links.
    Return null for any field you cannot find.
    Return strictly in valid JSON format:
    {
        "website_and_social_links": {
            "linkedin": "",
            "github": "",
            "portfolio": ""
        }
    }
    """
    failed_json = cv_failed_json['website_and_social_links']
    return extract_cv_with_fallback(query, llm_model, llm_model2, llm_model3, failed_json)


def extract_professional_summary(llm_model, llm_model2, llm_model3, cv_failed_json):
    # FIX: original prompt had malformed JSON template (missing opening brace)
    query = """
    You are an expert extraction algorithm.
    Extract from the candidate's CV: professional_summary.
    Return null if not found.
    Return strictly in JSON format:
    {
        "professional_summary": ""
    }
    """
    failed_json = cv_failed_json['professional_summary']
    return extract_cv_with_fallback(query, llm_model, llm_model2, llm_model3, failed_json)


def extract_work_experience(llm_model, llm_model2, llm_model3, cv_failed_json):
    query = """
    You are an expert extraction algorithm.
    Extract from the candidate's CV: work_experience.
    Return null for any field you cannot find.
    Return strictly in JSON format:
    {
        "work_experience": [
            {
                "job_title": "",
                "company": "",
                "location": "",
                "start_date": "",
                "end_date": "",
                "responsibilities": [],
                "achievements": []
            }
        ]
    }
    """
    failed_json = cv_failed_json['work_experience']
    return extract_cv_with_fallback(query, llm_model, llm_model2, llm_model3, failed_json)


def extract_education(llm_model, llm_model2, llm_model3, cv_failed_json):
    query = """
    You are an expert extraction algorithm.
    Extract from the candidate's CV: education.
    Return null for any field you cannot find.
    Return strictly in valid JSON format:
    {
        "education": [
            {
                "degree": "",
                "field_of_study": "",
                "school": "",
                "location": "",
                "start_year": "",
                "end_year": "",
                "gpa": null
            }
        ]
    }
    """
    # FIX: original prompt had Python `None` instead of JSON `null`
    failed_json = cv_failed_json['education']
    return extract_cv_with_fallback(query, llm_model, llm_model2, llm_model3, failed_json)


def extract_certification(llm_model, llm_model2, llm_model3, cv_failed_json):
    query = """
    You are an expert extraction algorithm.
    Extract certifications from the CV.
    If none exist, return exactly: {"certifications": []}
    Otherwise return:
    {
        "certifications": [
            {"name": "", "issuer": "", "issue_date": ""}
        ]
    }
    Do not invent or guess data.
    """
    failed_json = cv_failed_json['certifications']
    return extract_cv_with_fallback(query, llm_model, llm_model2, llm_model3, failed_json)


def extract_awards_and_achievements(llm_model, llm_model2, llm_model3, cv_failed_json):
    query = """
    You are an expert information extraction algorithm.
    Extract the candidate's awards and achievements from their CV.
    Return strictly one JSON object:
    {
        "awards_and_achievements": [
            {"title": "", "date": ""}
        ]
    }
    If none exist, return exactly: {"awards_and_achievements": []}
    Do NOT include any text outside the JSON.
    """
    failed_json = cv_failed_json['awards_and_achievements']
    return extract_cv_with_fallback(query, llm_model, llm_model2, llm_model3, failed_json)


def extract_projects(llm_model, llm_model2, llm_model3, cv_failed_json):
    query = """
    You are an expert extraction algorithm.
    Extract from the candidate's CV: projects.
    Return null for any field you cannot find.
    Return strictly in valid JSON format:
    {
        "projects": [
            {
                "name": "",
                "description": "",
                "link": ""
            }
        ]
    }
    """
    # FIX: original used single-quotes in the JSON template (invalid JSON)
    failed_json = cv_failed_json['projects']
    return extract_cv_with_fallback(query, llm_model, llm_model2, llm_model3, failed_json)


def extract_skills_and_interests(llm_model, llm_model2, llm_model3, cv_failed_json):
    query = """
    You are an expert extraction algorithm.
    Extract only from the "Skills" section of the CV.
    Focus on: technical_skills, soft_skills, languages, hobbies_and_interests.
    In "languages", include only spoken/written languages, not programming languages.
    Return strictly in JSON format:
    {
        "skills_and_interests": {
            "technical_skills": [],
            "soft_skills": [],
            "languages": [
                {"name": "", "proficiency": ""}
            ],
            "hobbies_and_interests": []
        }
    }
    """
    failed_json = cv_failed_json['skills_and_interests']
    return extract_cv_with_fallback(query, llm_model, llm_model2, llm_model3, failed_json)


def extract_volunteering(llm_model, llm_model2, llm_model3, cv_failed_json):
    query = """
    You are an expert extraction algorithm.
    Extract from the candidate's CV: volunteering.
    Return null for any field you cannot find.
    Return strictly in valid JSON format:
    {
        "volunteering": [
            {
                "organization": "",
                "role": "",
                "start_date": "",
                "end_date": "",
                "description": ""
            }
        ]
    }
    """
    failed_json = cv_failed_json['volunteering']
    return extract_cv_with_fallback(query, llm_model, llm_model2, llm_model3, failed_json)


def extract_publications(llm_model, llm_model2, llm_model3, cv_failed_json):
    query = """
    You are an expert extraction algorithm.
    Extract from the candidate's CV: publications.
    Return null for any field you cannot find.
    Return strictly in valid JSON format:
    {
        "publications": [
            {
                "title": "",
                "journal_or_conference": "",
                "publication_date": "",
                "url": "",
                "description": ""
            }
        ]
    }
    """
    failed_json = cv_failed_json['publications']
    return extract_cv_with_fallback(query, llm_model, llm_model2, llm_model3, failed_json)


# ─────────────────────────────────────────────
# CV ASSEMBLY
# ─────────────────────────────────────────────

def safe_extract_section(section_name, section_data, default_value):
    if isinstance(section_data, dict):
        if section_name in section_data:
            return section_data[section_name]
        return section_data
    logger.warning(f"Section '{section_name}' missing or invalid. Using default value.")
    return default_value


def build_cv(sections):
    defaults = {
        "personal_information": {"full_name": None, "email": None, "phone": None},
        "education": [],
        "website_and_social_links": {},
        "professional_summary": None,
        "work_experience": [],
        "certifications": [],
        "awards_and_achievements": [],
        "projects": [],
        "skills_and_interests": {
            "technical_skills": [],
            "soft_skills": [],
            "languages": [],
            "hobbies_and_interests": []
        },
        "volunteering": [],
        "publications": []
    }

    cv = {}
    for key, default in defaults.items():
        cv[key] = safe_extract_section(key, sections.get(key, {}), default)
    return cv


def verification(parsed_data, cv, failed_json):
    # FIX: original had no guard against parsed_data being None,
    # which causes a crash when RapidAPI fails
    if not parsed_data or "data" not in parsed_data:
        logger.warning("RapidAPI data unavailable; skipping verification step.")
        return cv

    if cv["personal_information"] == failed_json['personal_information']:
        cv["personal_information"]["full_name"] = parsed_data["data"].get("name")
        cv["personal_information"]["email"] = parsed_data["data"].get("email")
        cv["personal_information"]["phone"] = parsed_data["data"].get("phone")

    if cv['education'] == failed_json['education']:
        for edu in parsed_data["data"].get("education", []):
            cv['education'].append({
                'degree': edu.get('degree'),
                'field_of_study': edu.get('field_of_study'),
                'school': edu.get('institution'),
                'location': edu.get('location'),
                'start_year': edu.get('start_year'),
                'end_year': edu.get('end_year'),
                'gpa': edu.get('gpa')
            })

    if cv['work_experience'] == failed_json['work_experience']:
        for exp in parsed_data["data"].get("experience", []):
            cv['work_experience'].append({
                'job_title': exp.get('title'),
                'company': exp.get('company'),
                'location': exp.get('location'),
                'start_date': exp.get('start_date'),
                'end_date': exp.get('end_date'),
                'responsibilities': exp.get('description'),
                'achievements': exp.get('achievements', [])
            })

    if cv['skills_and_interests'] == failed_json['skills_and_interests']:
        cv['skills_and_interests']['technical_skills'] = parsed_data["data"].get('skills', [])

    return cv


def clean_vector_store(vector_store):
    try:
        result = vector_store.get()
        all_ids = result.get("ids", []) if result else []
        if all_ids:
            vector_store.delete(ids=all_ids)
            logger.info("Deleted %d embeddings from 'cv_collection'.", len(all_ids))
        else:
            logger.info("Collection already empty.")
    except Exception as e:
        logger.error("Could not clean Chroma DB: %s", e)


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────

def extract_cv(path):
    vector_store, raw_documents = to_embeddings_pdf(path)

    # FIX: setup_llm now receives vector_store as a parameter
    llm_model = setup_llm(api_key, model_name, vector_store)
    llm_model2 = setup_llm(api_key2, model_name, vector_store)
    llm_model3 = setup_llm(api_key3, model_name, vector_store)

    personal_information = extract_personal_info(llm_model, llm_model2, llm_model3, cv_failed_json)
    website_and_social_links = extract_links(llm_model, llm_model2, llm_model3, cv_failed_json)
    professional_summary = extract_professional_summary(llm_model, llm_model2, llm_model3, cv_failed_json)
    work_experience = extract_work_experience(llm_model, llm_model2, llm_model3, cv_failed_json)
    education = extract_education(llm_model, llm_model2, llm_model3, cv_failed_json)
    certification = extract_certification(llm_model, llm_model2, llm_model3, cv_failed_json)
    awards_and_achievements = extract_awards_and_achievements(llm_model, llm_model2, llm_model3, cv_failed_json)
    projects = extract_projects(llm_model, llm_model2, llm_model3, cv_failed_json)
    skills_and_interests = extract_skills_and_interests(llm_model, llm_model2, llm_model3, cv_failed_json)
    volunteering = extract_volunteering(llm_model, llm_model2, llm_model3, cv_failed_json)
    publications = extract_publications(llm_model, llm_model2, llm_model3, cv_failed_json)

    parsed_data = parse_with_rapidapi(path, API_KEY, API_HOST, API_URL)

    sections = {
        "personal_information": personal_information,
        "education": education,
        "website_and_social_links": website_and_social_links,
        "professional_summary": professional_summary,
        "work_experience": work_experience,
        "certifications": certification,
        "awards_and_achievements": awards_and_achievements,
        "projects": projects,
        "skills_and_interests": skills_and_interests,
        "volunteering": volunteering,
        "publications": publications
    }

    cv = build_cv(sections)
    logger.info("CV built successfully")

    cv = verification(parsed_data, cv, cv_failed_json)

    cv['pages'] = len(raw_documents) if isinstance(raw_documents, list) else 0

    # FIX: original only used page 0; concatenate all pages instead
    cv["full_cv_text"] = clean_text(
        " ".join(doc.page_content for doc in raw_documents)
    )

    clean_vector_store(vector_store)

    # FIX: original returned None — should return the built cv
    return cv