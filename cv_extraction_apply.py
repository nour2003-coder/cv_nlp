import streamlit as st
import os
from datetime import datetime, timezone
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, PyMongoError
import CV_extracton.extraire_cv as extract_cv

# ─────────────────────────────────────────────
# MongoDB connection
# ─────────────────────────────────────────────

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB_NAME", "cv_database")
COLLECTION_NAME = os.getenv("MONGO_COLLECTION", "cvs")

@st.cache_resource
def get_mongo_collection():
    """Create a cached MongoDB client and return the target collection."""
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")  # verify connection
        return client[DB_NAME][COLLECTION_NAME]
    except ConnectionFailure as e:
        st.error(f"Could not connect to MongoDB: {e}")
        return None

def save_cv_to_mongo(cv: dict) -> bool:
    """Insert a CV document into MongoDB. Returns True on success."""
    collection = get_mongo_collection()
    if collection is None:
        return False
    try:
        cv_doc = {**cv, "uploaded_at": datetime.now(timezone.utc)}
        result = collection.insert_one(cv_doc)
        return result.acknowledged
    except PyMongoError as e:
        st.error(f"Failed to save CV to MongoDB: {e}")
        return False

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────

st.title("Welcome!")
st.write("Upload your CV to apply.")

SAVE_DIR = "uploaded_cvs"
os.makedirs(SAVE_DIR, exist_ok=True)

cv_file = st.file_uploader("Upload CV (PDF)", type=["pdf"])

if cv_file is not None:
    file_path = os.path.join(SAVE_DIR, cv_file.name)

    # Save file to disk
    with open(file_path, "wb") as f:
        f.write(cv_file.read())

    st.success("CV uploaded successfully!")
    st.write("Saved at:", file_path)

    # Extract CV data
    with st.spinner("Extracting CV information..."):
        cv = extract_cv.extract_cv(file_path)

    if cv:
        st.subheader("Extracted Information")
        st.json(cv)

        # Save to MongoDB
        with st.spinner("Saving to database..."):
            saved = save_cv_to_mongo(cv)

        if saved:
            st.success("CV saved to MongoDB successfully!")
        else:
            st.error("Extraction succeeded but saving to MongoDB failed.")
    else:
        st.error("CV extraction failed. Please try again with a different file.")