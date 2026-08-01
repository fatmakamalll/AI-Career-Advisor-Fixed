"""
RAG Job Title Recommendation System - Applied AI & Data Analytics
====================================================================

Pipeline:
  1) clean_prepare - Clean merged_jobs.csv and build 'document' column
                      -> jobs_with_documents.csv
  2) ingest         - Embed documents (sentence-transformers, local) and store in ChromaDB
  3) query          - Interactive RAG app: retrieve similar jobs + generate recommendation (Groq LLM)

Usage:
    python rag_job_recommender.py clean_prepare
    python rag_job_recommender.py ingest
    python rag_job_recommender.py query
    python rag_job_recommender.py all

Requirements:
    pip install pandas chromadb groq sentence-transformers tqdm

Groq API key: set env var GROQ_API_KEY, or (inside Streamlit) st.secrets["GROQ_API_KEY"].
"""

import argparse
import json
import os
import re
import sys

import pandas as pd

# ---------------------------------------------------------------------------
# Shared configuration
# ---------------------------------------------------------------------------

RAW_CSV = "merged_jobs.csv"
DOCS_CSV = "jobs_with_documents.csv"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "job_titles"

EMBED_MODEL = "all-MiniLM-L6-v2"     # local sentence-transformers model (no API key needed)
CHAT_MODEL = "llama-3.1-8b-instant"  # valid Groq model id, used for generation only
TOP_K = 5                            # number of closest jobs to use as context

SKILLS_VOCAB_PATH = "skills_vocab.json"
MIN_MATCHED_SKILLS = 1
NO_MATCH_MESSAGE = "No Job Title Recommended"

# ---------------------------------------------------------------------------
# Retrieval-quality configuration (relevance threshold, dedup, lexical boost)
# ---------------------------------------------------------------------------
# Chroma collection now uses cosine distance (see step_ingest). Cosine distance
# ranges from 0 (identical direction) to 2 (opposite direction); 1.0 is a
# reasonable "not related at all" cutoff for MiniLM sentence embeddings on
# short job-description texts. Tune this against your own data if needed.
DISTANCE_THRESHOLD = 1.0

# Relative floor: a candidate is kept only if its similarity is at least this
# fraction of the *best* candidate's similarity in the same query. This is
# the missing half of Lab 8's build_context_package() filtering
# (min_absolute_score + min_score_ratio=0.40): an absolute floor alone still
# lets through a pool of uniformly mediocre matches when nothing strong
# exists; the relative floor catches that case.
MIN_SCORE_RATIO = 0.40

# We ask Chroma for more neighbors than we actually need, so that after
# dropping irrelevant hits and de-duplicating repeated job titles we still
# have TOP_K genuinely distinct, relevant jobs left to show the LLM.
CANDIDATE_POOL_MULTIPLIER = 4

# Real hybrid retrieval weight: final ranking score = HYBRID_ALPHA * (min-max
# normalized embedding similarity) + (1 - HYBRID_ALPHA) * (min-max normalized
# BM25 score). 0.6 matches the alpha used for the same embedding/BM25 hybrid
# in Lab 7. This replaces the earlier "+0.15 per exact skill match" heuristic,
# which was not real BM25 (no term frequency, no corpus-wide IDF, no per-system
# normalization, no weighted merge formula).
HYBRID_ALPHA = 0.6

REQUIRED_COLUMNS = [
    "job_id",
    "job_title",
    "job_category",
    "experience_level",
    "education_required",
    "required_skills",
    "industry",
    "remote_work",
    "company_size",
    "salary_tier",
    "source_dataset",
]


# ---------------------------------------------------------------------------
# Step 1: Clean + Prepare
# ---------------------------------------------------------------------------

MANUAL_TITLE_MAP = {
    "Teamleitung Automatisierung": "Automation Team Lead",
    "Datawarehouse/DWH-Entwickler (w/m/d)": "Data Warehouse Developer",
    "HR Automation/RPA Analyst (m/f/d) - HR Digital Transformation": "HR Automation / RPA Analyst",
}

GENDER_TAG_RE = re.compile(r"\(?\s*[mwfdMWFD]\s*/\s*[mwfdMWFD](\s*/\s*[mwfdMWFD])?\s*\)?")
TRAILING_NUMBER_RE = re.compile(r"\s+\d+$")
TRAILING_JUNIOR_RE = re.compile(r"\s+Junior$", re.IGNORECASE)
MULTI_SPACE_RE = re.compile(r"\s{2,}")


def clean_job_title(title: str) -> str:
    title = str(title).strip()

    if title in MANUAL_TITLE_MAP:
        return MANUAL_TITLE_MAP[title]

    title = GENDER_TAG_RE.sub("", title)
    title = title.replace("Sr.", "Senior")
    title = title.replace("Jr.", "Junior")
    title = title.replace("Machine Learning", "ML")
    title = TRAILING_JUNIOR_RE.sub("", title)
    title = TRAILING_NUMBER_RE.sub("", title)
    title = MULTI_SPACE_RE.sub(" ", title).strip()
    title = title.rstrip("-").strip()

    return title


SKILL_SPLIT_RE = re.compile(r"[|,]")


def clean_skills(skills: str) -> str:
    """Splits on '|' OR ',' (not one-or-the-other). The previous version chose
    a single delimiter for the whole string based on whether '|' appeared at
    all, so a row mixing both separators (e.g. "Python, SQL|Excel") would not
    be split on the comma. Splitting on the character class handles both in
    one pass regardless of which one(s) a given row actually uses."""
    raw = str(skills)
    parts = SKILL_SPLIT_RE.split(raw)

    seen = set()
    deduped = []
    for part in parts:
        skill = part.strip()
        if skill and skill not in seen:
            seen.add(skill)
            deduped.append(skill)

    return ", ".join(deduped)


def build_document(row: pd.Series) -> str:
    return (
        f"Job Title: {row['job_title']}. "
        f"Category: {row['job_category']}. "
        f"Experience Level: {row['experience_level']}. "
        f"Education Required: {row['education_required']}. "
        f"Required Skills: {row['required_skills']}. "
        f"Industry: {row['industry']}. "
        f"Remote Work: {row['remote_work']}. "
        f"Company Size: {row['company_size']}. "
        f"Salary Tier: {row['salary_tier']}. "
        f"Source: {row['source_dataset']}."
    )


def step_clean_prepare():
    if not os.path.exists(RAW_CSV):
        sys.exit(f"[clean_prepare] Raw input file not found: {RAW_CSV}")

    df = pd.read_csv(RAW_CSV)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        sys.exit(f"[clean_prepare] CSV is missing required columns: {missing}")

    before_rows = len(df)

    df = df.drop_duplicates()
    df = df.dropna(subset=["job_id"])

    text_cols = df.select_dtypes(include=["object", "string"]).columns
    df[text_cols] = df[text_cols].apply(lambda col: col.str.strip())

    df["job_title"] = df["job_title"].apply(clean_job_title)
    df["required_skills"] = df["required_skills"].apply(clean_skills)

    df["document"] = df.apply(build_document, axis=1)

    df.to_csv(DOCS_CSV, index=False)

    print(f"[clean_prepare] Started with {before_rows} rows, kept {len(df)} rows")
    print(f"[clean_prepare] Documents created: {len(df)}")
    print(f"[clean_prepare] Saved to: {DOCS_CSV}")
    print("\n[clean_prepare] Example:")
    print(df["document"].iloc[0])


# ---------------------------------------------------------------------------
# Shared helpers: embeddings + Groq key resolution
# ---------------------------------------------------------------------------

_EMBEDDER = None


def get_embedder():
    """Loads the sentence-transformers model once and caches it."""
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer(EMBED_MODEL)
    return _EMBEDDER


def embed_texts(texts: list) -> list:
    """Returns a list of embedding vectors (as plain lists) for the given texts.

    normalize_embeddings=True makes every vector unit-length, so cosine
    similarity becomes equivalent to a plain dot product. This must match the
    "hnsw:space": "cosine" setting used when the Chroma collection is created
    in step_ingest() -- otherwise the stored vectors and the collection's
    distance metric silently disagree with each other.
    """
    embedder = get_embedder()
    return embedder.encode(
        list(texts), show_progress_bar=False, normalize_embeddings=True
    ).tolist()


def get_groq_api_key() -> str:
    """Reads the Groq API key from the environment first (works for CLI use),
    falling back to st.secrets when running inside Streamlit."""
    key = os.getenv("GROQ_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        sys.exit(
            "[error] GROQ_API_KEY not found. Set it as an environment variable "
            "or in Streamlit secrets."
        )


# ---------------------------------------------------------------------------
# Step 2: Ingest (embed + store in ChromaDB)
# ---------------------------------------------------------------------------

def build_skills_vocab(df: pd.DataFrame) -> list:
    vocab = set()
    for skills in df["required_skills"].dropna():
        for skill in str(skills).split(","):
            skill = skill.strip()
            if skill:
                vocab.add(skill)
    return sorted(vocab)


def step_ingest():
    import chromadb
    from tqdm import tqdm

    if not os.path.exists(DOCS_CSV):
        sys.exit(f"[ingest] {DOCS_CSV} not found. Run 'clean_prepare' first.")

    df = pd.read_csv(DOCS_CSV)

    vocab = build_skills_vocab(df)
    with open(SKILLS_VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"[ingest] Saved {len(vocab)} known skills to {SKILLS_VOCAB_PATH}")

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    # hnsw:space="cosine" makes Chroma's distance metric match the normalized
    # embeddings produced by embed_texts(). Without this, Chroma silently
    # falls back to L2 (Euclidean) distance, which is not what the vectors
    # were normalized for.
    collection = client.create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    print(f"[ingest] Converting {len(df)} jobs to embeddings (sentence-transformers)...")

    batch_size = 50
    for start in tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[start:start + batch_size]

        embeddings = embed_texts(batch["document"].tolist())

        collection.add(
            ids=[str(i) for i in batch["job_id"]],
            embeddings=embeddings,
            documents=batch["document"].tolist(),
            metadatas=[
                {
                    "job_title": str(row["job_title"]),
                    "job_category": str(row["job_category"]),
                    "required_skills": str(row["required_skills"]),
                    "experience_level": str(row["experience_level"]),
                    "education_required": str(row["education_required"]),
                    "industry": str(row["industry"]),
                    "remote_work": str(row["remote_work"]),
                    "company_size": str(row["company_size"]),
                    "salary_tier": str(row["salary_tier"]),
                }
                for _, row in batch.iterrows()
            ],
        )
    print(f"\n[ingest] Done. Stored {collection.count()} jobs in the database")
    print(f"[ingest] Database saved at: {CHROMA_PATH}")


# ---------------------------------------------------------------------------
# Step 3: Query app (retrieval + generation)
# ---------------------------------------------------------------------------

_SKILLS_VOCAB_CACHE = None


def load_skills_vocab() -> list:
    global _SKILLS_VOCAB_CACHE
    if _SKILLS_VOCAB_CACHE is not None:
        return _SKILLS_VOCAB_CACHE

    if not os.path.exists(SKILLS_VOCAB_PATH):
        sys.exit(f"[query] {SKILLS_VOCAB_PATH} not found. Run 'ingest' first.")

    with open(SKILLS_VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    vocab.sort(key=len, reverse=True)
    _SKILLS_VOCAB_CACHE = [
        (skill, re.compile(r"(?<!\w)" + re.escape(skill.lower()) + r"(?!\w)"))
        for skill in vocab
    ]
    return _SKILLS_VOCAB_CACHE


def find_matched_skills(user_input: str) -> list:
    text = user_input.lower()
    matched = [skill for skill, pattern in load_skills_vocab() if pattern.search(text)]
    return matched


_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list:
    """Simple whole-word tokenizer used for BM25 (lowercase, alphanumeric
    tokens only). BM25 needs tokenized text, not raw strings."""
    return _WORD_RE.findall(str(text).lower())


_BM25_INDEX = None
_BM25_JOB_IDS = None


def get_bm25_index():
    """Builds (once, cached for the process) a BM25Okapi index over the ENTIRE
    job corpus (1,532 documents) -- not just a retrieved candidate pool.

    This is the part the earlier version was missing: real BM25 relevance
    (term frequency weighted by inverse document frequency) only means
    something when document frequency is computed across the whole corpus.
    A skill that appears in 3 out of 1,532 postings should score very
    differently from one that appears in 900 of them -- that signal collapses
    to nothing if BM25 is computed only over a pre-filtered pool of ~20
    candidates.
    """
    global _BM25_INDEX, _BM25_JOB_IDS
    if _BM25_INDEX is not None:
        return _BM25_INDEX, _BM25_JOB_IDS

    if not os.path.exists(DOCS_CSV):
        raise Exception(f"{DOCS_CSV} not found. Run 'clean_prepare' first.")

    from rank_bm25 import BM25Okapi

    df = pd.read_csv(DOCS_CSV)
    corpus_tokens = [_tokenize(doc) for doc in df["document"].tolist()]
    _BM25_JOB_IDS = [str(jid) for jid in df["job_id"].tolist()]
    _BM25_INDEX = BM25Okapi(corpus_tokens)
    return _BM25_INDEX, _BM25_JOB_IDS


def _min_max_normalize(values: dict) -> dict:
    """Min-max normalizes a dict of {key: raw_score} into {key: score in [0,1]}.
    Embedding similarity and BM25 scores live on completely different scales
    (cosine similarity in [-1,1] vs. unbounded BM25 scores), so they must each
    be normalized independently, per query, before they can be combined with
    a single weighted formula."""
    if not values:
        return {}
    lo, hi = min(values.values()), max(values.values())
    if hi - lo < 1e-9:
        return {k: 0.0 for k in values}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


def retrieve_similar_jobs(user_profile: str, n_results: int = TOP_K) -> list:
    """Retrieves relevant, distinct jobs for a user profile.

    Returns a list of dicts: {"document", "metadata", "distance"}, already
    filtered by a relevance threshold, re-ranked with a lexical boost, and
    de-duplicated by job_title. Returns an empty list when nothing in the
    database is a good enough match -- callers must handle that case instead
    of assuming Chroma always returns usable neighbors.
    """
    import chromadb

    if not os.path.isdir(CHROMA_PATH):
        raise Exception("Chroma database not found. Run ingest first.")

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    query_embedding = embed_texts([user_profile])[0]

    # Ask for a larger candidate pool than we need, since some candidates will
    # be dropped for being irrelevant (distance) or redundant (duplicate title).
    pool_size = min(n_results * CANDIDATE_POOL_MULTIPLIER, collection.count())
    if pool_size == 0:
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=pool_size,
        include=["documents", "metadatas", "distances"],
    )

    docs = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    ids = results["ids"][0]

    # Pass 1: absolute floor. Drop anything past DISTANCE_THRESHOLD outright.
    # This filter runs on the RAW embedding distance, before any BM25/lexical
    # signal is mixed in -- a job that is semantically unrelated must not be
    # able to "buy" its way into the result set just because it shares an
    # exact keyword with the user profile. BM25 only re-ranks *within* the
    # survivors of this stage, it never overrides it.
    absolute_survivors = [
        (doc, meta, distance, job_id)
        for doc, meta, distance, job_id in zip(docs, metadatas, distances, ids)
        if distance <= DISTANCE_THRESHOLD
    ]
    if not absolute_survivors:
        return []

    # Pass 2: relative floor (the previously-missing half of Lab 8's
    # dual-threshold design). Even among candidates that pass the absolute
    # floor, if the *best* candidate is still only a mediocre match, weaker
    # candidates in that same pool shouldn't be treated as equally credible.
    # cosine distance -> similarity: similarity = 1 - distance.
    best_distance = min(d for _, _, d, _ in absolute_survivors)
    best_similarity = 1 - best_distance

    filtered = []
    for doc, meta, distance, job_id in absolute_survivors:
        similarity = 1 - distance
        if best_similarity <= 0 or similarity < MIN_SCORE_RATIO * best_similarity:
            continue
        filtered.append({
            "document": doc,
            "metadata": meta,
            "distance": distance,
            "similarity": similarity,
            "job_id": job_id,
        })
    if not filtered:
        return []

    # Real hybrid re-ranking: BM25 computed over the FULL corpus (not just
    # this pool), min-max normalized independently from the embedding
    # similarity, then merged with a weighted formula -- this is the part
    # that was previously a rough "+0.15 per matched skill" approximation.
    bm25_index, bm25_job_ids = get_bm25_index()
    query_tokens = _tokenize(user_profile)
    all_bm25_scores = bm25_index.get_scores(query_tokens)
    bm25_lookup = dict(zip(bm25_job_ids, all_bm25_scores))

    sim_raw = {i: c["similarity"] for i, c in enumerate(filtered)}
    bm25_raw = {i: bm25_lookup.get(c["job_id"], 0.0) for i, c in enumerate(filtered)}
    sim_norm = _min_max_normalize(sim_raw)
    bm25_norm = _min_max_normalize(bm25_raw)

    for i, c in enumerate(filtered):
        c["hybrid_score"] = (
            HYBRID_ALPHA * sim_norm[i] + (1 - HYBRID_ALPHA) * bm25_norm[i]
        )

    filtered.sort(key=lambda c: c["hybrid_score"], reverse=True)

    # De-duplicate by job_title: with most titles repeated in the dataset,
    # naive top-k can return the same title 3-4 times. Keep only the first
    # (best-ranked, post-hybrid) occurrence of each title.
    seen_titles = set()
    deduped = []
    for c in filtered:
        title = c["metadata"].get("job_title", "")
        if title in seen_titles:
            continue
        seen_titles.add(title)
        deduped.append(c)
        if len(deduped) >= n_results:
            break

    return deduped


def build_prompt(user_profile: str, retrieved_docs: list, metadatas: list) -> str:
    """Builds the LLM prompt with numbered, cited context and explicit
    grounding/refusal rules so the model cannot hallucinate a confident
    recommendation when the retrieved evidence is weak."""
    context = ""
    for i, (doc, meta) in enumerate(zip(retrieved_docs, metadatas), start=1):
        # Inject the already-known structured fields directly instead of
        # asking the LLM to re-extract them from prose -- cheaper and more
        # reliable than trusting free-text reconstruction.
        context += (
            f"\n[Source {i}]\n"
            f"{doc}\n"
            f"(job_title={meta.get('job_title','')}; "
            f"experience_level={meta.get('experience_level','')}; "
            f"industry={meta.get('industry','')}; "
            f"remote_work={meta.get('remote_work','')}; "
            f"salary_tier={meta.get('salary_tier','')})\n"
        )

    prompt = f"""You are an AI Career Advisor specialized in Data and AI careers.

User Profile:
{user_profile}

Retrieved Career Options (numbered sources):
{context}

Grounding rules (follow strictly):
- Use ONLY the retrieved sources above as evidence. Do not invent job titles,
  skills, or facts that are not present in the sources.
- If none of the sources are a reasonably good match for the user profile,
  respond with exactly: "{NO_MATCH_MESSAGE}" and nothing else.
- When you recommend a role, cite which source number(s) it came from.
- Do not overstate confidence: if the match is partial, say so explicitly in
  "Why this matches" instead of presenting it as a perfect fit.

Task:
Recommend the single best job title for this user, grounded in the sources.
Return the answer in this exact format:
## Recommended Role
(Job Title) - based on Source X

## Why this matches
- Skill match
- Experience match
- Domain match

## Required Skills
(list skills, from the source)

## Missing Skills
(skills the user profile does not mention)

## Career Information
Experience:
Industry:
Remote:
Salary:

Keep the answer concise and professional.
"""
    return prompt


def get_recommendation(user_profile: str) -> str:
    from groq import Groq

    candidates = retrieve_similar_jobs(user_profile)

    # (c) Real relevance floor: if nothing survived the distance threshold,
    # don't call the LLM with empty/irrelevant context at all.
    if not candidates:
        return NO_MATCH_MESSAGE

    docs = [c["document"] for c in candidates]
    metadatas = [c["metadata"] for c in candidates]

    prompt = build_prompt(user_profile, docs, metadatas)

    client = Groq(api_key=get_groq_api_key())
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def step_query():
    print("=" * 60)
    print("Job Title Recommendation System - Applied AI & Data Analytics")
    print("=" * 60)
    print("Enter your skills (example: SQL advanced, Python intermediate, n8n, Excel, dashboards, analytics)")
    print("Type 'exit' to quit\n")

    while True:
        try:
            user_input = input("Your skills: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if user_input.lower() == "exit":
            break
        if not user_input:
            continue

        matched_skills = find_matched_skills(user_input)
        if len(matched_skills) < MIN_MATCHED_SKILLS:
            print(f"\n{NO_MATCH_MESSAGE}\n")
            print("-" * 60 + "\n")
            continue

        print(f"\nRecognized skills: {', '.join(matched_skills)}")
        print("Analyzing...\n")
        try:
            recommendation = get_recommendation(user_input)
        except Exception as e:
            print(f"[query] Error while generating recommendation: {e}")
            continue

        print(recommendation)
        print("\n" + "-" * 60 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RAG job title recommendation pipeline")
    parser.add_argument(
        "step",
        choices=["clean_prepare", "ingest", "query", "all"],
        help="Which pipeline step to run",
    )
    args = parser.parse_args()

    if args.step in ("clean_prepare", "all"):
        step_clean_prepare()
    if args.step in ("ingest", "all"):
        step_ingest()
    if args.step in ("query", "all"):
        step_query()


if __name__ == "__main__":
    main()
