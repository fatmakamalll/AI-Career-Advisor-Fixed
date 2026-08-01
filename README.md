# AI Career Advisor
### Retrieval-Augmented Generation (RAG) for AI & Data Analytics Career Recommendation

An AI-powered career recommendation system that suggests the most suitable **Job Title** based on a user's skills, experience, and profile.

The system uses a **Retrieval-Augmented Generation (RAG)** pipeline by retrieving the most relevant careers from a vector database (**ChromaDB**) and generating personalized career recommendations using **Llama 3.1** served through the **Groq API**.

---

# Project Overview

This project aims to help students and professionals identify the most suitable career path in the fields of:

- Applied Artificial Intelligence
- Data Analytics
- Data Science
- Business Intelligence
- Machine Learning

Instead of relying only on a Large Language Model, the application first retrieves similar careers from a structured jobs database, then generates recommendations grounded in those retrieved results.

---

# Dataset

The project contains a curated dataset of **1,532 job records** collected from AI Engineering and Data Analytics sources.

Each job includes:

- Job Title
- Job Category
- Experience Level
- Education Required
- Required Skills
- Industry
- Remote Work
- Company Size
- Salary Tier
- Source Dataset

During preprocessing, the data is cleaned by:

- Removing duplicate jobs
- Normalizing job titles
- Removing duplicate skills
- Standardizing formatting
- Building a descriptive document for every job

---

# RAG Pipeline

The recommendation process follows these steps:

1. Clean and prepare the dataset.
2. Convert every job description into embeddings.
3. Store embeddings in ChromaDB.
4. Receive the user's profile.
5. Retrieve the Top-K most similar careers.
6. Send the retrieved context to Llama 3.1 through Groq.
7. Generate a personalized career recommendation.

---

# Technologies

- Python
- Streamlit
- ChromaDB
- Groq API
- Llama 3.1
- Pandas
- BM25 (`rank_bm25`) for lexical hybrid retrieval
- Retrieval-Augmented Generation (RAG)

---

# Project Structure

```
AI-Career-Advisor/
│
├── app.py
├── rag_job_recommender.py
├── merged_jobs.csv
├── jobs_with_documents.csv
├── skills_vocab.json
├── requirements.txt
├── README.md
├── .gitignore
└── chroma_db/
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/USERNAME/AI-Career-Advisor.git
cd AI-Career-Advisor
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Build the Vector Database

Prepare the documents

```bash
python rag_job_recommender.py clean_prepare
```

Generate embeddings and build ChromaDB

```bash
python rag_job_recommender.py ingest
```

---

# Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

# Streamlit Cloud Deployment

After uploading the project to GitHub:

1. Create a new app on Streamlit Cloud.
2. Select your repository.
3. Set the main file to:

```
app.py
```

4. Add your Groq API key from:

```
Settings → Secrets
```

Example:

```toml
GROQ_API_KEY="YOUR_API_KEY"
```

Deploy the application.

---

# Example Output

The system returns:

- Recommended Job Title
- Match Explanation
- Required Skills
- Missing Skills
- Suggested Career Path
- Experience Level
- Industry
- Salary Tier
- Remote Work Availability

---

# Future Improvements

- Resume Upload (PDF)
- CV Parsing
- LinkedIn Profile Analysis
- Career Roadmap Generator
- Learning Resource Recommendation
- Interview Preparation Suggestions
- Multi-language Support

---

# Fixes Applied (Code Review Pass)

This version fixes issues found during a code review against the course's RAG
labs (Lab 7 / Lab 8):

1. **Normalized embeddings + cosine distance.** `embed_texts()` now sets
   `normalize_embeddings=True`, and the Chroma collection is created with
   `metadata={"hnsw:space": "cosine"}`, so the distance metric actually matches
   the embeddings being stored.
2. **Real relevance threshold.** `retrieve_similar_jobs()` now drops any
   candidate above `DISTANCE_THRESHOLD` instead of always returning `n_results`
   neighbors no matter how irrelevant they are.
3. **De-duplication by job title.** Since ~97% of job titles in the dataset
   repeat, retrieval now keeps only the best-ranked occurrence of each title
   so the LLM sees distinct options, not 5 copies of the same job.
4. **Lightweight lexical boost (superseded — see Round 3 below).** Exact skill
   matches (e.g. "SQL", "CI/CD") between the user's text and a job's
   `required_skills` gave a small distance discount, compensating for dense
   embeddings being weak on exact tokens/abbreviations. *This heuristic has
   since been replaced by real BM25 hybrid retrieval — see "Round 3 fixes"
   below.*
5. **Grounding + refusal rules in the prompt.** `build_prompt()` now instructs
   the model to use only the retrieved sources, cite source numbers, and
   explicitly say `"No Job Title Recommended"` when there is no good match,
   instead of always producing a confident-sounding answer.
6. **`app.py` now applies the same skill-validation gate the CLI already had**
   (`find_matched_skills` / `MIN_MATCHED_SKILLS`) before calling
   `get_recommendation`, so the deployed Streamlit app is no longer
   unguarded.

## ⚠️ You must re-run ingest after pulling this version

Because the embeddings themselves changed (they are now normalized) and the
Chroma collection's distance space changed (cosine instead of the old default
L2), the **old `chroma_db/` folder is stale and must be rebuilt**:

```bash
# delete the old vector store, then rebuild it
rm -rf chroma_db
python rag_job_recommender.py clean_prepare
python rag_job_recommender.py ingest
```

Querying against the old `chroma_db/` with the new code will not error, but
the distances Chroma returns will be on the old (L2) scale, so
`DISTANCE_THRESHOLD` will not behave correctly until you re-ingest.

## Round 2 fixes (self-review against Lab 8's actual threshold design)

Two real, verified gaps were fixed after re-checking the first round of
fixes against the labs' own code (no re-ingest needed for these two --
they only change post-retrieval filtering logic, not the stored embeddings):

1. **Missing relative threshold.** Lab 8's `build_context_package()` filters
   candidates with *two* rules: `min_absolute_score` AND `min_score_ratio`
   (score must be at least 40% of the best candidate's score in the same
   query). The first round of fixes only implemented the absolute half
   (`DISTANCE_THRESHOLD`). `retrieve_similar_jobs()` now also enforces
   `MIN_SCORE_RATIO = 0.40`, so a pool of uniformly mediocre matches is no
   longer treated as equally credible just because each one individually
   clears the absolute floor.
2. **Unhandled exception in `app.py`.** `step_query()` (the CLI) already
   wraps `get_recommendation()` in `try/except`; `app.py` did not, so a Groq
   API failure (rate limit, bad key, network issue) would crash the whole
   Streamlit page with a raw traceback. `app.py` now catches this and shows
   a readable `st.error(...)` message instead.

One thing considered during this review was **not** changed because it is a
deliberate, disclosed simplification rather than a bug:
- Deduplication keys on `job_title` rather than `job_id` + skill similarity,
  which is a reasonable simplification given ~97% of titles repeat in this
  dataset, but can in rare cases discard a better-fitting duplicate.

## Round 3 fixes (real BM25 hybrid retrieval)

The "lightweight lexical boost" from Round 1 (item 4 above) has now been
fully replaced with real BM25 hybrid retrieval, matching the
`ALPHA`-weighted fusion design used in Lab 7/8/9. This was a large enough
change that it gets its own section rather than being folded into "Round 2":

1. **Real BM25 index over the full corpus.** `get_bm25_index()` builds a
   `BM25Okapi` index over all 1,532 job documents (not just a retrieved
   candidate pool), because BM25's inverse-document-frequency weighting
   only means something when computed against the whole corpus. This index
   is built once per process and cached.
2. **Independent min-max normalization per query.** Embedding similarity
   (cosine, roughly in `[-1, 1]`) and raw BM25 scores (unbounded, roughly
   `0`–`15` on this corpus) live on completely different scales.
   `_min_max_normalize()` rescales each to `[0, 1]` independently, per
   query, before they are combined.
3. **Weighted merge formula.** The final ranking score is:
   ```
   hybrid_score = HYBRID_ALPHA * normalized_embedding_similarity
                + (1 - HYBRID_ALPHA) * normalized_BM25_score
   ```
   with `HYBRID_ALPHA = 0.6` (60% semantic, 40% lexical). This replaces the
   old "+0.15 per matched skill" heuristic entirely — there is no trace of
   it left in `retrieve_similar_jobs()`.
4. **BM25 runs only on the survivors of both threshold filters**, not the
   raw Chroma pool: absolute (`DISTANCE_THRESHOLD`) and relative
   (`MIN_SCORE_RATIO`) filtering happen first, and only the candidates that
   pass both get a BM25 score and enter the final hybrid ranking.

`HYBRID_ALPHA = 0.6` is a reasonable starting value based on wanting a
slight lean toward semantic matching over exact lexical matching, **not** a
value tuned against this dataset with a metric like Precision@K — that
measurement is a natural next step, not something this round claims to have
done.

---

# Author

**Fatma Kamal**

Applied AI & Data Analytics