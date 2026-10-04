# 🚀 Deploying IntelliPDF to Streamlit Community Cloud

Streamlit Community Cloud (SCC) hosts the **UI only**, so IntelliPDF is built to
run as a **single process**: the Streamlit app mounts the FastAPI application
in-process (see [frontend/api_client.py](frontend/api_client.py)). No second
server, no persistent disk, no paid services.

---

## 1. Push the code to a dedicated GitHub repo

SCC installs the `requirements.txt` that sits in the **repository root**, so the
recommended layout is a repo dedicated to IntelliPDF:

```
<your-repo>/                      <- repo root
├── requirements.txt              (the full list, used by SCC)
├── .streamlit/config.toml
├── .python-version
├── backend/
├── frontend/
│   ├── app.py                    <- the entry point you select in the UI
│   └── api_client.py
└── evaluation/, tests/
```

Push from this project directory:

```bash
cd /path/to/intellipdf
git init -b main                 # if it is not already a git repo
git add .
git commit -m "Add IntelliPDF"
git remote add origin https://github.com/<you>/<your-repo>.git
git push -u origin main
```

---

## 2. Create the app on Streamlit Community Cloud

1. Go to <https://share.streamlit.io> → **New app** → **Deploy from GitHub**.
2. Connect the repository you just pushed.
3. Deployment settings:

   | Field | Value |
   |---|---|
   | Main file path | `frontend/app.py` |
   | Python version | read from `.python-version` (`3.12`) |
   | Requirements file | `requirements.txt` (auto-detected) |
   | Streamlit version | latest (auto-detected) |

4. Click **Deploy**. The first build downloads torch + the embedding model, so
   expect **4-8 minutes**. Subsequent redeploys are faster.

---

## 3. Add your API keys (App → Settings → Secrets)

```toml
GROQ_API_KEY = "gsk_..."
EMBEDDING_BACKEND = "hashing"     # optional, see the memory note below
```

Secrets set in `st.secrets` are copied into the environment by `frontend/app.py`
before the backend config loads, so **no `.env` file is needed on the server**
(and none should ever be committed).

Optional secrets: `GEMINI_API_KEY`, `DEFAULT_LLM`, `SIMILARITY_THRESHOLD`,
`HYBRID_DENSE_WEIGHT`, `HYBRID_LEXICAL_WEIGHT`, `TOP_K`, `MAX_UPLOAD_MB`.

Without any key the app still runs in offline extractive mode.

---

## 4. Memory note (important)

SCC's free tier gives roughly **1 GB of RAM**. `sentence-transformers` pulls in
torch (~800 MB of RAM on import), which can exceed that.

Measured on this project (`INTELLIPDF_INPROCESS=1`, upload + ask + abstention):

| Setup | Startup | Python heap peak | torch imported | Retrieval quality |
|---|---|---|---|---|
| `EMBEDDING_BACKEND = "auto"` (default) | 21.8 s | ~1 GB | yes | best (MiniLM semantics) |
| `EMBEDDING_BACKEND = "hashing"` | 7.1 s | **49 MB** | **no** | lexical only |

Both paths answered correctly and abstained on an off-topic question; the
hashing path is the one to use on a ~1 GB host.

Set the secret to `"hashing"` if the app crashes on boot. Everything still works
— retrieval switches from dense cosine to hashed n-gram overlap, and the
`SIMILARITY_THRESHOLD` value in `.env.example` should be raised a little
(`0.12` → `0.2`) because hashed scores are on a different scale. Run
`python evaluation/threshold_sweep.py <your.pdf>` to re-calibrate for your setup.

---

## 5. Known Cloud limitations (be honest about these in a demo)

* **No persistent disk.** Uploads, the FAISS indexes and `intellipdf.db` live in
  the container filesystem, so every redeploy (or idle restart) wipes them.
  Re-upload your PDF after each redeploy.
* **Disk quota ≈ 1 GB.** Keep the repository lean: the two bundled sample PDFs
  are ~35 KB, but do not commit large datasets.
* **Cold starts.** The first request after an idle period re-imports torch and
  re-reads the FAISS index.
* **Local-only URLs.** The `🔗 Open PDF at page X` chip points at
  `http://127.0.0.1:8000/...`, which does not exist on Cloud. The page numbers
  and snippets still render; only the deep-link target is inert there.
* **Concurrency.** One shared SQLite file and one in-process FAISS cache — fine
  for a demo or a 2-3 person team, not for a busy public app.

---

## 6. Verify the deployment

```bash
# after deploying, point the smoke test at the Cloud URL
INTELLIPDF_INPROCESS=1 python tests/e2e_check.py sample_features.pdf
```

In the browser: the sidebar must show **API online**, and the Upload tab should
report `Indexed N pages into M chunks`.

---

## 7. Alternatives if Cloud is too tight

| Option | RAM | Notes |
|---|---|---|
| **Streamlit Community Cloud** | ~1 GB | Free, as above |
| **Koyeb / Render / Fly.io** free tier | 512 MB–1 GB | Needs `EMBEDDING_BACKEND=hashing` |
| **A free VPS (Oracle Cloud Always Free)** | 1–4 GB ARM | Full fidelity, `docker compose up` |
| **Local only** | your laptop | `./run.sh`, no constraints at all |

For the last two, keep the two-process layout — start the API with
`uvicorn backend.main:app --port 8000` and point the UI at it with
`API_URL=http://127.0.0.1:8000`.