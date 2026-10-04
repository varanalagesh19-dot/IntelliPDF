# 🌐 Deploying IntelliPDF on Netlify

This repository includes a dedicated, high-performance static landing page and interactive application embed in the [`site/`](site/) directory, pre-configured with [`netlify.toml`](netlify.toml).

---

## 🏗️ Architecture Overview

| Component | Platform | Role | Free Tier |
|---|---|---|---|
| **Frontend Showcase & Embed** | **Netlify** | Ultra-fast CDN landing page, custom domain, live application viewport | Unlimited bandwidth / 100GB |
| **RAG & Streamlit Engine** | **Streamlit Community Cloud** or **Hugging Face Spaces** | Python runtime, FAISS vector search, PyMuPDF, Groq/Gemini LLM inference | 100% Free |

---

## 🚀 Method 1: Deploy via Netlify Dashboard (Recommended — 2 Clicks)

Because your code is already pushed to GitHub (`varanalagesh19-dot/IntelliPDF`), Netlify can deploy it automatically:

1. **Log in to Netlify**: Go to [app.netlify.com](https://app.netlify.com).
2. Click **"Add new site"** ➔ **"Import an existing project"**.
3. Choose **GitHub** and authorize access.
4. Select repository: **`varanalagesh19-dot/IntelliPDF`**.
5. **Verify Build Settings** (Netlify will auto-detect from `netlify.toml`):
   * **Branch to deploy**: `main`
   * **Publish directory**: `site`
   * **Build command**: *(Leave blank)*
6. Click **"Deploy IntelliPDF"**.

Your website will be live at `https://<your-custom-name>.netlify.app` within 30 seconds!

---

## 💻 Method 2: Deploy via Netlify CLI (Direct from Terminal)

You can also deploy directly from your local terminal using `npx` (no global installation required):

```bash
# 1. Login to your Netlify account in browser
npx netlify-cli login

# 2. Deploy the site directly to production
npx netlify-cli deploy --prod --dir=site
```

Follow the interactive prompts to link or create a new site.

---

## 🔗 Connecting Your Live Streamlit Application

1. Deploy your Streamlit backend following [DEPLOY.md](DEPLOY.md) on [Streamlit Community Cloud](https://share.streamlit.io).
2. Copy your live Streamlit URL (e.g. `https://your-app.streamlit.app`).
3. Open your deployed Netlify website:
   * Click **"Configure URL"** in the browser mockup header.
   * Paste your live Streamlit link.
   * It will instantly embed and persist in your browser for visitors!
