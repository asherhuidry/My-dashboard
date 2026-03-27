# GitHub Actions Secrets Setup

## Required Secrets

Add these at: **GitHub repo → Settings → Secrets and variables → Actions → New repository secret**

| Secret Name | Where to get it | Required for |
|---|---|---|
| `SUPABASE_URL` | Supabase Dashboard → Settings → API → Project URL | All DB writes |
| `SUPABASE_KEY` | Supabase Dashboard → Settings → API → `service_role` key | All DB writes |
| `FRED_API_KEY` | [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html) (free) | Macro data ingestion + discovery |
| `NEO4J_URI` | Neo4j Aura Console → Connection URI | Graph materialization |
| `NEO4J_USER` | Neo4j Aura Console → Username | Graph materialization |
| `NEO4J_PASSWORD` | Neo4j Aura Console → Password | Graph materialization |

### Optional (for extended ingestion)

| Secret Name | Where to get it | Required for |
|---|---|---|
| `ALPHA_VANTAGE_KEY` | [alphavantage.co/support/#api-key](https://alphavantage.co/support/#api-key) (free tier) | Supplemental equity data |
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) → API Keys | AI Chat, Research |
| `QDRANT_URL` | Qdrant Cloud → Cluster URL | Vector embeddings |
| `QDRANT_API_KEY` | Qdrant Cloud → API Key | Vector embeddings |

## Step-by-step

1. Go to your repository on GitHub
2. Click **Settings** tab (top nav)
3. In the left sidebar: **Secrets and variables** → **Actions**
4. Click **New repository secret** for each secret above
5. Paste the value and save

## Verify Secrets Are Working

After adding the 6 required secrets, trigger the weekly pipeline manually:
- **GitHub** → **Actions** tab → **Weekly Market Graph Pipeline** → **Run workflow**

## Active Workflows

| Workflow | Schedule | What it does |
|---|---|---|
| `comprehensive_ingest.yml` | Saturday 8 AM ET | FRED macro + ECB + discovery → persist → graph materialize |
| `ingest_hourly.yml` | Every hour | Price/macro data archival to Supabase |
| `daily_analysis.yml` | Weekdays 6:30 AM UTC | Noise filter + data quality check |
| `python-app.yml` | Every push | Lint + test suite |

## Local .env File

For local development, create `.env` in the project root:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=eyJ...
FRED_API_KEY=your_fred_key
NEO4J_URI=neo4j+s://...
NEO4J_USER=neo4j
NEO4J_PASSWORD=...
ALPHA_VANTAGE_KEY=your_av_key
ANTHROPIC_API_KEY=sk-ant-...
QDRANT_URL=https://...
QDRANT_API_KEY=...
```

> **Never commit `.env` to git** — it's already in `.gitignore`
