# GitHub Actions Secrets Setup

## Required Secrets

Add these at: **GitHub repo → Settings → Secrets and variables → Actions → New repository secret**

| Secret Name | Where to get it | Required for |
|---|---|---|
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) → API Keys | AI Chat, Research, backtest insights |
| `SUPABASE_URL` | Supabase Dashboard → Settings → API → Project URL | All DB writes |
| `SUPABASE_KEY` | Supabase Dashboard → Settings → API → `service_role` key | All DB writes |
| `FRED_API_KEY` | [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html) (free) | Macro data ingestion |
| `ALPHA_VANTAGE_KEY` | [alphavantage.co/support/#api-key](https://alphavantage.co/support/#api-key) (free tier) | Additional market data |

## Step-by-step

1. Go to your repository on GitHub
2. Click **Settings** tab (top nav)
3. In the left sidebar: **Secrets and variables** → **Actions**
4. Click **New repository secret** for each secret above
5. Paste the value and save

## Verify Secrets Are Working

After adding secrets, push any commit to `main` to trigger the workflow, or manually run:
- **GitHub** → **Actions** tab → Select workflow → **Run workflow**

## Workflows That Use Secrets

| Workflow | Schedule | Secrets Used |
|---|---|---|
| `ingest_hourly.yml` | Every hour | SUPABASE_URL, SUPABASE_KEY, FRED_API_KEY, ALPHA_VANTAGE_KEY |
| `daily_analysis.yml` | Weekdays 6:30 AM UTC | SUPABASE_URL, SUPABASE_KEY |
| `weekly_retrain.yml` | Sundays 4 AM UTC | SUPABASE_URL, SUPABASE_KEY |

## Local .env File

For local development, create `.env` in the project root:

```env
ANTHROPIC_API_KEY=sk-ant-...
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=eyJ...
FRED_API_KEY=your_fred_key
ALPHA_VANTAGE_KEY=your_av_key
NEO4J_URI=neo4j+s://...
NEO4J_USER=neo4j
NEO4J_PASSWORD=...
QDRANT_URL=https://...
QDRANT_API_KEY=...
```

> **Never commit `.env` to git** — it's already in `.gitignore`
