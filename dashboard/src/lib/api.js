import axios from 'axios'

const BASE = 'http://localhost:8000'

const api = axios.create({ baseURL: BASE, timeout: 90_000 })  // 90s for research/backtest

// ── Existing routes ────────────────────────────────────────────────────────────
export const fetchHealth       = ()              => api.get('/api/health').then(r => r.data)
export const fetchSearch       = (q)             => api.get('/api/search', { params: { q } }).then(r => r.data)
export const fetchPrices       = (sym, days=365) => api.get(`/api/prices/${sym}`, { params: { days } }).then(r => r.data)
export const fetchAnalysis     = (sym, days=365) => api.get(`/api/analyze/${sym}`, { params: { days } }).then(r => r.data)
export const fetchDbStats      = ()              => api.get('/api/db-stats').then(r => r.data)
export const fetchEvolution    = (limit=50)      => api.get('/api/evolution-log', { params: { limit } }).then(r => r.data)
export const fetchRoadmap      = ()              => api.get('/api/roadmap').then(r => r.data)
export const fetchCorrelations = (days=63, threshold=0.3) =>
  api.get('/api/graph/correlations', { params: { days, threshold } }).then(r => r.data)
export const fetchKnowledge    = ()              => api.get('/api/graph/knowledge').then(r => r.data)
export const fetchScreener     = (filters={})   => api.get('/api/screener', { params: filters }).then(r => r.data)

// ── New intelligence routes ────────────────────────────────────────────────────
export const fetchPredict      = (sym, days=365) =>
  api.get(`/api/predict/${sym}`, { params: { days } }).then(r => r.data)

export const fetchBacktest     = (sym, days=730) =>
  api.get(`/api/backtest/${sym}`, { params: { days } }).then(r => r.data)

export const postResearch      = (body)         =>
  api.post('/api/research', body).then(r => r.data)

export const fetchNews         = (sym)          =>
  api.get(`/api/news/${sym}`).then(r => r.data)

export const fetchFundamentals = (sym)          =>
  api.get(`/api/fundamentals/${sym}`).then(r => r.data)

// ── Intelligence routes ────────────────────────────────────────────────────────
export const fetchCorrelationsFor = (sym, maxPairs=20) =>
  api.get(`/api/intelligence/correlations/${sym}`, { params: { max_pairs: maxPairs } }).then(r => r.data)

export const fetchMacroRegime     = ()    =>
  api.get('/api/intelligence/regime').then(r => r.data)

export const fetchSupplyChain     = (sym) =>
  api.get(`/api/intelligence/supply-chain/${sym}`).then(r => r.data)

export const fetchLeadIndicators  = (sym) =>
  api.get(`/api/intelligence/lead-indicators/${sym}`).then(r => r.data)

export const fetchSocialIntel     = (sym) =>
  api.get(`/api/intelligence/social/${sym}`).then(r => r.data)

export const fetchDeepFundamentals = (sym) =>
  api.get(`/api/intelligence/fundamentals-deep/${sym}`).then(r => r.data)

export const fetchMacroDashboard  = ()    =>
  api.get('/api/intelligence/macro-dashboard').then(r => r.data)

export const fetchGraphStats      = ()    =>
  api.get('/api/intelligence/graph-stats').then(r => r.data)

export const postSemanticSearch   = (query, symbol=null, limit=10) =>
  api.post('/api/intelligence/semantic-search', {
    query, limit, filters: symbol ? { symbol } : null,
  }).then(r => r.data)

// ── Experiments ────────────────────────────────────────────────────────────────
export const fetchExperiments = (params={}) =>
  api.get('/api/experiments', { params }).then(r => r.data)

export const fetchExperimentSummary = () =>
  api.get('/api/experiments/summary').then(r => r.data)

export const fetchBestExperiment = (metric='accuracy', modelType=null, higherIsBetter=true) =>
  api.get('/api/experiments/best', {
    params: { metric, model_type: modelType, higher_is_better: higherIsBetter },
  }).then(r => r.data)

export const fetchExperiment = (id) =>
  api.get(`/api/experiments/${id}`).then(r => r.data)

// ── Discoveries ──────────────────────────────────────────────────────────────
export const fetchDiscoveries = (params={}) =>
  api.get('/api/discoveries', { params }).then(r => r.data)

export const fetchDiscoverySummary = () =>
  api.get('/api/discoveries/summary').then(r => r.data)

// ── Sources ─────────────────────────────────────────────────────────────────────
export const fetchSources = (params={}) =>
  api.get('/api/sources', { params }).then(r => r.data)

export const fetchSourceSummary = () =>
  api.get('/api/sources/summary').then(r => r.data)

// ── Chat ───────────────────────────────────────────────────────────────────────
export const postChat = (message, symbol=null, context=null) =>
  api.post('/api/chat', { message, symbol, context }).then(r => r.data)
