import axios from 'axios'

const BASE = 'http://localhost:8000'

const api = axios.create({ baseURL: BASE, timeout: 30000 })

export const fetchHealth      = ()              => api.get('/api/health').then(r => r.data)
export const fetchSearch      = (q)             => api.get('/api/search', { params: { q } }).then(r => r.data)
export const fetchPrices      = (sym, days=365) => api.get(`/api/prices/${sym}`, { params: { days } }).then(r => r.data)
export const fetchAnalysis    = (sym, days=365) => api.get(`/api/analyze/${sym}`, { params: { days } }).then(r => r.data)
export const fetchDbStats     = ()              => api.get('/api/db-stats').then(r => r.data)
export const fetchEvolution   = (limit=50)      => api.get('/api/evolution-log', { params: { limit } }).then(r => r.data)
export const fetchRoadmap     = ()              => api.get('/api/roadmap').then(r => r.data)
export const fetchCorrelations = (days=63, threshold=0.3) => api.get('/api/graph/correlations', { params: { days, threshold } }).then(r => r.data)
export const fetchKnowledge   = ()              => api.get('/api/graph/knowledge').then(r => r.data)
