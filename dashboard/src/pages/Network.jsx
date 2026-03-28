import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { fetchCorrelations, fetchKnowledge, fetchNodeDetail, fetchGraphIntelligence } from '../lib/api'
import Header from '../components/Layout/Header'
import ForceGraph from '../components/Network/ForceGraph'
import Spinner from '../components/UI/Spinner'
import { motion } from 'framer-motion'
import { GitBranch, Network as NetworkIcon, AlertTriangle, TrendingUp, Zap, Filter, X, Layers, Calendar } from 'lucide-react'

const TABS = [
  { id:'graph', label:'Market Graph',  icon: GitBranch },
  { id:'corr',  label:'Correlations',  icon: NetworkIcon },
]

const REL_TYPES = ['SENSITIVE_TO', 'CORRELATED_WITH', 'BELONGS_TO', 'PART_OF', 'REPORTS', 'HAS_FEATURES']
const ASSET_CLASSES = ['equity', 'crypto', 'forex', 'commodity']
const REGIMES = ['all', 'bear', 'stress']

// ── Filter bar ──────────────────────────────────────────────────
function GraphFilters({ filters, setFilters }) {
  const active = Object.values(filters).some(Boolean)

  const set = (key, val) =>
    setFilters(prev => ({ ...prev, [key]: val || null }))

  return (
    <div className="flex items-center gap-2.5 flex-wrap">
      <Filter size={10} className="text-text-muted" />

      <select value={filters.rel_type ?? ''}
        onChange={e => set('rel_type', e.target.value)}
        className="text-[10px] bg-bg-hover border border-border rounded-lg px-2 py-1 text-text-secondary focus:border-accent/40 outline-none">
        <option value="">All edges</option>
        {REL_TYPES.map(t => <option key={t} value={t}>{t}</option>)}
      </select>

      <select value={filters.asset_class ?? ''}
        onChange={e => set('asset_class', e.target.value)}
        className="text-[10px] bg-bg-hover border border-border rounded-lg px-2 py-1 text-text-secondary focus:border-accent/40 outline-none">
        <option value="">All classes</option>
        {ASSET_CLASSES.map(c => <option key={c} value={c} className="capitalize">{c}</option>)}
      </select>

      <select value={filters.regime ?? ''}
        onChange={e => set('regime', e.target.value)}
        className="text-[10px] bg-bg-hover border border-border rounded-lg px-2 py-1 text-text-secondary focus:border-accent/40 outline-none">
        <option value="">All regimes</option>
        {REGIMES.map(r => <option key={r} value={r}>{r}</option>)}
      </select>

      {active && (
        <button onClick={() => setFilters({})}
          className="text-[9px] text-text-muted hover:text-text flex items-center gap-1 transition-colors">
          <X size={9} /> Clear
        </button>
      )}
    </div>
  )
}

// ── Correlation controls ────────────────────────────────────────
function CorrelationControls({ days, setDays, threshold, setThreshold }) {
  return (
    <div className="flex items-center gap-3 flex-wrap">
      <div className="flex items-center gap-1.5">
        <span className="text-[10px] text-text-muted">Window:</span>
        {[21,63,126,252].map(d => (
          <button key={d} onClick={() => setDays(d)}
            className={`text-[10px] px-2 py-1 rounded-lg border transition-all ${
              days===d ? 'border-accent/30 bg-accent/10 text-accent' : 'border-border text-text-muted hover:text-text'
            }`}>
            {d}D
          </button>
        ))}
      </div>
      <div className="flex items-center gap-1.5">
        <span className="text-[10px] text-text-muted">Min |r|:</span>
        <input type="range" min="0" max="0.9" step="0.05" value={threshold}
          onChange={e => setThreshold(parseFloat(e.target.value))}
          className="w-20 accent-accent" />
        <span className="ticker-value text-[10px] text-accent">{threshold.toFixed(2)}</span>
      </div>
    </div>
  )
}

// ── Node detail panel ───────────────────────────────────────────
const REL_COLORS = {
  SENSITIVE_TO:    '#f59e0b',
  CORRELATED_WITH: '#10b981',
  BELONGS_TO:      '#8b5cf6',
  PART_OF:         '#a78bfa',
  REPORTS:         '#06b6d4',
  HAS_FEATURES:    '#3b82f6',
}

function NodeDetailPanel({ nodeName, onClose }) {
  const { data, isLoading } = useQuery({
    queryKey: ['node-detail', nodeName],
    queryFn:  () => fetchNodeDetail(nodeName),
    staleTime: 120_000,
    enabled: !!nodeName,
  })

  if (isLoading) {
    return (
      <motion.div initial={{ x: 300, opacity: 0 }} animate={{ x: 0, opacity: 1 }}
        className="w-72 border-l border-border bg-bg-secondary p-4 flex items-center justify-center">
        <Spinner size="sm" text="Loading..." />
      </motion.div>
    )
  }

  if (!data?.found) {
    return (
      <motion.div initial={{ x: 300, opacity: 0 }} animate={{ x: 0, opacity: 1 }}
        className="w-72 border-l border-border bg-bg-secondary p-4">
        <div className="flex items-start justify-between mb-3">
          <h3 className="text-sm font-semibold text-text">{nodeName}</h3>
          <button onClick={onClose} className="text-text-muted hover:text-text text-xs">{'\u2715'}</button>
        </div>
        <p className="text-[10px] text-text-muted">No detail available</p>
      </motion.div>
    )
  }

  const { node, relationships, total_edges } = data

  return (
    <motion.div initial={{ x: 300, opacity: 0 }} animate={{ x: 0, opacity: 1 }}
      className="w-72 border-l border-border bg-bg-secondary overflow-y-auto">

      <div className="sticky top-0 bg-bg-secondary border-b border-border p-4 z-10">
        <div className="flex items-start justify-between">
          <div>
            <h3 className="text-sm font-semibold text-text">{node.name}</h3>
            <div className="text-[9px] text-text-muted mt-0.5">
              {node.type}{node.class ? ` \u00B7 ${node.class}` : ''}{node.sector ? ` \u00B7 ${node.sector}` : ''}
            </div>
          </div>
          <button onClick={onClose} className="text-text-muted hover:text-text text-xs">{'\u2715'}</button>
        </div>
        <div className="text-[9px] text-text-muted mt-1">{total_edges} edge{total_edges !== 1 ? 's' : ''}</div>
      </div>

      <div className="p-4 space-y-4">
        {Object.entries(relationships).map(([relType, rels]) => (
          <div key={relType}>
            <div className="flex items-center gap-2 mb-2">
              <div className="w-2 h-2 rounded-full" style={{ background: REL_COLORS[relType] ?? '#475569' }} />
              <span className="text-[9px] font-semibold uppercase tracking-wider"
                style={{ color: REL_COLORS[relType] ?? '#94a3b8' }}>
                {relType.replace(/_/g, ' ')}
              </span>
              <span className="text-[9px] text-text-muted ml-auto">{rels.length}</span>
            </div>

            <div className="space-y-1.5">
              {rels.map((rel, i) => (
                <div key={i} className="rounded-lg border border-border/50 bg-bg-hover/30 px-2.5 py-2 text-[10px]">
                  <div className="flex items-center justify-between">
                    <span className="font-medium text-text">{rel.neighbor}</span>
                    <span className="text-text-muted text-[9px]">
                      {rel.direction === 'outgoing' ? '\u2192' : '\u2190'} {rel.neighbor_type}
                    </span>
                  </div>

                  {rel.confidence != null && (
                    <div className="mt-1 flex items-center gap-1.5">
                      <span className="text-[9px] text-text-muted w-5">conf</span>
                      <div className="flex-1 h-1 bg-border/40 rounded-full overflow-hidden">
                        <div className="h-full rounded-full transition-all" style={{
                          width: `${Math.round(rel.confidence * 100)}%`,
                          background: rel.confidence >= 0.7 ? '#10b981' : rel.confidence >= 0.4 ? '#f59e0b' : '#ef4444',
                        }} />
                      </div>
                      <span className="text-[9px] ticker-value" style={{
                        color: rel.confidence >= 0.7 ? '#10b981' : rel.confidence >= 0.4 ? '#f59e0b' : '#ef4444',
                      }}>{(rel.confidence * 100).toFixed(0)}%</span>
                      {rel.evidence_count > 1 && (
                        <span className="text-[8px] text-accent bg-accent/10 px-1 rounded">{rel.evidence_count}x</span>
                      )}
                    </div>
                  )}

                  <div className="flex flex-wrap gap-x-3 gap-y-0.5 mt-1">
                    {rel.beta != null && (
                      <span className="text-text-secondary">
                        <span className="text-text-muted">{'\u03B2'} </span>
                        <span className={rel.beta >= 0 ? 'text-positive' : 'text-negative'}>
                          {rel.beta >= 0 ? '+' : ''}{Number(rel.beta).toFixed(3)}
                        </span>
                      </span>
                    )}
                    {rel.correlation != null && (
                      <span className="text-text-secondary">
                        <span className="text-text-muted">r </span>
                        <span className={rel.correlation >= 0 ? 'text-positive' : 'text-negative'}>
                          {Number(rel.correlation).toFixed(3)}
                        </span>
                      </span>
                    )}
                    {rel.regime && <span className="text-text-muted">{rel.regime}</span>}
                    {rel.factor_group && <span className="text-text-muted">{rel.factor_group}</span>}
                    {rel.strength && (
                      <span className="text-text-secondary">
                        <span className="text-text-muted">str </span>{rel.strength}
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </motion.div>
  )
}

// ── Graph health panel ──────────────────────────────────────────
function GraphHealthPanel({ intelligence }) {
  const { reasoning, confidence } = intelligence ?? {}
  if (!reasoning && !confidence) return null

  const health = reasoning?.structural_health
  const topFactor = reasoning?.most_influential_factor
  const stressed = reasoning?.most_stressed_sector

  const gradeColor = {
    excellent: '#10b981', good: '#3b82f6', fair: '#f59e0b', poor: '#ef4444',
  }

  return (
    <div className="absolute top-4 right-4 glass-card rounded-xl px-3.5 py-3 text-xs space-y-2.5 w-52">
      <div className="text-[9px] text-text-muted uppercase tracking-wider font-semibold">Graph Health</div>

      {health && (
        <div className="space-y-1">
          <div className="flex items-center justify-between">
            <span className="text-text-secondary text-[10px]">Overall</span>
            <span className="ticker-value text-[10px] font-semibold" style={{ color: gradeColor[health.grade] ?? '#94a3b8' }}>
              {(health.score * 100).toFixed(0)}% {health.grade}
            </span>
          </div>
          {Object.entries(health.components ?? {}).map(([k, v]) => (
            <div key={k} className="flex items-center gap-1.5">
              <span className="text-[9px] text-text-muted w-14 truncate capitalize">{k.replace(/_/g, ' ')}</span>
              <div className="flex-1 h-1 bg-border/40 rounded-full overflow-hidden">
                <div className="h-full bg-accent/60 rounded-full" style={{ width: `${Math.round(v * 100)}%` }} />
              </div>
              <span className="text-[9px] ticker-value text-text-secondary w-6 text-right">{(v * 100).toFixed(0)}%</span>
            </div>
          ))}
        </div>
      )}

      {confidence && (
        <div className="pt-1.5 border-t border-border/50 space-y-0.5 text-[10px]">
          <div className="flex justify-between">
            <span className="text-text-muted">Confidence</span>
            <span className="ticker-value text-text">{(confidence.mean_confidence * 100).toFixed(0)}%</span>
          </div>
          <div className="flex justify-between">
            <span className="text-text-muted">Scored</span>
            <span className="ticker-value text-text">{confidence.scored_edges}</span>
          </div>
        </div>
      )}

      {topFactor && (
        <div className="pt-1.5 border-t border-border/50">
          <div className="text-[9px] text-text-muted">Top bridge</div>
          <div className="text-[10px] text-text font-medium">{topFactor.factor_id}</div>
        </div>
      )}

      {stressed && (
        <div className="pt-1.5 border-t border-border/50">
          <div className="text-[9px] text-text-muted">Most stressed</div>
          <div className="text-[10px] font-medium" style={{ color: stressed.stress_score >= 0.5 ? '#f87171' : '#fbbf24' }}>
            {stressed.sector}
          </div>
        </div>
      )}
    </div>
  )
}

// ── Insights strip ──────────────────────────────────────────────
const PRIORITY_STYLES = {
  1: { border: 'border-negative/25', bg: 'bg-negative/5', icon: 'text-negative' },
  2: { border: 'border-warning/25',  bg: 'bg-warning/5',  icon: 'text-warning' },
  3: { border: 'border-accent/25',   bg: 'bg-accent/5',   icon: 'text-accent' },
}

function InsightIcon({ type }) {
  if (type.includes('anomaly'))  return <AlertTriangle size={10} />
  if (type.includes('bridge'))   return <Zap size={10} />
  if (type.includes('sector'))   return <Layers size={10} />
  if (type.includes('earning'))  return <Calendar size={10} />
  return <TrendingUp size={10} />
}

function InsightsPanel({ insights }) {
  if (!insights?.length) return null
  const top = insights.slice(0, 5)
  return (
    <div className="border-t border-border bg-bg-secondary/60 px-5 py-2.5">
      <div className="text-[9px] text-text-muted uppercase tracking-wider font-semibold mb-1.5">Top Insights</div>
      <div className="flex gap-2 overflow-x-auto pb-1">
        {top.map((ins, i) => {
          const s = PRIORITY_STYLES[ins.priority] ?? PRIORITY_STYLES[3]
          return (
            <div key={i} className={`flex-shrink-0 rounded-lg px-3 py-2 border ${s.border} ${s.bg} max-w-[280px]`}>
              <div className="flex items-center gap-1.5 mb-0.5">
                <span className={s.icon}><InsightIcon type={ins.type} /></span>
                <span className="text-[10px] font-semibold text-text truncate">{ins.title}</span>
              </div>
              <div className="text-[9px] text-text-muted truncate">{ins.detail}</div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

// ── Main page ───────────────────────────────────────────────────
export default function Network() {
  const [tab, setTab]             = useState('graph')
  const [days, setDays]           = useState(63)
  const [threshold, setThreshold] = useState(0.3)
  const [selectedNode, setSelectedNode] = useState(null)
  const [filters, setFilters]     = useState({})

  const { data: graphData, isLoading: graphLoading } = useQuery({
    queryKey: ['knowledge', filters],
    queryFn:  () => fetchKnowledge(filters),
    staleTime: 600_000,
    enabled:   tab === 'graph',
  })

  const { data: corrData, isLoading: corrLoading } = useQuery({
    queryKey: ['correlations', days, threshold],
    queryFn:  () => fetchCorrelations(days, threshold),
    staleTime: 300_000,
    enabled:   tab === 'corr',
  })

  const { data: intelligence } = useQuery({
    queryKey: ['graph-intelligence'],
    queryFn:  fetchGraphIntelligence,
    staleTime: 600_000,
  })

  const loading = tab === 'graph' ? graphLoading : corrLoading
  const data    = tab === 'graph' ? graphData : corrData
  const isEmpty = data && data.nodes?.length === 0

  const handleNodeClick = (node) => {
    const name = node.label ?? node.id
    setSelectedNode(name)
  }

  return (
    <div className="flex flex-col h-full">
      <Header title="Market Graph" subtitle="Live market structure from Neo4j" />
      <div className="flex-1 flex flex-col overflow-hidden">

        {/* Tab bar + controls */}
        <div className="px-5 py-2 border-b border-border bg-bg-secondary/50 flex items-center gap-4 flex-wrap">
          <div className="flex items-center gap-1">
            {TABS.map(t => (
              <button key={t.id} onClick={() => { setTab(t.id); setSelectedNode(null); setFilters({}) }}
                className={`flex items-center gap-1.5 text-[10px] px-3 py-1.5 rounded-lg border transition-all ${
                  tab===t.id ? 'border-accent/30 bg-accent/10 text-accent' : 'border-border text-text-muted hover:text-text'
                }`}>
                <t.icon size={10} /> {t.label}
              </button>
            ))}
          </div>
          {tab === 'graph' && <GraphFilters filters={filters} setFilters={setFilters} />}
          {tab === 'corr' && (
            <CorrelationControls days={days} setDays={setDays} threshold={threshold} setThreshold={setThreshold} />
          )}
          {tab === 'graph' && data?.source && (
            <span className={`text-[9px] px-2 py-0.5 rounded-lg border ml-auto ${
              data.source === 'neo4j'
                ? 'border-positive/25 text-positive bg-positive/6'
                : 'border-text-muted/25 text-text-muted bg-bg-hover'
            }`}>
              {data.source === 'neo4j' ? 'Live from Neo4j' : 'Neo4j unavailable'}
            </span>
          )}
        </div>

        {/* Graph area */}
        <div className="flex-1 flex overflow-hidden">
          <div className="flex-1 relative">
            {loading ? (
              <div className="flex items-center justify-center h-full">
                <Spinner size="lg" text="Loading graph..." />
              </div>
            ) : isEmpty ? (
              <div className="flex flex-col items-center justify-center h-full text-text-muted gap-3">
                <GitBranch size={32} className="text-text-muted/30" />
                <div className="text-sm">No graph data available</div>
                <div className="text-[10px] text-text-muted/60 max-w-sm text-center">
                  {tab === 'graph'
                    ? 'Neo4j is not connected or the graph has not been materialized yet.'
                    : 'Could not fetch correlation data.'}
                </div>
              </div>
            ) : data ? (
              <ForceGraph nodes={data.nodes} edges={data.edges} onNodeClick={handleNodeClick} />
            ) : null}

            {tab === 'graph' && intelligence && <GraphHealthPanel intelligence={intelligence} />}

            {data && !loading && !isEmpty && (
              <div className="absolute top-4 left-4 glass-card rounded-xl px-3 py-2.5 text-[10px] space-y-0.5">
                <div className="flex items-center gap-1.5">
                  <span className="text-text-secondary">Nodes:</span>
                  <span className="ticker-value text-text font-semibold">{data.nodes?.length}</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="text-text-secondary">Edges:</span>
                  <span className="ticker-value text-text font-semibold">{data.edges?.length}</span>
                </div>
                {tab === 'graph' && data.meta?.rel_types && (
                  <div className="pt-1 mt-1 border-t border-border/40 space-y-0.5">
                    {Object.entries(data.meta.rel_types).map(([rt, count]) => (
                      <div key={rt} className="flex items-center gap-1.5">
                        <div className="w-1.5 h-1.5 rounded-full" style={{ background: REL_COLORS[rt] ?? '#475569' }} />
                        <span className="text-text-muted">{rt.replace(/_/g, ' ')}</span>
                        <span className="ticker-value text-text-secondary ml-auto">{count}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>

          {selectedNode && tab === 'graph' && (
            <NodeDetailPanel nodeName={selectedNode} onClose={() => setSelectedNode(null)} />
          )}

          {selectedNode && tab === 'corr' && (
            <motion.div
              initial={{ x: 300, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              className="w-64 border-l border-border bg-bg-secondary p-4 space-y-3 overflow-y-auto"
            >
              <div className="flex items-start justify-between">
                <h3 className="text-sm font-semibold text-text">{selectedNode}</h3>
                <button onClick={() => setSelectedNode(null)} className="text-text-muted hover:text-text text-xs">{'\u2715'}</button>
              </div>
              <div className="pt-2 border-t border-border">
                <p className="text-[10px] text-text-muted">
                  Rolling return correlations. Thick = stronger. Green = positive, Red = inverse.
                </p>
              </div>
            </motion.div>
          )}
        </div>

        {intelligence?.insights && <InsightsPanel insights={intelligence.insights} />}
      </div>
    </div>
  )
}
