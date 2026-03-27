import { useQuery } from '@tanstack/react-query'
import { fetchHealth, fetchSourceSummary, fetchDiscoverySummary, fetchGraphStats, fetchGraphIntelligence } from '../lib/api'
import Header from '../components/Layout/Header'
import {
  Shield, Network, CheckCircle2, XCircle, AlertTriangle, TrendingUp, Zap, Globe, Compass,
} from 'lucide-react'
import { motion } from 'framer-motion'

const fade = (delay = 0) => ({
  initial: { opacity: 0, y: 12 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.35, delay },
})


// ── Graph status: sources → nodes → edges → research ────────────
function GraphStatus({ srcSummary, discSummary, graphStats }) {
  const sources    = srcSummary?.total ?? 0
  const validated  = srcSummary?.by_status?.validated ?? 0
  const discoveries = discSummary?.total_discoveries ?? 0
  const strong     = discSummary?.by_strength?.strong ?? 0
  const nodes      = graphStats?.total_nodes ?? 0
  const edges      = graphStats?.total_edges ?? 0
  const assetNodes = graphStats?.nodes?.Asset ?? 0
  const macroNodes = graphStats?.nodes?.MacroIndicator ?? 0
  const runs       = discSummary?.run_count ?? 0

  const stages = [
    { label: 'Sources',     color: '#3b82f6', items: [
      { k: 'Registered', v: sources },
      { k: 'Validated',  v: validated },
    ]},
    { label: 'Graph Nodes', color: '#10b981', items: [
      { k: 'Total',  v: nodes },
      { k: 'Assets', v: assetNodes },
      { k: 'Macro',  v: macroNodes },
    ]},
    { label: 'Graph Edges', color: '#8b5cf6', items: [
      { k: 'Neo4j edges',  v: edges },
      { k: 'Discoveries',  v: discoveries },
      { k: 'Strong',       v: strong },
    ]},
    { label: 'Pipeline',    color: '#f59e0b', items: [
      { k: 'Discovery runs', v: runs },
    ]},
  ]

  return (
    <motion.div {...fade(0.05)} className="glass-card rounded-xl border border-border p-5">
      <div className="flex items-center gap-2 mb-4">
        <div className="w-7 h-7 rounded-lg bg-accent/10 border border-accent/20 flex items-center justify-center">
          <Network size={13} className="text-accent" />
        </div>
        <span className="text-xs font-semibold text-text uppercase tracking-wider">Graph State</span>
        {graphStats?.source === 'neo4j' && (
          <span className="text-[9px] px-1.5 py-0.5 rounded border border-positive/30 text-positive bg-positive/8 ml-auto">Live</span>
        )}
      </div>
      <div className="flex items-stretch gap-2 overflow-x-auto">
        {stages.map((stage) => (
          <div key={stage.label} className="flex-1 min-w-[100px]">
            <div className="text-[9px] font-semibold uppercase tracking-wider mb-2"
              style={{ color: stage.color }}>{stage.label}</div>
            <div className="space-y-1">
              {stage.items.map(item => (
                <div key={item.k}
                  className="rounded-lg px-2.5 py-1.5 text-[10px] font-medium text-text-secondary
                    border flex items-center justify-between"
                  style={{ background: `${stage.color}0d`, borderColor: `${stage.color}25` }}>
                  <span>{item.k}</span>
                  <span className="font-mono font-bold" style={{ color: stage.color }}>
                    {typeof item.v === 'number' ? item.v : item.v}
                  </span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </motion.div>
  )
}


// ── Service health (compact) ────────────────────────────────────
function ServiceHealth({ checks }) {
  const entries = Object.entries(checks)
  if (!entries.length) return null
  const up = entries.filter(([, v]) => v).length

  return (
    <motion.div {...fade(0.1)} className="glass-card rounded-xl border border-border p-5">
      <div className="flex items-center gap-2 mb-3">
        <div className="w-7 h-7 rounded-lg bg-positive/10 border border-positive/20 flex items-center justify-center">
          <Shield size={13} className="text-positive" />
        </div>
        <span className="text-xs font-semibold text-text uppercase tracking-wider">Services</span>
        <span className="text-[10px] text-text-muted ml-auto">{up}/{entries.length} online</span>
      </div>
      <div className="flex flex-wrap gap-2">
        {entries.map(([k, v]) => (
          <div key={k} className="flex items-center gap-1.5 text-xs">
            {v
              ? <CheckCircle2 size={11} className="text-positive" />
              : <XCircle size={11} className="text-negative" />
            }
            <span className="text-text-secondary">{k}</span>
          </div>
        ))}
      </div>
    </motion.div>
  )
}


// ── Structural insights ─────────────────────────────────────────
const PRIORITY_COLORS = {
  1: { border: 'border-negative/25', bg: 'bg-negative/6', badge: 'bg-negative/15 text-negative' },
  2: { border: 'border-warning/25',  bg: 'bg-warning/6',  badge: 'bg-warning/15 text-warning' },
  3: { border: 'border-accent/25',   bg: 'bg-accent/6',   badge: 'bg-accent/15 text-accent' },
}

function InsightIcon({ type }) {
  if (type.includes('anomaly'))  return <AlertTriangle size={12} />
  if (type.includes('bridge'))   return <Zap size={12} />
  return <TrendingUp size={12} />
}

function Insights({ intelligence }) {
  const insights = intelligence?.insights
  if (!insights?.length) return null

  const anomalyResult = intelligence?.anomalies
  const hasAnomalies  = anomalyResult?.anomalies?.length > 0

  return (
    <motion.div {...fade(0.15)} className="glass-card rounded-xl border border-border p-5">
      <div className="flex items-center gap-2 mb-4">
        <div className="w-7 h-7 rounded-lg bg-warning/10 border border-warning/20 flex items-center justify-center">
          <AlertTriangle size={13} className="text-warning" />
        </div>
        <span className="text-xs font-semibold text-text uppercase tracking-wider">Structural Insights</span>
        <span className="text-[10px] text-text-muted ml-auto">{insights.length} findings</span>
      </div>

      {/* Anomaly summary banner */}
      {hasAnomalies && (
        <div className="rounded-lg border border-negative/25 bg-negative/6 px-3 py-2 mb-3 flex items-center gap-2">
          <AlertTriangle size={12} className="text-negative flex-shrink-0" />
          <span className="text-[10px] text-text-secondary">
            <strong className="text-negative">{anomalyResult.anomalies.length} anomal{anomalyResult.anomalies.length === 1 ? 'y' : 'ies'}</strong> detected in structural metrics
            (rolling history: {anomalyResult.history_depth} snapshots)
          </span>
        </div>
      )}

      <div className="space-y-2">
        {insights.slice(0, 8).map((ins, i) => {
          const c = PRIORITY_COLORS[ins.priority] ?? PRIORITY_COLORS[3]
          return (
            <div key={i} className={`rounded-lg border ${c.border} ${c.bg} px-3 py-2`}>
              <div className="flex items-start gap-2">
                <span className={c.badge + ' mt-0.5'}><InsightIcon type={ins.type} /></span>
                <div className="min-w-0">
                  <div className="text-xs font-medium text-text">{ins.title}</div>
                  <div className="text-[10px] text-text-muted mt-0.5">{ins.detail}</div>
                </div>
                <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded ${c.badge} flex-shrink-0`}>
                  P{ins.priority}
                </span>
              </div>
            </div>
          )
        })}
      </div>

      {intelligence?.previous_timestamp && (
        <div className="text-[9px] text-text-muted/60 mt-3 pt-2 border-t border-border/50">
          Compared against snapshot from {new Date(intelligence.previous_timestamp).toLocaleDateString()}
        </div>
      )}
    </motion.div>
  )
}


// ── Provenance summary ──────────────────────────────────────────
function ProvenanceSummary({ provenance }) {
  if (!provenance || Object.keys(provenance).length === 0) return null

  const entries = Object.entries(provenance)
  const bySrc = {}
  for (const [, info] of entries) {
    const src = info.source_name ?? info.source_id ?? 'unknown'
    bySrc[src] = (bySrc[src] ?? 0) + 1
  }

  return (
    <motion.div {...fade(0.2)} className="glass-card rounded-xl border border-border p-5">
      <div className="flex items-center gap-2 mb-3">
        <div className="w-7 h-7 rounded-lg bg-accent/10 border border-accent/20 flex items-center justify-center">
          <Globe size={13} className="text-accent" />
        </div>
        <span className="text-xs font-semibold text-text uppercase tracking-wider">Data Provenance</span>
        <span className="text-[10px] text-text-muted ml-auto">{entries.length} series tracked</span>
      </div>
      <div className="flex flex-wrap gap-2">
        {Object.entries(bySrc).map(([src, count]) => (
          <div key={src} className="rounded-lg px-3 py-1.5 border border-border bg-bg-hover text-[10px] text-text-secondary">
            <span className="font-semibold text-text">{count}</span> from {src}
          </div>
        ))}
      </div>
    </motion.div>
  )
}


// ── Main page ───────────────────────────────────────────────────
export default function Overview() {
  const { data: health }       = useQuery({ queryKey: ['health'],               queryFn: fetchHealth,             staleTime: 30_000 })
  const { data: srcSummary }   = useQuery({ queryKey: ['sources-summary'],      queryFn: fetchSourceSummary,      staleTime: 60_000 })
  const { data: discSummary }  = useQuery({ queryKey: ['discoveries-summary'],  queryFn: fetchDiscoverySummary,   staleTime: 60_000 })
  const { data: graphStats }   = useQuery({ queryKey: ['graph-stats'],          queryFn: fetchGraphStats,         staleTime: 60_000 })
  const { data: intelligence } = useQuery({ queryKey: ['graph-intelligence'],   queryFn: fetchGraphIntelligence,  staleTime: 300_000 })

  const checks = health?.checks ?? {}

  return (
    <div className="flex flex-col h-full">
      <Header title="Overview" subtitle="Structural state of the market graph" />

      <div className="flex-1 overflow-y-auto p-5 space-y-4">
        {/* Graph state */}
        <GraphStatus srcSummary={srcSummary} discSummary={discSummary} graphStats={graphStats} />

        {/* Two-column: insights + services/provenance */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <div className="lg:col-span-2">
            <Insights intelligence={intelligence} />
          </div>
          <div className="space-y-4">
            <ServiceHealth checks={checks} />
            <ProvenanceSummary provenance={intelligence?.provenance} />
          </div>
        </div>
      </div>
    </div>
  )
}
