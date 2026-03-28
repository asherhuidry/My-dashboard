import { useQuery } from '@tanstack/react-query'
import { fetchHealth, fetchSourceSummary, fetchDiscoverySummary, fetchGraphStats, fetchGraphIntelligence, fetchGraphAnalysis } from '../lib/api'
import Header from '../components/Layout/Header'
import {
  Shield, Network, AlertTriangle, TrendingUp, Zap, Globe,
  Activity, BarChart3, Calendar, Layers, ChevronRight, Target,
} from 'lucide-react'
import { motion } from 'framer-motion'

const fade = (delay = 0) => ({
  initial: { opacity: 0, y: 10 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.3, delay },
})


// ── Metric card ─────────────────────────────────────────────────
function MetricCard({ label, value, sub, color, icon: Icon, delay = 0 }) {
  return (
    <motion.div {...fade(delay)} className="glass-card-hover rounded-xl p-4 relative overflow-hidden">
      <div className="absolute top-0 left-0 right-0 h-px" style={{ background: `linear-gradient(90deg, transparent, ${color}40, transparent)` }} />
      <div className="flex items-start justify-between mb-2">
        <span className="text-[10px] font-semibold text-text-muted uppercase tracking-wider">{label}</span>
        {Icon && <Icon size={13} style={{ color }} className="opacity-60" />}
      </div>
      <div className="metric-lg" style={{ color }}>{value}</div>
      {sub && <div className="text-[10px] text-text-muted mt-1">{sub}</div>}
    </motion.div>
  )
}


// ── Graph pipeline status ───────────────────────────────────────
function PipelineStatus({ srcSummary, discSummary, graphStats }) {
  const sources    = srcSummary?.total ?? 0
  const validated  = srcSummary?.by_status?.validated ?? 0
  const discoveries = discSummary?.total_discoveries ?? 0
  const strong     = discSummary?.by_strength?.strong ?? 0
  const nodes      = graphStats?.total_nodes ?? 0
  const edges      = graphStats?.total_edges ?? 0
  const assetNodes = graphStats?.nodes?.Asset ?? 0
  const macroNodes = graphStats?.nodes?.MacroIndicator ?? 0
  const sectorNodes = graphStats?.nodes?.Sector ?? 0
  const runs       = discSummary?.run_count ?? 0

  return (
    <motion.div {...fade(0.05)} className="glass-card rounded-xl border border-border p-5">
      <div className="flex items-center gap-2 mb-4">
        <div className="w-7 h-7 rounded-lg bg-accent/10 border border-accent/20 flex items-center justify-center">
          <Network size={13} className="text-accent" />
        </div>
        <span className="text-xs font-semibold text-text uppercase tracking-wider">Graph Pipeline</span>
        {graphStats?.source === 'neo4j' && (
          <span className="text-[9px] px-1.5 py-0.5 rounded border border-positive/30 text-positive bg-positive/8 ml-auto">
            Live
          </span>
        )}
      </div>

      <div className="grid grid-cols-4 gap-3">
        {/* Sources */}
        <div className="space-y-1.5">
          <div className="text-[9px] font-semibold uppercase tracking-wider text-accent">Sources</div>
          <div className="rounded-lg px-2.5 py-1.5 border border-accent/15 bg-accent/5 flex justify-between items-center">
            <span className="text-[10px] text-text-secondary">Registered</span>
            <span className="ticker-value text-xs font-bold text-accent">{sources}</span>
          </div>
          <div className="rounded-lg px-2.5 py-1.5 border border-accent/15 bg-accent/5 flex justify-between items-center">
            <span className="text-[10px] text-text-secondary">Validated</span>
            <span className="ticker-value text-xs font-bold text-accent">{validated}</span>
          </div>
        </div>

        {/* Nodes */}
        <div className="space-y-1.5">
          <div className="text-[9px] font-semibold uppercase tracking-wider text-positive">Nodes</div>
          <div className="rounded-lg px-2.5 py-1.5 border border-positive/15 bg-positive/5 flex justify-between items-center">
            <span className="text-[10px] text-text-secondary">Total</span>
            <span className="ticker-value text-xs font-bold text-positive">{nodes}</span>
          </div>
          <div className="rounded-lg px-2.5 py-1.5 border border-positive/15 bg-positive/5 flex justify-between items-center">
            <span className="text-[10px] text-text-secondary">Assets</span>
            <span className="ticker-value text-xs font-bold text-positive">{assetNodes}</span>
          </div>
          <div className="rounded-lg px-2.5 py-1.5 border border-positive/15 bg-positive/5 flex justify-between items-center">
            <span className="text-[10px] text-text-secondary">Sectors</span>
            <span className="ticker-value text-xs font-bold text-positive">{sectorNodes}</span>
          </div>
        </div>

        {/* Edges */}
        <div className="space-y-1.5">
          <div className="text-[9px] font-semibold uppercase tracking-wider text-purple">Edges</div>
          <div className="rounded-lg px-2.5 py-1.5 border border-purple/15 bg-purple/5 flex justify-between items-center">
            <span className="text-[10px] text-text-secondary">Graph</span>
            <span className="ticker-value text-xs font-bold text-purple">{edges}</span>
          </div>
          <div className="rounded-lg px-2.5 py-1.5 border border-purple/15 bg-purple/5 flex justify-between items-center">
            <span className="text-[10px] text-text-secondary">Discoveries</span>
            <span className="ticker-value text-xs font-bold text-purple">{discoveries}</span>
          </div>
          <div className="rounded-lg px-2.5 py-1.5 border border-purple/15 bg-purple/5 flex justify-between items-center">
            <span className="text-[10px] text-text-secondary">Strong</span>
            <span className="ticker-value text-xs font-bold text-purple">{strong}</span>
          </div>
        </div>

        {/* Pipeline */}
        <div className="space-y-1.5">
          <div className="text-[9px] font-semibold uppercase tracking-wider text-warning">Pipeline</div>
          <div className="rounded-lg px-2.5 py-1.5 border border-warning/15 bg-warning/5 flex justify-between items-center">
            <span className="text-[10px] text-text-secondary">Runs</span>
            <span className="ticker-value text-xs font-bold text-warning">{runs}</span>
          </div>
          <div className="rounded-lg px-2.5 py-1.5 border border-warning/15 bg-warning/5 flex justify-between items-center">
            <span className="text-[10px] text-text-secondary">Macro</span>
            <span className="ticker-value text-xs font-bold text-warning">{macroNodes}</span>
          </div>
        </div>
      </div>
    </motion.div>
  )
}


// ── Sector stress heatmap ───────────────────────────────────────
function SectorStress({ analysis }) {
  const sectors = analysis?.sector_stress?.sectors
  if (!sectors?.length) return null

  const maxStress = Math.max(...sectors.map(s => s.stress_score))

  function stressColor(score) {
    if (score >= 0.6) return { bg: 'rgba(239,68,68,0.12)', border: 'rgba(239,68,68,0.3)', text: '#f87171' }
    if (score >= 0.35) return { bg: 'rgba(245,158,11,0.12)', border: 'rgba(245,158,11,0.3)', text: '#fbbf24' }
    return { bg: 'rgba(16,185,129,0.08)', border: 'rgba(16,185,129,0.2)', text: '#34d399' }
  }

  return (
    <motion.div {...fade(0.15)} className="glass-card rounded-xl border border-border p-5">
      <div className="flex items-center gap-2 mb-4">
        <div className="w-7 h-7 rounded-lg bg-negative/10 border border-negative/20 flex items-center justify-center">
          <Layers size={13} className="text-negative" />
        </div>
        <span className="text-xs font-semibold text-text uppercase tracking-wider">Sector Stress</span>
        <span className="text-[10px] text-text-muted ml-auto">{sectors.length} sectors</span>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-3 gap-2">
        {sectors.map(s => {
          const c = stressColor(s.stress_score)
          return (
            <div key={s.sector} className="rounded-lg px-3 py-2.5 transition-all hover:scale-[1.02]"
              style={{ background: c.bg, border: `1px solid ${c.border}` }}>
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-[10px] font-semibold text-text truncate">{s.sector}</span>
                <span className="ticker-value text-[10px] font-bold" style={{ color: c.text }}>
                  {(s.stress_score * 100).toFixed(0)}%
                </span>
              </div>
              <div className="w-full h-1 rounded-full bg-border/50 overflow-hidden">
                <div className="h-full rounded-full transition-all" style={{
                  width: `${Math.min(100, (s.stress_score / Math.max(maxStress, 0.01)) * 100)}%`,
                  background: c.text,
                  opacity: 0.7,
                }} />
              </div>
              <div className="flex justify-between mt-1.5">
                <span className="text-[9px] text-text-muted">{s.asset_count} assets</span>
                {s.divergence_breadth != null && (
                  <span className="text-[9px] text-text-muted">div {(s.divergence_breadth * 100).toFixed(0)}%</span>
                )}
              </div>
            </div>
          )
        })}
      </div>
    </motion.div>
  )
}


// ── Earnings calendar ───────────────────────────────────────────
function EarningsCalendar({ analysis }) {
  const earnings = analysis?.earnings_exposure
  if (!earnings) return null

  const upcoming = earnings.upcoming ?? []
  const recent   = earnings.recent_actuals ?? []

  if (!upcoming.length && !recent.length) return null

  return (
    <motion.div {...fade(0.2)} className="glass-card rounded-xl border border-border p-5">
      <div className="flex items-center gap-2 mb-4">
        <div className="w-7 h-7 rounded-lg bg-cyan/10 border border-cyan/20 flex items-center justify-center">
          <Calendar size={13} className="text-cyan" />
        </div>
        <span className="text-xs font-semibold text-text uppercase tracking-wider">Earnings Events</span>
        <span className="text-[10px] text-text-muted ml-auto">
          {upcoming.length} upcoming
        </span>
      </div>

      {upcoming.length > 0 && (
        <div className="space-y-1.5 mb-3">
          <div className="text-[9px] font-semibold text-text-muted uppercase tracking-wider">Upcoming</div>
          {upcoming.slice(0, 6).map((e, i) => (
            <div key={i} className="flex items-center gap-3 rounded-lg px-2.5 py-2 border border-border/50 bg-bg-hover/30">
              <span className="ticker-value text-[10px] font-bold text-text w-12">{e.asset}</span>
              <span className="text-[10px] text-text-muted flex-1">{e.event_date}</span>
              {e.importance_score != null && (
                <div className="flex items-center gap-1">
                  <Target size={9} className="text-cyan" />
                  <span className="ticker-value text-[10px] text-cyan">{(e.importance_score * 100).toFixed(0)}</span>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {recent.length > 0 && (
        <div className="space-y-1.5">
          <div className="text-[9px] font-semibold text-text-muted uppercase tracking-wider">Recent Results</div>
          {recent.slice(0, 4).map((e, i) => (
            <div key={i} className="flex items-center gap-3 rounded-lg px-2.5 py-2 border border-border/50 bg-bg-hover/30">
              <span className="ticker-value text-[10px] font-bold text-text w-12">{e.asset}</span>
              <span className="text-[10px] text-text-muted flex-1">{e.event_date}</span>
              {e.eps_surprise_pct != null && (
                <span className={`ticker-value text-[10px] font-bold ${e.eps_surprise_pct >= 0 ? 'text-positive' : 'text-negative'}`}>
                  {e.eps_surprise_pct >= 0 ? '+' : ''}{e.eps_surprise_pct.toFixed(1)}%
                </span>
              )}
            </div>
          ))}
        </div>
      )}
    </motion.div>
  )
}


// ── Structural insights ─────────────────────────────────────────
const PRIORITY_COLORS = {
  1: { border: 'border-negative/25', bg: 'bg-negative/6', badge: 'bg-negative/15 text-negative', dot: '#ef4444' },
  2: { border: 'border-warning/25',  bg: 'bg-warning/6',  badge: 'bg-warning/15 text-warning', dot: '#f59e0b' },
  3: { border: 'border-accent/25',   bg: 'bg-accent/6',   badge: 'bg-accent/15 text-accent', dot: '#3b82f6' },
}

function InsightIcon({ type }) {
  if (type.includes('anomaly'))  return <AlertTriangle size={11} />
  if (type.includes('bridge'))   return <Zap size={11} />
  if (type.includes('sector'))   return <Layers size={11} />
  if (type.includes('earning'))  return <Calendar size={11} />
  return <TrendingUp size={11} />
}

function Insights({ intelligence }) {
  const insights = intelligence?.insights
  if (!insights?.length) return null

  const anomalyResult = intelligence?.anomalies
  const hasAnomalies  = anomalyResult?.anomalies?.length > 0

  return (
    <motion.div {...fade(0.1)} className="glass-card rounded-xl border border-border p-5">
      <div className="flex items-center gap-2 mb-4">
        <div className="w-7 h-7 rounded-lg bg-warning/10 border border-warning/20 flex items-center justify-center">
          <AlertTriangle size={13} className="text-warning" />
        </div>
        <span className="text-xs font-semibold text-text uppercase tracking-wider">Structural Insights</span>
        <span className="text-[10px] text-text-muted ml-auto">{insights.length} findings</span>
      </div>

      {hasAnomalies && (
        <div className="rounded-lg border border-negative/25 bg-negative/6 px-3 py-2 mb-3 flex items-center gap-2">
          <AlertTriangle size={12} className="text-negative flex-shrink-0" />
          <span className="text-[10px] text-text-secondary">
            <strong className="text-negative">{anomalyResult.anomalies.length} anomal{anomalyResult.anomalies.length === 1 ? 'y' : 'ies'}</strong> detected
            <span className="text-text-muted ml-1">(depth: {anomalyResult.history_depth} snapshots)</span>
          </span>
        </div>
      )}

      <div className="space-y-1.5">
        {insights.slice(0, 10).map((ins, i) => {
          const c = PRIORITY_COLORS[ins.priority] ?? PRIORITY_COLORS[3]
          return (
            <div key={i} className={`rounded-lg border ${c.border} ${c.bg} px-3 py-2`}>
              <div className="flex items-start gap-2">
                <span className={`${c.badge} mt-0.5 p-0.5 rounded`}><InsightIcon type={ins.type} /></span>
                <div className="min-w-0 flex-1">
                  <div className="text-[11px] font-medium text-text">{ins.title}</div>
                  <div className="text-[10px] text-text-muted mt-0.5 leading-relaxed">{ins.detail}</div>
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
          vs snapshot from {new Date(intelligence.previous_timestamp).toLocaleDateString()}
        </div>
      )}
    </motion.div>
  )
}


// ── Service health ──────────────────────────────────────────────
function ServiceHealth({ checks }) {
  const entries = Object.entries(checks)
  if (!entries.length) return null
  const up = entries.filter(([, v]) => v).length

  return (
    <motion.div {...fade(0.25)} className="glass-card rounded-xl border border-border p-4">
      <div className="flex items-center gap-2 mb-3">
        <Shield size={12} className="text-positive" />
        <span className="text-[10px] font-semibold text-text uppercase tracking-wider">Services</span>
        <span className="text-[10px] text-text-muted ml-auto">{up}/{entries.length}</span>
      </div>
      <div className="grid grid-cols-2 gap-1.5">
        {entries.map(([k, v]) => (
          <div key={k} className={`flex items-center gap-1.5 text-[10px] px-2 py-1 rounded border ${
            v ? 'border-positive/15 bg-positive/5 text-positive' : 'border-negative/15 bg-negative/5 text-negative'
          }`}>
            <div className={`w-1.5 h-1.5 rounded-full ${v ? 'bg-positive' : 'bg-negative'}`} />
            {k}
          </div>
        ))}
      </div>
    </motion.div>
  )
}


// ── Graph reasoning summary ─────────────────────────────────────
function ReasoningSummary({ intelligence }) {
  const reasoning = intelligence?.reasoning
  if (!reasoning) return null

  const health = reasoning.structural_health
  const topFactor = reasoning.most_influential_factor
  const topExposed = reasoning.most_exposed_asset
  const stressed = reasoning.most_stressed_sector

  const gradeColors = {
    excellent: '#10b981', good: '#3b82f6', fair: '#f59e0b', poor: '#ef4444',
  }

  return (
    <motion.div {...fade(0.25)} className="glass-card rounded-xl border border-border p-4 space-y-3">
      <div className="flex items-center gap-2">
        <Activity size={12} className="text-accent" />
        <span className="text-[10px] font-semibold text-text uppercase tracking-wider">Graph Health</span>
      </div>

      {health && (
        <div className="space-y-1.5">
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-text-secondary">Overall</span>
            <span className="ticker-value text-xs font-bold" style={{ color: gradeColors[health.grade] ?? '#94a3b8' }}>
              {(health.score * 100).toFixed(0)}% {health.grade}
            </span>
          </div>
          {Object.entries(health.components ?? {}).map(([k, v]) => (
            <div key={k} className="flex items-center gap-1.5">
              <span className="text-[9px] text-text-muted w-20 truncate capitalize">{k.replace(/_/g, ' ')}</span>
              <div className="flex-1 h-1 bg-border/50 rounded-full overflow-hidden">
                <div className="h-full bg-accent/60 rounded-full" style={{ width: `${Math.round(v * 100)}%` }} />
              </div>
              <span className="text-[9px] ticker-value text-text-secondary w-7 text-right">{(v * 100).toFixed(0)}%</span>
            </div>
          ))}
        </div>
      )}

      {topFactor && (
        <div className="pt-2 border-t border-border/50">
          <div className="text-[9px] text-text-muted mb-0.5">Top Bridge Factor</div>
          <div className="text-[10px] text-text font-medium">{topFactor.factor_id}</div>
          <div className="text-[9px] text-text-muted">{topFactor.asset_count} assets, {topFactor.class_count} classes</div>
        </div>
      )}

      {topExposed && (
        <div className="pt-2 border-t border-border/50">
          <div className="text-[9px] text-text-muted mb-0.5">Most Exposed</div>
          <div className="text-[10px] text-text font-medium">{topExposed.asset}</div>
        </div>
      )}

      {stressed && (
        <div className="pt-2 border-t border-border/50">
          <div className="text-[9px] text-text-muted mb-0.5">Most Stressed Sector</div>
          <div className="text-[10px] font-medium" style={{ color: stressed.stress_score >= 0.5 ? '#f87171' : '#fbbf24' }}>
            {stressed.sector}
          </div>
          <div className="text-[9px] text-text-muted">{stressed.narrative}</div>
        </div>
      )}
    </motion.div>
  )
}


// ── Provenance ──────────────────────────────────────────────────
function ProvenanceSummary({ provenance }) {
  if (!provenance || Object.keys(provenance).length === 0) return null

  const entries = Object.entries(provenance)
  const bySrc = {}
  for (const [, info] of entries) {
    const src = info.source_name ?? info.source_id ?? 'unknown'
    bySrc[src] = (bySrc[src] ?? 0) + 1
  }

  return (
    <motion.div {...fade(0.3)} className="glass-card rounded-xl border border-border p-4">
      <div className="flex items-center gap-2 mb-3">
        <Globe size={12} className="text-accent" />
        <span className="text-[10px] font-semibold text-text uppercase tracking-wider">Provenance</span>
        <span className="text-[10px] text-text-muted ml-auto">{entries.length} series</span>
      </div>
      <div className="flex flex-wrap gap-1.5">
        {Object.entries(bySrc).map(([src, count]) => (
          <div key={src} className="rounded-lg px-2.5 py-1 border border-border bg-bg-hover text-[10px] text-text-secondary">
            <span className="font-semibold text-text">{count}</span> {src}
          </div>
        ))}
      </div>
    </motion.div>
  )
}


// ── Main page ───────────────────────────────────────────────────
export default function Overview() {
  const { data: health }       = useQuery({ queryKey: ['health'],              queryFn: fetchHealth,            staleTime: 30_000 })
  const { data: srcSummary }   = useQuery({ queryKey: ['sources-summary'],     queryFn: fetchSourceSummary,     staleTime: 60_000 })
  const { data: discSummary }  = useQuery({ queryKey: ['discoveries-summary'], queryFn: fetchDiscoverySummary,  staleTime: 60_000 })
  const { data: graphStats }   = useQuery({ queryKey: ['graph-stats'],         queryFn: fetchGraphStats,        staleTime: 60_000 })
  const { data: intelligence } = useQuery({ queryKey: ['graph-intelligence'],  queryFn: fetchGraphIntelligence, staleTime: 300_000 })
  const { data: analysis }     = useQuery({ queryKey: ['graph-analysis'],      queryFn: () => fetchGraphAnalysis(20), staleTime: 300_000 })

  const checks = health?.checks ?? {}

  return (
    <div className="flex flex-col h-full">
      <Header title="Command Center" subtitle="Market graph state and structural intelligence" />

      <div className="flex-1 overflow-y-auto p-5 space-y-4 bg-grid">
        {/* Pipeline status */}
        <PipelineStatus srcSummary={srcSummary} discSummary={discSummary} graphStats={graphStats} />

        {/* Main grid: insights + sector stress + earnings + health */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
          {/* Insights — main column */}
          <div className="lg:col-span-5">
            <Insights intelligence={intelligence} />
          </div>

          {/* Sector stress */}
          <div className="lg:col-span-4">
            <SectorStress analysis={analysis} />
          </div>

          {/* Right sidebar: health + reasoning + provenance */}
          <div className="lg:col-span-3 space-y-4">
            <ReasoningSummary intelligence={intelligence} />
            <ServiceHealth checks={checks} />
            <ProvenanceSummary provenance={intelligence?.provenance} />
          </div>
        </div>

        {/* Earnings calendar */}
        <EarningsCalendar analysis={analysis} />
      </div>
    </div>
  )
}
