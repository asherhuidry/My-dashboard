import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { fetchDbStats, fetchHealth, fetchPrices, fetchSourceSummary, fetchDiscoverySummary } from '../lib/api'
import { fmt, pctColor } from '../lib/utils'
import SearchBar from '../components/UI/SearchBar'
import Header from '../components/Layout/Header'
import {
  Database, TrendingUp, TrendingDown, Activity, Layers,
  Brain, Cpu, Network, BarChart2, Shield, Clock, CheckCircle2, XCircle,
  Globe, Compass,
} from 'lucide-react'
import { motion } from 'framer-motion'

const WATCHLIST = ['AAPL','NVDA','MSFT','TSLA','META','AMZN','BTC-USD','ETH-USD','SOL-USD','SPY']


const fade = (delay = 0) => ({
  initial: { opacity: 0, y: 16 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.4, delay },
})

function WatchCard({ symbol }) {
  const navigate = useNavigate()
  const { data, isLoading } = useQuery({
    queryKey: ['prices', symbol, 7],
    queryFn: () => fetchPrices(symbol, 7),
    staleTime: 60_000,
  })

  const label = symbol.replace('-USD','').replace('=X','')
  const isCrypto = symbol.includes('-USD')
  const up = data ? data.change_pct >= 0 : null

  return (
    <motion.div
      whileHover={{ y: -2, scale: 1.01 }}
      onClick={() => navigate(`/analyze?symbol=${symbol}`)}
      className="glass-card rounded-xl p-3.5 border border-border hover:border-border-bright
        cursor-pointer transition-all duration-200 relative overflow-hidden group"
    >
      {/* Hover glow */}
      <div className={`absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none
        bg-gradient-to-br ${up === null ? 'from-accent/4' : up ? 'from-positive/4' : 'from-negative/4'} to-transparent`} />

      <div className="flex items-start justify-between mb-2">
        <div>
          <div className="text-xs font-bold text-text">{label}</div>
          <div className="text-[9px] text-text-muted uppercase tracking-wider mt-0.5">
            {isCrypto ? 'Crypto' : symbol.includes('=X') ? 'Forex' : 'Equity'}
          </div>
        </div>
        {isLoading
          ? <div className="w-3 h-3 border border-accent/50 border-t-transparent rounded-full animate-spin mt-0.5" />
          : up != null && (
            <div className={`flex items-center gap-0.5 text-[9px] font-bold px-1.5 py-0.5 rounded-md ${
              up ? 'bg-positive/12 text-positive border border-positive/20' : 'bg-negative/12 text-negative border border-negative/20'
            }`}>
              {up ? <TrendingUp size={8} /> : <TrendingDown size={8} />}
              {fmt.pct(data.change_pct)}
            </div>
          )
        }
      </div>

      {data ? (
        <div className="ticker-value text-base font-bold text-text">
          ${fmt.price(data.latest)}
        </div>
      ) : (
        <div className="shimmer h-5 w-16 rounded mt-1" />
      )}
    </motion.div>
  )
}

function StatCard({ label, value, sub, icon: Icon, color, delay }) {
  const colorMap = {
    accent:   { bg: 'bg-accent/10',   border: 'border-accent/20',   icon: 'text-accent',   val: 'text-accent'   },
    positive: { bg: 'bg-positive/10', border: 'border-positive/20', icon: 'text-positive', val: 'text-positive' },
    purple:   { bg: 'bg-purple/10',   border: 'border-purple/20',   icon: 'text-purple',   val: 'text-purple'   },
    warning:  { bg: 'bg-warning/10',  border: 'border-warning/20',  icon: 'text-warning',  val: 'text-warning'  },
    negative: { bg: 'bg-negative/10', border: 'border-negative/20', icon: 'text-negative', val: 'text-negative' },
    cyan:     { bg: 'bg-cyan/10',     border: 'border-cyan/20',     icon: 'text-cyan',     val: 'text-cyan'     },
  }
  const c = colorMap[color] ?? colorMap.accent

  return (
    <motion.div {...fade(delay)}
      className={`glass-card rounded-xl p-4 border ${c.border} relative overflow-hidden group hover:scale-[1.01] transition-transform`}>
      <div className={`absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity ${c.bg} pointer-events-none`} />
      <div className="relative">
        <div className="flex items-center justify-between mb-3">
          <div className={`w-8 h-8 rounded-lg ${c.bg} border ${c.border} flex items-center justify-center`}>
            <Icon size={15} className={c.icon} />
          </div>
          <div className="w-1.5 h-1.5 rounded-full bg-positive pulse-dot" />
        </div>
        <div className={`ticker-value text-2xl font-bold ${c.val} leading-none`}>{value}</div>
        <div className="text-xs text-text-muted mt-1">{label}</div>
        {sub && <div className="text-[10px] text-text-muted/60 mt-0.5">{sub}</div>}
      </div>
    </motion.div>
  )
}

function ServiceRow({ label, ok, latency }) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-border/50 last:border-0">
      <div className="flex items-center gap-2">
        {ok
          ? <CheckCircle2 size={12} className="text-positive" />
          : <XCircle size={12} className="text-negative" />
        }
        <span className="text-xs text-text-secondary">{label}</span>
      </div>
      <div className="flex items-center gap-2">
        {latency != null && (
          <span className="ticker-value text-[10px] text-text-muted">{latency}ms</span>
        )}
        <span className={`text-[9px] font-semibold uppercase px-1.5 py-0.5 rounded ${
          ok ? 'bg-positive/10 text-positive border border-positive/20'
             : 'bg-negative/10 text-negative border border-negative/20'
        }`}>{ok ? 'OK' : 'Down'}</span>
      </div>
    </div>
  )
}

function GraphStatusVisual({ srcSummary, discSummary }) {
  const sources = srcSummary?.total ?? 0
  const validated = srcSummary?.by_status?.validated ?? 0
  const discoveries = discSummary?.total_discoveries ?? 0
  const strong = discSummary?.by_strength?.strong ?? 0
  const uniqueSeries = discSummary?.unique_series ?? 0
  const runs = discSummary?.run_count ?? 0

  const stages = [
    { label: 'Sources',     color: '#3b82f6', items: [
      { k: 'Registered', v: sources },
      { k: 'Validated',  v: validated },
      { k: 'Categories', v: Object.keys(srcSummary?.by_category ?? {}).length },
    ]},
    { label: 'Nodes',       color: '#10b981', items: [
      { k: 'Unique series', v: uniqueSeries },
      { k: 'Asset classes', v: (srcSummary?.by_category ? Object.keys(srcSummary.by_category).length : 0) },
    ]},
    { label: 'Edges',       color: '#8b5cf6', items: [
      { k: 'Discoveries',  v: discoveries },
      { k: 'Strong',       v: strong },
      { k: 'Runs',         v: runs },
    ]},
    { label: 'Research',    color: '#f59e0b', items: [
      { k: 'Analyzer',    v: 'live' },
      { k: 'Screener',    v: 'live' },
      { k: 'AI Research',  v: 'live' },
    ]},
  ]

  return (
    <div className="glass-card rounded-xl border border-border p-5">
      <div className="flex items-center gap-2 mb-5">
        <div className="w-7 h-7 rounded-lg bg-accent/10 border border-accent/20 flex items-center justify-center">
          <Network size={13} className="text-accent" />
        </div>
        <span className="text-xs font-semibold text-text uppercase tracking-wider">Market Graph Status</span>
      </div>
      <div className="flex items-stretch gap-2 overflow-x-auto pb-1">
        {stages.map((stage, si) => (
          <div key={stage.label} className="flex-1 min-w-[120px]">
            <div className="text-[9px] font-semibold uppercase tracking-wider mb-2"
              style={{ color: stage.color }}>{stage.label}</div>
            <div className="space-y-1.5">
              {stage.items.map(item => (
                <div key={item.k}
                  className="rounded-lg px-2.5 py-1.5 text-[10px] font-medium text-text-secondary
                    border transition-all hover:text-text flex items-center justify-between"
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
      <div className="flex items-center justify-around mt-3 px-1">
        {stages.slice(0,-1).map((_, i) => (
          <div key={i} className="flex-1 flex items-center justify-end pr-1 text-text-muted/40 text-xs">→</div>
        ))}
      </div>
    </div>
  )
}

export default function Overview() {
  const { data: stats }  = useQuery({ queryKey: ['db-stats'], queryFn: fetchDbStats, staleTime: 60_000 })
  const { data: health } = useQuery({ queryKey: ['health'],   queryFn: fetchHealth,  staleTime: 30_000 })
  const { data: srcSummary }  = useQuery({ queryKey: ['sources-summary'],     queryFn: fetchSourceSummary,    staleTime: 60_000 })
  const { data: discSummary } = useQuery({ queryKey: ['discoveries-summary'], queryFn: fetchDiscoverySummary, staleTime: 60_000 })

  const priceRows  = stats?.tables?.find(t => t.table === 'prices')?.rows ?? 0
  const macroRows  = stats?.tables?.find(t => t.table === 'macro_events')?.rows ?? 0
  const totalRows  = stats?.total_rows ?? 0
  const checks     = health?.checks ?? {}
  const servicesUp = Object.values(checks).filter(Boolean).length
  const totalSvc   = Object.keys(checks).length || 5

  return (
    <div className="flex flex-col h-full">
      <Header title="Overview" subtitle="FinBrain Market Graph Engine" />

      <div className="flex-1 overflow-y-auto bg-grid p-5 space-y-5">

        {/* ── Hero ──────────────────────────────────────────────────── */}
        <motion.div {...fade(0)}
          className="relative rounded-2xl overflow-hidden border border-accent/20"
          style={{
            background: 'linear-gradient(135deg, rgba(14,28,54,0.95) 0%, rgba(8,18,36,0.98) 60%, rgba(18,12,40,0.95) 100%)',
          }}
        >
          {/* Background pattern */}
          <div className="absolute inset-0 bg-grid opacity-60 pointer-events-none" />
          <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-accent/50 to-transparent" />
          <div className="absolute top-0 left-1/4 w-64 h-64 rounded-full bg-accent/5 blur-3xl pointer-events-none" />
          <div className="absolute top-0 right-1/4 w-48 h-48 rounded-full bg-purple/5 blur-3xl pointer-events-none" />

          <div className="relative z-10 p-8 text-center">
            <div className="inline-flex items-center gap-1.5 text-[9px] text-accent uppercase tracking-[0.15em]
              mb-4 px-3 py-1.5 rounded-full border border-accent/25 bg-accent/8">
              <Network size={9} /> Market Graph Engine
            </div>
            <h2 className="text-3xl font-bold text-text mb-2 leading-tight">
              <span className="gradient-text">Sources</span>{' · '}
              <span className="gradient-text-green">Nodes</span>{' · '}
              <span className="gradient-text">Edges</span>
            </h2>
            <p className="text-text-secondary text-sm mb-7 max-w-md mx-auto">
              Governed data sources · Persisted correlations · Evolving market structure
            </p>
            <div className="flex justify-center">
              <SearchBar placeholder="Search any asset: AAPL, BTC-USD, SPY..." className="w-full max-w-sm" />
            </div>
          </div>
        </motion.div>

        {/* ── Stat cards ────────────────────────────────────────────── */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          <StatCard label="Price Rows"      value={fmt.compact(priceRows)}          icon={BarChart2}   color="accent"    sub="TimescaleDB" delay={0.05} />
          <StatCard label="Macro Records"   value={fmt.compact(macroRows)}          icon={Activity}    color="positive"  sub="FRED events"  delay={0.10} />
          <StatCard label="Total DB Rows"   value={fmt.compact(totalRows)}          icon={Database}    color="purple"    sub="All tables"   delay={0.15} />
          <StatCard label="Services Online" value={`${servicesUp}/${totalSvc}`}     icon={Shield}
            color={servicesUp === totalSvc ? 'positive' : servicesUp > 0 ? 'warning' : 'negative'}
            sub={servicesUp === totalSvc ? 'All healthy' : 'Degraded'}
            delay={0.20} />
        </div>

        {/* ── Watchlist ─────────────────────────────────────────────── */}
        <motion.div {...fade(0.1)}>
          <div className="flex items-center gap-2 mb-3">
            <TrendingUp size={13} className="text-accent" />
            <h3 className="text-xs font-semibold text-text uppercase tracking-wider">Watchlist</h3>
            <span className="text-[9px] text-text-muted ml-1">Click to analyze</span>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-5 lg:grid-cols-10 gap-2">
            {WATCHLIST.map(sym => <WatchCard key={sym} symbol={sym} />)}
          </div>
        </motion.div>

        {/* ── Middle row: data flow + service health ─────────────────── */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <div className="lg:col-span-2">
            <GraphStatusVisual srcSummary={srcSummary} discSummary={discSummary} />
          </div>

          {/* Service health */}
          <motion.div {...fade(0.15)} className="glass-card rounded-xl border border-border p-5">
            <div className="flex items-center gap-2 mb-4">
              <div className="w-7 h-7 rounded-lg bg-positive/10 border border-positive/20 flex items-center justify-center">
                <Shield size={13} className="text-positive" />
              </div>
              <span className="text-xs font-semibold text-text uppercase tracking-wider">Service Health</span>
            </div>
            <div className="space-y-0">
              {Object.entries(checks).length > 0
                ? Object.entries(checks).map(([k, v]) => (
                    <ServiceRow key={k} label={k} ok={v} />
                  ))
                : ['Supabase','Qdrant','Neo4j','yFinance','API'].map(s => (
                    <ServiceRow key={s} label={s} ok={null} />
                  ))
              }
            </div>
          </motion.div>
        </div>

        {/* ── DB Table Health ───────────────────────────────────────── */}
        {stats?.tables && (
          <motion.div {...fade(0.2)}>
            <div className="flex items-center gap-2 mb-3">
              <Database size={13} className="text-accent" />
              <h3 className="text-xs font-semibold text-text uppercase tracking-wider">Database Tables</h3>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
              {stats.tables.map((t, i) => (
                <motion.div key={t.table} {...fade(0.2 + i * 0.03)}
                  className="glass rounded-xl p-3.5 border border-border hover:border-border-bright transition-all group">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-[9px] text-text-muted uppercase tracking-wider font-semibold truncate pr-1">
                      {t.table.replace(/_/g,' ')}
                    </span>
                    <div className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${t.status === 'ok' ? 'bg-positive' : 'bg-negative'}`} />
                  </div>
                  <div className="ticker-value text-lg font-bold text-text group-hover:text-accent transition-colors">
                    {fmt.compact(t.rows)}
                  </div>
                  <div className="text-[9px] text-text-muted mt-0.5">rows</div>
                  {/* Mini fill bar */}
                  <div className="mt-2 h-0.5 rounded-full bg-border overflow-hidden">
                    <div className="h-full bg-accent/40 rounded-full"
                      style={{ width: `${Math.min(100, (t.rows / (stats.total_rows || 1)) * 100 * 3)}%` }} />
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}

        {/* ── Bottom: system info strip ─────────────────────────────── */}
        <motion.div {...fade(0.25)}
          className="glass rounded-xl border border-border p-4 flex items-center flex-wrap gap-6">
          <div className="flex items-center gap-2">
            <Globe size={14} className="text-accent" />
            <span className="text-[10px] text-text-muted">Sources:</span>
            <span className="text-[10px] text-text-secondary font-medium">{srcSummary?.total ?? 0} registered · {srcSummary?.by_status?.validated ?? 0} validated</span>
          </div>
          <div className="flex items-center gap-2">
            <Compass size={14} className="text-purple" />
            <span className="text-[10px] text-text-muted">Edges:</span>
            <span className="text-[10px] text-text-secondary font-medium">{discSummary?.total_discoveries ?? 0} discoveries · {discSummary?.by_strength?.strong ?? 0} strong</span>
          </div>
          <div className="flex items-center gap-2">
            <Clock size={14} className="text-positive" />
            <span className="text-[10px] text-text-muted">Refresh:</span>
            <span className="text-[10px] text-text-secondary font-medium">30s prices · 60s graph state</span>
          </div>
          <div className="flex items-center gap-2">
            <Layers size={14} className="text-cyan" />
            <span className="text-[10px] text-text-muted">Coverage:</span>
            <span className="text-[10px] text-text-secondary font-medium">Equities · Crypto · Macro · Forex</span>
          </div>
        </motion.div>

      </div>
    </div>
  )
}
