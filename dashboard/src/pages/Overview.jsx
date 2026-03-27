import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { fetchDbStats, fetchHealth, fetchPrices } from '../lib/api'
import { fmt, pctColor } from '../lib/utils'
import SearchBar from '../components/UI/SearchBar'
import Header from '../components/Layout/Header'
import {
  Database, TrendingUp, TrendingDown, Activity, Layers, Zap,
  Brain, Cpu, Network, BarChart2, Shield, Clock, CheckCircle2, XCircle,
} from 'lucide-react'
import { motion } from 'framer-motion'

const WATCHLIST = ['AAPL','NVDA','MSFT','TSLA','META','AMZN','BTC-USD','ETH-USD','SOL-USD','SPY']

const SYSTEM_NODES = [
  { id: 'yfinance',  label: 'yFinance',   type: 'source',   color: '#3b82f6' },
  { id: 'fred',      label: 'FRED',       type: 'source',   color: '#8b5cf6' },
  { id: 'alpha',     label: 'AlphaV',     type: 'source',   color: '#06b6d4' },
  { id: 'timescale', label: 'TimescaleDB',type: 'database', color: '#10b981' },
  { id: 'supabase',  label: 'Supabase',   type: 'database', color: '#34d399' },
  { id: 'qdrant',    label: 'Qdrant',     type: 'vector',   color: '#f59e0b' },
  { id: 'neo4j',     label: 'Neo4j',      type: 'graph',    color: '#a78bfa' },
  { id: 'lstm',      label: 'LSTM',       type: 'model',    color: '#ef4444' },
  { id: 'features',  label: 'Features',   type: 'pipeline', color: '#3b82f6' },
  { id: 'signals',   label: 'Signals',    type: 'output',   color: '#10b981' },
]

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

function DataFlowVisual() {
  const layers = [
    { label: 'Data Sources', color: '#3b82f6', items: ['yFinance', 'FRED', 'AlphaV', 'CoinGecko'] },
    { label: 'Storage',      color: '#10b981', items: ['TimescaleDB', 'Supabase', 'Qdrant', 'Neo4j'] },
    { label: 'ML Pipeline',  color: '#8b5cf6', items: ['Features', 'LSTM', 'Signals', 'Backtest'] },
    { label: 'Output',       color: '#f59e0b', items: ['API', 'Dashboard', 'Alerts', 'GitHub'] },
  ]
  return (
    <div className="glass-card rounded-xl border border-border p-5">
      <div className="flex items-center gap-2 mb-5">
        <div className="w-7 h-7 rounded-lg bg-accent/10 border border-accent/20 flex items-center justify-center">
          <Network size={13} className="text-accent" />
        </div>
        <span className="text-xs font-semibold text-text uppercase tracking-wider">Data Flow Architecture</span>
      </div>
      <div className="flex items-stretch gap-2 overflow-x-auto pb-1">
        {layers.map((layer, li) => (
          <div key={layer.label} className="flex-1 min-w-[120px]">
            <div className="text-[9px] font-semibold uppercase tracking-wider mb-2"
              style={{ color: layer.color }}>{layer.label}</div>
            <div className="space-y-1.5">
              {layer.items.map(item => (
                <div key={item}
                  className="rounded-lg px-2.5 py-1.5 text-[10px] font-medium text-text-secondary
                    border transition-all hover:text-text"
                  style={{ background: `${layer.color}0d`, borderColor: `${layer.color}25` }}>
                  {item}
                </div>
              ))}
            </div>
            {/* Arrow between layers */}
            {li < layers.length - 1 && (
              <div className="hidden" /> /* spacer; arrows rendered via grid gap */
            )}
          </div>
        ))}
      </div>
      {/* Flow arrows */}
      <div className="flex items-center justify-around mt-3 px-1">
        {layers.slice(0,-1).map((_, i) => (
          <div key={i} className="flex-1 flex items-center justify-end pr-1 text-text-muted/40 text-xs">→</div>
        ))}
      </div>
    </div>
  )
}

export default function Overview() {
  const { data: stats }  = useQuery({ queryKey: ['db-stats'], queryFn: fetchDbStats, staleTime: 60_000 })
  const { data: health } = useQuery({ queryKey: ['health'],   queryFn: fetchHealth,  staleTime: 30_000 })

  const priceRows  = stats?.tables?.find(t => t.table === 'prices')?.rows ?? 0
  const macroRows  = stats?.tables?.find(t => t.table === 'macro_events')?.rows ?? 0
  const totalRows  = stats?.total_rows ?? 0
  const checks     = health?.checks ?? {}
  const servicesUp = Object.values(checks).filter(Boolean).length
  const totalSvc   = Object.keys(checks).length || 5

  return (
    <div className="flex flex-col h-full">
      <Header title="Overview" subtitle="FinBrain Autonomous Financial Intelligence" />

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
              <Zap size={9} /> AI-Powered Financial Intelligence
            </div>
            <h2 className="text-3xl font-bold text-text mb-2 leading-tight">
              Analyze any{' '}
              <span className="gradient-text">stock</span>{' '}or{' '}
              <span className="gradient-text-green">crypto</span>
            </h2>
            <p className="text-text-secondary text-sm mb-7 max-w-md mx-auto">
              74-feature ML pipeline · Real-time OHLCV · Signal consensus · LSTM predictions
            </p>
            <div className="flex justify-center">
              <SearchBar placeholder="Try AAPL, NVDA, BTC-USD, ETH-USD, MSFT…" className="w-full max-w-sm" />
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
            <DataFlowVisual />
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
            <Brain size={14} className="text-purple" />
            <span className="text-[10px] text-text-muted">Model:</span>
            <span className="text-[10px] text-text-secondary font-medium">Claude Haiku (AI Chat)</span>
          </div>
          <div className="flex items-center gap-2">
            <Cpu size={14} className="text-accent" />
            <span className="text-[10px] text-text-muted">ML:</span>
            <span className="text-[10px] text-text-secondary font-medium">FinBrain LSTM · 74 features</span>
          </div>
          <div className="flex items-center gap-2">
            <Clock size={14} className="text-positive" />
            <span className="text-[10px] text-text-muted">Refresh:</span>
            <span className="text-[10px] text-text-secondary font-medium">30s prices · 60s stats</span>
          </div>
          <div className="flex items-center gap-2">
            <Layers size={14} className="text-cyan" />
            <span className="text-[10px] text-text-muted">Assets:</span>
            <span className="text-[10px] text-text-secondary font-medium">Stocks · Crypto · Forex · Commodities</span>
          </div>
        </motion.div>

      </div>
    </div>
  )
}
