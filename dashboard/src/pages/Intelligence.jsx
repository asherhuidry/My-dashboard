import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { motion, AnimatePresence } from 'framer-motion'
import {
  TrendingUp, TrendingDown, Minus, Activity, Globe, Network,
  BarChart2, MessageSquare, Search, ChevronDown, ChevronRight,
  AlertTriangle, CheckCircle, Zap, Brain, Link, ArrowRight,
} from 'lucide-react'
import {
  fetchMacroDashboard, fetchCorrelationsFor, fetchSupplyChain,
  fetchSocialIntel, fetchGraphStats,
} from '../lib/api'

// ── Macro regime badge ─────────────────────────────────────────────────────────
const REGIME_COLORS = {
  goldilocks:    { bg: 'rgba(34,197,94,0.15)',  border: '#22c55e', text: '#4ade80', icon: TrendingUp },
  reflation:     { bg: 'rgba(234,179,8,0.15)',  border: '#eab308', text: '#facc15', icon: TrendingUp },
  stagflation:   { bg: 'rgba(239,68,68,0.15)',  border: '#ef4444', text: '#f87171', icon: AlertTriangle },
  deflation_risk:{ bg: 'rgba(99,102,241,0.15)', border: '#6366f1', text: '#818cf8', icon: TrendingDown },
  unknown:       { bg: 'rgba(71,85,105,0.15)',  border: '#475569', text: '#94a3b8', icon: Minus },
}

function RegimeBadge({ quadrant }) {
  const cfg = REGIME_COLORS[quadrant] || REGIME_COLORS.unknown
  const Icon = cfg.icon
  return (
    <span
      style={{ background: cfg.bg, border: `1px solid ${cfg.border}`, color: cfg.text }}
      className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-sm font-semibold"
    >
      <Icon size={14} />
      {quadrant?.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()) || 'Unknown'}
    </span>
  )
}

// ── Macro indicator card ───────────────────────────────────────────────────────
function MacroRow({ label, value, format = 'pct', colorize = true }) {
  if (value == null) return null
  const fmt = format === 'pct' ? `${value.toFixed(2)}%`
            : format === 'num' ? value.toFixed(2)
            : format === 'bp'  ? `${(value * 100).toFixed(0)}bp`
            : value.toFixed(2)

  const color = !colorize ? '#94a3b8'
    : value > 0 ? '#4ade80' : '#f87171'

  return (
    <div className="flex items-center justify-between py-1.5 border-b border-white/5 last:border-0">
      <span className="text-xs text-slate-400">{label}</span>
      <span className="text-xs font-mono font-semibold" style={{ color }}>{fmt}</span>
    </div>
  )
}

// ── Macro dashboard panel ──────────────────────────────────────────────────────
function MacroDashboard() {
  const { data, isLoading } = useQuery({
    queryKey: ['macro-dashboard'],
    queryFn:  fetchMacroDashboard,
    staleTime: 5 * 60_000,
  })

  if (isLoading) return (
    <div className="glass rounded-xl p-4 animate-pulse h-64 flex items-center justify-center">
      <span className="text-slate-500 text-sm">Loading macro data...</span>
    </div>
  )

  const regime = data?.regime || {}
  const cats   = data?.categories || {}
  const raw    = data?.raw || {}

  const categoryLabels = {
    rates:       'Interest Rates',
    inflation:   'Inflation',
    credit:      'Credit Stress',
    employment:  'Employment',
    commodities: 'Commodities',
    risk:        'Risk / Volatility',
    fed:         'Fed / Liquidity',
  }

  const seriesLabels = {
    DFF: 'Fed Funds', GS2: '2Y Yield', GS10: '10Y Yield', GS30: '30Y Yield',
    T10Y2Y: '10Y-2Y Spread', T10Y3M: '10Y-3M Spread',
    CPIAUCSL: 'CPI YoY', CPILFESL: 'Core CPI', T10YIE: '10Y Breakeven', T5YIE: '5Y Breakeven', DFII10: '10Y Real',
    BAMLH0A0HYM2: 'HY Spread', BAMLC0A0CM: 'IG Spread', STLFSI4: 'Fin. Stress', NFCI: 'Fin. Conditions',
    UNRATE: 'Unemployment', PAYEMS: 'Payrolls (k)', ICSA: 'Init. Claims', JTSJOL: 'Job Openings',
    DCOILWTICO: 'WTI Crude', DCOILBRENTEU: 'Brent', GOLDAMGBD228NLBM: 'Gold', DHHNGSP: 'Nat Gas',
    VIXCLS: 'VIX', OVXCLS: 'OVX (Oil)', GVZCLS: 'GVZ (Gold)', DTWEXBGS: 'USD Index',
    WALCL: 'Fed Balance Sheet', RRPONTSYD: 'Reverse Repos', M2SL: 'M2 Supply',
  }

  return (
    <div className="space-y-4">
      {/* Regime header */}
      <div className="glass rounded-xl p-5">
        <div className="flex items-start justify-between mb-3">
          <div>
            <h3 className="text-sm font-semibold text-slate-300 mb-1">Current Macro Regime</h3>
            <RegimeBadge quadrant={regime.quadrant} />
          </div>
          <div className="text-right">
            {regime.scores && Object.entries(regime.scores).map(([k, v]) => (
              <div key={k} className="flex items-center gap-2 justify-end">
                <span className="text-xs text-slate-500 capitalize">{k}</span>
                <div className="w-16 h-1.5 rounded-full bg-slate-700 relative overflow-hidden">
                  <div
                    className="absolute h-full rounded-full transition-all"
                    style={{
                      width: `${Math.abs(v) * 50}%`,
                      left: v < 0 ? `${50 - Math.abs(v) * 50}%` : '50%',
                      background: v > 0 ? '#4ade80' : '#f87171',
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
        {regime.description && (
          <p className="text-xs text-slate-400 border-t border-white/10 pt-3 mt-2">
            {regime.description}
          </p>
        )}
        {regime.key_levels && (
          <div className="grid grid-cols-4 gap-3 mt-3 pt-3 border-t border-white/10">
            {Object.entries(regime.key_levels).map(([k, v]) => (
              <div key={k} className="text-center">
                <div className="text-xs text-slate-500 capitalize mb-0.5">{k.replace(/_/g, ' ')}</div>
                <div className="text-sm font-mono font-bold text-slate-200">
                  {typeof v === 'number' ? v.toFixed(2) : v}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Category grids */}
      <div className="grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
        {Object.entries(cats).map(([cat, vals]) => (
          <div key={cat} className="glass rounded-xl p-4">
            <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
              {categoryLabels[cat] || cat}
            </h4>
            {Object.entries(vals).map(([sid, val]) => (
              <MacroRow
                key={sid}
                label={seriesLabels[sid] || sid}
                value={val}
                colorize={['T10Y2Y', 'T10Y3M'].includes(sid)}
              />
            ))}
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Correlation panel ──────────────────────────────────────────────────────────
const STRENGTH_COLORS = {
  very_strong: '#4ade80',
  strong:      '#86efac',
  moderate:    '#fbbf24',
  weak:        '#94a3b8',
}

function CorrelationPanel({ symbol }) {
  const { data, isLoading } = useQuery({
    queryKey: ['correlations', symbol],
    queryFn:  () => fetchCorrelationsFor(symbol),
    enabled:  !!symbol,
    staleTime: 10 * 60_000,
  })

  if (!symbol) return (
    <div className="glass rounded-xl p-6 flex items-center justify-center h-48">
      <span className="text-slate-500 text-sm">Enter a symbol to see correlations</span>
    </div>
  )

  if (isLoading) return (
    <div className="glass rounded-xl p-6 animate-pulse h-48 flex items-center justify-center">
      <span className="text-slate-500 text-sm">Computing correlations...</span>
    </div>
  )

  const corrs = data?.correlations || []

  return (
    <div className="glass rounded-xl p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-slate-300">
          Top Correlations — {symbol}
        </h3>
        <span className="text-xs text-slate-500">{corrs.length} found</span>
      </div>
      <div className="space-y-2">
        {corrs.length === 0 && (
          <p className="text-xs text-slate-500 text-center py-4">No significant correlations found</p>
        )}
        {corrs.slice(0, 15).map((c, i) => {
          const r = c.r_val ?? c.pearson_r ?? 0
          const partner = c.series_b ?? c.partner
          const strength = c.strength || 'weak'
          return (
            <div key={i} className="flex items-center gap-3 py-1.5 border-b border-white/5 last:border-0">
              <div className="flex-1 min-w-0">
                <span className="text-xs font-mono font-semibold text-slate-200">{partner}</span>
                {c.rel_type && (
                  <span className="ml-2 text-xs text-slate-500">{c.rel_type.replace(/_/g, ' ')}</span>
                )}
              </div>
              <div className="flex items-center gap-2 flex-shrink-0">
                {c.lag && c.lag !== 0 && (
                  <span className="text-xs text-slate-500">lag {c.lag}d</span>
                )}
                <div className="w-20 h-1.5 rounded-full bg-slate-700 overflow-hidden relative">
                  <div
                    className="absolute h-full rounded-full"
                    style={{
                      width: `${Math.abs(r) * 100}%`,
                      left: r < 0 ? `${100 - Math.abs(r) * 100}%` : '0',
                      background: r > 0 ? '#4ade80' : '#f87171',
                    }}
                  />
                </div>
                <span
                  className="text-xs font-mono w-14 text-right font-semibold"
                  style={{ color: STRENGTH_COLORS[strength] || '#94a3b8' }}
                >
                  {r >= 0 ? '+' : ''}{r.toFixed(3)}
                </span>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

// ── Supply chain panel ─────────────────────────────────────────────────────────
function SupplyChainPanel({ symbol }) {
  const { data, isLoading } = useQuery({
    queryKey: ['supply-chain', symbol],
    queryFn:  () => fetchSupplyChain(symbol),
    enabled:  !!symbol,
    staleTime: 60 * 60_000,
  })

  if (!symbol) return null
  if (isLoading) return (
    <div className="glass rounded-xl p-4 animate-pulse h-40" />
  )

  const suppliers = data?.suppliers || []
  const customers = data?.customers || []

  if (!suppliers.length && !customers.length) return (
    <div className="glass rounded-xl p-4">
      <h3 className="text-sm font-semibold text-slate-300 mb-2">Supply Chain — {symbol}</h3>
      <p className="text-xs text-slate-500">No supply chain data in knowledge graph</p>
    </div>
  )

  return (
    <div className="glass rounded-xl p-4">
      <h3 className="text-sm font-semibold text-slate-300 mb-4 flex items-center gap-2">
        <Link size={14} className="text-blue-400" />
        Supply Chain — {symbol}
      </h3>
      <div className="grid grid-cols-2 gap-4">
        {suppliers.length > 0 && (
          <div>
            <div className="text-xs text-slate-500 uppercase tracking-wider mb-2">Suppliers</div>
            {suppliers.map((s, i) => (
              <div key={i} className="flex items-center gap-2 py-1 text-xs">
                <ArrowRight size={10} className="text-green-400 flex-shrink-0" />
                <span className="font-mono font-semibold text-slate-200">{s.supplier}</span>
                <span className="text-slate-500 truncate">{s.product?.replace(/_/g, ' ')}</span>
                {s.rev_pct && (
                  <span className="text-slate-400 ml-auto">{(s.rev_pct * 100).toFixed(0)}%</span>
                )}
              </div>
            ))}
          </div>
        )}
        {customers.length > 0 && (
          <div>
            <div className="text-xs text-slate-500 uppercase tracking-wider mb-2">Customers</div>
            {customers.map((c, i) => (
              <div key={i} className="flex items-center gap-2 py-1 text-xs">
                <ArrowRight size={10} className="text-purple-400 flex-shrink-0" />
                <span className="font-mono font-semibold text-slate-200">{c.customer}</span>
                <span className="text-slate-500 truncate">{c.product?.replace(/_/g, ' ')}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

// ── Social sentiment panel ─────────────────────────────────────────────────────
function SocialPanel({ symbol }) {
  const { data, isLoading } = useQuery({
    queryKey: ['social', symbol],
    queryFn:  () => fetchSocialIntel(symbol),
    enabled:  !!symbol,
    staleTime: 15 * 60_000,
  })

  if (!symbol) return null
  if (isLoading) return <div className="glass rounded-xl p-4 animate-pulse h-40" />

  const reddit = data?.reddit || {}
  const st     = data?.stocktwits || {}
  const trends = data?.google_trends || {}
  const comp   = data?.composite || {}
  const pc     = data?.put_call || {}

  const LABEL_COLOR = { bullish: '#4ade80', bearish: '#f87171', neutral: '#94a3b8' }

  return (
    <div className="glass rounded-xl p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-slate-300 flex items-center gap-2">
          <MessageSquare size={14} className="text-purple-400" />
          Social Intelligence — {symbol}
        </h3>
        <span
          className="text-xs font-semibold px-2 py-0.5 rounded"
          style={{
            color: LABEL_COLOR[comp.label] || '#94a3b8',
            background: `${LABEL_COLOR[comp.label] || '#94a3b8'}20`,
          }}
        >
          {comp.label || 'neutral'}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div className="bg-white/5 rounded-lg p-3">
          <div className="text-xs text-slate-500 mb-2 font-medium">Reddit</div>
          <div className="text-lg font-bold text-slate-200">{reddit.mentions || 0}</div>
          <div className="text-xs text-slate-500">mentions / 48h</div>
          <div
            className="text-xs font-semibold mt-1"
            style={{ color: LABEL_COLOR[reddit.sentiment_label] || '#94a3b8' }}
          >
            {reddit.sentiment_label || 'neutral'}
          </div>
        </div>

        <div className="bg-white/5 rounded-lg p-3">
          <div className="text-xs text-slate-500 mb-2 font-medium">StockTwits</div>
          {st.bull_pct !== undefined ? (
            <>
              <div className="flex items-center gap-1.5 mb-1">
                <div className="flex-1 h-2 rounded-full bg-red-900/50 overflow-hidden">
                  <div
                    className="h-full rounded-full bg-green-500"
                    style={{ width: `${st.bull_pct}%` }}
                  />
                </div>
              </div>
              <div className="text-xs text-slate-400">
                <span className="text-green-400">{st.bull_pct}% bull</span>
                {' / '}
                <span className="text-red-400">{st.bear_pct}% bear</span>
              </div>
            </>
          ) : (
            <div className="text-xs text-slate-500">{st.error || 'unavailable'}</div>
          )}
        </div>

        {trends.avg_interest !== undefined && (
          <div className="bg-white/5 rounded-lg p-3">
            <div className="text-xs text-slate-500 mb-2 font-medium">Google Trends</div>
            <div className="text-lg font-bold text-slate-200">{trends.current_interest || '—'}</div>
            <div className="text-xs text-slate-500">interest score</div>
            <div className={`text-xs font-semibold mt-1 ${
              trends.trend === 'rising' ? 'text-green-400' :
              trends.trend === 'falling' ? 'text-red-400' : 'text-slate-400'
            }`}>
              {trends.trend || '—'}
              {trends.current_vs_avg_pct !== undefined && ` (${trends.current_vs_avg_pct > 0 ? '+' : ''}${trends.current_vs_avg_pct}% vs avg)`}
            </div>
          </div>
        )}

        {pc.equity_pc_ratio !== undefined && (
          <div className="bg-white/5 rounded-lg p-3">
            <div className="text-xs text-slate-500 mb-2 font-medium">Put/Call Ratio</div>
            <div className="text-lg font-bold text-slate-200">{pc.equity_pc_ratio?.toFixed(2) || '—'}</div>
            <div className="text-xs text-slate-500">{pc.interpretation || '—'}</div>
            <div className={`text-xs font-semibold mt-1 ${
              pc.contrarian_signal === 'bullish' ? 'text-green-400' :
              pc.contrarian_signal === 'bearish' ? 'text-red-400' : 'text-slate-400'
            }`}>
              Contrarian: {pc.contrarian_signal || '—'}
            </div>
          </div>
        )}
      </div>

      {reddit.top_posts?.length > 0 && (
        <div className="mt-3 pt-3 border-t border-white/10">
          <div className="text-xs text-slate-500 mb-2">Top Reddit Posts</div>
          {reddit.top_posts.slice(0, 3).map((p, i) => (
            <div key={i} className="py-1.5 border-b border-white/5 last:border-0">
              <a
                href={p.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-xs text-slate-300 hover:text-blue-400 line-clamp-1 transition-colors"
              >
                {p.title}
              </a>
              <div className="text-xs text-slate-600 mt-0.5">
                r/{p.subreddit} · ↑{p.score} · {p.comments} comments
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ── Knowledge graph stats ──────────────────────────────────────────────────────
function GraphStats() {
  const { data } = useQuery({
    queryKey: ['graph-stats'],
    queryFn:  fetchGraphStats,
    staleTime: 30 * 60_000,
  })

  if (!data) return null

  return (
    <div className="glass rounded-xl p-4">
      <h3 className="text-sm font-semibold text-slate-300 mb-3 flex items-center gap-2">
        <Network size={14} className="text-blue-400" />
        Knowledge Graph
      </h3>
      <div className="grid grid-cols-3 gap-3 mb-3">
        <div className="text-center">
          <div className="text-2xl font-bold text-blue-400">{data.total_nodes || 0}</div>
          <div className="text-xs text-slate-500">Nodes</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-purple-400">{data.total_edges || 0}</div>
          <div className="text-xs text-slate-500">Edges</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-green-400">
            {Object.keys(data.nodes || {}).length}
          </div>
          <div className="text-xs text-slate-500">Types</div>
        </div>
      </div>
      {data.edges && (
        <div className="space-y-1">
          {Object.entries(data.edges).map(([type, count]) => (
            <div key={type} className="flex items-center justify-between text-xs">
              <span className="text-slate-500 font-mono">{type}</span>
              <span className="text-slate-300 font-semibold">{count}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ── Tab navigation ─────────────────────────────────────────────────────────────
const TABS = [
  { id: 'macro',        label: 'Macro Regime',    icon: Globe },
  { id: 'correlations', label: 'Correlations',    icon: Activity },
  { id: 'supply_chain', label: 'Supply Chain',    icon: Link },
  { id: 'social',       label: 'Social Intel',    icon: MessageSquare },
]

// ── Main Intelligence Page ─────────────────────────────────────────────────────
export default function Intelligence() {
  const [activeTab, setActiveTab] = useState('macro')
  const [symbol, setSymbol] = useState('AAPL')
  const [inputVal, setInputVal] = useState('AAPL')

  function handleSymbolSubmit(e) {
    e.preventDefault()
    if (inputVal.trim()) setSymbol(inputVal.trim().toUpperCase())
  }

  return (
    <div className="min-h-screen p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-3">
            <Brain size={28} className="text-purple-400" />
            Intelligence Engine
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Macro regimes · supply chains · correlations · social sentiment · knowledge graph
          </p>
        </div>
        <GraphStats />
      </div>

      {/* Symbol input */}
      <form onSubmit={handleSymbolSubmit} className="flex gap-2">
        <div className="relative flex-1 max-w-xs">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
          <input
            value={inputVal}
            onChange={e => setInputVal(e.target.value.toUpperCase())}
            placeholder="Symbol (e.g. AAPL)"
            className="w-full pl-9 pr-4 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white placeholder-slate-500 focus:outline-none focus:border-blue-500/50 focus:ring-1 focus:ring-blue-500/30 transition-all"
          />
        </div>
        <button
          type="submit"
          className="px-4 py-2 bg-blue-500/20 hover:bg-blue-500/30 border border-blue-500/30 rounded-lg text-sm text-blue-300 font-medium transition-all"
        >
          Analyze
        </button>
      </form>

      {/* Tab bar */}
      <div className="flex gap-1 p-1 glass rounded-xl w-fit">
        {TABS.map(tab => {
          const Icon = tab.icon
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                activeTab === tab.id
                  ? 'bg-blue-500/20 text-blue-300 border border-blue-500/30'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Icon size={14} />
              {tab.label}
            </button>
          )
        })}
      </div>

      {/* Tab content */}
      <AnimatePresence mode="wait">
        <motion.div
          key={activeTab}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -8 }}
          transition={{ duration: 0.15 }}
        >
          {activeTab === 'macro'        && <MacroDashboard />}
          {activeTab === 'correlations' && <CorrelationPanel symbol={symbol} />}
          {activeTab === 'supply_chain' && <SupplyChainPanel symbol={symbol} />}
          {activeTab === 'social'       && <SocialPanel symbol={symbol} />}
        </motion.div>
      </AnimatePresence>
    </div>
  )
}
