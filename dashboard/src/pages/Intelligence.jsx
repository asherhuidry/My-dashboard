import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { motion, AnimatePresence } from 'framer-motion'
import Header from '../components/Layout/Header'
import {
  TrendingUp, TrendingDown, Minus, Activity, Globe, Network,
  MessageSquare, Search, AlertTriangle, Zap, Brain, Link, ArrowRight,
  Layers, BarChart3, Target,
} from 'lucide-react'
import {
  fetchMacroDashboard, fetchCorrelationsFor, fetchSupplyChain,
  fetchSocialIntel, fetchGraphStats, fetchGraphAnalysis, fetchGraphIntelligence,
} from '../lib/api'


// ── Macro regime badge ──────────────────────────────────────────
const REGIME_COLORS = {
  goldilocks:     { bg: 'rgba(16,185,129,0.1)',  border: '#10b981', text: '#34d399', icon: TrendingUp },
  reflation:      { bg: 'rgba(245,158,11,0.1)',  border: '#f59e0b', text: '#fbbf24', icon: TrendingUp },
  stagflation:    { bg: 'rgba(239,68,68,0.1)',   border: '#ef4444', text: '#f87171', icon: AlertTriangle },
  deflation_risk: { bg: 'rgba(99,102,241,0.1)',  border: '#6366f1', text: '#818cf8', icon: TrendingDown },
  unknown:        { bg: 'rgba(71,85,105,0.1)',   border: '#475569', text: '#94a3b8', icon: Minus },
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


// ── Macro indicator row ─────────────────────────────────────────
function MacroRow({ label, value, format = 'pct', colorize = true }) {
  if (value == null) return null
  const fmt = format === 'pct' ? `${value.toFixed(2)}%`
            : format === 'num' ? value.toFixed(2)
            : format === 'bp'  ? `${(value * 100).toFixed(0)}bp`
            : value.toFixed(2)

  const color = !colorize ? '#94a3b8'
    : value > 0 ? '#34d399' : '#f87171'

  return (
    <div className="flex items-center justify-between py-1.5 border-b border-border/30 last:border-0">
      <span className="text-[10px] text-text-secondary">{label}</span>
      <span className="ticker-value text-[10px] font-semibold" style={{ color }}>{fmt}</span>
    </div>
  )
}


// ── Macro dashboard panel ───────────────────────────────────────
function MacroDashboard() {
  const { data, isLoading } = useQuery({
    queryKey: ['macro-dashboard'],
    queryFn:  fetchMacroDashboard,
    staleTime: 5 * 60_000,
  })

  if (isLoading) return (
    <div className="glass-card rounded-xl p-6 h-64 flex items-center justify-center">
      <span className="text-text-muted text-sm">Loading macro data...</span>
    </div>
  )

  const regime = data?.regime || {}
  const cats   = data?.categories || {}

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
      <div className="glass-card rounded-xl p-5">
        <div className="flex items-start justify-between mb-3">
          <div>
            <h3 className="text-xs font-semibold text-text-secondary uppercase tracking-wider mb-2">Current Macro Regime</h3>
            <RegimeBadge quadrant={regime.quadrant} />
          </div>
          <div className="text-right space-y-1">
            {regime.scores && Object.entries(regime.scores).map(([k, v]) => (
              <div key={k} className="flex items-center gap-2 justify-end">
                <span className="text-[10px] text-text-muted capitalize">{k}</span>
                <div className="w-16 h-1.5 rounded-full bg-border/50 relative overflow-hidden">
                  <div
                    className="absolute h-full rounded-full transition-all"
                    style={{
                      width: `${Math.abs(v) * 50}%`,
                      left: v < 0 ? `${50 - Math.abs(v) * 50}%` : '50%',
                      background: v > 0 ? '#34d399' : '#f87171',
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
        {regime.description && (
          <p className="text-[10px] text-text-secondary border-t border-border/50 pt-3 mt-2 leading-relaxed">
            {regime.description}
          </p>
        )}
        {regime.key_levels && (
          <div className="grid grid-cols-4 gap-3 mt-3 pt-3 border-t border-border/50">
            {Object.entries(regime.key_levels).map(([k, v]) => (
              <div key={k} className="text-center">
                <div className="text-[9px] text-text-muted capitalize mb-0.5">{k.replace(/_/g, ' ')}</div>
                <div className="ticker-value text-sm font-bold text-text">
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
          <div key={cat} className="glass-card rounded-xl p-4">
            <h4 className="text-[10px] font-semibold text-text-muted uppercase tracking-wider mb-3">
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


// ── Bridge factors panel ────────────────────────────────────────
function BridgeFactors({ analysis }) {
  const bridges = analysis?.bridge_factors
  if (!bridges?.length) return (
    <div className="glass-card rounded-xl p-5">
      <div className="flex items-center gap-2 mb-3">
        <Zap size={13} className="text-warning" />
        <span className="text-xs font-semibold text-text uppercase tracking-wider">Bridge Factors</span>
      </div>
      <p className="text-[10px] text-text-muted">No bridge factor data available</p>
    </div>
  )

  return (
    <div className="glass-card rounded-xl p-5">
      <div className="flex items-center gap-2 mb-4">
        <div className="w-7 h-7 rounded-lg bg-warning/10 border border-warning/20 flex items-center justify-center">
          <Zap size={13} className="text-warning" />
        </div>
        <span className="text-xs font-semibold text-text uppercase tracking-wider">Bridge Factors</span>
        <span className="text-[10px] text-text-muted ml-auto">{bridges.length} factors</span>
      </div>
      <div className="space-y-2">
        {bridges.slice(0, 10).map((b, i) => (
          <div key={i} className="rounded-lg border border-border/50 bg-bg-hover/30 px-3 py-2.5">
            <div className="flex items-center justify-between mb-1">
              <span className="ticker-value text-[11px] font-bold text-text">{b.factor}</span>
              <div className="flex items-center gap-2">
                <span className="text-[9px] text-text-muted">{b.asset_count} assets</span>
                <span className="text-[9px] px-1.5 py-0.5 rounded border border-warning/20 bg-warning/8 text-warning">
                  {b.class_count} classes
                </span>
              </div>
            </div>
            {b.classes && (
              <div className="flex flex-wrap gap-1 mt-1">
                {b.classes.map(c => (
                  <span key={c} className="text-[9px] px-1.5 py-0.5 rounded bg-bg-hover border border-border text-text-muted">
                    {c}
                  </span>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}


// ── Sector stress detail panel ──────────────────────────────────
function SectorStressDetail({ analysis }) {
  const sectors = analysis?.sector_stress?.sectors
  if (!sectors?.length) return (
    <div className="glass-card rounded-xl p-5">
      <div className="flex items-center gap-2 mb-3">
        <Layers size={13} className="text-negative" />
        <span className="text-xs font-semibold text-text uppercase tracking-wider">Sector Stress</span>
      </div>
      <p className="text-[10px] text-text-muted">No sector stress data available</p>
    </div>
  )

  return (
    <div className="glass-card rounded-xl p-5">
      <div className="flex items-center gap-2 mb-4">
        <div className="w-7 h-7 rounded-lg bg-negative/10 border border-negative/20 flex items-center justify-center">
          <Layers size={13} className="text-negative" />
        </div>
        <span className="text-xs font-semibold text-text uppercase tracking-wider">Sector Stress Detail</span>
      </div>
      <div className="space-y-2">
        {sectors.map(s => {
          const color = s.stress_score >= 0.6 ? '#ef4444' : s.stress_score >= 0.35 ? '#f59e0b' : '#10b981'
          return (
            <div key={s.sector} className="rounded-lg border border-border/50 bg-bg-hover/30 px-3 py-2.5">
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-[11px] font-semibold text-text">{s.sector}</span>
                <span className="ticker-value text-[11px] font-bold" style={{ color }}>
                  {(s.stress_score * 100).toFixed(1)}%
                </span>
              </div>
              <div className="w-full h-1.5 rounded-full bg-border/40 overflow-hidden mb-1.5">
                <div className="h-full rounded-full transition-all" style={{ width: `${s.stress_score * 100}%`, background: color }} />
              </div>
              <div className="grid grid-cols-3 gap-2 text-[9px]">
                <div>
                  <span className="text-text-muted">Divergence</span>
                  <span className="ticker-value text-text-secondary ml-1">{(s.divergence_intensity ?? 0).toFixed(2)}</span>
                </div>
                <div>
                  <span className="text-text-muted">Bridges</span>
                  <span className="ticker-value text-text-secondary ml-1">{((s.bridge_exposure ?? 0) * 100).toFixed(0)}%</span>
                </div>
                <div>
                  <span className="text-text-muted">Assets</span>
                  <span className="ticker-value text-text-secondary ml-1">{s.asset_count}</span>
                </div>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}


// ── Correlation panel ───────────────────────────────────────────
const STRENGTH_COLORS = {
  very_strong: '#34d399',
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
    <div className="glass-card rounded-xl p-6 flex items-center justify-center h-48">
      <span className="text-text-muted text-sm">Enter a symbol to see correlations</span>
    </div>
  )

  if (isLoading) return (
    <div className="glass-card rounded-xl p-6 h-48 flex items-center justify-center shimmer" />
  )

  const corrs = data?.correlations || []

  return (
    <div className="glass-card rounded-xl p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-xs font-semibold text-text uppercase tracking-wider">
          Top Correlations — {symbol}
        </h3>
        <span className="text-[10px] text-text-muted">{corrs.length} found</span>
      </div>
      <div className="space-y-1.5">
        {corrs.length === 0 && (
          <p className="text-[10px] text-text-muted text-center py-4">No significant correlations found</p>
        )}
        {corrs.slice(0, 15).map((c, i) => {
          const r = c.r_val ?? c.pearson_r ?? 0
          const partner = c.series_b ?? c.partner
          const strength = c.strength || 'weak'
          return (
            <div key={i} className="flex items-center gap-3 py-1.5 border-b border-border/30 last:border-0">
              <div className="flex-1 min-w-0">
                <span className="ticker-value text-[10px] font-semibold text-text">{partner}</span>
                {c.rel_type && (
                  <span className="ml-2 text-[9px] text-text-muted">{c.rel_type.replace(/_/g, ' ')}</span>
                )}
              </div>
              <div className="flex items-center gap-2 flex-shrink-0">
                {c.lag && c.lag !== 0 && (
                  <span className="text-[9px] text-text-muted">lag {c.lag}d</span>
                )}
                <div className="w-20 h-1.5 rounded-full bg-border/40 overflow-hidden relative">
                  <div
                    className="absolute h-full rounded-full"
                    style={{
                      width: `${Math.abs(r) * 100}%`,
                      left: r < 0 ? `${100 - Math.abs(r) * 100}%` : '0',
                      background: r > 0 ? '#34d399' : '#f87171',
                    }}
                  />
                </div>
                <span
                  className="ticker-value text-[10px] w-14 text-right font-semibold"
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


// ── Supply chain panel ──────────────────────────────────────────
function SupplyChainPanel({ symbol }) {
  const { data, isLoading } = useQuery({
    queryKey: ['supply-chain', symbol],
    queryFn:  () => fetchSupplyChain(symbol),
    enabled:  !!symbol,
    staleTime: 60 * 60_000,
  })

  if (!symbol) return null
  if (isLoading) return <div className="glass-card rounded-xl p-4 shimmer h-40" />

  const suppliers = data?.suppliers || []
  const customers = data?.customers || []

  if (!suppliers.length && !customers.length) return (
    <div className="glass-card rounded-xl p-4">
      <h3 className="text-xs font-semibold text-text uppercase tracking-wider mb-2 flex items-center gap-2">
        <Link size={13} className="text-accent" /> Supply Chain — {symbol}
      </h3>
      <p className="text-[10px] text-text-muted">No supply chain data in knowledge graph</p>
    </div>
  )

  return (
    <div className="glass-card rounded-xl p-4">
      <h3 className="text-xs font-semibold text-text uppercase tracking-wider mb-4 flex items-center gap-2">
        <Link size={13} className="text-accent" /> Supply Chain — {symbol}
      </h3>
      <div className="grid grid-cols-2 gap-4">
        {suppliers.length > 0 && (
          <div>
            <div className="text-[9px] text-text-muted uppercase tracking-wider mb-2 font-semibold">Suppliers</div>
            {suppliers.map((s, i) => (
              <div key={i} className="flex items-center gap-2 py-1 text-[10px]">
                <ArrowRight size={10} className="text-positive flex-shrink-0" />
                <span className="ticker-value font-semibold text-text">{s.supplier}</span>
                <span className="text-text-muted truncate">{s.product?.replace(/_/g, ' ')}</span>
                {s.rev_pct && (
                  <span className="text-text-secondary ml-auto">{(s.rev_pct * 100).toFixed(0)}%</span>
                )}
              </div>
            ))}
          </div>
        )}
        {customers.length > 0 && (
          <div>
            <div className="text-[9px] text-text-muted uppercase tracking-wider mb-2 font-semibold">Customers</div>
            {customers.map((c, i) => (
              <div key={i} className="flex items-center gap-2 py-1 text-[10px]">
                <ArrowRight size={10} className="text-purple flex-shrink-0" />
                <span className="ticker-value font-semibold text-text">{c.customer}</span>
                <span className="text-text-muted truncate">{c.product?.replace(/_/g, ' ')}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}


// ── Social sentiment panel ──────────────────────────────────────
function SocialPanel({ symbol }) {
  const { data, isLoading } = useQuery({
    queryKey: ['social', symbol],
    queryFn:  () => fetchSocialIntel(symbol),
    enabled:  !!symbol,
    staleTime: 15 * 60_000,
  })

  if (!symbol) return null
  if (isLoading) return <div className="glass-card rounded-xl p-4 shimmer h-40" />

  const reddit = data?.reddit || {}
  const st     = data?.stocktwits || {}
  const trends = data?.google_trends || {}
  const comp   = data?.composite || {}
  const pc     = data?.put_call || {}

  const LABEL_COLOR = { bullish: '#34d399', bearish: '#f87171', neutral: '#94a3b8' }

  return (
    <div className="glass-card rounded-xl p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-xs font-semibold text-text uppercase tracking-wider flex items-center gap-2">
          <MessageSquare size={13} className="text-purple" /> Social Intelligence — {symbol}
        </h3>
        <span
          className="text-[10px] font-semibold px-2 py-0.5 rounded"
          style={{
            color: LABEL_COLOR[comp.label] || '#94a3b8',
            background: `${LABEL_COLOR[comp.label] || '#94a3b8'}20`,
          }}
        >
          {comp.label || 'neutral'}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div className="rounded-lg border border-border/50 bg-bg-hover/30 p-3">
          <div className="text-[9px] text-text-muted mb-2 font-semibold uppercase tracking-wider">Reddit</div>
          <div className="metric-md text-text">{reddit.mentions || 0}</div>
          <div className="text-[9px] text-text-muted">mentions / 48h</div>
          <div className="text-[10px] font-semibold mt-1"
            style={{ color: LABEL_COLOR[reddit.sentiment_label] || '#94a3b8' }}>
            {reddit.sentiment_label || 'neutral'}
          </div>
        </div>

        <div className="rounded-lg border border-border/50 bg-bg-hover/30 p-3">
          <div className="text-[9px] text-text-muted mb-2 font-semibold uppercase tracking-wider">StockTwits</div>
          {st.bull_pct !== undefined ? (
            <>
              <div className="flex items-center gap-1.5 mb-1">
                <div className="flex-1 h-2 rounded-full bg-negative/20 overflow-hidden">
                  <div className="h-full rounded-full bg-positive" style={{ width: `${st.bull_pct}%` }} />
                </div>
              </div>
              <div className="text-[10px] text-text-secondary">
                <span className="text-positive">{st.bull_pct}% bull</span>
                {' / '}
                <span className="text-negative">{st.bear_pct}% bear</span>
              </div>
            </>
          ) : (
            <div className="text-[10px] text-text-muted">{st.error || 'unavailable'}</div>
          )}
        </div>

        {trends.avg_interest !== undefined && (
          <div className="rounded-lg border border-border/50 bg-bg-hover/30 p-3">
            <div className="text-[9px] text-text-muted mb-2 font-semibold uppercase tracking-wider">Google Trends</div>
            <div className="metric-md text-text">{trends.current_interest || '\u2014'}</div>
            <div className="text-[9px] text-text-muted">interest score</div>
            <div className={`text-[10px] font-semibold mt-1 ${
              trends.trend === 'rising' ? 'text-positive' : trends.trend === 'falling' ? 'text-negative' : 'text-text-secondary'
            }`}>
              {trends.trend || '\u2014'}
              {trends.current_vs_avg_pct !== undefined && ` (${trends.current_vs_avg_pct > 0 ? '+' : ''}${trends.current_vs_avg_pct}% vs avg)`}
            </div>
          </div>
        )}

        {pc.equity_pc_ratio !== undefined && (
          <div className="rounded-lg border border-border/50 bg-bg-hover/30 p-3">
            <div className="text-[9px] text-text-muted mb-2 font-semibold uppercase tracking-wider">Put/Call</div>
            <div className="metric-md text-text">{pc.equity_pc_ratio?.toFixed(2) || '\u2014'}</div>
            <div className="text-[9px] text-text-muted">{pc.interpretation || '\u2014'}</div>
            <div className={`text-[10px] font-semibold mt-1 ${
              pc.contrarian_signal === 'bullish' ? 'text-positive' : pc.contrarian_signal === 'bearish' ? 'text-negative' : 'text-text-secondary'
            }`}>
              Contrarian: {pc.contrarian_signal || '\u2014'}
            </div>
          </div>
        )}
      </div>

      {reddit.top_posts?.length > 0 && (
        <div className="mt-3 pt-3 border-t border-border/50">
          <div className="text-[9px] text-text-muted uppercase tracking-wider font-semibold mb-2">Top Reddit Posts</div>
          {reddit.top_posts.slice(0, 3).map((p, i) => (
            <div key={i} className="py-1.5 border-b border-border/30 last:border-0">
              <a href={p.url} target="_blank" rel="noopener noreferrer"
                className="text-[10px] text-text-secondary hover:text-accent line-clamp-1 transition-colors">
                {p.title}
              </a>
              <div className="text-[9px] text-text-muted mt-0.5">
                r/{p.subreddit} · {'\u2191'}{p.score} · {p.comments} comments
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}


// ── Knowledge graph stats ───────────────────────────────────────
function GraphStatsPanel() {
  const { data } = useQuery({
    queryKey: ['graph-stats'],
    queryFn:  fetchGraphStats,
    staleTime: 30 * 60_000,
  })

  if (!data) return null

  return (
    <div className="glass-card rounded-xl p-4">
      <h3 className="text-xs font-semibold text-text uppercase tracking-wider mb-3 flex items-center gap-2">
        <Network size={13} className="text-accent" /> Knowledge Graph
      </h3>
      <div className="grid grid-cols-3 gap-3 mb-3">
        <div className="text-center">
          <div className="metric-lg text-accent">{data.total_nodes || 0}</div>
          <div className="text-[9px] text-text-muted mt-0.5">Nodes</div>
        </div>
        <div className="text-center">
          <div className="metric-lg text-purple">{data.total_edges || 0}</div>
          <div className="text-[9px] text-text-muted mt-0.5">Edges</div>
        </div>
        <div className="text-center">
          <div className="metric-lg text-positive">{Object.keys(data.nodes || {}).length}</div>
          <div className="text-[9px] text-text-muted mt-0.5">Types</div>
        </div>
      </div>
      {data.edges && (
        <div className="space-y-1 border-t border-border/50 pt-2">
          {Object.entries(data.edges).map(([type, count]) => (
            <div key={type} className="flex items-center justify-between text-[10px]">
              <span className="ticker-value text-text-muted">{type}</span>
              <span className="ticker-value text-text-secondary font-semibold">{count}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}


// ── Tab navigation ──────────────────────────────────────────────
const TABS = [
  { id: 'macro',        label: 'Macro Regime',    icon: Globe },
  { id: 'structure',    label: 'Graph Structure', icon: Network },
  { id: 'correlations', label: 'Correlations',    icon: Activity },
  { id: 'supply_chain', label: 'Supply Chain',    icon: Link },
  { id: 'social',       label: 'Social Intel',    icon: MessageSquare },
]


// ── Structure tab content ───────────────────────────────────────
function StructureTab() {
  const { data: analysis } = useQuery({
    queryKey: ['graph-analysis'],
    queryFn:  () => fetchGraphAnalysis(20),
    staleTime: 300_000,
  })

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      <SectorStressDetail analysis={analysis} />
      <BridgeFactors analysis={analysis} />
      <div className="lg:col-span-2">
        <GraphStatsPanel />
      </div>
    </div>
  )
}


// ── Main Intelligence Page ──────────────────────────────────────
export default function Intelligence() {
  const [activeTab, setActiveTab] = useState('macro')
  const [symbol, setSymbol]       = useState('AAPL')
  const [inputVal, setInputVal]   = useState('AAPL')

  function handleSymbolSubmit(e) {
    e.preventDefault()
    if (inputVal.trim()) setSymbol(inputVal.trim().toUpperCase())
  }

  const needsSymbol = ['correlations', 'supply_chain', 'social'].includes(activeTab)

  return (
    <div className="flex flex-col h-full">
      <Header title="Intelligence" subtitle="Macro regimes, graph structure, and market analysis" />

      <div className="flex-1 overflow-y-auto p-5 space-y-5 bg-grid">
        {/* Top bar: tabs + symbol input */}
        <div className="flex items-center gap-4 flex-wrap">
          <div className="flex gap-1 p-1 glass-card rounded-xl">
            {TABS.map(tab => {
              const Icon = tab.icon
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[11px] font-medium transition-all ${
                    activeTab === tab.id
                      ? 'bg-accent/15 text-accent border border-accent/25'
                      : 'text-text-muted hover:text-text-secondary border border-transparent'
                  }`}
                >
                  <Icon size={12} />
                  {tab.label}
                </button>
              )
            })}
          </div>

          {needsSymbol && (
            <form onSubmit={handleSymbolSubmit} className="flex gap-2">
              <div className="relative">
                <Search size={12} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-text-muted" />
                <input
                  value={inputVal}
                  onChange={e => setInputVal(e.target.value.toUpperCase())}
                  placeholder="Symbol"
                  className="w-32 pl-7 pr-3 py-1.5 bg-bg-hover border border-border rounded-lg text-[11px] text-text
                    placeholder-text-muted focus:outline-none focus:border-accent/40 transition-all"
                />
              </div>
              <button type="submit"
                className="px-3 py-1.5 bg-accent/15 hover:bg-accent/25 border border-accent/25 rounded-lg text-[11px] text-accent font-medium transition-all">
                Analyze
              </button>
            </form>
          )}
        </div>

        {/* Tab content */}
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            transition={{ duration: 0.15 }}
          >
            {activeTab === 'macro'        && <MacroDashboard />}
            {activeTab === 'structure'    && <StructureTab />}
            {activeTab === 'correlations' && <CorrelationPanel symbol={symbol} />}
            {activeTab === 'supply_chain' && <SupplyChainPanel symbol={symbol} />}
            {activeTab === 'social'       && <SocialPanel symbol={symbol} />}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  )
}
