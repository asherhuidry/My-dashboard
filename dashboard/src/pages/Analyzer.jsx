import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useSearchParams } from 'react-router-dom'
import { fetchAnalysis } from '../lib/api'
import { fmt, pctColor, signalBadge } from '../lib/utils'
import Header from '../components/Layout/Header'
import SearchBar from '../components/UI/SearchBar'
import Spinner from '../components/UI/Spinner'
import PriceChart from '../components/Charts/PriceChart'
import ErrorBoundary from '../components/ErrorBoundary'
import { RSIChart, MACDChart, StochasticChart, OBVChart } from '../components/Charts/IndicatorChart'
import FeatureHeatmap from '../components/Charts/FeatureHeatmap'
import { motion, AnimatePresence } from 'framer-motion'
import {
  TrendingUp, TrendingDown, Minus, ChevronDown, ChevronUp,
  BarChart2, Activity, Layers, Info, Target, Gauge,
} from 'lucide-react'

const OVERLAY_OPTIONS = [
  { key: 'bb_upper',  label: 'Bollinger Bands', keys: ['bb_upper','bb_lower','bb_mid'], color: '#8b5cf6' },
  { key: 'ema_9',     label: 'EMA 9/21',        keys: ['ema_9','ema_21'],               color: '#f59e0b' },
  { key: 'ema_50',    label: 'EMA 50/200',       keys: ['ema_50','ema_200'],             color: '#10b981' },
]

const INDICATOR_TABS = [
  { id: 'rsi',   label: 'RSI',         icon: Gauge    },
  { id: 'macd',  label: 'MACD',        icon: Activity },
  { id: 'stoch', label: 'Stochastic',  icon: Target   },
  { id: 'obv',   label: 'OBV',         icon: BarChart2 },
]

const DAY_OPTIONS = [
  { d: 90,  label: '3M' },
  { d: 180, label: '6M' },
  { d: 365, label: '1Y' },
  { d: 730, label: '2Y' },
]

function ReturnPill({ label, value }) {
  const pos = value > 0
  const neutral = value == null || value === 0
  return (
    <div className="flex flex-col items-center px-3 py-2 rounded-xl border transition-all
      bg-bg-card hover:border-border-bright border-border min-w-[56px]">
      <span className="text-[9px] text-text-muted uppercase tracking-wider font-semibold">{label}</span>
      <span className={`ticker-value text-xs font-bold mt-0.5 ${
        neutral ? 'text-text-muted' : pos ? 'text-positive' : 'text-negative'
      }`}>
        {value != null ? fmt.pct(value) : '—'}
      </span>
    </div>
  )
}

function SignalMeter({ score, overall }) {
  const pct   = Math.min(Math.max((score + 100) / 2, 0), 100)
  const color = overall === 'bullish' ? '#10b981' : overall === 'bearish' ? '#ef4444' : '#64748b'

  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <span className="text-[10px] text-text-muted uppercase tracking-wider">Signal Strength</span>
        <span className={`ticker-value text-sm font-bold ${
          overall === 'bullish' ? 'text-positive' : overall === 'bearish' ? 'text-negative' : 'text-text-muted'
        }`}>
          {score > 0 ? '+' : ''}{score}
        </span>
      </div>
      <div className="h-2.5 rounded-full overflow-hidden bg-bg relative">
        <div className="absolute inset-0 flex">
          <div className="flex-1 bg-negative/15 rounded-l-full" />
          <div className="flex-1 bg-text-muted/5 rounded-r-full" />
        </div>
        <div
          className="absolute top-0 left-0 h-full rounded-full transition-all duration-700"
          style={{
            width: `${pct}%`,
            background: `linear-gradient(90deg, #ef444460, ${color})`,
            boxShadow: `0 0 8px ${color}80`,
          }}
        />
        {/* Center line */}
        <div className="absolute top-0 left-1/2 w-px h-full bg-border/80" />
      </div>
      <div className="flex justify-between text-[9px] text-text-muted mt-1.5">
        <span>Bearish</span><span>Neutral</span><span>Bullish</span>
      </div>
    </div>
  )
}

function IndicatorRow({ label, value, format, highlight }) {
  if (value == null) return null
  const display = format ? format(value) : value
  return (
    <div className="flex items-center justify-between py-1.5 border-b border-border/40 last:border-0">
      <span className="text-[10px] text-text-muted">{label}</span>
      <span className={`ticker-value text-[11px] font-semibold ${highlight ?? 'text-text'}`}>{display}</span>
    </div>
  )
}

export default function Analyzer() {
  const [params]             = useSearchParams()
  const [symbol, setSymbol]  = useState(params.get('symbol') ?? '')
  const [days,   setDays]    = useState(365)
  const [overlays, setOverlays]     = useState(['bb_upper','bb_lower','bb_mid'])
  const [activeIndicator, setActive] = useState('rsi')
  const [showFeatures, setShowFeat]  = useState(false)

  const { data, isLoading, error } = useQuery({
    queryKey:  ['analysis', symbol, days],
    queryFn:   () => fetchAnalysis(symbol, days),
    enabled:   symbol.length > 0,
    staleTime: 120_000,
    retry: 1,
  })

  const toggleOverlay = (keys) => {
    setOverlays(prev => {
      const has = keys.every(k => prev.includes(k))
      return has ? prev.filter(k => !keys.includes(k)) : [...new Set([...prev, ...keys])]
    })
  }

  const sig    = data?.signal
  const SigIcon = sig?.overall === 'bullish' ? TrendingUp : sig?.overall === 'bearish' ? TrendingDown : Minus
  const sigColor = sig?.overall === 'bullish' ? 'text-positive' : sig?.overall === 'bearish' ? 'text-negative' : 'text-text-muted'

  return (
    <div className="flex flex-col h-full">
      <Header title="Stock Analyzer" subtitle="74-feature ML analysis · Signal consensus · Live OHLCV" />

      <div className="flex-1 overflow-y-auto bg-grid">

        {/* ── Search bar ─────────────────────────────────────────── */}
        <div className="px-5 py-3 border-b border-border bg-bg-secondary/60 backdrop-blur flex items-center gap-4 flex-wrap sticky top-0 z-20">
          <SearchBar onSelect={setSymbol} placeholder="Enter symbol (AAPL, NVDA, BTC-USD…)" />
          <div className="flex items-center gap-1 bg-bg/60 rounded-lg border border-border p-0.5">
            {DAY_OPTIONS.map(({ d, label }) => (
              <button key={d} onClick={() => setDays(d)}
                className={`px-2.5 py-1 rounded-md text-xs font-semibold transition-all ${
                  days === d
                    ? 'bg-accent/20 text-accent border border-accent/30 shadow-glow'
                    : 'text-text-muted hover:text-text'
                }`}>
                {label}
              </button>
            ))}
          </div>
        </div>

        {/* ── Empty state ────────────────────────────────────────── */}
        {!symbol && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
            className="flex flex-col items-center justify-center h-72 gap-4 text-center">
            <div className="w-16 h-16 rounded-2xl bg-accent/8 border border-accent/15
              flex items-center justify-center animate-float">
              <BarChart2 size={28} className="text-accent/60" />
            </div>
            <div>
              <p className="text-text-secondary text-sm font-medium">Start by searching a symbol</p>
              <p className="text-text-muted text-xs mt-1">Supports stocks, crypto, ETFs, and forex</p>
            </div>
            <div className="flex gap-2 flex-wrap justify-center">
              {['AAPL','NVDA','BTC-USD','ETH-USD','SPY'].map(s => (
                <button key={s} onClick={() => setSymbol(s)}
                  className="text-xs px-2.5 py-1 rounded-lg border border-border hover:border-accent/40
                    hover:bg-accent/8 text-text-muted hover:text-accent transition-all">
                  {s}
                </button>
              ))}
            </div>
          </motion.div>
        )}

        {/* ── Loading ────────────────────────────────────────────── */}
        {isLoading && (
          <div className="flex flex-col items-center justify-center h-64 gap-3">
            <Spinner size="lg" text={`Analyzing ${symbol}…`} />
            <p className="text-xs text-text-muted">Computing 74 ML features</p>
          </div>
        )}

        {/* ── Error ──────────────────────────────────────────────── */}
        {error && !isLoading && (
          <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
            className="m-5 glass rounded-xl p-5 border border-negative/25 text-center">
            <TrendingDown size={20} className="text-negative mx-auto mb-2" />
            <p className="text-negative text-sm font-medium">Could not load {symbol}</p>
            <p className="text-text-muted text-xs mt-1">Check the symbol and try again, or pick from suggestions above.</p>
          </motion.div>
        )}

        {/* ── Main content ────────────────────────────────────────── */}
        {data && !isLoading && (
          <div className="p-5 space-y-4">

            {/* ── Stock header ──────────────────────────────────── */}
            <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}
              className="glass-card rounded-2xl border border-border p-5 relative overflow-hidden">
              {/* Glow based on signal */}
              <div className={`absolute inset-0 pointer-events-none opacity-25 ${
                sig?.overall === 'bullish' ? 'bg-gradient-to-br from-positive/8 to-transparent' :
                sig?.overall === 'bearish' ? 'bg-gradient-to-br from-negative/8 to-transparent' : ''
              }`} />
              <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-border-bright to-transparent" />

              <div className="relative flex items-start justify-between flex-wrap gap-4">
                <div>
                  <div className="flex items-center gap-3 flex-wrap">
                    <h2 className="text-2xl font-bold text-text">{data.symbol}</h2>
                    {sig && (
                      <span className={`inline-flex items-center gap-1.5 text-[10px] font-bold px-2.5 py-1
                        rounded-full border uppercase tracking-wider ${signalBadge(sig.overall)}`}>
                        <SigIcon size={10} /> {sig.overall}
                      </span>
                    )}
                  </div>
                  <p className="text-sm text-text-secondary mt-1">{data.name}</p>
                  {(data.sector || data.industry) && (
                    <p className="text-xs text-text-muted mt-0.5">
                      {[data.sector, data.industry].filter(Boolean).join(' · ')}
                    </p>
                  )}
                </div>

                <div className="text-right">
                  <div className="ticker-value text-3xl font-bold text-text">${fmt.price(data.price)}</div>
                  <div className={`text-sm font-semibold mt-0.5 ${pctColor(data.returns?.['1d'])}`}>
                    {fmt.pct(data.returns?.['1d'])} <span className="text-xs font-normal text-text-muted">today</span>
                  </div>
                  {data.market_cap && (
                    <div className="text-xs text-text-muted mt-0.5">Mkt cap {fmt.big(data.market_cap)}</div>
                  )}
                </div>
              </div>

              {/* Returns strip */}
              <div className="relative mt-4 flex items-center gap-2 flex-wrap">
                <ReturnPill label="1D"  value={data.returns?.['1d']} />
                <ReturnPill label="5D"  value={data.returns?.['5d']} />
                <ReturnPill label="21D" value={data.returns?.['21d']} />
                <ReturnPill label="63D" value={data.returns?.['63d']} />
                <ReturnPill label="YTD" value={data.returns?.ytd} />
                {data.vol_52w != null && (
                  <div className="flex flex-col items-center px-3 py-2 rounded-xl border border-border min-w-[56px] bg-bg-card">
                    <span className="text-[9px] text-text-muted uppercase tracking-wider font-semibold">Vol</span>
                    <span className="ticker-value text-xs font-bold mt-0.5 text-warning">
                      {(data.vol_52w * 100).toFixed(1)}%
                    </span>
                  </div>
                )}
                {data.pe_ratio != null && (
                  <div className="flex flex-col items-center px-3 py-2 rounded-xl border border-border min-w-[56px] bg-bg-card">
                    <span className="text-[9px] text-text-muted uppercase tracking-wider font-semibold">P/E</span>
                    <span className="ticker-value text-xs font-bold mt-0.5 text-text">
                      {fmt.num(data.pe_ratio, 1)}x
                    </span>
                  </div>
                )}
              </div>
            </motion.div>

            {/* ── Chart + signal panel ───────────────────────────── */}
            <div className="grid grid-cols-1 xl:grid-cols-4 gap-4">

              {/* Charts column */}
              <div className="xl:col-span-3 space-y-3">
                {/* Overlay toggles */}
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-[10px] text-text-muted uppercase tracking-wider mr-1">Overlays:</span>
                  {OVERLAY_OPTIONS.map(opt => {
                    const active = opt.keys.every(k => overlays.includes(k))
                    return (
                      <button key={opt.key} onClick={() => toggleOverlay(opt.keys)}
                        className={`text-[10px] px-2.5 py-1 rounded-lg border transition-all font-medium ${
                          active
                            ? 'border-opacity-60 bg-opacity-15 text-current'
                            : 'border-border text-text-muted hover:text-text hover:border-border-bright'
                        }`}
                        style={active ? {
                          borderColor: `${opt.color}60`,
                          background:  `${opt.color}15`,
                          color:        opt.color,
                        } : {}}>
                        {opt.label}
                      </button>
                    )
                  })}
                </div>

                {/* Price chart */}
                <ErrorBoundary>
                  <div className="glass-card rounded-xl border border-border overflow-hidden" style={{ height: 360 }}>
                    <PriceChart
                      candles={data.candles}
                      indicators={data.indicators}
                      overlays={overlays}
                    />
                  </div>
                </ErrorBoundary>

                {/* Indicator tabs */}
                <div className="flex items-center gap-1 bg-bg/60 rounded-xl border border-border p-1">
                  {INDICATOR_TABS.map(tab => {
                    const Icon = tab.icon
                    return (
                      <button key={tab.id} onClick={() => setActive(tab.id)}
                        className={`flex items-center gap-1.5 text-[10px] font-semibold px-3 py-1.5 rounded-lg
                          flex-1 justify-center transition-all ${
                          activeIndicator === tab.id
                            ? 'bg-accent/15 text-accent border border-accent/25 shadow-glow'
                            : 'text-text-muted hover:text-text'
                        }`}>
                        <Icon size={11} /> {tab.label}
                      </button>
                    )
                  })}
                </div>

                {/* Indicator chart */}
                <ErrorBoundary>
                  <div className="glass-card rounded-xl border border-border overflow-hidden" style={{ height: 190 }}>
                    {activeIndicator === 'rsi'   && <RSIChart data14={data.indicators?.rsi_14} data28={data.indicators?.rsi_28} />}
                    {activeIndicator === 'macd'  && <MACDChart macd={data.indicators?.macd} signal={data.indicators?.macd_signal} hist={data.indicators?.macd_hist} />}
                    {activeIndicator === 'stoch' && <StochasticChart k={data.indicators?.stoch_k} d={data.indicators?.stoch_d} />}
                    {activeIndicator === 'obv'   && <OBVChart data={data.indicators?.obv} />}
                  </div>
                </ErrorBoundary>
              </div>

              {/* Signal + indicators sidebar */}
              <div className="space-y-3">

                {/* Signal panel */}
                {sig && (
                  <motion.div initial={{ opacity: 0, x: 10 }} animate={{ opacity: 1, x: 0 }}
                    className={`glass-card rounded-xl p-4 border relative overflow-hidden ${
                      sig.overall === 'bullish' ? 'border-positive/25'
                      : sig.overall === 'bearish' ? 'border-negative/25'
                      : 'border-border'
                    }`}>
                    <div className={`absolute inset-0 pointer-events-none ${
                      sig.overall === 'bullish' ? 'bg-gradient-to-br from-positive/6 to-transparent'
                      : sig.overall === 'bearish' ? 'bg-gradient-to-br from-negative/6 to-transparent'
                      : ''
                    }`} />
                    <div className="relative">
                      <div className="flex items-center gap-2 mb-3">
                        <div className={`w-7 h-7 rounded-lg flex items-center justify-center border ${
                          sig.overall === 'bullish' ? 'bg-positive/15 border-positive/30'
                          : sig.overall === 'bearish' ? 'bg-negative/15 border-negative/30'
                          : 'bg-bg border-border'
                        }`}>
                          <SigIcon size={14} className={sigColor} />
                        </div>
                        <span className={`text-sm font-bold capitalize ${sigColor}`}>
                          {sig.overall} Signal
                        </span>
                      </div>

                      <SignalMeter score={sig.score} overall={sig.overall} />

                      <div className="mt-3 grid grid-cols-2 gap-2">
                        <div className="text-center p-2.5 rounded-xl bg-positive/8 border border-positive/20">
                          <div className="ticker-value text-xl font-bold text-positive">{sig.bull_count}</div>
                          <div className="text-[9px] text-text-muted mt-0.5 uppercase tracking-wider">Bullish</div>
                        </div>
                        <div className="text-center p-2.5 rounded-xl bg-negative/8 border border-negative/20">
                          <div className="ticker-value text-xl font-bold text-negative">{sig.bear_count}</div>
                          <div className="text-[9px] text-text-muted mt-0.5 uppercase tracking-wider">Bearish</div>
                        </div>
                      </div>

                      {/* Individual signals */}
                      <div className="mt-3 space-y-1.5">
                        {(sig.signals ?? []).map((s, i) => (
                          <div key={i} className="flex items-center gap-2 text-[10px]">
                            <div className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${
                              s === 'bullish' ? 'bg-positive' : s === 'bearish' ? 'bg-negative' : 'bg-text-muted'
                            }`} />
                            <span className="text-text-muted capitalize">{s}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </motion.div>
                )}

                {/* Key indicators */}
                <div className="glass-card rounded-xl p-4 border border-border">
                  <div className="flex items-center gap-1.5 mb-3">
                    <Gauge size={12} className="text-accent" />
                    <h4 className="text-[10px] font-semibold text-text uppercase tracking-wider">Key Indicators</h4>
                  </div>
                  <IndicatorRow label="RSI 14"   value={data.features?.rsi_14}   format={v => v.toFixed(1)}
                    highlight={data.features?.rsi_14 < 30 ? 'text-positive' : data.features?.rsi_14 > 70 ? 'text-negative' : 'text-text'} />
                  <IndicatorRow label="RSI 28"   value={data.features?.rsi_28}   format={v => v.toFixed(1)} />
                  <IndicatorRow label="ADX 14"   value={data.features?.adx_14}   format={v => v.toFixed(1)}
                    highlight={data.features?.adx_14 > 25 ? 'text-accent' : 'text-text'} />
                  <IndicatorRow label="ATR"      value={data.atr}                format={v => `$${v.toFixed(2)}`} />
                  <IndicatorRow label="BB %"     value={data.features?.bb_pct}   format={v => `${(v*100).toFixed(1)}%`}
                    highlight={data.features?.bb_pct > 0.8 ? 'text-negative' : data.features?.bb_pct < 0.2 ? 'text-positive' : 'text-text'} />
                  <IndicatorRow label="Vol 5D"   value={data.features?.realized_vol_5d} format={v => `${(v*100).toFixed(1)}%`} highlight="text-warning" />
                </div>

                {/* About */}
                {data.description && (
                  <div className="glass-card rounded-xl p-4 border border-border">
                    <div className="flex items-center gap-1.5 mb-2">
                      <Info size={11} className="text-text-muted" />
                      <h4 className="text-[10px] font-semibold text-text uppercase tracking-wider">About</h4>
                    </div>
                    <p className="text-[10px] text-text-muted leading-relaxed line-clamp-6">{data.description}</p>
                  </div>
                )}
              </div>
            </div>

            {/* ── Feature heatmap (expandable) ─────────────────── */}
            <div className="glass-card rounded-xl border border-border overflow-hidden">
              <button
                onClick={() => setShowFeat(f => !f)}
                className="w-full flex items-center justify-between px-5 py-3.5 hover:bg-bg-hover/50 transition-colors"
              >
                <div className="flex items-center gap-2.5">
                  <div className="w-6 h-6 rounded-lg bg-accent/10 border border-accent/20 flex items-center justify-center">
                    <Layers size={12} className="text-accent" />
                  </div>
                  <span className="text-sm font-semibold text-text">ML Feature Heatmap</span>
                  <span className="text-[10px] text-text-muted px-2 py-0.5 rounded-md border border-border bg-bg">
                    74 features
                  </span>
                </div>
                {showFeatures
                  ? <ChevronUp size={14} className="text-text-muted" />
                  : <ChevronDown size={14} className="text-text-muted" />
                }
              </button>
              <AnimatePresence>
                {showFeatures && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.25 }}
                    className="border-t border-border px-5 py-4 overflow-hidden"
                  >
                    <FeatureHeatmap features={data.features} />
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

          </div>
        )}
      </div>
    </div>
  )
}
