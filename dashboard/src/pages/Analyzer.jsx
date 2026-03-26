import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useSearchParams } from 'react-router-dom'
import { fetchAnalysis } from '../lib/api'
import { fmt, pctColor, signalBadge } from '../lib/utils'
import Header from '../components/Layout/Header'
import SearchBar from '../components/UI/SearchBar'
import Spinner from '../components/UI/Spinner'
import PriceChart from '../components/Charts/PriceChart'
import { RSIChart, MACDChart, StochasticChart } from '../components/Charts/IndicatorChart'
import FeatureHeatmap from '../components/Charts/FeatureHeatmap'
import { motion, AnimatePresence } from 'framer-motion'
import { TrendingUp, TrendingDown, Minus, ChevronDown, ChevronUp, BarChart2, Activity, Layers } from 'lucide-react'

const OVERLAY_OPTIONS = [
  { key: 'bb_upper',  label: 'Bollinger Bands', keys: ['bb_upper','bb_lower','bb_mid'] },
  { key: 'ema_9',     label: 'EMA 9/21',        keys: ['ema_9','ema_21'] },
  { key: 'ema_50',    label: 'EMA 50/200',       keys: ['ema_50','ema_200'] },
]

function ReturnBadge({ label, value }) {
  const color = pctColor(value)
  return (
    <div className="glass rounded-lg px-3 py-2 text-center border border-border min-w-[70px]">
      <div className="text-[10px] text-text-muted mb-0.5 uppercase tracking-wider">{label}</div>
      <div className={`ticker-value text-sm font-semibold ${color}`}>{fmt.pct(value)}</div>
    </div>
  )
}

function SignalMeter({ score, overall }) {
  const pct   = Math.min(Math.max((score + 100) / 2, 0), 100)
  const color = overall === 'bullish' ? '#10b981' : overall === 'bearish' ? '#ef4444' : '#94a3b8'

  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs text-text-muted">Signal Strength</span>
        <span className={`text-xs font-semibold ${overall==='bullish'?'text-positive':overall==='bearish'?'text-negative':'text-text-secondary'}`}>
          {score > 0 ? '+' : ''}{score}
        </span>
      </div>
      <div className="h-2 bg-bg rounded-full overflow-hidden">
        <div className="h-full rounded-full transition-all duration-700"
          style={{ width: `${pct}%`, background: `linear-gradient(90deg, #ef4444, ${color})` }} />
      </div>
      <div className="flex justify-between text-[9px] text-text-muted mt-1">
        <span>Bearish</span><span>Neutral</span><span>Bullish</span>
      </div>
    </div>
  )
}

export default function Analyzer() {
  const [params] = useSearchParams()
  const [symbol, setSymbol] = useState(params.get('symbol') ?? '')
  const [days,   setDays]   = useState(365)
  const [overlays, setOverlays] = useState(['bb_upper','bb_lower','bb_mid'])
  const [activeIndicator, setActiveIndicator] = useState('rsi')
  const [showFeatures, setShowFeatures] = useState(false)

  const { data, isLoading, error } = useQuery({
    queryKey: ['analysis', symbol, days],
    queryFn:  () => fetchAnalysis(symbol, days),
    enabled:  symbol.length > 0,
    staleTime: 120_000,
  })

  const toggleOverlay = (keys) => {
    setOverlays(prev => {
      const has = keys.every(k => prev.includes(k))
      return has ? prev.filter(k => !keys.includes(k)) : [...new Set([...prev, ...keys])]
    })
  }

  const sig = data?.signal
  const SigIcon = sig?.overall === 'bullish' ? TrendingUp : sig?.overall === 'bearish' ? TrendingDown : Minus

  return (
    <div className="flex flex-col h-full">
      <Header title="Stock Analyzer" subtitle="Full technical analysis with 74 ML features" />
      <div className="flex-1 overflow-y-auto bg-grid">

        {/* Search bar */}
        <div className="px-6 py-4 border-b border-border bg-bg-secondary/50 flex items-center gap-4 flex-wrap">
          <SearchBar onSelect={setSymbol} placeholder="Enter symbol…" />
          <div className="flex items-center gap-2">
            {[90,180,365,730].map(d => (
              <button key={d}
                onClick={() => setDays(d)}
                className={`px-2.5 py-1 rounded text-xs font-medium transition-colors ${days===d ? 'bg-accent/20 text-accent border border-accent/30' : 'text-text-muted hover:text-text border border-transparent'}`}
              >{d}D</button>
            ))}
          </div>
        </div>

        {!symbol && (
          <div className="flex flex-col items-center justify-center h-64 text-center">
            <BarChart2 size={40} className="text-border mb-3" />
            <p className="text-text-secondary text-sm">Search a symbol to begin full analysis</p>
            <p className="text-text-muted text-xs mt-1">Try AAPL, NVDA, BTC-USD, ETH-USD, MSFT</p>
          </div>
        )}

        {isLoading && (
          <div className="flex items-center justify-center h-64">
            <Spinner size="lg" text={`Analyzing ${symbol}…`} />
          </div>
        )}

        {error && (
          <div className="m-6 glass rounded-xl p-6 border border-negative/30 text-center">
            <p className="text-negative text-sm">Could not load {symbol}. Check the symbol and try again.</p>
          </div>
        )}

        {data && (
          <div className="p-6 space-y-4">
            {/* Header info */}
            <motion.div initial={{ opacity:0 }} animate={{ opacity:1 }} className="flex items-start justify-between flex-wrap gap-4">
              <div>
                <div className="flex items-center gap-3">
                  <h2 className="text-2xl font-bold text-text">{data.symbol}</h2>
                  {sig && (
                    <span className={`flex items-center gap-1.5 text-xs font-semibold px-2.5 py-1 rounded-full ${signalBadge(sig.overall)}`}>
                      <SigIcon size={11} /> {sig.overall.toUpperCase()}
                    </span>
                  )}
                </div>
                <p className="text-text-secondary text-sm mt-0.5">{data.name}</p>
                <p className="text-text-muted text-xs">{data.sector}{data.industry ? ` · ${data.industry}` : ''}</p>
              </div>
              <div className="text-right">
                <div className="ticker-value text-3xl font-bold text-text">${fmt.price(data.price)}</div>
                <div className={`text-sm font-medium ${pctColor(data.returns?.['1d'])}`}>
                  {fmt.pct(data.returns?.['1d'])} today
                </div>
                {data.market_cap && <div className="text-xs text-text-muted">{fmt.big(data.market_cap)} mkt cap</div>}
              </div>
            </motion.div>

            {/* Returns strip */}
            <div className="flex items-center gap-2 flex-wrap">
              <ReturnBadge label="1D"  value={data.returns?.['1d']} />
              <ReturnBadge label="5D"  value={data.returns?.['5d']} />
              <ReturnBadge label="21D" value={data.returns?.['21d']} />
              <ReturnBadge label="63D" value={data.returns?.['63d']} />
              <ReturnBadge label="YTD" value={data.returns?.ytd} />
              {data.vol_52w != null && (
                <div className="glass rounded-lg px-3 py-2 text-center border border-border min-w-[70px]">
                  <div className="text-[10px] text-text-muted mb-0.5 uppercase tracking-wider">Vol 21D</div>
                  <div className="ticker-value text-sm font-semibold text-warning">{(data.vol_52w * 100).toFixed(1)}%</div>
                </div>
              )}
              {data.pe_ratio != null && (
                <div className="glass rounded-lg px-3 py-2 text-center border border-border min-w-[70px]">
                  <div className="text-[10px] text-text-muted mb-0.5 uppercase tracking-wider">P/E</div>
                  <div className="ticker-value text-sm font-semibold text-text">{fmt.num(data.pe_ratio, 1)}x</div>
                </div>
              )}
            </div>

            {/* Main layout: chart + signal panel */}
            <div className="grid grid-cols-1 xl:grid-cols-4 gap-4">
              {/* Chart col */}
              <div className="xl:col-span-3 space-y-4">
                {/* Overlay toggles */}
                <div className="flex items-center gap-2 flex-wrap">
                  {OVERLAY_OPTIONS.map(opt => {
                    const active = opt.keys.every(k => overlays.includes(k))
                    return (
                      <button key={opt.key} onClick={() => toggleOverlay(opt.keys)}
                        className={`text-xs px-2.5 py-1 rounded-lg border transition-all ${active ? 'border-accent/40 bg-accent/10 text-accent' : 'border-border text-text-muted hover:text-text'}`}>
                        {opt.label}
                      </button>
                    )
                  })}
                </div>

                {/* Price chart */}
                <div className="glass rounded-xl border border-border overflow-hidden" style={{ height: 340 }}>
                  <PriceChart candles={data.candles} indicators={data.indicators} overlays={overlays} />
                </div>

                {/* Indicator selector */}
                <div className="flex items-center gap-2 border-b border-border pb-3">
                  {[
                    { id:'rsi',   label:'RSI' },
                    { id:'macd',  label:'MACD' },
                    { id:'stoch', label:'Stochastic' },
                  ].map(tab => (
                    <button key={tab.id} onClick={() => setActiveIndicator(tab.id)}
                      className={`text-xs px-3 py-1.5 rounded-lg border transition-all ${activeIndicator===tab.id ? 'border-accent/40 bg-accent/10 text-accent' : 'border-border text-text-muted hover:text-text'}`}>
                      {tab.label}
                    </button>
                  ))}
                </div>

                {/* Indicator panel */}
                <div className="glass rounded-xl border border-border overflow-hidden" style={{ height: 180 }}>
                  {activeIndicator === 'rsi'   && <RSIChart data14={data.indicators.rsi_14} data28={data.indicators.rsi_28} />}
                  {activeIndicator === 'macd'  && <MACDChart macd={data.indicators.macd} signal={data.indicators.macd_signal} hist={data.indicators.macd_hist} />}
                  {activeIndicator === 'stoch' && <StochasticChart k={data.indicators.stoch_k} d={data.indicators.stoch_d} />}
                </div>
              </div>

              {/* Signal panel */}
              <div className="space-y-4">
                {sig && (
                  <div className={`glass rounded-xl p-4 border ${sig.overall==='bullish'?'border-positive/30 glow-green':sig.overall==='bearish'?'border-negative/30 glow-red':'border-border'}`}>
                    <div className="flex items-center gap-2 mb-4">
                      <SigIcon size={16} className={sig.overall==='bullish'?'text-positive':sig.overall==='bearish'?'text-negative':'text-text-secondary'} />
                      <span className="text-sm font-semibold text-text capitalize">{sig.overall} Signal</span>
                    </div>
                    <SignalMeter score={sig.score} overall={sig.overall} />
                    <div className="mt-4 grid grid-cols-2 gap-2">
                      <div className="text-center p-2 rounded-lg bg-positive/10 border border-positive/20">
                        <div className="text-lg font-bold text-positive">{sig.bull_count}</div>
                        <div className="text-[10px] text-text-muted">Bullish</div>
                      </div>
                      <div className="text-center p-2 rounded-lg bg-negative/10 border border-negative/20">
                        <div className="text-lg font-bold text-negative">{sig.bear_count}</div>
                        <div className="text-[10px] text-text-muted">Bearish</div>
                      </div>
                    </div>
                    <div className="mt-3 space-y-1">
                      {sig.signals.map((s,i) => (
                        <div key={i} className="flex items-center gap-2 text-xs">
                          <div className={`w-1.5 h-1.5 rounded-full ${s==='bullish'?'bg-positive':s==='bearish'?'bg-negative':'bg-text-muted'}`} />
                          <span className="text-text-muted capitalize">{s}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Key features */}
                <div className="glass rounded-xl p-4 border border-border">
                  <h4 className="text-xs font-semibold text-text mb-3 uppercase tracking-wider">Key Indicators</h4>
                  <div className="space-y-2">
                    {[
                      { label:'RSI 14',    val: data.features?.rsi_14,    fmt: v => v?.toFixed(1) },
                      { label:'RSI 28',    val: data.features?.rsi_28,    fmt: v => v?.toFixed(1) },
                      { label:'ADX 14',    val: data.features?.adx_14,    fmt: v => v?.toFixed(1) },
                      { label:'ATR',       val: data.atr,                  fmt: v => `$${v?.toFixed(2)}` },
                      { label:'BB %',      val: data.features?.bb_pct,    fmt: v => `${(v*100)?.toFixed(1)}%` },
                      { label:'Vol 5D',    val: data.features?.realized_vol_5d,  fmt: v => `${(v*100)?.toFixed(1)}%` },
                    ].map(row => row.val != null && (
                      <div key={row.label} className="flex items-center justify-between text-xs">
                        <span className="text-text-muted">{row.label}</span>
                        <span className="ticker-value text-text font-medium">{row.fmt(row.val)}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Description */}
                {data.description && (
                  <div className="glass rounded-xl p-4 border border-border">
                    <h4 className="text-xs font-semibold text-text mb-2 uppercase tracking-wider">About</h4>
                    <p className="text-xs text-text-muted leading-relaxed">{data.description}</p>
                  </div>
                )}
              </div>
            </div>

            {/* Feature heatmap (expandable) */}
            <div className="glass rounded-xl border border-border overflow-hidden">
              <button
                onClick={() => setShowFeatures(f => !f)}
                className="w-full flex items-center justify-between px-5 py-3 hover:bg-bg-hover transition-colors"
              >
                <div className="flex items-center gap-2">
                  <Layers size={14} className="text-accent" />
                  <span className="text-sm font-medium text-text">Feature Heatmap</span>
                  <span className="text-[10px] text-text-muted">74 computed ML features</span>
                </div>
                {showFeatures ? <ChevronUp size={14} className="text-text-muted" /> : <ChevronDown size={14} className="text-text-muted" />}
              </button>
              <AnimatePresence>
                {showFeatures && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    className="border-t border-border px-5 py-4"
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
