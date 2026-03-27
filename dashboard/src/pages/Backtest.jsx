import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { fetchBacktest } from '../lib/api'
import { fmt, pctColor } from '../lib/utils'
import Header from '../components/Layout/Header'
import SearchBar from '../components/UI/SearchBar'
import Spinner from '../components/UI/Spinner'
import ErrorBoundary from '../components/ErrorBoundary'
import { motion, AnimatePresence } from 'framer-motion'
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine, CartesianGrid,
} from 'recharts'
import {
  TrendingUp, TrendingDown, BarChart2, Activity, Target, Shield,
  ChevronUp, ChevronDown, ArrowRight, CheckCircle, XCircle,
} from 'lucide-react'

const DAY_OPTIONS = [
  { d: 365, label: '1Y'  },
  { d: 730, label: '2Y'  },
  { d: 1000, label: '3Y' },
  { d: 1500, label: '5Y' },
]

function MetricCard({ label, value, sub, color, icon: Icon, positive, delay }) {
  const isPos  = positive ?? (typeof value === 'number' ? value > 0 : null)
  const colors = {
    accent:   ['bg-accent/10',   'border-accent/20',   'text-accent'   ],
    positive: ['bg-positive/10', 'border-positive/20', 'text-positive' ],
    negative: ['bg-negative/10', 'border-negative/20', 'text-negative' ],
    warning:  ['bg-warning/10',  'border-warning/20',  'text-warning'  ],
    purple:   ['bg-purple/10',   'border-purple/20',   'text-purple'   ],
    cyan:     ['bg-cyan/10',     'border-cyan/20',     'text-cyan'     ],
    neutral:  ['bg-bg-hover',    'border-border',      'text-text'     ],
  }
  const [bg, border, tc] = colors[color] ?? colors.neutral

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}
      transition={{ delay: delay ?? 0 }}
      className={`glass-card rounded-xl p-4 border ${border} relative overflow-hidden`}
    >
      <div className="flex items-center justify-between mb-3">
        <span className="text-[9px] text-text-muted uppercase tracking-wider font-semibold">{label}</span>
        {Icon && <Icon size={13} className={tc} />}
      </div>
      <div className={`ticker-value text-xl font-bold ${tc}`}>{value}</div>
      {sub && <div className="text-[10px] text-text-muted mt-1">{sub}</div>}
    </motion.div>
  )
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  const d = payload[0]?.payload
  return (
    <div className="glass-card rounded-xl px-3 py-2 border border-border text-[10px] space-y-1">
      <div className="text-text-muted">{new Date(label * 1000).toLocaleDateString()}</div>
      {payload.map(p => (
        <div key={p.name} className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full" style={{ background: p.color }} />
          <span className="text-text-muted">{p.name}:</span>
          <span className="text-text font-semibold ticker-value">${fmt.price(p.value)}</span>
        </div>
      ))}
      {d?.signal === 1 && <div className="text-positive font-semibold mt-0.5">● Long</div>}
    </div>
  )
}

function EquityCurve({ data, symbol, metrics }) {
  if (!data?.length) return null

  const initial = data[0]?.equity ?? 10000
  const finalEq = data[data.length - 1]?.equity ?? initial
  const finalBm = data[data.length - 1]?.benchmark ?? initial
  const stratUp = finalEq >= initial
  const bmUp    = finalBm >= initial

  return (
    <div className="glass-card rounded-xl border border-border p-5">
      <div className="flex items-center justify-between mb-4 flex-wrap gap-3">
        <div>
          <div className="text-sm font-semibold text-text">{symbol} — Signal Strategy vs Buy & Hold</div>
          <div className="text-[10px] text-text-muted mt-0.5">Starting value: $10,000</div>
        </div>
        <div className="flex items-center gap-4 text-[10px]">
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-0.5 bg-accent rounded" />
            <span className="text-text-secondary">Strategy</span>
            <span className={`ticker-value font-bold ${stratUp ? 'text-positive' : 'text-negative'}`}>
              {stratUp ? '+' : ''}{fmt.pct(metrics.total_return_pct / 100 * 100 - 100, 1)}
            </span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-0.5 bg-text-muted/40 rounded" style={{ borderStyle: 'dashed' }} />
            <span className="text-text-secondary">Buy & Hold</span>
            <span className={`ticker-value font-bold ${bmUp ? 'text-positive' : 'text-negative'}`}>
              {bmUp ? '+' : ''}{fmt.pct(metrics.benchmark_return_pct / 100 * 100 - 100, 1)}
            </span>
          </div>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={280}>
        <AreaChart data={data} margin={{ top: 4, right: 4, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="stratGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%"   stopColor="#3b82f6" stopOpacity={0.25} />
              <stop offset="95%"  stopColor="#3b82f6" stopOpacity={0.0}  />
            </linearGradient>
            <linearGradient id="bmGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%"  stopColor="#64748b" stopOpacity={0.12} />
              <stop offset="95%" stopColor="#64748b" stopOpacity={0.0}  />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(26,39,64,0.5)" />
          <XAxis dataKey="time"
            tickFormatter={t => new Date(t * 1000).toLocaleDateString('en-US', { month:'short', year:'2-digit' })}
            tick={{ fill:'#475569', fontSize:9 }} axisLine={false} tickLine={false} />
          <YAxis tickFormatter={v => `$${(v/1000).toFixed(0)}k`}
            tick={{ fill:'#475569', fontSize:9 }} axisLine={false} tickLine={false} width={42} />
          <Tooltip content={<CustomTooltip />} />
          <ReferenceLine y={10000} stroke="#475569" strokeDasharray="4 4" strokeWidth={1} />
          <Area type="monotone" dataKey="benchmark" name="Buy & Hold"
            stroke="#475569" strokeWidth={1} strokeDasharray="4 4"
            fill="url(#bmGrad)" dot={false} />
          <Area type="monotone" dataKey="equity" name="Strategy"
            stroke="#3b82f6" strokeWidth={2}
            fill="url(#stratGrad)" dot={false} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}

function TradeTable({ trades }) {
  const [show, setShow] = useState(false)
  if (!trades?.length) return null

  return (
    <div className="glass-card rounded-xl border border-border overflow-hidden">
      <button onClick={() => setShow(s => !s)}
        className="w-full flex items-center justify-between px-5 py-3.5 hover:bg-bg-hover/50 transition-colors">
        <div className="flex items-center gap-2">
          <Activity size={13} className="text-accent" />
          <span className="text-sm font-semibold text-text">Trade Log</span>
          <span className="text-[10px] text-text-muted px-2 py-0.5 rounded border border-border bg-bg">
            {trades.length} trades shown (last 20)
          </span>
        </div>
        {show ? <ChevronUp size={13} className="text-text-muted" /> : <ChevronDown size={13} className="text-text-muted" />}
      </button>
      <AnimatePresence>
        {show && (
          <motion.div initial={{ height: 0 }} animate={{ height: 'auto' }} exit={{ height: 0 }}
            className="overflow-hidden border-t border-border">
            <div className="overflow-x-auto">
              <table className="w-full text-[10px] data-table">
                <thead>
                  <tr className="border-b border-border">
                    {['Entry Date','Exit Date','Entry $','Exit $','Return','Days','Result'].map(h => (
                      <th key={h} className="text-left px-4 py-2 text-text-muted uppercase tracking-wider font-semibold">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {[...trades].reverse().map((t, i) => (
                    <tr key={i} className="border-b border-border/40">
                      <td className="px-4 py-2 text-text-secondary">{t.entry_date}</td>
                      <td className="px-4 py-2 text-text-secondary">{t.exit_date}</td>
                      <td className="px-4 py-2 ticker-value text-text">${fmt.price(t.entry_price)}</td>
                      <td className="px-4 py-2 ticker-value text-text">${fmt.price(t.exit_price)}</td>
                      <td className={`px-4 py-2 ticker-value font-semibold ${t.return_pct > 0 ? 'text-positive' : 'text-negative'}`}>
                        {t.return_pct > 0 ? '+' : ''}{t.return_pct}%
                      </td>
                      <td className="px-4 py-2 text-text-muted">{t.bars_held}d</td>
                      <td className="px-4 py-2">
                        {t.win
                          ? <span className="flex items-center gap-1 text-positive"><CheckCircle size={10} /> Win</span>
                          : <span className="flex items-center gap-1 text-negative"><XCircle size={10} /> Loss</span>
                        }
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export default function Backtest() {
  const [symbol, setSymbol] = useState('')
  const [days,   setDays]   = useState(730)

  const { data, isLoading, error } = useQuery({
    queryKey:  ['backtest', symbol, days],
    queryFn:   () => fetchBacktest(symbol, days),
    enabled:   symbol.length > 0,
    staleTime: 300_000,
    retry: 1,
  })

  const m = data?.metrics

  return (
    <div className="flex flex-col h-full">
      <Header title="Strategy Backtest" subtitle="Walk-forward signal simulation vs buy & hold benchmark" />

      <div className="flex-1 overflow-y-auto bg-grid">

        {/* Search bar */}
        <div className="px-5 py-3 border-b border-border bg-bg-secondary/60 backdrop-blur flex items-center gap-4 flex-wrap sticky top-0 z-20">
          <SearchBar onSelect={setSymbol} placeholder="Enter symbol to backtest…" />
          <div className="flex items-center gap-1 bg-bg/60 rounded-lg border border-border p-0.5">
            {DAY_OPTIONS.map(({ d, label }) => (
              <button key={d} onClick={() => setDays(d)}
                className={`px-2.5 py-1 rounded-md text-xs font-semibold transition-all ${
                  days === d ? 'bg-accent/20 text-accent border border-accent/30 shadow-glow' : 'text-text-muted hover:text-text'
                }`}>{label}</button>
            ))}
          </div>
        </div>

        {/* Empty state */}
        {!symbol && (
          <div className="flex flex-col items-center justify-center h-72 gap-4 text-center">
            <div className="w-16 h-16 rounded-2xl bg-accent/8 border border-accent/15 flex items-center justify-center animate-float">
              <BarChart2 size={28} className="text-accent/60" />
            </div>
            <div>
              <p className="text-text-secondary text-sm font-medium">Enter a symbol to run a backtest</p>
              <p className="text-text-muted text-xs mt-1">Tests signal consensus strategy vs buy & hold</p>
            </div>
            <div className="flex gap-2 flex-wrap justify-center">
              {['AAPL','NVDA','BTC-USD','SPY','TSLA'].map(s => (
                <button key={s} onClick={() => setSymbol(s)}
                  className="text-xs px-2.5 py-1 rounded-lg border border-border hover:border-accent/40 hover:bg-accent/8 text-text-muted hover:text-accent transition-all">
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {isLoading && (
          <div className="flex flex-col items-center justify-center h-64 gap-3">
            <Spinner size="lg" text={`Backtesting ${symbol}…`} />
            <p className="text-xs text-text-muted">Computing signals across {days} trading days</p>
          </div>
        )}

        {error && !isLoading && (
          <div className="m-5 glass rounded-xl p-5 border border-negative/25 text-center">
            <p className="text-negative text-sm">Backtest failed for {symbol}</p>
            <p className="text-text-muted text-xs mt-1">{error.message}</p>
          </div>
        )}

        {data && m && (
          <div className="p-5 space-y-4">

            {/* Header */}
            <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}
              className="glass-card rounded-2xl border border-border p-5">
              <div className="flex items-start justify-between flex-wrap gap-4">
                <div>
                  <h2 className="text-2xl font-bold text-text">{data.symbol}</h2>
                  <p className="text-text-muted text-xs mt-1">
                    {m.years_backtested}Y backtest · {m.days_backtested} trading days ·{' '}
                    ${fmt.price(data.latest_price)} current price
                  </p>
                </div>
                <div className="text-right">
                  <div className={`ticker-value text-3xl font-bold ${m.total_return_pct > 0 ? 'text-positive' : 'text-negative'}`}>
                    {m.total_return_pct > 0 ? '+' : ''}{m.total_return_pct}%
                  </div>
                  <div className="text-xs text-text-muted mt-0.5">Strategy total return</div>
                  <div className={`text-xs mt-1 ${m.alpha_pct > 0 ? 'text-positive' : 'text-negative'}`}>
                    Alpha vs benchmark: {m.alpha_pct > 0 ? '+' : ''}{m.alpha_pct}%
                  </div>
                </div>
              </div>
            </motion.div>

            {/* Metric grid */}
            <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-3">
              <MetricCard label="Total Return"  value={`${m.total_return_pct > 0 ? '+' : ''}${m.total_return_pct}%`}
                color={m.total_return_pct > 0 ? 'positive' : 'negative'} icon={TrendingUp} delay={0.05} />
              <MetricCard label="Benchmark"     value={`${m.benchmark_return_pct > 0 ? '+' : ''}${m.benchmark_return_pct}%`}
                color={m.benchmark_return_pct > 0 ? 'positive' : 'negative'} icon={ArrowRight} delay={0.08} />
              <MetricCard label="Sharpe Ratio"  value={m.sharpe_ratio.toFixed(2)}
                color={m.sharpe_ratio > 1 ? 'positive' : m.sharpe_ratio > 0.5 ? 'warning' : 'negative'} icon={Activity} delay={0.11} />
              <MetricCard label="Max Drawdown"  value={`${m.max_drawdown_pct}%`}
                color={m.max_drawdown_pct > -15 ? 'warning' : 'negative'} icon={Shield} delay={0.14} />
              <MetricCard label="Win Rate"      value={`${m.win_rate_pct}%`}
                color={m.win_rate_pct > 55 ? 'positive' : m.win_rate_pct > 45 ? 'warning' : 'negative'} icon={Target} delay={0.17} />
              <MetricCard label="Total Trades"  value={m.total_trades}
                color="accent" icon={BarChart2} sub={`Avg ${m.avg_trade_pct > 0 ? '+' : ''}${m.avg_trade_pct}% / trade`} delay={0.20} />
            </div>

            {/* Secondary metrics */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <MetricCard label="Annualised Return" value={`${m.annualised_return_pct > 0 ? '+' : ''}${m.annualised_return_pct}%`}
                color={m.annualised_return_pct > 10 ? 'positive' : 'warning'} delay={0.23} />
              <MetricCard label="Sortino Ratio"     value={m.sortino_ratio.toFixed(2)}
                color={m.sortino_ratio > 1 ? 'positive' : 'warning'} delay={0.26} />
              <MetricCard label="Calmar Ratio"      value={m.calmar_ratio.toFixed(2)}
                color={m.calmar_ratio > 1 ? 'positive' : 'warning'} delay={0.29} />
              <MetricCard label="Alpha vs Benchmark" value={`${m.alpha_pct > 0 ? '+' : ''}${m.alpha_pct}%`}
                color={m.alpha_pct > 0 ? 'positive' : 'negative'} delay={0.32} />
            </div>

            {/* Equity curve */}
            <ErrorBoundary>
              <EquityCurve data={data.equity_curve} symbol={data.symbol} metrics={m} />
            </ErrorBoundary>

            {/* Trade log */}
            <TradeTable trades={data.trades} />

          </div>
        )}
      </div>
    </div>
  )
}
