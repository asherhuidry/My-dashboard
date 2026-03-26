import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import axios from 'axios'
import { motion, AnimatePresence } from 'framer-motion'
import { SlidersHorizontal, TrendingUp, TrendingDown, Minus, ArrowUpDown, Loader2, Filter } from 'lucide-react'
import Header from '../components/Layout/Header'
import { fmt, pctColor, signalBadge, ASSET_COLORS } from '../lib/utils'

const fetchScreener = (params) =>
  axios.get('http://localhost:8000/api/screener', { params }).then(r => r.data)

const SORT_COLS = [
  { key:'ret_1d',       label:'1D %'       },
  { key:'ret_5d',       label:'5D %'       },
  { key:'ret_21d',      label:'21D %'      },
  { key:'rsi_14',       label:'RSI'        },
  { key:'adx_14',       label:'ADX'        },
  { key:'volume_ratio', label:'Vol Ratio'  },
  { key:'sharpe_21',    label:'Sharpe'     },
  { key:'realized_vol', label:'Volatility' },
]

function SignalChip({ signal }) {
  const Icon = signal==='bullish' ? TrendingUp : signal==='bearish' ? TrendingDown : Minus
  return (
    <span className={`flex items-center gap-1 text-[10px] font-semibold px-2 py-0.5 rounded-full ${signalBadge(signal)}`}>
      <Icon size={9} /> {signal}
    </span>
  )
}

function GaugeBar({ value, min=0, max=100, colorClass='bg-accent' }) {
  if (value == null) return <span className="text-text-muted">—</span>
  const pct = Math.min(Math.max((value - min) / (max - min) * 100, 0), 100)
  return (
    <div className="flex items-center gap-2 w-full">
      <div className="flex-1 h-1.5 bg-bg rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${colorClass}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="ticker-value text-xs text-text w-10 text-right">{value?.toFixed(1)}</span>
    </div>
  )
}

export default function Screener() {
  const navigate = useNavigate()
  const [assetClass, setAssetClass] = useState('all')
  const [signal,     setSignal]     = useState('all')
  const [minRsi,     setMinRsi]     = useState(0)
  const [maxRsi,     setMaxRsi]     = useState(100)
  const [minAdx,     setMinAdx]     = useState(0)
  const [sortBy,     setSortBy]     = useState('ret_1d')
  const [sortDesc,   setSortDesc]   = useState(true)
  const [showFilters, setShowFilters] = useState(true)

  const params = {
    asset_class: assetClass,
    signal,
    min_rsi:   minRsi,
    max_rsi:   maxRsi,
    min_adx:   minAdx,
    sort_by:   sortBy,
    sort_desc: sortDesc,
    limit:     50,
  }

  const { data, isLoading, refetch } = useQuery({
    queryKey: ['screener', params],
    queryFn:  () => fetchScreener(params),
    staleTime: 120_000,
  })

  const toggleSort = (col) => {
    if (sortBy === col) setSortDesc(d => !d)
    else { setSortBy(col); setSortDesc(true) }
  }

  return (
    <div className="flex flex-col h-full">
      <Header title="Screener" subtitle="Filter & rank assets by technical signals" />
      <div className="flex-1 overflow-hidden flex flex-col bg-grid">

        {/* Filter bar */}
        <div className="border-b border-border bg-bg-secondary/50 px-6 py-3">
          <div className="flex items-center gap-3 flex-wrap">
            {/* Asset class */}
            <div className="flex items-center gap-1">
              {['all','equity','crypto','etf','forex'].map(c => (
                <button key={c} onClick={() => setAssetClass(c)}
                  className={`text-xs px-2.5 py-1 rounded-lg border transition-all capitalize ${assetClass===c?'border-accent/40 bg-accent/10 text-accent':'border-border text-text-muted hover:text-text'}`}>
                  {c}
                </button>
              ))}
            </div>

            <div className="w-px h-4 bg-border" />

            {/* Signal filter */}
            <div className="flex items-center gap-1">
              {['all','bullish','bearish','neutral'].map(s => (
                <button key={s} onClick={() => setSignal(s)}
                  className={`text-xs px-2.5 py-1 rounded-lg border transition-all capitalize ${signal===s?'border-accent/40 bg-accent/10 text-accent':'border-border text-text-muted hover:text-text'}`}>
                  {s}
                </button>
              ))}
            </div>

            <div className="w-px h-4 bg-border" />

            {/* RSI range */}
            <div className="flex items-center gap-2 text-xs text-text-muted">
              <span>RSI</span>
              <input type="number" value={minRsi} onChange={e=>setMinRsi(+e.target.value)} min={0} max={100}
                className="w-12 bg-bg border border-border rounded px-1.5 py-0.5 text-text text-xs text-center" />
              <span>–</span>
              <input type="number" value={maxRsi} onChange={e=>setMaxRsi(+e.target.value)} min={0} max={100}
                className="w-12 bg-bg border border-border rounded px-1.5 py-0.5 text-text text-xs text-center" />
            </div>

            {/* ADX min */}
            <div className="flex items-center gap-2 text-xs text-text-muted">
              <span>Min ADX</span>
              <input type="number" value={minAdx} onChange={e=>setMinAdx(+e.target.value)} min={0}
                className="w-12 bg-bg border border-border rounded px-1.5 py-0.5 text-text text-xs text-center" />
            </div>

            <button onClick={() => refetch()}
              className="ml-auto text-xs text-text-muted hover:text-text border border-border px-3 py-1.5 rounded-lg hover:bg-bg-hover transition-colors flex items-center gap-1.5">
              <Filter size={11} /> Refresh
            </button>
          </div>
        </div>

        {/* Table */}
        <div className="flex-1 overflow-auto">
          {isLoading ? (
            <div className="flex flex-col items-center justify-center h-48 gap-3">
              <Loader2 size={28} className="text-accent animate-spin" />
              <p className="text-sm text-text-secondary">Scanning {assetClass === 'all' ? 'all assets' : assetClass}…</p>
              <p className="text-xs text-text-muted">Computing 74 features per asset</p>
            </div>
          ) : (
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-bg-secondary border-b border-border">
                <tr>
                  <th className="text-left px-4 py-2.5 text-text-muted font-medium w-32">Symbol</th>
                  <th className="text-right px-3 py-2.5 text-text-muted font-medium">Price</th>
                  {[{k:'ret_1d',l:'1D'},{k:'ret_5d',l:'5D'},{k:'ret_21d',l:'21D'}].map(c => (
                    <th key={c.k} className="text-right px-3 py-2.5 cursor-pointer hover:text-text transition-colors"
                      onClick={() => toggleSort(c.k)}>
                      <span className={`flex items-center justify-end gap-1 font-medium ${sortBy===c.k?'text-accent':'text-text-muted'}`}>
                        {c.l} <ArrowUpDown size={9} />
                      </span>
                    </th>
                  ))}
                  <th className="text-center px-3 py-2.5 text-text-muted font-medium">Signal</th>
                  <th className="px-3 py-2.5 text-text-muted font-medium w-32 cursor-pointer hover:text-text"
                    onClick={() => toggleSort('rsi_14')}>
                    <span className={`flex items-center gap-1 font-medium ${sortBy==='rsi_14'?'text-accent':'text-text-muted'}`}>
                      RSI <ArrowUpDown size={9} />
                    </span>
                  </th>
                  <th className="px-3 py-2.5 text-text-muted font-medium w-32 cursor-pointer hover:text-text"
                    onClick={() => toggleSort('adx_14')}>
                    <span className={`flex items-center gap-1 font-medium ${sortBy==='adx_14'?'text-accent':'text-text-muted'}`}>
                      ADX <ArrowUpDown size={9} />
                    </span>
                  </th>
                  <th className="text-right px-3 py-2.5 text-text-muted font-medium cursor-pointer hover:text-text"
                    onClick={() => toggleSort('realized_vol')}>
                    <span className={`flex items-center justify-end gap-1 font-medium ${sortBy==='realized_vol'?'text-accent':'text-text-muted'}`}>
                      Vol <ArrowUpDown size={9} />
                    </span>
                  </th>
                  <th className="text-right px-3 py-2.5 text-text-muted font-medium">52W</th>
                </tr>
              </thead>
              <tbody>
                {data?.results?.map((r, i) => (
                  <motion.tr
                    key={r.full_symbol}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: i * 0.015 }}
                    onClick={() => navigate(`/analyze?symbol=${r.full_symbol}`)}
                    className="border-b border-border/50 hover:bg-bg-hover cursor-pointer transition-colors group"
                  >
                    <td className="px-4 py-2.5">
                      <div className="flex items-center gap-2">
                        <div className="w-6 h-6 rounded flex items-center justify-center text-[9px] font-bold flex-shrink-0"
                          style={{ background: `${ASSET_COLORS[r.asset_class]??'#3b82f6'}20`, color: ASSET_COLORS[r.asset_class]??'#3b82f6' }}>
                          {r.symbol.slice(0,2)}
                        </div>
                        <div>
                          <div className="font-semibold text-text group-hover:text-accent transition-colors">{r.symbol}</div>
                          <div className="text-[9px] text-text-muted capitalize">{r.asset_class}</div>
                        </div>
                      </div>
                    </td>
                    <td className="px-3 py-2.5 text-right ticker-value text-text font-medium">${fmt.price(r.price)}</td>
                    {[r.ret_1d, r.ret_5d, r.ret_21d].map((v,j) => (
                      <td key={j} className={`px-3 py-2.5 text-right ticker-value font-medium ${pctColor(v)}`}>
                        {v != null ? `${v>0?'+':''}${v.toFixed(2)}%` : '—'}
                      </td>
                    ))}
                    <td className="px-3 py-2.5 text-center">
                      <SignalChip signal={r.signal} />
                    </td>
                    <td className="px-3 py-2.5 w-32">
                      <GaugeBar value={r.rsi_14} min={0} max={100}
                        colorClass={r.rsi_14<30?'bg-positive':r.rsi_14>70?'bg-negative':'bg-accent'} />
                    </td>
                    <td className="px-3 py-2.5 w-32">
                      <GaugeBar value={r.adx_14} min={0} max={60} colorClass="bg-purple" />
                    </td>
                    <td className="px-3 py-2.5 text-right ticker-value text-warning">
                      {r.realized_vol != null ? `${(r.realized_vol*100).toFixed(1)}%` : '—'}
                    </td>
                    <td className="px-3 py-2.5 text-right">
                      {r.pct_from_high != null && (
                        <span className={`ticker-value text-xs ${r.pct_from_high > -5 ? 'text-positive' : r.pct_from_high < -20 ? 'text-negative' : 'text-text-secondary'}`}>
                          {r.pct_from_high.toFixed(1)}%
                        </span>
                      )}
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          )}

          {!isLoading && data?.results?.length === 0 && (
            <div className="flex flex-col items-center justify-center h-32 text-text-muted text-sm">
              No assets match these filters
            </div>
          )}
        </div>

        {/* Footer */}
        {data && !isLoading && (
          <div className="px-4 py-2 border-t border-border bg-bg-secondary/50 flex items-center justify-between text-[10px] text-text-muted">
            <span>{data.total} assets matched · click any row to open full analysis</span>
            <span>Sorted by {sortBy.replace('_',' ')} {sortDesc ? '▼' : '▲'}</span>
          </div>
        )}
      </div>
    </div>
  )
}
