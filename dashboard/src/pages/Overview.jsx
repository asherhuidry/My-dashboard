import { useQuery } from '@tanstack/react-query'
import { fetchDbStats, fetchHealth, fetchPrices } from '../lib/api'
import { fmt, pctColor } from '../lib/utils'
import StatCard from '../components/UI/StatCard'
import SearchBar from '../components/UI/SearchBar'
import Spinner from '../components/UI/Spinner'
import Header from '../components/Layout/Header'
import { Database, TrendingUp, Activity, Layers, Search, Zap } from 'lucide-react'
import { motion } from 'framer-motion'

const WATCHLIST = ['AAPL','NVDA','MSFT','TSLA','BTC-USD','ETH-USD','SOL-USD','SPY']

function WatchCard({ symbol }) {
  const { data, isLoading } = useQuery({
    queryKey: ['prices', symbol, 7],
    queryFn:  () => fetchPrices(symbol, 7),
    staleTime: 60_000,
  })

  const label = symbol.replace('-USD','').replace('=X','')
  const up    = data ? data.change_pct >= 0 : null

  return (
    <motion.div
      whileHover={{ scale: 1.02 }}
      className="glass rounded-xl p-4 border border-border hover:border-border-bright transition-all cursor-pointer group"
    >
      <div className="flex items-start justify-between mb-2">
        <div>
          <div className="text-xs font-bold text-text">{label}</div>
          <div className="text-[10px] text-text-muted mt-0.5">
            {symbol.includes('-USD') ? 'Crypto' : symbol.includes('=X') ? 'Forex' : 'Equity'}
          </div>
        </div>
        {isLoading
          ? <div className="w-3 h-3 border border-accent border-t-transparent rounded-full animate-spin" />
          : up != null && (
            <span className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${up ? 'badge-bullish' : 'badge-bearish'}`}>
              {up ? '▲' : '▼'}
            </span>
          )
        }
      </div>
      {data ? (
        <>
          <div className="ticker-value text-lg font-semibold text-text">
            ${fmt.price(data.latest)}
          </div>
          <div className={`text-xs font-medium mt-0.5 ${up ? 'text-positive' : 'text-negative'}`}>
            {fmt.pct(data.change_pct)}
          </div>
        </>
      ) : (
        <div className="shimmer h-6 w-20 rounded mt-1" />
      )}
    </motion.div>
  )
}

export default function Overview() {
  const { data: stats } = useQuery({ queryKey: ['db-stats'], queryFn: fetchDbStats, staleTime: 60_000 })
  const { data: health } = useQuery({ queryKey: ['health'],   queryFn: fetchHealth,  staleTime: 30_000 })

  const priceRows  = stats?.tables?.find(t => t.table === 'prices')?.rows ?? 0
  const macroRows  = stats?.tables?.find(t => t.table === 'macro_events')?.rows ?? 0
  const totalRows  = stats?.total_rows ?? 0
  const servicesUp = health?.checks ? Object.values(health.checks).filter(Boolean).length : 0
  const totalSvc   = health?.checks ? Object.keys(health.checks).length : 5

  return (
    <div className="flex flex-col h-full">
      <Header title="Overview" subtitle="FinBrain Autonomous Financial Intelligence" />
      <div className="flex-1 overflow-y-auto p-6 space-y-6 bg-grid">

        {/* Hero search */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="gradient-border rounded-2xl p-8 text-center relative overflow-hidden"
        >
          <div className="absolute inset-0 bg-gradient-to-br from-accent/5 via-purple/5 to-cyan/5" />
          <div className="relative z-10">
            <div className="inline-flex items-center gap-2 text-[10px] text-accent uppercase tracking-widest mb-3 px-3 py-1 rounded-full border border-accent/30 bg-accent/10">
              <Zap size={10} /> AI-Powered Analysis
            </div>
            <h2 className="text-3xl font-bold text-text mb-2">
              Search any <span className="neon-blue">stock</span> or <span className="neon-green">crypto</span>
            </h2>
            <p className="text-text-secondary text-sm mb-6">74-feature ML analysis · Live price data · Signal consensus</p>
            <div className="flex justify-center">
              <SearchBar placeholder="Try AAPL, NVDA, BTC-USD, MSFT…" />
            </div>
          </div>
        </motion.div>

        {/* Stats row */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <StatCard label="Price Rows"    value={fmt.compact(priceRows)}  icon={TrendingUp} color="accent"    delay={0.05} />
          <StatCard label="Macro Records" value={fmt.compact(macroRows)}  icon={Activity}   color="positive"  delay={0.10} />
          <StatCard label="Total DB Rows" value={fmt.compact(totalRows)}  icon={Database}   color="purple"    delay={0.15} />
          <StatCard label="Services Live" value={`${servicesUp}/${totalSvc}`} icon={Layers} color={servicesUp===totalSvc?'positive':'warning'} delay={0.20} />
        </div>

        {/* Watchlist */}
        <div>
          <h3 className="text-sm font-semibold text-text mb-3 flex items-center gap-2">
            <TrendingUp size={14} className="text-accent" /> Watchlist
          </h3>
          <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-3">
            {WATCHLIST.map(sym => <WatchCard key={sym} symbol={sym} />)}
          </div>
        </div>

        {/* DB Table Health */}
        {stats?.tables && (
          <div>
            <h3 className="text-sm font-semibold text-text mb-3 flex items-center gap-2">
              <Database size={14} className="text-accent" /> Database Health
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
              {stats.tables.map(t => (
                <div key={t.table} className="glass rounded-xl p-3 border border-border">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-[10px] text-text-muted uppercase tracking-wider font-medium truncate">{t.table.replace('_',' ')}</span>
                    <span className={`w-1.5 h-1.5 rounded-full ${t.status==='ok' ? 'bg-positive' : 'bg-negative'}`} />
                  </div>
                  <div className="ticker-value text-base font-semibold text-text">{fmt.compact(t.rows)}</div>
                  <div className="text-[10px] text-text-muted">rows</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
