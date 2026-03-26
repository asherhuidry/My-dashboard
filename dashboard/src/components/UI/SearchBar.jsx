import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { Search, X, TrendingUp } from 'lucide-react'
import { fetchSearch } from '../../lib/api'
import { ASSET_COLORS } from '../../lib/utils'
import { motion, AnimatePresence } from 'framer-motion'

export default function SearchBar({ onSelect, placeholder = 'Search any symbol… AAPL, BTC, NVDA', autoFocus }) {
  const [q, setQ]           = useState('')
  const [results, setResults] = useState([])
  const [open, setOpen]     = useState(false)
  const [loading, setLoading] = useState(false)
  const [focused, setFocused] = useState(false)
  const navigate             = useNavigate()
  const ref                  = useRef()
  const timer                = useRef()

  useEffect(() => {
    clearTimeout(timer.current)
    if (!q) { setResults([]); return }
    setLoading(true)
    timer.current = setTimeout(async () => {
      try {
        const data = await fetchSearch(q)
        setResults(data.results ?? [])
        setOpen(true)
      } catch { setResults([]) }
      finally { setLoading(false) }
    }, 200)
  }, [q])

  // Load popular on focus
  useEffect(() => {
    if (focused && !q) {
      fetchSearch('').then(d => { setResults(d.results ?? []); setOpen(true) }).catch(() => {})
    }
  }, [focused, q])

  useEffect(() => {
    const handler = (e) => { if (!ref.current?.contains(e.target)) setOpen(false) }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const select = (sym) => {
    setQ('')
    setOpen(false)
    if (onSelect) onSelect(sym)
    else navigate(`/analyze?symbol=${sym}`)
  }

  const typeLabel = { equity: 'Stock', crypto: 'Crypto', forex: 'Forex', commodity: 'ETF/Commodity', etf: 'ETF' }

  return (
    <div ref={ref} className="relative w-full max-w-xl">
      <div className={`flex items-center gap-2 px-4 py-2.5 rounded-xl border transition-all duration-200 ${
        focused ? 'border-accent/60 bg-bg-hover shadow-glow' : 'border-border bg-bg-card'
      }`}>
        <Search size={15} className={focused ? 'text-accent' : 'text-text-muted'} />
        <input
          autoFocus={autoFocus}
          value={q}
          onChange={e => setQ(e.target.value)}
          onFocus={() => setFocused(true)}
          onBlur={() => setTimeout(() => setFocused(false), 200)}
          placeholder={placeholder}
          className="flex-1 bg-transparent text-sm text-text placeholder-text-muted outline-none"
        />
        {loading && <div className="w-3 h-3 border border-accent border-t-transparent rounded-full animate-spin" />}
        {q && !loading && <button onClick={() => { setQ(''); setOpen(false) }}><X size={13} className="text-text-muted hover:text-text" /></button>}
      </div>

      <AnimatePresence>
        {open && results.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            className="absolute top-full mt-2 w-full glass-bright rounded-xl border border-border shadow-card overflow-hidden z-50"
          >
            {!q && <div className="px-4 py-2 border-b border-border"><p className="text-[10px] text-text-muted uppercase tracking-wider">Popular</p></div>}
            {results.map((r, i) => (
              <button
                key={i}
                onClick={() => select(r.symbol)}
                className="w-full flex items-center gap-3 px-4 py-2.5 hover:bg-bg-hover transition-colors text-left"
              >
                <div className="w-7 h-7 rounded-lg flex items-center justify-center text-[10px] font-bold"
                  style={{ background: `${ASSET_COLORS[r.type] ?? '#3b82f6'}20`, color: ASSET_COLORS[r.type] ?? '#3b82f6' }}>
                  {r.symbol.replace('-USD','').replace('=X','').slice(0,3)}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium text-text">{r.symbol}</div>
                  <div className="text-xs text-text-muted truncate">{r.name}</div>
                </div>
                <div className="flex items-center gap-2">
                  {r.price && <span className="ticker-value text-xs text-text-secondary">${r.price}</span>}
                  <span className="text-[10px] px-1.5 py-0.5 rounded text-text-muted bg-bg border border-border">
                    {typeLabel[r.type] ?? r.type}
                  </span>
                </div>
              </button>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
