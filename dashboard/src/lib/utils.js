export const fmt = {
  price:   (v, d=2) => v == null ? '—' : Number(v).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d }),
  pct:     (v, d=2) => v == null ? '—' : `${v > 0 ? '+' : ''}${Number(v).toFixed(d)}%`,
  big:     (v)      => { if (v == null) return '—'; if (v >= 1e12) return `$${(v/1e12).toFixed(2)}T`; if (v >= 1e9) return `$${(v/1e9).toFixed(2)}B`; if (v >= 1e6) return `$${(v/1e6).toFixed(2)}M`; return `$${v}`; },
  num:     (v, d=2) => v == null ? '—' : Number(v).toFixed(d),
  date:    (s)      => new Date(s).toLocaleDateString('en-US', { month:'short', day:'numeric', year:'numeric' }),
  time:    (s)      => new Date(s).toLocaleTimeString('en-US', { hour:'2-digit', minute:'2-digit' }),
  compact: (v)      => v == null ? '—' : Intl.NumberFormat('en', { notation:'compact' }).format(v),
}

export const signalColor = (s) => ({ bullish:'text-positive', bearish:'text-negative', neutral:'text-text-secondary' }[s] ?? 'text-text-secondary')
export const signalBadge = (s) => ({ bullish:'badge-bullish', bearish:'badge-bearish', neutral:'badge-neutral' }[s] ?? 'badge-neutral')
export const pctColor    = (v) => v == null ? '' : v > 0 ? 'text-positive' : v < 0 ? 'text-negative' : 'text-text-secondary'

export const ASSET_COLORS = {
  equity:    '#3b82f6',
  crypto:    '#f59e0b',
  forex:     '#10b981',
  commodity: '#8b5cf6',
  DataSource:'#06b6d4',
  Database:  '#3b82f6',
  VectorDB:  '#3b82f6',
  GraphDB:   '#3b82f6',
  Asset:     '#a78bfa',
  Pipeline:  '#10b981',
  Model:     '#ef4444',
  Signal:    '#f59e0b',
  MacroIndicator:'#10b981',
  Sector:    '#8b5cf6',
  Agent:     '#f97316',
  etf:       '#06b6d4',
}

export const sleep = (ms) => new Promise(r => setTimeout(r, ms))
