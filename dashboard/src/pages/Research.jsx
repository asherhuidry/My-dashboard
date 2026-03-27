import { useState, useRef, useEffect } from 'react'
import { useMutation } from '@tanstack/react-query'
import { postResearch } from '../lib/api'
import Header from '../components/Layout/Header'
import SearchBar from '../components/UI/SearchBar'
import Spinner from '../components/UI/Spinner'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Brain, Search, Globe, Database, BarChart2, TrendingUp, Shield,
  Newspaper, Zap, ChevronDown, ChevronUp, Copy, CheckCheck,
  AlertTriangle, Activity,
} from 'lucide-react'

const SUGGESTIONS = [
  { label: "Full deep-dive analysis",     query: "Conduct a comprehensive analysis of $SYMBOL including technical signals, fundamentals, recent news, and key risks." },
  { label: "Bull vs Bear case",           query: "What is the bull case and bear case for $SYMBOL right now? Include specific data points." },
  { label: "Earnings outlook",            query: "Analyze $SYMBOL's upcoming earnings — what are analyst expectations, historical surprises, and key metrics to watch?" },
  { label: "Macro tailwinds/headwinds",   query: "What macro factors are currently tailwinds or headwinds for $SYMBOL?" },
  { label: "Sector comparison",           query: "How does $SYMBOL compare to its peers technically and fundamentally?" },
  { label: "Options market insight",      query: "What is the options market pricing in for $SYMBOL? Analyze put/call ratio and implied move." },
]

const QUICK_TOPICS = [
  "What sectors are showing strength in the current market?",
  "Summarize the current state of crypto markets.",
  "What is the current macro environment and how does it affect equities?",
  "Explain the Fear & Greed index and what it signals right now.",
  "What are the most important economic indicators to watch this week?",
]

const SOURCE_ICONS = {
  "yfinance+features": BarChart2,
  "news+web":          Newspaper,
  "sec+yfinance":      Database,
  "web_search":        Globe,
  "macro":             Activity,
  "yfinance":          TrendingUp,
}

const TOOL_LABELS = {
  get_technical_analysis: "Technical Analysis",
  get_news:               "News & Sentiment",
  get_fundamentals:       "Fundamental Data",
  web_search:             "Web Search",
  get_macro_context:      "Macro Context",
}

function SourceBadge({ source }) {
  const Icon = SOURCE_ICONS[source.type] ?? Database
  return (
    <div className="flex items-center gap-1.5 text-[9px] px-2 py-1 rounded-lg
      border border-border bg-bg-hover text-text-muted">
      <Icon size={9} className="text-accent/70" />
      {TOOL_LABELS[source.tool] ?? source.tool}
    </div>
  )
}

function ReportView({ report, sources, dataUsed, symbol, query, tokens }) {
  const [copied, setCopied] = useState(false)
  const [showMeta, setShowMeta] = useState(false)

  const copyReport = () => {
    navigator.clipboard.writeText(report)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  // Convert markdown to styled sections
  const renderReport = (text) => {
    const lines = text.split('\n')
    return lines.map((line, i) => {
      if (line.startsWith('## ')) {
        return (
          <h3 key={i} className="text-sm font-bold text-text mt-5 mb-2 pb-1 border-b border-border/50 flex items-center gap-2">
            <div className="w-1 h-4 bg-accent rounded-full" />
            {line.replace('## ', '')}
          </h3>
        )
      }
      if (line.startsWith('### ')) {
        return <h4 key={i} className="text-xs font-semibold text-text-secondary mt-3 mb-1">{line.replace('### ', '')}</h4>
      }
      if (line.startsWith('- ') || line.startsWith('* ')) {
        return (
          <div key={i} className="flex items-start gap-2 text-xs text-text-secondary leading-relaxed mb-1">
            <div className="w-1 h-1 rounded-full bg-accent/60 mt-1.5 flex-shrink-0" />
            <span>{line.replace(/^[-*] /, '').replace(/\*\*(.*?)\*\*/g, '<b>$1</b>')}</span>
          </div>
        )
      }
      if (line.trim() === '') return <div key={i} className="h-1" />
      // Bold text
      const bolded = line.replace(/\*\*(.*?)\*\*/g, (_, m) => `<strong class="text-text font-semibold">${m}</strong>`)
      return (
        <p key={i} className="text-xs text-text-secondary leading-relaxed mb-1.5"
          dangerouslySetInnerHTML={{ __html: bolded }} />
      )
    })
  }

  return (
    <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}
      className="glass-card rounded-2xl border border-border overflow-hidden">

      {/* Report header */}
      <div className="flex items-center justify-between px-5 py-4 border-b border-border bg-bg-secondary/40">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-xl bg-purple/10 border border-purple/20 flex items-center justify-center">
            <Brain size={15} className="text-purple" />
          </div>
          <div>
            <div className="text-sm font-semibold text-text">
              {symbol ? `Research Report — ${symbol}` : 'Research Report'}
            </div>
            <div className="text-[10px] text-text-muted mt-0.5 max-w-sm truncate">{query}</div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={() => setShowMeta(s => !s)}
            className="text-[10px] px-2 py-1 rounded-lg border border-border text-text-muted hover:text-text transition-colors flex items-center gap-1">
            <Zap size={9} /> {tokens.toLocaleString()} tokens
            {showMeta ? <ChevronUp size={9} /> : <ChevronDown size={9} />}
          </button>
          <button onClick={copyReport}
            className="text-[10px] px-2.5 py-1 rounded-lg border border-border text-text-muted hover:text-text transition-colors flex items-center gap-1.5">
            {copied ? <><CheckCheck size={10} className="text-positive" /> Copied</> : <><Copy size={10} /> Copy</>}
          </button>
        </div>
      </div>

      {/* Meta: tools used */}
      <AnimatePresence>
        {showMeta && (
          <motion.div initial={{ height: 0 }} animate={{ height: 'auto' }} exit={{ height: 0 }}
            className="overflow-hidden border-b border-border bg-bg-secondary/20">
            <div className="px-5 py-3">
              <div className="text-[9px] text-text-muted uppercase tracking-wider mb-2">Data Sources Used</div>
              <div className="flex flex-wrap gap-1.5">
                {sources.map((s, i) => <SourceBadge key={i} source={s} />)}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Report body */}
      <div className="px-6 py-5 max-h-[65vh] overflow-y-auto">
        {renderReport(report)}
      </div>
    </motion.div>
  )
}

export default function Research() {
  const [symbol, setSymbol]   = useState('')
  const [query,  setQuery]    = useState('')
  const [depth,  setDepth]    = useState('standard')
  const inputRef              = useRef(null)

  const mutation = useMutation({
    mutationFn: ({ symbol, query, depth }) => postResearch({ symbol, query, depth }),
  })

  const run = (q = query) => {
    if (!q.trim()) return
    const finalQuery = q.replace('$SYMBOL', symbol || 'the market')
    mutation.mutate({ symbol: symbol || null, query: finalQuery, depth })
  }

  const handleKey = (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) run()
  }

  return (
    <div className="flex flex-col h-full">
      <Header title="AI Research" subtitle="Multi-source intelligence — Claude + live data + web search" />

      <div className="flex-1 overflow-y-auto bg-grid p-5 space-y-4">

        {/* Input panel */}
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}
          className="glass-card rounded-2xl border border-border p-5 relative overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-br from-purple/5 via-transparent to-accent/5 pointer-events-none" />
          <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-purple/40 to-transparent" />

          <div className="relative">
            <div className="flex items-center gap-2 mb-4">
              <div className="w-8 h-8 rounded-xl bg-purple/10 border border-purple/20 flex items-center justify-center">
                <Brain size={15} className="text-purple" />
              </div>
              <div>
                <div className="text-sm font-semibold text-text">FinBrain Intelligence Engine</div>
                <div className="text-[10px] text-text-muted">
                  Powered by Claude Sonnet · Real-time data · Web search · Multi-source synthesis
                </div>
              </div>
            </div>

            {/* Symbol + depth row */}
            <div className="flex items-center gap-3 mb-3 flex-wrap">
              <div className="flex-1 min-w-[180px]">
                <SearchBar
                  onSelect={setSymbol}
                  placeholder="Symbol (optional)"
                  compact
                />
              </div>
              <div className="flex items-center gap-1 bg-bg/60 rounded-lg border border-border p-0.5">
                {[
                  { d: 'quick',    label: 'Quick',    sub: '~30s'  },
                  { d: 'standard', label: 'Standard', sub: '~60s'  },
                  { d: 'deep',     label: 'Deep',     sub: '~2min' },
                ].map(({ d, label, sub }) => (
                  <button key={d} onClick={() => setDepth(d)}
                    className={`px-2.5 py-1.5 rounded-md text-[10px] font-semibold transition-all flex flex-col items-center ${
                      depth === d ? 'bg-purple/20 text-purple border border-purple/30' : 'text-text-muted hover:text-text'
                    }`}>
                    <span>{label}</span>
                    <span className="text-[8px] opacity-60">{sub}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* Query textarea */}
            <div className="relative">
              <textarea
                ref={inputRef}
                value={query}
                onChange={e => setQuery(e.target.value)}
                onKeyDown={handleKey}
                placeholder="Ask anything about a stock, market, sector, strategy, or macro environment…"
                className="w-full bg-bg border border-border rounded-xl px-4 py-3 text-sm text-text
                  placeholder-text-muted resize-none focus:outline-none focus:border-purple/50
                  focus:bg-bg-hover transition-all"
                rows={3}
              />
              <div className="absolute bottom-2 right-3 flex items-center gap-2">
                <span className="text-[9px] text-text-muted">Ctrl+Enter to run</span>
              </div>
            </div>

            {/* Run button */}
            <div className="flex items-center justify-between mt-3">
              <div className="flex items-center gap-2 flex-wrap">
                {[Globe, Database, BarChart2, Newspaper, Activity].map((Icon, i) => (
                  <div key={i} className="flex items-center gap-1 text-[9px] text-text-muted">
                    <Icon size={9} className="text-accent/50" />
                    {['Web Search', 'Fundamentals', 'Technical', 'News', 'Macro'][i]}
                  </div>
                ))}
              </div>
              <button
                onClick={() => run()}
                disabled={!query.trim() || mutation.isPending}
                className="flex items-center gap-2 px-4 py-2 rounded-xl font-semibold text-sm
                  bg-gradient-to-r from-purple/80 to-accent/80 text-white border border-purple/40
                  hover:from-purple hover:to-accent disabled:opacity-40 disabled:cursor-not-allowed
                  shadow-glow transition-all"
              >
                {mutation.isPending
                  ? <><Activity size={13} className="animate-spin" /> Researching…</>
                  : <><Search size={13} /> Research</>
                }
              </button>
            </div>
          </div>
        </motion.div>

        {/* Quick suggestions */}
        {!mutation.data && !mutation.isPending && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.1 }}>
            <div className="text-[10px] text-text-muted uppercase tracking-wider mb-2 flex items-center gap-1.5">
              <Zap size={9} /> Quick research templates
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
              {SUGGESTIONS.map(s => (
                <button key={s.label}
                  onClick={() => { setQuery(s.query); inputRef.current?.focus() }}
                  className="text-left px-3 py-2.5 rounded-xl border border-border hover:border-purple/30
                    bg-bg-card hover:bg-purple/5 text-[10px] text-text-secondary hover:text-text
                    transition-all group">
                  <div className="font-semibold group-hover:text-purple transition-colors mb-0.5">{s.label}</div>
                  <div className="text-text-muted line-clamp-1 text-[9px]">{s.query.slice(0, 60)}…</div>
                </button>
              ))}
            </div>

            <div className="text-[10px] text-text-muted uppercase tracking-wider mt-4 mb-2 flex items-center gap-1.5">
              <Globe size={9} /> Market-wide questions
            </div>
            <div className="flex flex-wrap gap-2">
              {QUICK_TOPICS.map(t => (
                <button key={t} onClick={() => { setQuery(t); run(t) }}
                  className="text-[10px] px-2.5 py-1.5 rounded-lg border border-border hover:border-accent/40
                    text-text-muted hover:text-accent bg-bg-card hover:bg-accent/5 transition-all">
                  {t.length > 50 ? t.slice(0, 50) + '…' : t}
                </button>
              ))}
            </div>
          </motion.div>
        )}

        {/* Loading state */}
        {mutation.isPending && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
            className="glass-card rounded-2xl border border-purple/20 p-8 text-center">
            <div className="flex flex-col items-center gap-4">
              <div className="w-14 h-14 rounded-2xl bg-purple/10 border border-purple/20 flex items-center justify-center">
                <Brain size={24} className="text-purple animate-pulse" />
              </div>
              <div>
                <p className="text-sm font-semibold text-text mb-1">FinBrain is researching…</p>
                <p className="text-xs text-text-muted">
                  Calling tools: technical analysis · news · fundamentals · web search
                </p>
              </div>
              <div className="flex items-center gap-2 flex-wrap justify-center">
                {[Globe, Database, BarChart2, Newspaper, Activity].map((Icon, i) => (
                  <div key={i} className="flex items-center gap-1 px-2 py-1 rounded-lg bg-bg border border-border text-[9px] text-text-muted">
                    <Icon size={9} className="text-purple/70 animate-pulse" style={{ animationDelay: `${i * 0.3}s` }} />
                    {['Web Search','Fundamentals','Technicals','News','Macro'][i]}
                  </div>
                ))}
              </div>
            </div>
          </motion.div>
        )}

        {/* Error state */}
        {mutation.isError && (
          <div className="glass rounded-xl p-5 border border-negative/25 flex items-start gap-3">
            <AlertTriangle size={16} className="text-negative mt-0.5 flex-shrink-0" />
            <div>
              <p className="text-sm font-medium text-negative">Research failed</p>
              <p className="text-xs text-text-muted mt-1">{mutation.error?.message}</p>
            </div>
          </div>
        )}

        {/* Result */}
        {mutation.data && (
          <ReportView
            report={mutation.data.report}
            sources={mutation.data.sources}
            dataUsed={mutation.data.data_used}
            symbol={mutation.data.symbol}
            query={mutation.data.query}
            tokens={mutation.data.tokens}
          />
        )}

      </div>
    </div>
  )
}
