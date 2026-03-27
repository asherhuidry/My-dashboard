import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { fetchCorrelations, fetchKnowledge, fetchGraphIntelligence } from '../lib/api'
import Header from '../components/Layout/Header'
import ForceGraph from '../components/Network/ForceGraph'
import Spinner from '../components/UI/Spinner'
import { motion } from 'framer-motion'
import { GitBranch, Network as NetworkIcon, AlertTriangle, TrendingUp, Zap } from 'lucide-react'

const TABS = [
  { id:'graph', label:'Market Graph',  icon: GitBranch },
  { id:'corr',  label:'Correlations',  icon: NetworkIcon },
]

function CorrelationControls({ days, setDays, threshold, setThreshold }) {
  return (
    <div className="flex items-center gap-4 flex-wrap">
      <div className="flex items-center gap-2">
        <span className="text-xs text-text-muted">Window:</span>
        {[21,63,126,252].map(d => (
          <button key={d} onClick={() => setDays(d)}
            className={`text-xs px-2 py-1 rounded border transition-all ${days===d?'border-accent/40 bg-accent/10 text-accent':'border-border text-text-muted hover:text-text'}`}>
            {d}D
          </button>
        ))}
      </div>
      <div className="flex items-center gap-2">
        <span className="text-xs text-text-muted">Min |corr|:</span>
        <input type="range" min="0" max="0.9" step="0.05" value={threshold}
          onChange={e => setThreshold(parseFloat(e.target.value))}
          className="w-24 accent-accent" />
        <span className="ticker-value text-xs text-accent w-6">{threshold.toFixed(2)}</span>
      </div>
    </div>
  )
}

const PRIORITY_STYLES = {
  1: { border: 'border-negative/30', bg: 'bg-negative/6', icon: 'text-negative' },
  2: { border: 'border-warning/30',  bg: 'bg-warning/6',  icon: 'text-warning' },
  3: { border: 'border-accent/30',   bg: 'bg-accent/6',   icon: 'text-accent' },
}

function InsightIcon({ type }) {
  if (type.includes('anomaly'))  return <AlertTriangle size={11} />
  if (type.includes('bridge'))   return <Zap size={11} />
  return <TrendingUp size={11} />
}

function InsightsPanel({ insights }) {
  if (!insights?.length) return null
  const top = insights.slice(0, 5)
  return (
    <div className="border-t border-border bg-bg-secondary/60 px-5 py-3">
      <div className="text-[9px] text-text-muted uppercase tracking-wider font-semibold mb-2">Top Structural Insights</div>
      <div className="flex gap-2 overflow-x-auto pb-1">
        {top.map((ins, i) => {
          const s = PRIORITY_STYLES[ins.priority] ?? PRIORITY_STYLES[3]
          return (
            <div key={i} className={`flex-shrink-0 rounded-lg px-3 py-2 border ${s.border} ${s.bg} max-w-[280px]`}>
              <div className="flex items-center gap-1.5 mb-0.5">
                <span className={s.icon}><InsightIcon type={ins.type} /></span>
                <span className="text-[10px] font-semibold text-text truncate">{ins.title}</span>
              </div>
              <div className="text-[9px] text-text-muted truncate">{ins.detail}</div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default function Network() {
  const [tab, setTab]             = useState('graph')
  const [days, setDays]           = useState(63)
  const [threshold, setThreshold] = useState(0.3)
  const [selectedNode, setSelectedNode] = useState(null)

  const { data: graphData, isLoading: graphLoading } = useQuery({
    queryKey: ['knowledge'],
    queryFn:  fetchKnowledge,
    staleTime: 600_000,
    enabled:   tab === 'graph',
  })

  const { data: corrData, isLoading: corrLoading } = useQuery({
    queryKey: ['correlations', days, threshold],
    queryFn:  () => fetchCorrelations(days, threshold),
    staleTime: 300_000,
    enabled:   tab === 'corr',
  })

  const { data: intelligence } = useQuery({
    queryKey: ['graph-intelligence'],
    queryFn:  fetchGraphIntelligence,
    staleTime: 600_000,
  })

  const loading = tab === 'graph' ? graphLoading : corrLoading
  const data    = tab === 'graph' ? graphData : corrData
  const isEmpty = data && data.nodes?.length === 0

  return (
    <div className="flex flex-col h-full">
      <Header title="Market Graph" subtitle="Live market structure from Neo4j" />
      <div className="flex-1 flex flex-col overflow-hidden">

        {/* Tab bar + controls */}
        <div className="px-5 py-2.5 border-b border-border bg-bg-secondary/50 flex items-center gap-5 flex-wrap">
          <div className="flex items-center gap-1">
            {TABS.map(t => (
              <button key={t.id} onClick={() => { setTab(t.id); setSelectedNode(null) }}
                className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg border transition-all ${tab===t.id?'border-accent/40 bg-accent/10 text-accent':'border-border text-text-muted hover:text-text'}`}>
                <t.icon size={11} /> {t.label}
              </button>
            ))}
          </div>
          {tab === 'corr' && (
            <CorrelationControls days={days} setDays={setDays} threshold={threshold} setThreshold={setThreshold} />
          )}
          {tab === 'graph' && data?.source && (
            <span className={`text-[10px] px-2 py-0.5 rounded border ${
              data.source === 'neo4j'
                ? 'border-positive/30 text-positive bg-positive/8'
                : 'border-text-muted/30 text-text-muted bg-bg-hover'
            }`}>
              {data.source === 'neo4j' ? 'Live from Neo4j' : 'Neo4j unavailable'}
            </span>
          )}
        </div>

        {/* Graph area */}
        <div className="flex-1 flex overflow-hidden">
          <div className="flex-1 relative">
            {loading ? (
              <div className="flex items-center justify-center h-full">
                <Spinner size="lg" text="Loading graph..." />
              </div>
            ) : isEmpty ? (
              <div className="flex flex-col items-center justify-center h-full text-text-muted gap-3">
                <GitBranch size={32} className="text-text-muted/40" />
                <div className="text-sm">No graph data available</div>
                <div className="text-xs text-text-muted/60 max-w-sm text-center">
                  {tab === 'graph'
                    ? 'Neo4j is not connected or the graph has not been materialized yet. Run the discovery pipeline to populate the market graph.'
                    : 'Could not fetch correlation data.'}
                </div>
              </div>
            ) : data ? (
              <ForceGraph
                nodes={data.nodes}
                edges={data.edges}
                onNodeClick={setSelectedNode}
              />
            ) : null}

            {/* Stats overlay */}
            {data && !loading && !isEmpty && (
              <div className="absolute top-4 left-4 glass rounded-lg px-3 py-2 text-xs space-y-0.5">
                <div className="flex items-center gap-1.5"><span className="text-text-secondary">Nodes:</span><span className="ticker-value text-text">{data.nodes?.length}</span></div>
                <div className="flex items-center gap-1.5"><span className="text-text-secondary">Edges:</span><span className="ticker-value text-text">{data.edges?.length}</span></div>
                {tab === 'corr' && data.meta && (
                  <div className="flex items-center gap-1.5"><span className="text-text-secondary">Window:</span><span className="ticker-value text-text">{data.meta.days}D</span></div>
                )}
              </div>
            )}
          </div>

          {/* Node detail panel */}
          {selectedNode && (
            <motion.div
              initial={{ x: 300, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              className="w-64 border-l border-border bg-bg-secondary p-4 space-y-3 overflow-y-auto"
            >
              <div className="flex items-start justify-between">
                <h3 className="text-sm font-semibold text-text">{selectedNode.label ?? selectedNode.id}</h3>
                <button onClick={() => setSelectedNode(null)} className="text-text-muted hover:text-text text-xs">✕</button>
              </div>

              {selectedNode.type && (
                <div>
                  <span className="text-[10px] text-text-muted uppercase tracking-wider">Type</span>
                  <p className="text-sm text-text capitalize mt-0.5">{selectedNode.type}</p>
                </div>
              )}
              {selectedNode.class && (
                <div>
                  <span className="text-[10px] text-text-muted uppercase tracking-wider">Asset Class</span>
                  <p className="text-sm text-text capitalize mt-0.5">{selectedNode.class}</p>
                </div>
              )}
              {selectedNode.price != null && (
                <div>
                  <span className="text-[10px] text-text-muted uppercase tracking-wider">Price</span>
                  <p className="ticker-value text-lg font-semibold text-text mt-0.5">${selectedNode.price}</p>
                </div>
              )}
              {selectedNode.ret_1d != null && (
                <div>
                  <span className="text-[10px] text-text-muted uppercase tracking-wider">1D Return</span>
                  <p className={`ticker-value text-sm font-medium mt-0.5 ${selectedNode.ret_1d>=0?'text-positive':'text-negative'}`}>
                    {selectedNode.ret_1d >= 0 ? '+' : ''}{selectedNode.ret_1d}%
                  </p>
                </div>
              )}

              <div className="pt-2 border-t border-border">
                <p className="text-[10px] text-text-muted">
                  {tab === 'graph'
                    ? 'Edges show factor sensitivities and correlations discovered in the market graph.'
                    : 'Edges show rolling return correlations. Thick = stronger. Green = positive, Red = inverse.'}
                </p>
              </div>
            </motion.div>
          )}
        </div>

        {/* Insights strip */}
        {intelligence?.insights && <InsightsPanel insights={intelligence.insights} />}
      </div>
    </div>
  )
}
