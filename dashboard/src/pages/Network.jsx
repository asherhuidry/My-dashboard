import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { fetchCorrelations, fetchKnowledge } from '../lib/api'
import Header from '../components/Layout/Header'
import ForceGraph from '../components/Network/ForceGraph'
import Spinner from '../components/UI/Spinner'
import { motion } from 'framer-motion'
import { GitBranch, Network as NetworkIcon, Sliders, Info } from 'lucide-react'

const TABS = [
  { id:'corr',  label:'Correlation Network', icon: NetworkIcon },
  { id:'know',  label:'Knowledge Graph',     icon: GitBranch },
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

export default function Network() {
  const [tab, setTab]           = useState('corr')
  const [days, setDays]         = useState(63)
  const [threshold, setThreshold] = useState(0.3)
  const [selectedNode, setSelectedNode] = useState(null)

  const { data: corrData, isLoading: corrLoading } = useQuery({
    queryKey: ['correlations', days, threshold],
    queryFn:  () => fetchCorrelations(days, threshold),
    staleTime: 300_000,
    enabled:   tab === 'corr',
  })

  const { data: knowData, isLoading: knowLoading } = useQuery({
    queryKey: ['knowledge'],
    queryFn:  fetchKnowledge,
    staleTime: 600_000,
    enabled:   tab === 'know',
  })

  const loading = tab === 'corr' ? corrLoading : knowLoading
  const graphData = tab === 'corr' ? corrData : knowData

  return (
    <div className="flex flex-col h-full">
      <Header title="Network" subtitle="Asset correlations · Knowledge graph · Data relationships" />
      <div className="flex-1 flex flex-col overflow-hidden">

        {/* Tab bar + controls */}
        <div className="px-6 py-3 border-b border-border bg-bg-secondary/50 flex items-center gap-6 flex-wrap">
          <div className="flex items-center gap-1">
            {TABS.map(t => (
              <button key={t.id} onClick={() => setTab(t.id)}
                className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg border transition-all ${tab===t.id?'border-accent/40 bg-accent/10 text-accent':'border-border text-text-muted hover:text-text'}`}>
                <t.icon size={11} /> {t.label}
              </button>
            ))}
          </div>
          {tab === 'corr' && (
            <CorrelationControls days={days} setDays={setDays} threshold={threshold} setThreshold={setThreshold} />
          )}
        </div>

        {/* Graph area */}
        <div className="flex-1 flex overflow-hidden">
          <div className="flex-1 relative">
            {loading ? (
              <div className="flex items-center justify-center h-full">
                <Spinner size="lg" text="Building network graph…" />
              </div>
            ) : graphData ? (
              <ForceGraph
                nodes={graphData.nodes}
                edges={graphData.edges}
                onNodeClick={setSelectedNode}
              />
            ) : (
              <div className="flex items-center justify-center h-full text-text-muted text-sm">No data</div>
            )}

            {/* Stats overlay */}
            {graphData && !loading && (
              <div className="absolute top-4 left-4 glass rounded-lg px-3 py-2 text-xs space-y-0.5">
                <div className="text-text-muted uppercase tracking-wider text-[9px] mb-1">Graph Stats</div>
                <div className="flex items-center gap-1.5"><span className="text-text-secondary">Nodes:</span><span className="ticker-value text-text">{graphData.nodes?.length}</span></div>
                <div className="flex items-center gap-1.5"><span className="text-text-secondary">Edges:</span><span className="ticker-value text-text">{graphData.edges?.length}</span></div>
                {tab === 'corr' && graphData.meta && (
                  <div className="flex items-center gap-1.5"><span className="text-text-secondary">Window:</span><span className="ticker-value text-text">{graphData.meta.days}D</span></div>
                )}
                {tab === 'know' && graphData.source && (
                  <div className="flex items-center gap-1.5"><span className="text-text-secondary">Source:</span><span className="ticker-value text-accent capitalize">{graphData.source}</span></div>
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
                  {tab === 'corr'
                    ? 'Edges show rolling correlations. Thick = stronger. Green = positive, Red = inverse.'
                    : 'Arrows show data flow and relationships in the FinBrain knowledge graph.'}
                </p>
              </div>
            </motion.div>
          )}
        </div>
      </div>
    </div>
  )
}
