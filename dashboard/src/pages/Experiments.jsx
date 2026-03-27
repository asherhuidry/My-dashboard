/**
 * Experiments page — minimal table view of experiment registry entries.
 *
 * Shows all training runs with their status, model type, key metrics, and
 * backtest results.  Read-only: no training is triggered from the UI.
 */
import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { motion, AnimatePresence } from 'framer-motion'
import {
  FlaskConical, ChevronDown, ChevronUp, RefreshCw,
  CheckCircle2, XCircle, Clock, Star, Archive,
} from 'lucide-react'
import { fetchExperiments, fetchExperimentSummary } from '../lib/api'

// ── Status badge ──────────────────────────────────────────────────────────────

const STATUS_CONFIG = {
  running:   { label: 'Running',   color: 'text-yellow-400 bg-yellow-400/10 border-yellow-400/20', Icon: Clock },
  completed: { label: 'Completed', color: 'text-blue-400  bg-blue-400/10  border-blue-400/20',   Icon: CheckCircle2 },
  failed:    { label: 'Failed',    color: 'text-red-400   bg-red-400/10   border-red-400/20',     Icon: XCircle },
  promoted:  { label: 'Promoted',  color: 'text-green-400 bg-green-400/10 border-green-400/20',  Icon: Star },
  archived:  { label: 'Archived',  color: 'text-gray-400  bg-gray-400/10  border-gray-400/20',   Icon: Archive },
}

function StatusBadge({ status }) {
  const cfg = STATUS_CONFIG[status] || STATUS_CONFIG.running
  const { label, color, Icon } = cfg
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold border ${color}`}>
      <Icon size={9} />
      {label}
    </span>
  )
}

// ── Metric cell ───────────────────────────────────────────────────────────────

function MetricPill({ label, value, pct = false }) {
  if (value == null) return <span className="text-text-muted text-xs">—</span>
  const display = pct ? `${(value * 100).toFixed(1)}%` : value.toFixed(3)
  return (
    <span className="inline-block text-xs font-mono text-text-secondary">
      <span className="text-text-muted text-[10px] mr-1">{label}</span>{display}
    </span>
  )
}

// ── Expanded detail row ───────────────────────────────────────────────────────

function DetailRow({ exp }) {
  const bt = exp.backtest
  return (
    <motion.tr
      initial={{ opacity: 0, height: 0 }}
      animate={{ opacity: 1, height: 'auto' }}
      exit={{ opacity: 0, height: 0 }}
      className="bg-bg-hover/30"
    >
      <td colSpan={7} className="px-6 py-4">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">

          {/* Hyperparams */}
          <div>
            <div className="text-text-muted uppercase tracking-wider text-[9px] mb-2">Hyperparameters</div>
            <div className="space-y-1">
              {Object.entries(exp.hyperparams || {}).slice(0, 8).map(([k, v]) => (
                <div key={k} className="flex justify-between">
                  <span className="text-text-muted">{k}</span>
                  <span className="text-text-secondary font-mono">{JSON.stringify(v)}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Dataset info */}
          <div>
            <div className="text-text-muted uppercase tracking-wider text-[9px] mb-2">Dataset</div>
            <div className="space-y-1">
              {Object.entries(exp.dataset_info || {}).slice(0, 8).map(([k, v]) => (
                <div key={k} className="flex justify-between">
                  <span className="text-text-muted">{k}</span>
                  <span className="text-text-secondary font-mono text-[10px]">
                    {Array.isArray(v) ? `[${v.length} items]` : String(v)}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Backtest */}
          <div>
            <div className="text-text-muted uppercase tracking-wider text-[9px] mb-2">Backtest</div>
            {bt ? (
              <div className="space-y-1">
                {[
                  ['Period', `${bt.period_start?.slice(0,10)} → ${bt.period_end?.slice(0,10)}`],
                  ['Return', `${((bt.cumulative_return||0)*100).toFixed(1)}%`],
                  ['Benchmark', `${((bt.benchmark_return||0)*100).toFixed(1)}%`],
                  ['Hit rate', `${((bt.hit_rate||0)*100).toFixed(1)}%`],
                  ['Max DD', `${((bt.max_drawdown||0)*100).toFixed(1)}%`],
                  ['Sharpe', (bt.sharpe||0).toFixed(2)],
                  ['Trades', bt.trade_count],
                ].map(([k, v]) => (
                  <div key={k} className="flex justify-between">
                    <span className="text-text-muted">{k}</span>
                    <span className="text-text-secondary font-mono">{v}</span>
                  </div>
                ))}
              </div>
            ) : (
              <span className="text-text-muted">No backtest attached</span>
            )}
          </div>
        </div>

        {/* Notes / checkpoint */}
        {(exp.notes || exp.checkpoint_path) && (
          <div className="mt-3 pt-3 border-t border-border/50 flex gap-6 text-[10px] text-text-muted">
            {exp.checkpoint_path && (
              <span>
                <span className="text-text-muted mr-1">Checkpoint:</span>
                <span className="font-mono text-text-secondary">{exp.checkpoint_path.split(/[/\\]/).pop()}</span>
              </span>
            )}
            {exp.notes && (
              <span>
                <span className="text-text-muted mr-1">Notes:</span>
                <span className="text-text-secondary">{exp.notes.slice(0, 120)}{exp.notes.length > 120 ? '…' : ''}</span>
              </span>
            )}
          </div>
        )}
      </td>
    </motion.tr>
  )
}

// ── Summary cards ─────────────────────────────────────────────────────────────

function SummaryCard({ label, value, sub }) {
  return (
    <div className="rounded-xl bg-bg-secondary border border-border p-4">
      <div className="text-2xl font-bold text-text">{value ?? '—'}</div>
      <div className="text-xs font-semibold text-text-secondary mt-1">{label}</div>
      {sub && <div className="text-[10px] text-text-muted mt-0.5">{sub}</div>}
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────

const STATUS_FILTERS = ['all', 'running', 'completed', 'promoted', 'failed', 'archived']

export default function Experiments() {
  const [statusFilter, setStatusFilter] = useState('all')
  const [modelFilter,  setModelFilter]  = useState('')
  const [expandedId,   setExpandedId]   = useState(null)

  const params = {}
  if (statusFilter !== 'all') params.status = statusFilter
  if (modelFilter)            params.model_type = modelFilter

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['experiments', statusFilter, modelFilter],
    queryFn:  () => fetchExperiments({ ...params, limit: 100 }),
    refetchInterval: 30_000,
  })

  const { data: summary } = useQuery({
    queryKey: ['experiments-summary'],
    queryFn:  fetchExperimentSummary,
    refetchInterval: 30_000,
  })

  const experiments = data?.experiments ?? []

  return (
    <div className="h-full overflow-y-auto p-6 space-y-6">

      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-accent/15 border border-accent/25 flex items-center justify-center">
            <FlaskConical size={16} className="text-accent" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-text">Experiments</h1>
            <p className="text-xs text-text-muted">Model training runs · evaluation · backtest history</p>
          </div>
        </div>
        <button
          onClick={() => refetch()}
          className="p-2 rounded-lg border border-border hover:border-border-bright text-text-muted hover:text-text transition-colors"
        >
          <RefreshCw size={14} />
        </button>
      </div>

      {/* Summary cards */}
      {summary && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <SummaryCard label="Total runs"    value={summary.total}          sub="all time" />
          <SummaryCard label="Promoted"      value={summary.promoted}       sub="production candidates" />
          <SummaryCard label="With backtest" value={summary.with_backtest}  sub="evaluated" />
          <SummaryCard
            label="By model"
            value={Object.keys(summary.by_model ?? {}).join(', ') || '—'}
            sub="model families"
          />
        </div>
      )}

      {/* Filters */}
      <div className="flex flex-wrap gap-2">
        <div className="flex gap-1 rounded-xl bg-bg-secondary border border-border p-1">
          {STATUS_FILTERS.map(s => (
            <button
              key={s}
              onClick={() => setStatusFilter(s)}
              className={`px-3 py-1 rounded-lg text-xs font-semibold transition-colors ${
                statusFilter === s
                  ? 'bg-accent text-white'
                  : 'text-text-muted hover:text-text'
              }`}
            >
              {s.charAt(0).toUpperCase() + s.slice(1)}
            </button>
          ))}
        </div>
        <input
          type="text"
          placeholder="Filter by model type…"
          value={modelFilter}
          onChange={e => setModelFilter(e.target.value)}
          className="px-3 py-1.5 rounded-xl bg-bg-secondary border border-border text-xs text-text placeholder:text-text-muted focus:outline-none focus:border-accent/50"
        />
      </div>

      {/* Table */}
      <div className="rounded-xl bg-bg-secondary border border-border overflow-hidden">
        {isLoading ? (
          <div className="py-16 text-center text-text-muted text-sm">Loading experiments…</div>
        ) : error ? (
          <div className="py-16 text-center text-red-400 text-sm">
            Could not load experiments — is the API running?
          </div>
        ) : experiments.length === 0 ? (
          <div className="py-16 text-center text-text-muted text-sm">
            No experiments found.
            <div className="text-xs mt-2">
              Run <code className="font-mono bg-bg-hover px-1.5 py-0.5 rounded">python -m ml.patterns.train_mlp --symbol AAPL</code> to create one.
            </div>
          </div>
        ) : (
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-border text-text-muted uppercase tracking-wider text-[9px]">
                <th className="text-left px-4 py-3 font-semibold">Name</th>
                <th className="text-left px-4 py-3 font-semibold">Model</th>
                <th className="text-left px-4 py-3 font-semibold">Status</th>
                <th className="text-left px-4 py-3 font-semibold">Accuracy</th>
                <th className="text-left px-4 py-3 font-semibold">Backtest</th>
                <th className="text-left px-4 py-3 font-semibold">Started</th>
                <th className="px-4 py-3" />
              </tr>
            </thead>
            <tbody>
              {experiments.map(exp => {
                const isExpanded = expandedId === exp.experiment_id
                const bt = exp.backtest
                return (
                  <>
                    <tr
                      key={exp.experiment_id}
                      className="border-b border-border/50 hover:bg-bg-hover/40 cursor-pointer transition-colors"
                      onClick={() => setExpandedId(isExpanded ? null : exp.experiment_id)}
                    >
                      <td className="px-4 py-3">
                        <div className="font-semibold text-text leading-tight truncate max-w-[180px]" title={exp.name}>
                          {exp.name}
                        </div>
                        <div className="text-[9px] text-text-muted font-mono mt-0.5 truncate">
                          {exp.experiment_id.slice(0, 8)}…
                        </div>
                        {exp.tags?.length > 0 && (
                          <div className="flex gap-1 mt-1 flex-wrap">
                            {exp.tags.slice(0, 3).map(t => (
                              <span key={t} className="text-[9px] px-1.5 py-0.5 rounded bg-accent/8 text-accent/70 border border-accent/15">
                                {t}
                              </span>
                            ))}
                          </div>
                        )}
                      </td>
                      <td className="px-4 py-3 font-mono text-text-secondary uppercase text-[10px]">
                        {exp.model_type}
                      </td>
                      <td className="px-4 py-3">
                        <StatusBadge status={exp.status} />
                      </td>
                      <td className="px-4 py-3">
                        <MetricPill label="acc" value={exp.metrics?.accuracy} pct />
                        <div className="mt-0.5">
                          <MetricPill label="auc" value={exp.metrics?.auc} pct />
                        </div>
                      </td>
                      <td className="px-4 py-3">
                        {bt ? (
                          <>
                            <MetricPill label="ret" value={bt.cumulative_return} pct />
                            <div className="mt-0.5">
                              <MetricPill label="hit" value={bt.hit_rate} pct />
                            </div>
                          </>
                        ) : (
                          <span className="text-text-muted">—</span>
                        )}
                      </td>
                      <td className="px-4 py-3 text-text-muted text-[10px] font-mono">
                        {exp.started_at?.slice(0, 16).replace('T', ' ')}
                      </td>
                      <td className="px-4 py-3 text-text-muted">
                        {isExpanded
                          ? <ChevronUp size={12} />
                          : <ChevronDown size={12} />}
                      </td>
                    </tr>
                    <AnimatePresence>
                      {isExpanded && <DetailRow key={`detail-${exp.experiment_id}`} exp={exp} />}
                    </AnimatePresence>
                  </>
                )
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
