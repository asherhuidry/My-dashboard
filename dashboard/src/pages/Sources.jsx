/**
 * Sources page — source registry lifecycle visibility.
 *
 * Shows what the scout / probe / validation pipeline has discovered,
 * which sources are validated, and overall pipeline health.
 */
import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import {
  Globe, RefreshCw, Shield, ShieldCheck, ShieldX, Search,
  CheckCircle2, XCircle, AlertTriangle, Eye, Lock, Unlock,
} from 'lucide-react'
import { fetchSources, fetchSourceSummary } from '../lib/api'
import StatCard from '../components/UI/StatCard'

// ── Status styling ───────────────────────────────────────────────────────────

const STATUS_STYLE = {
  approved:    'text-green-400  bg-green-400/10  border-green-400/20',
  validated:   'text-blue-400   bg-blue-400/10   border-blue-400/20',
  sampled:     'text-cyan-400   bg-cyan-400/10   border-cyan-400/20',
  discovered:  'text-gray-400   bg-gray-400/10   border-gray-400/20',
  quarantined: 'text-yellow-400 bg-yellow-400/10 border-yellow-400/20',
  rejected:    'text-red-400    bg-red-400/10    border-red-400/20',
}

const STATUS_ICON = {
  approved:    CheckCircle2,
  validated:   ShieldCheck,
  sampled:     Eye,
  discovered:  Search,
  quarantined: AlertTriangle,
  rejected:    XCircle,
}

function StatusBadge({ status }) {
  const cls = STATUS_STYLE[status] || STATUS_STYLE.discovered
  const Icon = STATUS_ICON[status] || Search
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold border ${cls}`}>
      <Icon size={10} />
      {status}
    </span>
  )
}

// ── Score bar ────────────────────────────────────────────────────────────────

function ScoreBar({ score }) {
  const pct = Math.round(score * 100)
  const color = score >= 0.7 ? 'bg-green-400' : score >= 0.4 ? 'bg-blue-400' : 'bg-gray-400'
  const textColor = score >= 0.7 ? 'text-green-400' : score >= 0.4 ? 'text-blue-400' : 'text-gray-400'
  return (
    <div className="flex items-center gap-2">
      <span className={`font-mono text-xs ${textColor}`}>{pct}%</span>
      <div className="w-14 h-1.5 bg-bg-hover rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}

// ── Filter tabs ──────────────────────────────────────────────────────────────

const STATUS_OPTIONS = ['all', 'approved', 'validated', 'sampled', 'discovered', 'quarantined', 'rejected']

// ── Top sources mini-list ────────────────────────────────────────────────────

function TopList({ title, items, emptyText }) {
  if (!items || items.length === 0) {
    return (
      <div className="rounded-xl bg-bg-secondary border border-border p-4">
        <div className="text-xs text-text-muted uppercase tracking-wider font-semibold mb-3">{title}</div>
        <div className="text-xs text-text-muted">{emptyText}</div>
      </div>
    )
  }
  return (
    <div className="rounded-xl bg-bg-secondary border border-border p-4">
      <div className="text-xs text-text-muted uppercase tracking-wider font-semibold mb-3">{title}</div>
      <div className="space-y-2">
        {items.map((s, i) => (
          <div key={s.source_id} className="flex items-center justify-between">
            <div className="flex items-center gap-2 min-w-0">
              <span className="text-[10px] text-text-muted w-4">{i + 1}.</span>
              <span className="text-xs text-text font-medium truncate">{s.name}</span>
              <span className="text-[10px] text-text-muted">{s.category}</span>
            </div>
            <div className="flex items-center gap-2 flex-shrink-0">
              <StatusBadge status={s.status} />
              <ScoreBar score={s.score} />
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Main page ────────────────────────────────────────────────────────────────

export default function Sources() {
  const [statusFilter, setStatusFilter] = useState('all')
  const [searchQuery, setSearchQuery] = useState('')

  const queryParams = {}
  if (statusFilter !== 'all') queryParams.status = statusFilter
  if (searchQuery.trim()) queryParams.search = searchQuery.trim()
  queryParams.limit = 200

  const { data, isLoading, error, refetch } = useQuery({
    queryKey:  ['sources', statusFilter, searchQuery],
    queryFn:   () => fetchSources(queryParams),
    staleTime: 60_000,
  })

  const { data: summary } = useQuery({
    queryKey:  ['sources-summary'],
    queryFn:   fetchSourceSummary,
    staleTime: 60_000,
  })

  const rows = data?.sources ?? []
  const byStatus = summary?.by_status ?? {}

  return (
    <div className="h-full overflow-y-auto p-6 space-y-6">

      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-accent/15 border border-accent/25 flex items-center justify-center">
            <Globe size={16} className="text-accent" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-text">Sources</h1>
            <p className="text-xs text-text-muted">Data origins — governed sources feeding the market graph</p>
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
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
          <StatCard
            label="Total sources"
            value={summary.total}
            icon={Globe}
            sub="in registry"
            color="accent"
            delay={0}
          />
          <StatCard
            label="Approved"
            value={byStatus.approved ?? 0}
            icon={CheckCircle2}
            sub="active in pipeline"
            color="positive"
            delay={0.05}
          />
          <StatCard
            label="Validated"
            value={byStatus.validated ?? 0}
            icon={ShieldCheck}
            sub="passed quality checks"
            color="cyan"
            delay={0.1}
          />
          <StatCard
            label="Sampled"
            value={byStatus.sampled ?? 0}
            icon={Eye}
            sub="data fetched"
            color="purple"
            delay={0.15}
          />
          <StatCard
            label="Discovered"
            value={byStatus.discovered ?? 0}
            icon={Search}
            sub="awaiting probe"
            color="warning"
            delay={0.2}
          />
          <StatCard
            label="Rejected / Quarantined"
            value={(byStatus.rejected ?? 0) + (byStatus.quarantined ?? 0)}
            icon={ShieldX}
            sub="failed or flagged"
            color="negative"
            delay={0.25}
          />
        </div>
      )}

      {/* Top lists */}
      {summary && (summary.top_validated?.length > 0 || summary.top_scored?.length > 0) && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <TopList
            title="Top validated sources"
            items={summary.top_validated}
            emptyText="No validated sources yet"
          />
          <TopList
            title="Highest scored sources"
            items={summary.top_scored}
            emptyText="No scored sources yet"
          />
        </div>
      )}

      {/* Filters row */}
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex gap-1 rounded-xl bg-bg-secondary border border-border p-1">
          {STATUS_OPTIONS.map(s => (
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
              {s !== 'all' && byStatus[s] ? ` (${byStatus[s]})` : ''}
            </button>
          ))}
        </div>
        <div className="relative">
          <Search size={12} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-text-muted" />
          <input
            type="text"
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            placeholder="Search sources..."
            className="pl-7 pr-3 py-1.5 rounded-lg bg-bg-secondary border border-border text-xs text-text placeholder:text-text-muted focus:outline-none focus:border-accent/40 w-48"
          />
        </div>
      </div>

      {/* Table */}
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2, duration: 0.4 }}
        className="rounded-xl bg-bg-secondary border border-border overflow-hidden"
      >
        {isLoading ? (
          <div className="py-16 text-center text-text-muted text-sm">Loading sources...</div>
        ) : error ? (
          <div className="py-16 text-center text-red-400 text-sm">
            Could not load sources — is the API running?
          </div>
        ) : rows.length === 0 ? (
          <div className="py-16 text-center text-text-muted text-sm">
            {data?.registry_missing ? (
              <>No source registry found. Run the scout pipeline to populate it.</>
            ) : (
              <>
                No sources found{statusFilter !== 'all' ? ` with status "${statusFilter}"` : ''}{searchQuery ? ` matching "${searchQuery}"` : ''}.
                <div className="text-xs mt-2">
                  Sources are populated by the scout / probe / validation pipeline.
                </div>
              </>
            )}
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-border text-text-muted uppercase tracking-wider text-[9px]">
                  <th className="text-left px-4 py-3 font-semibold">Name</th>
                  <th className="text-left px-4 py-3 font-semibold">Category</th>
                  <th className="text-left px-4 py-3 font-semibold">Status</th>
                  <th className="text-left px-4 py-3 font-semibold">Score</th>
                  <th className="text-left px-4 py-3 font-semibold">Method</th>
                  <th className="text-left px-4 py-3 font-semibold">Auth</th>
                  <th className="text-left px-4 py-3 font-semibold">Free</th>
                  <th className="text-left px-4 py-3 font-semibold">Frequency</th>
                  <th className="text-left px-4 py-3 font-semibold">Last checked</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r, i) => (
                  <motion.tr
                    key={r.source_id}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: i * 0.015 }}
                    className="border-b border-border/50 hover:bg-bg-hover/40 transition-colors"
                  >
                    <td className="px-4 py-2.5">
                      <div className="font-semibold text-text">{r.name}</div>
                      <div className="text-[10px] text-text-muted font-mono">{r.source_id}</div>
                    </td>
                    <td className="px-4 py-2.5 text-text-secondary">{r.category}</td>
                    <td className="px-4 py-2.5"><StatusBadge status={r.status} /></td>
                    <td className="px-4 py-2.5"><ScoreBar score={r.reliability_score} /></td>
                    <td className="px-4 py-2.5 text-text-muted font-mono text-[10px]">{r.acquisition_method}</td>
                    <td className="px-4 py-2.5">
                      {r.auth_required
                        ? <Lock size={12} className="text-yellow-400" />
                        : <Unlock size={12} className="text-text-muted" />}
                    </td>
                    <td className="px-4 py-2.5">
                      {r.free_tier
                        ? <span className="text-green-400 text-[10px] font-semibold">FREE</span>
                        : <span className="text-text-muted text-[10px]">paid</span>}
                    </td>
                    <td className="px-4 py-2.5 text-text-muted text-[10px]">{r.update_frequency || '--'}</td>
                    <td className="px-4 py-2.5 text-text-muted text-[10px] font-mono">
                      {r.last_checked_at
                        ? new Date(r.last_checked_at).toLocaleDateString()
                        : '--'}
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </motion.div>

      {/* Footer count */}
      {rows.length > 0 && (
        <div className="text-[10px] text-text-muted text-right">
          Showing {rows.length} of {data?.total ?? '?'} sources
        </div>
      )}
    </div>
  )
}
