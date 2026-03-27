/**
 * Discoveries page — persisted correlation research output.
 *
 * Shows accumulated correlation findings from the weekly correlation hunter:
 * summary cards + filterable table of strongest relationships.
 */
import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import { Compass, RefreshCw, ArrowRight } from 'lucide-react'
import { fetchDiscoveries, fetchDiscoverySummary } from '../lib/api'
import StatCard from '../components/UI/StatCard'

// ── Strength badge ──────────────────────────────────────────────────────────

const STRENGTH_STYLE = {
  strong:   'text-green-400 bg-green-400/10 border-green-400/20',
  moderate: 'text-blue-400  bg-blue-400/10  border-blue-400/20',
  weak:     'text-gray-400  bg-gray-400/10  border-gray-400/20',
}

function StrengthBadge({ strength }) {
  const cls = STRENGTH_STYLE[strength] || STRENGTH_STYLE.weak
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-semibold border ${cls}`}>
      {strength}
    </span>
  )
}

// ── Correlation bar ─────────────────────────────────────────────────────────

function CorrelationBar({ value }) {
  const abs = Math.min(Math.abs(value), 1)
  const pct = (abs * 100).toFixed(0)
  const color = value > 0 ? 'bg-green-400' : 'bg-red-400'
  return (
    <div className="flex items-center gap-2">
      <span className={`font-mono text-xs ${value > 0 ? 'text-green-400' : 'text-red-400'}`}>
        {value > 0 ? '+' : ''}{value.toFixed(3)}
      </span>
      <div className="w-16 h-1.5 bg-bg-hover rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}

// ── Strength filter ─────────────────────────────────────────────────────────

const STRENGTH_OPTIONS = ['all', 'strong', 'moderate', 'weak']

// ── Main page ───────────────────────────────────────────────────────────────

export default function Discoveries() {
  const [strengthFilter, setStrengthFilter] = useState('strong')

  const queryParams = {}
  if (strengthFilter !== 'all') queryParams.strength = strengthFilter
  queryParams.limit = 100

  const { data, isLoading, error, refetch } = useQuery({
    queryKey:  ['discoveries', strengthFilter],
    queryFn:   () => fetchDiscoveries(queryParams),
    staleTime: 120_000,
  })

  const { data: summary } = useQuery({
    queryKey:  ['discoveries-summary'],
    queryFn:   fetchDiscoverySummary,
    staleTime: 120_000,
  })

  const rows = data?.discoveries ?? []

  return (
    <div className="h-full overflow-y-auto p-6 space-y-6">

      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-accent/15 border border-accent/25 flex items-center justify-center">
            <Compass size={16} className="text-accent" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-text">Discoveries</h1>
            <p className="text-xs text-text-muted">Persisted edges — measured correlations between market nodes</p>
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
          <StatCard
            label="Total discoveries"
            value={summary.total_discoveries}
            sub="accumulated findings"
            color="accent"
            delay={0}
          />
          <StatCard
            label="Strong"
            value={summary.by_strength?.strong ?? 0}
            sub={`of ${summary.total_discoveries} total`}
            color="positive"
            delay={0.05}
          />
          <StatCard
            label="Unique series"
            value={summary.unique_series}
            sub="assets & macro indicators"
            color="purple"
            delay={0.1}
          />
          <StatCard
            label="Discovery runs"
            value={summary.run_count}
            sub={summary.latest_run_id ? `latest: ${summary.latest_run_id.slice(0, 8)}...` : 'no runs yet'}
            color="cyan"
            delay={0.15}
          />
        </div>
      )}

      {/* Strength filter */}
      <div className="flex gap-1 rounded-xl bg-bg-secondary border border-border p-1 w-fit">
        {STRENGTH_OPTIONS.map(s => (
          <button
            key={s}
            onClick={() => setStrengthFilter(s)}
            className={`px-3 py-1 rounded-lg text-xs font-semibold transition-colors ${
              strengthFilter === s
                ? 'bg-accent text-white'
                : 'text-text-muted hover:text-text'
            }`}
          >
            {s.charAt(0).toUpperCase() + s.slice(1)}
          </button>
        ))}
      </div>

      {/* Table */}
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2, duration: 0.4 }}
        className="rounded-xl bg-bg-secondary border border-border overflow-hidden"
      >
        {isLoading ? (
          <div className="py-16 text-center text-text-muted text-sm">Loading discoveries...</div>
        ) : error ? (
          <div className="py-16 text-center text-red-400 text-sm">
            Could not load discoveries — is the API running?
          </div>
        ) : rows.length === 0 ? (
          <div className="py-16 text-center text-text-muted text-sm">
            No discoveries found{strengthFilter !== 'all' ? ` with strength "${strengthFilter}"` : ''}.
            <div className="text-xs mt-2">
              Discoveries accumulate from weekly correlation hunter runs.
            </div>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-border text-text-muted uppercase tracking-wider text-[9px]">
                  <th className="text-left px-4 py-3 font-semibold">Series A</th>
                  <th className="text-left px-1 py-3 font-semibold" />
                  <th className="text-left px-4 py-3 font-semibold">Series B</th>
                  <th className="text-left px-4 py-3 font-semibold">Pearson r</th>
                  <th className="text-left px-4 py-3 font-semibold">Lag</th>
                  <th className="text-left px-4 py-3 font-semibold">Granger p</th>
                  <th className="text-left px-4 py-3 font-semibold">Strength</th>
                  <th className="text-left px-4 py-3 font-semibold">Type</th>
                  <th className="text-left px-4 py-3 font-semibold">Regime</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r, i) => (
                  <motion.tr
                    key={r.id || `${r.series_a}-${r.series_b}-${i}`}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: i * 0.02 }}
                    className="border-b border-border/50 hover:bg-bg-hover/40 transition-colors"
                  >
                    <td className="px-4 py-2.5 font-semibold text-text font-mono">{r.series_a}</td>
                    <td className="px-1 py-2.5 text-text-muted"><ArrowRight size={10} /></td>
                    <td className="px-4 py-2.5 font-semibold text-text font-mono">{r.series_b}</td>
                    <td className="px-4 py-2.5"><CorrelationBar value={r.pearson_r} /></td>
                    <td className="px-4 py-2.5 text-text-secondary font-mono">
                      {r.lag_days}d
                    </td>
                    <td className="px-4 py-2.5 font-mono text-text-secondary">
                      {r.granger_p != null ? (
                        <span className={r.granger_p < 0.05 ? 'text-green-400' : 'text-text-muted'}>
                          {r.granger_p.toFixed(4)}
                        </span>
                      ) : (
                        <span className="text-text-muted">--</span>
                      )}
                    </td>
                    <td className="px-4 py-2.5"><StrengthBadge strength={r.strength} /></td>
                    <td className="px-4 py-2.5 text-text-muted text-[10px]">
                      {r.relationship_type === 'discovered' ? (
                        <span className="text-accent">discovered</span>
                      ) : (
                        r.relationship_type
                      )}
                    </td>
                    <td className="px-4 py-2.5 text-text-muted text-[10px]">{r.regime}</td>
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
          Showing {rows.length} of {data?.total ?? '?'} discoveries
        </div>
      )}
    </div>
  )
}
