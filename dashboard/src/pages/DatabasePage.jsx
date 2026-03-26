import { useQuery } from '@tanstack/react-query'
import { fetchDbStats } from '../lib/api'
import { fmt } from '../lib/utils'
import Header from '../components/Layout/Header'
import Spinner from '../components/UI/Spinner'
import { motion } from 'framer-motion'
import { Database, CheckCircle, XCircle, Table } from 'lucide-react'

const TABLE_DESCRIPTIONS = {
  prices:        'OHLCV price data for all assets',
  macro_events:  'FRED macro indicator releases',
  evolution_log: 'Agent activity and system changes',
  roadmap:       'Development tasks and milestones',
  signals:       'Generated trading signals',
  model_registry:'Trained ML model versions',
  api_sources:   'External data source registry',
  system_health: 'System performance metrics',
  agent_runs:    'Individual agent execution records',
  quarantine:    'Flagged anomalous data records',
}

const TABLE_COLORS = {
  prices:        'accent',
  macro_events:  'positive',
  evolution_log: 'purple',
  signals:       'warning',
  model_registry:'cyan',
  agent_runs:    'purple',
  quarantine:    'negative',
}

const C = {
  accent:   { border:'border-accent/20',   bg:'bg-accent/10',   text:'text-accent'   },
  positive: { border:'border-positive/20', bg:'bg-positive/10', text:'text-positive' },
  purple:   { border:'border-purple/20',   bg:'bg-purple/10',   text:'text-purple'   },
  warning:  { border:'border-warning/20',  bg:'bg-warning/10',  text:'text-warning'  },
  cyan:     { border:'border-cyan/20',     bg:'bg-cyan/10',     text:'text-cyan'     },
  negative: { border:'border-negative/20', bg:'bg-negative/10', text:'text-negative' },
  default:  { border:'border-border',      bg:'bg-bg-hover',    text:'text-text'     },
}

function TableCard({ t, index }) {
  const c = C[TABLE_COLORS[t.table] ?? 'default']
  const isOk = t.status === 'ok'

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.04 }}
      className={`glass rounded-xl p-5 border ${c.border} hover:border-opacity-60 transition-all`}
    >
      <div className="flex items-start justify-between mb-3">
        <div className={`w-9 h-9 rounded-lg ${c.bg} flex items-center justify-center`}>
          <Table size={14} className={c.text} />
        </div>
        {isOk
          ? <CheckCircle size={14} className="text-positive" />
          : <XCircle    size={14} className="text-negative" />
        }
      </div>

      <h3 className="text-sm font-semibold text-text mb-0.5">{t.table.replace(/_/g,' ')}</h3>
      <p className="text-[10px] text-text-muted mb-3 leading-relaxed">{TABLE_DESCRIPTIONS[t.table] ?? ''}</p>

      <div className="flex items-end justify-between">
        <div>
          <div className={`ticker-value text-2xl font-bold ${c.text}`}>{fmt.compact(t.rows)}</div>
          <div className="text-[10px] text-text-muted">rows</div>
        </div>
        <div className={`text-xs px-2 py-1 rounded-md ${c.bg} ${c.text} border ${c.border}`}>
          {isOk ? 'ONLINE' : 'ERROR'}
        </div>
      </div>

      {!isOk && t.error && (
        <p className="mt-2 text-[10px] text-negative truncate">{t.error}</p>
      )}
    </motion.div>
  )
}

export default function DatabasePage() {
  const { data, isLoading, refetch } = useQuery({
    queryKey:    ['db-stats'],
    queryFn:     fetchDbStats,
    staleTime:   60_000,
    refetchInterval: 30_000,
  })

  const totalRows = data?.total_rows ?? 0
  const okTables  = data?.tables?.filter(t => t.status === 'ok').length ?? 0
  const allTables = data?.tables?.length ?? 0

  return (
    <div className="flex flex-col h-full">
      <Header title="Database" subtitle="Live Supabase table stats" />
      <div className="flex-1 overflow-y-auto p-6 space-y-6 bg-grid">

        {/* Summary bar */}
        <motion.div
          initial={{ opacity:0, y:10 }}
          animate={{ opacity:1, y:0 }}
          className="grid grid-cols-3 gap-4"
        >
          <div className="glass rounded-xl p-4 border border-accent/20">
            <p className="text-xs text-text-muted uppercase tracking-wider mb-1">Total Rows</p>
            <p className="ticker-value text-3xl font-bold text-accent">{fmt.compact(totalRows)}</p>
          </div>
          <div className="glass rounded-xl p-4 border border-positive/20">
            <p className="text-xs text-text-muted uppercase tracking-wider mb-1">Tables Online</p>
            <p className="ticker-value text-3xl font-bold text-positive">{okTables}/{allTables}</p>
          </div>
          <div className="glass rounded-xl p-4 border border-border">
            <p className="text-xs text-text-muted uppercase tracking-wider mb-1">Database</p>
            <p className="text-sm font-semibold text-text mt-2">Supabase<br /><span className="text-xs text-text-muted font-normal">TimescaleDB extension</span></p>
          </div>
        </motion.div>

        {isLoading && <div className="flex justify-center py-12"><Spinner size="lg" text="Loading database stats…" /></div>}

        {data?.tables && (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {data.tables.map((t, i) => <TableCard key={t.table} t={t} index={i} />)}
          </div>
        )}

        <div className="text-center">
          <button onClick={() => refetch()} className="text-xs text-text-muted hover:text-text border border-border px-3 py-1.5 rounded-lg hover:bg-bg-hover transition-colors">
            Refresh
          </button>
        </div>
      </div>
    </div>
  )
}
