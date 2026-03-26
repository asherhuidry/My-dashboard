import { useQuery } from '@tanstack/react-query'
import { fetchEvolution, fetchRoadmap } from '../lib/api'
import { fmt } from '../lib/utils'
import Header from '../components/Layout/Header'
import Spinner from '../components/UI/Spinner'
import { motion } from 'framer-motion'
import { Activity, CheckCircle, Clock, AlertCircle, Map } from 'lucide-react'

function StatusIcon({ status }) {
  if (status === 'completed' || status === 'success') return <CheckCircle size={13} className="text-positive flex-shrink-0" />
  if (status === 'failed'    || status === 'error')   return <AlertCircle  size={13} className="text-negative flex-shrink-0" />
  return <Clock size={13} className="text-warning flex-shrink-0" />
}

function LogEntry({ log, index }) {
  return (
    <motion.div
      initial={{ opacity:0, x:-10 }}
      animate={{ opacity:1, x:0 }}
      transition={{ delay: index * 0.02 }}
      className="flex gap-4 group"
    >
      <div className="flex flex-col items-center">
        <div className={`w-7 h-7 rounded-full flex items-center justify-center border ${
          log.status==='completed'||log.status==='success' ? 'border-positive/40 bg-positive/10'
          : log.status==='failed'||log.status==='error' ? 'border-negative/40 bg-negative/10'
          : 'border-warning/40 bg-warning/10'
        }`}>
          <StatusIcon status={log.status} />
        </div>
        <div className="flex-1 w-px bg-border mt-1 mb-1" />
      </div>
      <div className="pb-4 flex-1 min-w-0">
        <div className="flex items-start justify-between gap-2">
          <p className="text-sm text-text font-medium truncate">{log.agent_name ?? log.action ?? 'System'}</p>
          <span className="text-[10px] text-text-muted flex-shrink-0">{fmt.time(log.created_at ?? log.timestamp)}</span>
        </div>
        {log.summary && <p className="text-xs text-text-secondary mt-0.5 line-clamp-2">{log.summary}</p>}
        {log.details && <pre className="text-[10px] text-text-muted mt-1 font-mono bg-bg rounded p-2 overflow-x-auto max-h-16">{typeof log.details === 'object' ? JSON.stringify(log.details, null, 2) : log.details}</pre>}
      </div>
    </motion.div>
  )
}

const ROADMAP_PHASES = [
  { step:1,  label:'Folder structure',           done:true  },
  { step:2,  label:'Supabase tables',            done:true  },
  { step:3,  label:'TimescaleDB hypertables',    done:true  },
  { step:4,  label:'Qdrant collections',         done:true  },
  { step:5,  label:'Neo4j schema',               done:true  },
  { step:6,  label:'Data connectors',            done:true  },
  { step:7,  label:'Noise filter agent',         done:true  },
  { step:8,  label:'First data ingestion',       done:true  },
  { step:9,  label:'Feature engineering (74)',   done:true  },
  { step:10, label:'Train first LSTM',           done:false },
  { step:11, label:'First backtest',             done:false },
  { step:12, label:'FastAPI backend',            done:false },
  { step:13, label:'React dashboard',            done:true  },
  { step:14, label:'GitHub Actions workflows',   done:false },
  { step:15, label:'Master architect agent',     done:false },
]

export default function EvolutionLog() {
  const { data: logs,    isLoading: logsLoading  } = useQuery({ queryKey:['evolution'], queryFn: fetchEvolution,  staleTime: 30_000 })
  const { data: roadmap, isLoading: roadLoading  } = useQuery({ queryKey:['roadmap'],   queryFn: fetchRoadmap,    staleTime: 60_000 })

  const done  = ROADMAP_PHASES.filter(p => p.done).length
  const total = ROADMAP_PHASES.length
  const pct   = Math.round(done / total * 100)

  return (
    <div className="flex flex-col h-full">
      <Header title="Evolution Log" subtitle="Agent activity and build progress" />
      <div className="flex-1 overflow-y-auto p-6 bg-grid">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

          {/* Build progress */}
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <Map size={14} className="text-accent" />
              <h3 className="text-sm font-semibold text-text">Phase 1 Roadmap</h3>
              <span className="text-xs text-text-muted ml-auto">{done}/{total} complete</span>
            </div>

            {/* Progress bar */}
            <div className="glass rounded-xl p-4 border border-accent/20">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-text-muted">Overall progress</span>
                <span className="ticker-value text-lg font-bold text-accent">{pct}%</span>
              </div>
              <div className="h-2 bg-bg rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${pct}%` }}
                  transition={{ duration: 1, delay: 0.3 }}
                  className="h-full rounded-full bg-gradient-to-r from-accent to-cyan"
                />
              </div>
            </div>

            {/* Steps */}
            <div className="space-y-1">
              {ROADMAP_PHASES.map((step, i) => (
                <motion.div
                  key={step.step}
                  initial={{ opacity:0, x:-8 }}
                  animate={{ opacity:1, x:0 }}
                  transition={{ delay: i * 0.03 }}
                  className={`flex items-center gap-3 px-3 py-2 rounded-lg border transition-colors ${
                    step.done ? 'border-positive/20 bg-positive/5' : 'border-border bg-transparent'
                  }`}
                >
                  <div className={`w-5 h-5 rounded-full flex items-center justify-center text-[9px] font-bold border ${
                    step.done ? 'border-positive/40 bg-positive/20 text-positive' : 'border-border bg-bg text-text-muted'
                  }`}>{step.step}</div>
                  <span className={`text-xs flex-1 ${step.done ? 'text-text' : 'text-text-muted'}`}>{step.label}</span>
                  {step.done && <CheckCircle size={11} className="text-positive flex-shrink-0" />}
                </motion.div>
              ))}
            </div>
          </div>

          {/* Activity log */}
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <Activity size={14} className="text-accent" />
              <h3 className="text-sm font-semibold text-text">Activity Feed</h3>
            </div>

            {logsLoading && <Spinner text="Loading activity…" />}

            {logs?.logs?.length > 0 ? (
              <div className="space-y-0">
                {logs.logs.map((log, i) => <LogEntry key={log.id ?? i} log={log} index={i} />)}
              </div>
            ) : !logsLoading && (
              <div className="glass rounded-xl p-8 border border-border text-center">
                <Activity size={24} className="text-border mx-auto mb-2" />
                <p className="text-sm text-text-secondary">No agent activity yet</p>
                <p className="text-xs text-text-muted mt-1">Activity will appear here as agents run</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
