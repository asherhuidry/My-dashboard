import { useQuery } from '@tanstack/react-query'
import { fetchHealth } from '../../lib/api'
import { Wifi, WifiOff, Activity, Server, Database, Brain, Cpu } from 'lucide-react'
import { useState, useEffect } from 'react'

function LiveClock() {
  const [t, setT] = useState(new Date())
  useEffect(() => { const id = setInterval(() => setT(new Date()), 1000); return () => clearInterval(id) }, [])
  const pad = n => String(n).padStart(2,'0')
  return (
    <div className="flex items-center gap-2">
      <div className="w-1.5 h-1.5 rounded-full bg-positive status-dot" style={{ background: '#10b981' }} />
      <span className="ticker-value text-[10px] text-text-secondary">
        {pad(t.getUTCHours())}:{pad(t.getUTCMinutes())}:{pad(t.getUTCSeconds())}{' '}
        <span className="text-text-muted">UTC</span>
      </span>
    </div>
  )
}

const SERVICE_ICONS = { supabase: Database, qdrant: Brain, neo4j: Activity, yfinance: Cpu, api: Server }
const SERVICE_LABELS = { supabase: 'DB', qdrant: 'Vec', neo4j: 'Graph', yfinance: 'Feed', api: 'API' }

export default function Header({ title, subtitle }) {
  const { data: health } = useQuery({ queryKey: ['health'], queryFn: fetchHealth, refetchInterval: 30_000 })
  const ok = health?.status === 'healthy'
  const checks = health?.checks ?? {}

  return (
    <header className="sticky top-0 z-30 flex items-center justify-between px-6 py-3
      bg-bg-secondary/85 backdrop-blur-2xl border-b border-border">
      {/* Left: title */}
      <div className="flex items-center gap-4">
        <div>
          <h1 className="text-sm font-semibold text-text leading-tight">{title}</h1>
          {subtitle && <p className="text-[9px] text-text-muted leading-tight mt-0.5">{subtitle}</p>}
        </div>
      </div>

      {/* Right: status + clock */}
      <div className="flex items-center gap-4">
        {Object.keys(checks).length > 0 && (
          <div className="hidden lg:flex items-center gap-2.5">
            {Object.entries(checks).map(([k, v]) => {
              const Icon = SERVICE_ICONS[k] ?? Activity
              return (
                <div key={k} title={`${k}: ${v ? 'online' : 'offline'}`}
                  className="flex items-center gap-1 text-[9px] uppercase tracking-wider">
                  <Icon size={10} className={v ? 'text-positive' : 'text-negative'} />
                  <span className={v ? 'text-positive/70' : 'text-negative/70'}>
                    {SERVICE_LABELS[k] ?? k}
                  </span>
                </div>
              )
            })}
          </div>
        )}

        <div className="h-4 w-px bg-border hidden lg:block" />

        <LiveClock />

        <div className={`flex items-center gap-1.5 text-[9px] font-medium px-2 py-1 rounded-full border transition-all ${
          ok
            ? 'text-positive border-positive/25 bg-positive/6'
            : 'text-negative border-negative/25 bg-negative/6'
        }`}>
          {ok ? <Wifi size={9} /> : <WifiOff size={9} />}
          <span>{ok ? 'Live' : 'Offline'}</span>
        </div>
      </div>
    </header>
  )
}
