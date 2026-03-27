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
      <div className="w-1.5 h-1.5 rounded-full bg-positive pulse-dot" />
      <span className="ticker-value text-xs text-text-secondary">
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
      bg-bg-secondary/80 backdrop-blur-xl border-b border-border">
      {/* Left: title */}
      <div className="flex items-center gap-4">
        <div>
          <h1 className="text-base font-semibold text-text leading-tight">{title}</h1>
          {subtitle && <p className="text-[10px] text-text-muted leading-tight mt-0.5">{subtitle}</p>}
        </div>
      </div>

      {/* Right: status + clock */}
      <div className="flex items-center gap-5">
        {/* Per-service health dots */}
        {Object.keys(checks).length > 0 && (
          <div className="hidden lg:flex items-center gap-3">
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

        {/* Overall status pill */}
        <div className={`flex items-center gap-1.5 text-[10px] font-medium px-2.5 py-1 rounded-full border transition-all ${
          ok
            ? 'text-positive border-positive/30 bg-positive/8'
            : 'text-negative border-negative/30 bg-negative/8'
        }`}>
          {ok ? <Wifi size={10} /> : <WifiOff size={10} />}
          <span>{ok ? 'Live' : 'Offline'}</span>
        </div>
      </div>
    </header>
  )
}
