import { useQuery } from '@tanstack/react-query'
import { fetchHealth } from '../../lib/api'
import { Clock, Wifi, WifiOff } from 'lucide-react'
import { useState, useEffect } from 'react'

function Clock24() {
  const [t, setT] = useState(new Date())
  useEffect(() => { const id = setInterval(() => setT(new Date()), 1000); return () => clearInterval(id) }, [])
  return (
    <span className="ticker-value text-xs text-text-secondary">
      {t.toUTCString().slice(17,25)} UTC
    </span>
  )
}

export default function Header({ title, subtitle }) {
  const { data: health } = useQuery({ queryKey: ['health'], queryFn: fetchHealth, refetchInterval: 30000 })
  const ok = health?.status === 'healthy'

  return (
    <header className="sticky top-0 z-30 flex items-center justify-between px-6 py-3 bg-bg-secondary/80 backdrop-blur border-b border-border">
      <div>
        <h1 className="text-lg font-semibold text-text">{title}</h1>
        {subtitle && <p className="text-xs text-text-muted">{subtitle}</p>}
      </div>

      <div className="flex items-center gap-4">
        <Clock24 />

        {/* System health indicators */}
        {health && (
          <div className="hidden md:flex items-center gap-2">
            {Object.entries(health.checks ?? {}).map(([k, v]) => (
              <div key={k} title={k} className={`w-1.5 h-1.5 rounded-full ${v ? 'bg-positive' : 'bg-negative'}`} />
            ))}
          </div>
        )}

        <div className={`flex items-center gap-1.5 text-xs px-2 py-1 rounded-md border ${ok ? 'text-positive border-positive/30 bg-positive/10' : 'text-negative border-negative/30 bg-negative/10'}`}>
          {ok ? <Wifi size={11} /> : <WifiOff size={11} />}
          <span>{ok ? 'Live' : 'Offline'}</span>
        </div>
      </div>
    </header>
  )
}
