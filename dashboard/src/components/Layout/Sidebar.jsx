import { NavLink } from 'react-router-dom'
import { LayoutDashboard, LineChart, Database, Activity, Network, Brain } from 'lucide-react'

const nav = [
  { to:'/',          icon: LayoutDashboard, label: 'Overview'   },
  { to:'/analyze',   icon: LineChart,       label: 'Analyzer'   },
  { to:'/network',   icon: Network,         label: 'Network'    },
  { to:'/database',  icon: Database,        label: 'Database'   },
  { to:'/evolution', icon: Activity,        label: 'Evolution'  },
]

export default function Sidebar() {
  return (
    <aside className="fixed left-0 top-0 h-full w-16 lg:w-56 z-40 flex flex-col bg-bg-secondary border-r border-border">
      {/* Logo */}
      <div className="flex items-center gap-3 px-4 py-5 border-b border-border">
        <div className="relative flex-shrink-0">
          <div className="w-8 h-8 rounded-lg bg-accent/20 border border-accent/40 flex items-center justify-center animate-glow">
            <Brain size={16} className="text-accent" />
          </div>
        </div>
        <div className="hidden lg:block">
          <div className="text-sm font-semibold text-text neon-blue">FinBrain</div>
          <div className="text-[10px] text-text-muted uppercase tracking-widest">Intelligence</div>
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 py-4 space-y-1 px-2">
        {nav.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-all duration-200 group ${
                isActive
                  ? 'bg-accent/15 text-accent border border-accent/25 shadow-glow'
                  : 'text-text-secondary hover:text-text hover:bg-bg-hover'
              }`
            }
          >
            <Icon size={16} className="flex-shrink-0" />
            <span className="hidden lg:block font-medium">{label}</span>
          </NavLink>
        ))}
      </nav>

      {/* Footer */}
      <div className="p-3 border-t border-border hidden lg:block">
        <div className="text-[10px] text-text-muted text-center uppercase tracking-widest">
          Phase 1 · Build {import.meta.env.VITE_BUILD ?? '1.0'}
        </div>
      </div>
    </aside>
  )
}
