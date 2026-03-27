import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard, LineChart, Database, Activity, Network,
  Brain, SlidersHorizontal, Cpu, FlaskConical, Search, Zap, TestTube2,
} from 'lucide-react'

const nav = [
  { to:'/',              icon: LayoutDashboard,  label: 'Overview',      desc: 'System status'   },
  { to:'/analyze',       icon: LineChart,         label: 'Analyzer',      desc: 'Deep analysis'   },
  { to:'/intelligence',  icon: Zap,               label: 'Intelligence',  desc: 'Macro · supply chain' },
  { to:'/research',      icon: Search,            label: 'Research',      desc: 'AI intelligence' },
  { to:'/experiments',   icon: TestTube2,          label: 'Experiments',   desc: 'Model runs'      },
  { to:'/backtest',      icon: FlaskConical,      label: 'Backtest',      desc: 'Signal testing'  },
  { to:'/screener',      icon: SlidersHorizontal, label: 'Screener',      desc: 'Asset scanner'   },
  { to:'/network',       icon: Network,           label: 'Network',       desc: 'Data graph'      },
  { to:'/database',      icon: Database,          label: 'Database',      desc: 'DB explorer'     },
  { to:'/evolution',     icon: Activity,          label: 'Evolution',     desc: 'Agent logs'      },
]

export default function Sidebar() {
  return (
    <aside className="fixed left-0 top-0 h-full w-16 lg:w-60 z-40 flex flex-col
      bg-bg-secondary border-r border-border overflow-hidden">

      {/* Top accent line */}
      <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-accent/60 to-transparent" />

      {/* Logo */}
      <div className="flex items-center gap-3 px-4 py-5 border-b border-border relative">
        <div className="relative flex-shrink-0">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-accent/30 to-purple/20
            border border-accent/30 flex items-center justify-center animate-glow shadow-glow">
            <Brain size={17} className="text-accent" />
          </div>
          <div className="absolute -top-0.5 -right-0.5 w-2.5 h-2.5 rounded-full bg-positive border-2 border-bg-secondary" />
        </div>
        <div className="hidden lg:block overflow-hidden">
          <div className="text-sm font-bold gradient-text leading-tight">FinBrain</div>
          <div className="text-[9px] text-text-muted uppercase tracking-[0.15em] leading-tight mt-0.5">
            Intelligence v1
          </div>
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 py-3 space-y-0.5 px-2 overflow-y-auto">
        {nav.map(({ to, icon: Icon, label, desc }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm transition-all duration-200 group relative overflow-hidden ${
                isActive
                  ? 'bg-accent/12 text-accent border border-accent/20 shadow-glow'
                  : 'text-text-secondary hover:text-text hover:bg-bg-hover border border-transparent'
              }`
            }
          >
            {({ isActive }) => (
              <>
                {isActive && (
                  <div className="absolute inset-0 bg-gradient-to-r from-accent/8 to-transparent pointer-events-none" />
                )}
                <div className={`relative flex-shrink-0 w-7 h-7 rounded-lg flex items-center justify-center transition-all ${
                  isActive
                    ? 'bg-accent/20 border border-accent/30'
                    : 'bg-bg-hover/50 border border-border group-hover:border-border-bright'
                }`}>
                  <Icon size={14} className={isActive ? 'text-accent' : 'text-text-muted group-hover:text-text-secondary'} />
                </div>
                <div className="hidden lg:block overflow-hidden">
                  <div className={`text-xs font-semibold leading-tight transition-colors ${
                    isActive ? 'text-accent' : 'text-text-secondary group-hover:text-text'
                  }`}>{label}</div>
                  <div className="text-[9px] text-text-muted leading-tight mt-0.5">{desc}</div>
                </div>
                {isActive && (
                  <div className="absolute right-2 hidden lg:block">
                    <div className="w-1 h-1 rounded-full bg-accent" />
                  </div>
                )}
              </>
            )}
          </NavLink>
        ))}
      </nav>

      {/* Bottom: model badge */}
      <div className="p-3 border-t border-border hidden lg:block">
        <div className="rounded-xl p-3 bg-gradient-to-br from-accent/6 to-purple/6 border border-accent/15">
          <div className="flex items-center gap-2 mb-1.5">
            <Cpu size={11} className="text-accent/70" />
            <span className="text-[9px] font-semibold text-text-muted uppercase tracking-wider">AI Engine</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-1.5 h-1.5 rounded-full bg-positive pulse-dot" />
            <span className="text-[10px] text-text-secondary font-medium">Claude Sonnet Active</span>
          </div>
          <div className="text-[9px] text-text-muted mt-1">Research · Chat · Predictions</div>
        </div>
      </div>
    </aside>
  )
}
