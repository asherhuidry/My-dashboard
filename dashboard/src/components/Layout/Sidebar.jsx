import { NavLink } from 'react-router-dom'
import { LayoutDashboard, Network, Compass, Globe, Brain } from 'lucide-react'

const nav = [
  { to:'/',              icon: LayoutDashboard,  label: 'Command',        desc: 'System overview' },
  { to:'/network',       icon: Network,          label: 'Graph',          desc: 'Market structure' },
  { to:'/intelligence',  icon: Brain,            label: 'Intelligence',   desc: 'Macro & analysis' },
  { to:'/discoveries',   icon: Compass,          label: 'Discoveries',    desc: 'Edges & signals' },
  { to:'/sources',       icon: Globe,            label: 'Sources',        desc: 'Data origins' },
]

function NavSection({ items }) {
  return (
    <div className="space-y-0.5">
      {items.map(({ to, icon: Icon, label, desc }) => (
        <NavLink
          key={to}
          to={to}
          end={to === '/'}
          className={({ isActive }) =>
            `flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm transition-all duration-200 group relative overflow-hidden ${
              isActive
                ? 'bg-accent/10 text-accent border border-accent/20 shadow-glow'
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
    </div>
  )
}

export default function Sidebar() {
  return (
    <aside className="fixed left-0 top-0 h-full w-16 lg:w-60 z-40 flex flex-col sidebar-bg">
      {/* Top accent line */}
      <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-accent/50 to-transparent" />

      {/* Logo */}
      <div className="flex items-center gap-3 px-4 py-5 border-b border-border relative">
        <div className="relative flex-shrink-0">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-accent/25 to-purple/15
            border border-accent/25 flex items-center justify-center shadow-glow">
            <Brain size={17} className="text-accent-glow" />
          </div>
          <div className="absolute -top-0.5 -right-0.5 w-2.5 h-2.5 rounded-full bg-positive border-2 border-bg-secondary" />
        </div>
        <div className="hidden lg:block overflow-hidden">
          <div className="text-sm font-bold gradient-text leading-tight">FinBrain</div>
          <div className="text-[9px] text-text-muted uppercase tracking-[0.15em] leading-tight mt-0.5">
            Market Graph Engine
          </div>
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 py-3 px-2 overflow-y-auto">
        <NavSection items={nav} />
      </nav>

      {/* Bottom status */}
      <div className="p-3 border-t border-border hidden lg:block">
        <div className="flex items-center gap-2 px-2">
          <div className="w-1.5 h-1.5 rounded-full bg-positive status-dot" style={{ background: '#10b981' }} />
          <span className="text-[10px] text-text-muted">Graph Engine Active</span>
        </div>
      </div>
    </aside>
  )
}
