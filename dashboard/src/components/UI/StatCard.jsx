import { motion } from 'framer-motion'

export default function StatCard({ label, value, sub, icon: Icon, color = 'accent', trend, delay = 0 }) {
  const colors = {
    accent:   { border: 'border-accent/20',   bg: 'bg-accent/10',   text: 'text-accent',   glow: 'glow-blue'  },
    positive: { border: 'border-positive/20', bg: 'bg-positive/10', text: 'text-positive', glow: 'glow-green' },
    negative: { border: 'border-negative/20', bg: 'bg-negative/10', text: 'text-negative', glow: 'glow-red'   },
    purple:   { border: 'border-purple/20',   bg: 'bg-purple/10',   text: 'text-purple',   glow: ''           },
    cyan:     { border: 'border-cyan/20',      bg: 'bg-cyan/10',     text: 'text-cyan',     glow: ''           },
    warning:  { border: 'border-warning/20',  bg: 'bg-warning/10',  text: 'text-warning',  glow: ''           },
  }
  const c = colors[color] ?? colors.accent

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.4 }}
      className={`glass rounded-xl p-4 border ${c.border} ${c.glow}`}
    >
      <div className="flex items-start justify-between mb-3">
        <p className="text-xs text-text-muted uppercase tracking-wider font-medium">{label}</p>
        {Icon && (
          <div className={`w-7 h-7 rounded-lg ${c.bg} flex items-center justify-center`}>
            <Icon size={13} className={c.text} />
          </div>
        )}
      </div>
      <div className="ticker-value text-2xl font-semibold text-text">{value}</div>
      {sub && <p className="mt-1 text-xs text-text-muted">{sub}</p>}
      {trend != null && (
        <p className={`mt-1 text-xs font-medium ${trend > 0 ? 'text-positive' : trend < 0 ? 'text-negative' : 'text-text-muted'}`}>
          {trend > 0 ? '▲' : trend < 0 ? '▼' : '—'} {Math.abs(trend).toFixed(2)}%
        </p>
      )}
    </motion.div>
  )
}
